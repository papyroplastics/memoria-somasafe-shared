# Backend server internals

How the FastAPI gateway, Celery worker and storage fit together. See
[submission-type.md](submission-type.md) for the upload paths and
[versioning.md](versioning.md) for what each version field means.

## Gateway + worker split

The server is a FastAPI **gateway** in front of a Celery **worker**; the gateway never
runs ML work directly, so it stays fast to start (no TensorFlow import). Both run on the
host with `uv`; only Postgres and Redis run in containers. They don't need a shared
filesystem — only the same database — which is what makes the worker actually
distributable instead of assuming a co-located disk.

- **Gateway (`api/`):** serves model artifacts (from the DB, keyed by the active
  `GlobalWeights` row), accepts weight-delta uploads, persists them, enqueues worker jobs,
  and exposes a result endpoint the client polls.
- **Worker (`worker/tasks.py`):** restores uploaded weights into the model, converts it to
  a signed int8 `.tflite`, validates submit-only uploads, and runs the daily federated
  aggregation. Each forked worker child builds every available model post-fork (TensorFlow
  isn't fork-safe once initialized, so building it in the parent would deadlock every
  child).

## Storage decisions (thesis scope, no production deployment)

- **PostgreSQL** holds weight submissions, quantization jobs, and every served blob
  (`WeightsArtifact`, `QuantizationResult`, `Firmware.data`), each in its own table keyed
  by the row that owns it — blobs are read almost never, owning rows constantly, so they
  don't sit in the same table. Object storage (MinIO) was tried and dropped: the database
  is already the shared state both processes reach, so a bucket only added an untransacted
  second source of truth (upload between `flush()`/`commit()`, orphaned objects on a failed
  commit, a `stat` call needed on every download route). Blobs stay within what Postgres
  handles comfortably at this scale (a few MB to ~13 MB per artifact).
- **Redis** is the Celery broker and also backs the result cache (so a caller can await a
  queued task's return value) and rate limiting.
- **Everything served is zstd-compressed**, stored compressed and served as-is (the client
  decompresses). Signatures cover the raw bytes, so the server compresses after signing.
- **The gateway always proxies** — it reads the blob server-side and returns it, so auth
  and rate limiting apply to every byte served and no presigned URL bypasses the download
  cooldown.

**Result lifecycle.** A `done` quantization result is served on request and stamped
`served_at`; a Celery beat sweep deletes it (and nulls the job's signature) once a served
result is older than `SERVE_GRACE_SECONDS` (5 min) or an unclaimed one older than
`RESULT_TTL_SECONDS` (1 h), flipping the job to `expired`. Weight submissions themselves
are never reaped.

## Federated aggregation

A beat task runs one round per initialized model, on `FED_AGG_INTERVAL_SECONDS` (default
24 h):

1. **Strategy** — chosen by the latest version's `submission_type` (see
   [submission-type.md](submission-type.md)); a type with no strategy is skipped.
2. **Window** — the deltas whose `base_weights_id` is the version's active snapshot, one
   per client (latest submission wins).
3. **Validation** — a structural check (weight count matches, buffer finite), normally
   already cached by the quantize/validate tasks as submissions arrive.
4. **Round threshold** — fewer than `FED_MIN_SUBMISSIONS` (default 1) valid submissions
   skips the model until next round.
5. **Aggregation** — `trimmed_mean` drops the smallest/largest `FED_TRIM_RATIO` (default
   0.1) fraction of each coordinate before averaging the rest; this is the round's only
   Byzantine defense. Uniform weighting (not weighted by claimed dataset size) removes the
   incentive to lie about how much data was trained on.
6. **Artifact baking** — the averaged weights are restored into the cached model and both
   serving artifacts (trainable + signed int8) are re-exported and committed in the same
   transaction as the new `GlobalWeights` row, so a visible snapshot always has its
   artifacts.
7. **Rate-limit reset** — a successful round clears the model's download/submission
   counters for every user.

If a round makes a model worse, flipping its `valid` flag to false rolls clients back to
the previous snapshot atomically (artifacts + weights together). Schema changes are
handled by wiping the database and re-running the seed script — no migrations.

A round can be queued by hand: `uv run -m scripts.integration.queue_aggregation [model]`.

**Headless federated run.** `scripts/integration/fed_client.py` drives the whole stack
over the real HTTP API: downloads the trainable artifact once, then per subject (as user
`test_N`) logs in, pulls the current weight buffer, trains one local pass, uploads the
update, and logs out — then queues a round, waits for the new weights, and scores it on
the held-out subjects, repeating for `--rounds`. See
[secure-aggregation.md](secure-aggregation.md) for the masked variant of this same script.

```bash
uv run -m scripts.system.seed_db --test-users
uv run -m scripts.integration.fed_client feature-ae --rounds 5 --eval-subjects 2
```

## Auth & rate limiting

See [authentication.md](authentication.md) for sessions and
[device-attestation.md](device-attestation.md) for device ownership. All `/model/*`
routes require login; the rate-limited ones additionally require verified device
ownership. Rate limiting is per-user, per-resource, enforced two-phase (checked up front,
spent only after the work happens, so a rejected request never counts against quota):

| Endpoint | Limit |
|----------|-------|
| `GET /model/list`, `GET /model/versions/{key}` | authed only |
| `GET /model/download/{trainable,quantized}/{key}` | device-owner; one per model per `DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) |
| `GET /model/weights/{key}` | device-owner; same cooldown, separate counter from the artifact download |
| `POST /model/submit/quantize/{key}/{weights_id}` | device-owner; `QUANTIZE_DAILY_LIMIT` (default 2) per model per rolling 24 h |
| `POST /model/submit/raw/{key}/{weights_id}` | device-owner; `SUBMIT_DAILY_LIMIT` (default 2) per model per rolling 24 h |
| `GET /model/quantize/result/{job_id}` | authed; only the submitting user |
| `POST /model/secure/join/{key}` | device-owner; `404` unless the model is `secure`-typed |
| `POST /model/secure/submit/{round_id}` | device-owner + round member; one vector per member |
| `GET /ota/download/{interface}/{version}` | device-owner; one per interface per `OTA_DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) |

Accounts are seeded, not self-registered: `uv run -m scripts.system.seed_db` bootstraps a
default user and the model registry rows (idempotent). `--reseed` re-points each seeded
model at the artifacts currently on disk, dropping its `GlobalWeights` history and
everything anchored to it — the way to re-seed after changing a model rather than merely
retraining one.

## Model download endpoints

| Method | Path | Response |
|--------|------|----------|
| GET | `/model/download/{trainable,quantized}/{key}` | the artifact, keyed by the version's active snapshot; echoes fingerprint/version/weights headers (quantized also carries the signature + contract version) |
| GET | `/model/weights/{key}` | just the flat weight buffer of the same active snapshot — lighter than re-pulling the whole trainable artifact when only the weights moved |

Both are zstd-compressed bodies; the client decompresses before verifying/using them.
