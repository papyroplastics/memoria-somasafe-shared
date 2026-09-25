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
- **Worker (`worker/`):** quantizes and signs uploads, runs the aggregation rounds and
  sweeps expired results. Tasks go to a `light` queue (dispatcher, secure-round sweep,
  cleanup) or a `heavy` one (quantization, aggregation), so cheap SQL never waits behind a
  TFLite conversion. Each forked child builds a model on first use and caches it (variants
  sharing an architecture share one graph) — post-fork, since TensorFlow isn't fork-safe
  once initialized. Beat runs as its own single-replica process (`make beat-run`); inside
  a scaled-out worker it would fire every tick once per replica.

## Worker task pattern

Every task **claims** its work (committed at once), **reads** what it needs into plain
values, **computes** with no database session open, **writes** the result and the row's
terminal status in one transaction, then runs **effects** that are safe to lose (clearing
rate limits, dispatching follow-ups). A row with a lifecycle (`QuantizationJob`,
`SecureRound`) is its own lock: every status change is a conditional `UPDATE … WHERE
status = :from` whose row count says whether it happened (`ClaimableModel`), so a
redelivered task finds its row already claimed and returns. An exception after the claim
records a failure verdict; a worker killed outright can't, so the periodic sweeps fail any
row claimed for longer than the hard time limit ("worker lost"). Delivery is at-least-once
(late acks, one prefetched task per process).

## Storage decisions (thesis scope, no production deployment)

- **PostgreSQL** holds weight submissions, quantization jobs, and every served blob
  (`WeightsArtifact`, `QuantizationResult`, `Firmware.data`), each in its own table keyed
  by the row that owns it — blobs are read almost never, owning rows constantly, so they
  don't sit in the same table. Object storage (MinIO) was tried and dropped: the database
  is already the shared state both processes reach, so a bucket only added an untransacted
  second source of truth (upload between `flush()`/`commit()`, orphaned objects on a failed
  commit, a `stat` call needed on every download route). Blobs stay within what Postgres
  handles comfortably at this scale (a few MB to ~13 MB per artifact).
- **Redis** backs the Celery broker/result cache (so a caller can await a queued task's
  return value) and, separately, session/rate-limit state. Both default to the same
  `REDIS_URL` instance for local single-instance setups, but the broker side can be pointed
  at a different instance via `CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND` — the two have
  different access patterns and don't need to share an instance.
- **Everything served is zstd-compressed**, stored compressed and served as-is (the client
  decompresses). Signatures cover the raw bytes, so the server compresses after signing.
- **The gateway always proxies** — it reads the blob server-side and returns it, so auth
  and rate limiting apply to every byte served and no presigned URL bypasses the download
  cooldown.

**Result lifecycle.** A `done` quantization result is served on request and stamped
`served_at`. The cleanup task deletes it (and nulls the job's signature) once a served
result is older than `SERVE_GRACE_SECONDS` (5 min) or an unclaimed one older than
`RESULT_TTL_SECONDS` (1 h), flipping the job to `expired`, in batches of
`CLEANUP_BATCH_SIZE`. The same pass fails jobs a dead worker left `running` and expires
`pending` jobs whose task never ran. Weight submissions themselves are never reaped.

## Federated aggregation

Every `FED_AGG_INTERVAL_SECONDS` (default 24 h) beat fires a dispatcher that queues one
round per `raw`/`quantize` model (see [submission-type.md](submission-type.md)), so models
aggregate in parallel across workers; `secure` models aggregate per sealed round instead
(see [secure-aggregation.md](secure-aggregation.md)). Each round:

1. **Lock** — takes a per-model Redis lock (`FED_LOCK_TTL_SECONDS`); a round that finds it
   held is skipped.
2. **Cohort** — the newest valid delta per client among those based on the version's
   active snapshot, capped to the newest `FED_AGG_MEMORY_BYTES // (weight_count × 4)`
   clients (500 MB default) so memory stays bounded. Fewer than `FED_MIN_SUBMISSIONS`
   (default 1) skips the model until next tick. Structural checks happen at the gateway
   before a delta is stored, so `valid` is only a manual revocation switch.
3. **Aggregation** — `trimmed_mean` drops the smallest/largest `FED_TRIM_RATIO` (default
   0.2) of each coordinate before averaging the rest; this is the round's only Byzantine
   defense. Uniform weighting removes the incentive to lie about dataset size.
4. **Artifact baking** — both serving artifacts (trainable + signed int8) are exported
   from the new weights; if that fails, nothing is written.
5. **Commit** — the new `GlobalWeights` row and its artifacts land in one transaction, so a
   visible snapshot always has its artifacts. The row records the snapshot it aggregated
   from (`parent_weights_id`), and a partial unique index allows one valid child per
   parent, so if the lock ever lapses the second round to commit fails as a duplicate. The
   lock saves wasted work; the index guarantees correctness.
6. **Rate-limit reset** — the model's download/submission counters are cleared for every
   user.

The round returns its outcome, cohort size and per-phase timings as structured data.

If a round makes a model worse, flipping its `valid` flag to false rolls clients back to
the previous snapshot atomically (artifacts + weights together), and since the index only
counts valid rows the next round can aggregate from that parent again. Schema changes are
handled by wiping the database and re-running the seed script — no migrations.

A round can be queued by hand: `uv run -m scripts.integration.queue_aggregation [model]`
(without a model it runs the dispatcher).

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
| `GET /model/download/{trainable,quantized}/{key}` | device-owner; one per (model, artifact) per `DOWNLOAD_COOLDOWN_SECONDS` (default 300 s) — trainable and quantized cool down independently |
| `GET /model/weights/{key}` | device-owner; same cooldown, separate counter from the artifact download |
| `POST /model/submit/quantize\|raw/{key}/{weights_id}`, `POST /model/secure/submit/{round_id}` | device-owner; `SUBMIT_DAILY_LIMIT` (default 2) per model per rolling 24 h, shared across all three submission paths — one weight-submission budget per model regardless of which endpoint delivers it |
| `GET /model/quantize/result/{job_id}` | authed; only the submitting user |
| `POST /model/secure/join/{key}` | device-owner; `404` unless the model is `secure`-typed; one join per model per `SECURE_JOIN_COOLDOWN_SECONDS` (default 300 s) |
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
