# Submission types

Every `ModelVersion` carries a `submission_type`, sourced from the code registry
(`backend/ml/model_list.py`) and seeded into the DB. It is a per-model property that
decides two things at once: **which upload path** a model accepts a federated update on,
and **which aggregation strategy** the server runs for it. There are three types today,
and the design leaves room for more (sparse, differential-privacy) — each future format
adds its own type plus its own endpoint rather than overloading an existing one.

| Type | Upload path | What the client gets back | Aggregation |
|------|-------------|---------------------------|-------------|
| `raw` | `POST /model/submit/raw/{key}/{weights_id}` | nothing (`202`, silent) | dense FedAvg |
| `quantize` | `POST /model/submit/quantize/{key}/{weights_id}` | a signed int8 `.tflite` of the uploaded model | dense FedAvg |
| `secure` | `POST /model/secure/*` (sealed round) | nothing (`202`) | masked-sum FedAvg |

`raw` and `quantize` are byte-identical dense weight-delta vectors and share the same
FedAvg path, so the `raw` (submit-only) path accepts both (`quantize`'s dense body is
compatible and submit-only is the least work); the `quantize` path accepts only
`quantize`-typed models. A model uploaded on a path it doesn't accept gets `404` (not
`403`, so the path stays unguessable). `secure` carries an incompatible masked,
non-float32 body and aggregates only inside a sealed round, so it lives entirely on its
own endpoints — see [secure-aggregation.md](secure-aggregation.md).

## `raw` — submit-only

The client uploads its weight-delta (Δ = local − global), the server persists it for the
next aggregation round, and **nothing comes back**. Rejection is fully silent (a Byzantine
client never learns its update was filtered). Because it neither reveals a verdict nor
round-trips a full-weights artifact, it is the natural host for future privacy-preserving
submission formats (sampled weights, differential privacy).

## `quantize` — personalized model for the firmware

The client uploads its weight-delta and the server returns a **signed int8 `.tflite`** of
the resulting model, ready to load onto the ESP32. Its purpose is to give the system the
flexibility to deploy **models that benefit from personalization on a user's own data** to
the firmware. This matters for two reasons:

- Not every model benefits from personalization — it is a per-model property. `quantize`
  exists so that a model which *does* benefit can ship its personalized weights to the
  device. `FeatureMLP` (evaluated in `backend/scripts/distill_eval.py`, where the
  personalized model scores marginally better than the global one) is only an **example**
  of such a model, not the justification for the feature; the justification is the
  flexibility itself.
- Federated aggregation can make a model *worse* on a given user's local data (the global
  average pulls it away from that user's distribution). The `quantize` path lets a user
  obtain a version of the model quantized from the weights **before** aggregation — i.e.
  their own locally-trained parameters — instead of the global ones.

This feature exists **only to run personalized models on the firmware**. The phone already
personalizes by default: it runs float32 models and swaps their parameters directly
through LiteRT, so it never needs the server to build it a personalized artifact.

### Why the server quantizes, not the phone

Ideally quantization would happen on the phone itself — the delta was trained there, and a
round-trip to the server would be avoided. Two problems make the server the practical
place instead:

- **The conversion is impractical on-device.** TensorFlow's tuning + quantization +
  `.tflite` artifact creation is implemented in Python, so on-device conversion would mean
  embedding a Python interpreter. It also operates on TensorFlow `SavedModel` artifacts,
  not `.tflite` files — the phone would have to implement trainable-`.tflite` → quantized-
  `.tflite` conversion itself, or the server would have to ship the `SavedModel` with its
  quantization params pre-computed. Running a `SavedModel` on the phone is unrealistic
  anyway: it needs the full TensorFlow runtime, which would defeat the whole point of
  using LiteRT.
- **The client can't sign the model.** Only the server holds the private key the firmware
  verifies against (see [model-signing.md](model-signing.md)), so a phone-quantized model
  would fail to load on the ESP32. This is the softer of the two blockers: firmware-side
  signature verification is not critical and could be dropped or loosened, since the
  app↔firmware BLE channel already requires authentication (see
  [device-attestation.md](device-attestation.md)). The conversion cost is the real reason
  the work stays on the server.

## `secure` — masked aggregation

The server only ever sees the **sum** of the round's updates, never an individual one, via
a pairwise-masking protocol over a sealed cohort. This buys privacy against an
honest-but-curious server at the cost of per-client validation (the MSE gate and outlier
filter become structurally impossible). The full construction, its invariants and the
sealed-round lifecycle are in [secure-aggregation.md](secure-aggregation.md).
