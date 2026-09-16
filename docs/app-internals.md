# Android app internals

Implementation details of `application/` that don't belong in a high-level README. See
[architecture.md](architecture.md) for the app's role in the system.

## Project structure

Each feature module is an MVC trio: `ui/` (Compose screens/state), `domain/` (business
logic) and `data/` (APIs, databases, Bluetooth, filesystem).

```
app/src/main/
  kotlin/app/somasafe/
    bluetooth/   BLE relay to the ESP32: GATT connection, the firmware's BLE contract,
                 capture-session orchestration
    capture/     Capture storage + on-device preprocessing (Room db, feature extraction,
                 per-wearer normalization, .ssds dataset import)
    training/    On-device LiteRT training: JNI wrapper + local-epoch trainer
    backend/     Server session, model list/download/introspection, upload actions
  cpp/
    litert/      Git submodule with LiteRT abstractions
    jni_bridge.cpp
```

## Navigation

One `NavHost` with a nested graph per bottom-nav tab, so each tab keeps its own back
stack (switching tabs restores where you left off). `BackendTab` starts at a login
screen that redirects to a home screen once a session exists; the home screen has a
"Models" and a "Firmware" section, each with a local-management screen. The live
Bluetooth session (device + connection) is owned above the `NavHost` so it survives tab
switches, and is only torn down when the user navigates back off the device-detail
screen.

## On-device model storage

Models live under `context.filesDir/models/<key>/`:

```
trainable.tflite     trainable LiteRT flatbuffer with the global weights baked in (zstd)
trainable.json       RemoteModel snapshot incl. weights_id / weights_version
base_weights.bin     global snapshot training started from, raw LE float32
trained_weights.bin  absolute trained weights, raw LE float32
quantized.tflite      int8 tflite from the backend (zstd)
quantized.json       signed fields: signature, contract version, architecture version
```

Served artifacts are stored exactly as received (compressed) and decompressed by dedicated
readers; the two `.bin` weight blobs are written locally by training and never compressed.
Firmware images live alongside under `context.filesDir/firmware/<version>/`.

### Two weight-refresh paths

`downloadTrainable` (whole architecture + baked-in weights) is the only path that can move
the architecture itself, so it clears both `.bin` files — any local training state was
computed against the superseded `.tflite`. `downloadWeights` (just the flat buffer) is the
lighter path when only the weights moved: it overwrites `base_weights.bin` and drops
`trained_weights.bin`, leaving the trainable file untouched. Both screens that show model
status compare local `trainable.json` against `/model/list` by **equality**, never
ordering, since a federated round can roll the active weights backward after the fact
(see "Federated aggregation" in [server-internals.md](server-internals.md)).

The two `.bin` files hold only a locally trained update; the submission body is the delta
`trained − base`, computed at upload time and never stored. `weightsStatus` reports
`MISSING` (not trained, or weights just refreshed), `OUTDATED` (trainable file is newer
than `trained_weights.bin`) or `CURRENT`.

### Quantized model and federated uploads

The device runs int8 inference, so the quantized model is what gets staged during
capture. Two ways to get it, matching the backend's two submission paths: "Download
quantized" (the current global artifact, no training required) and "Upload & quantize"
(submits the trained delta, backend enqueues a job, app polls for the personalized int8
result). "Submit only" posts the same delta with nothing coming back. Which paths a model
accepts is set by its `submission_type` — see
[submission-type.md](submission-type.md). `quantStatus` flags the quantized artifact
stale when `trained_weights.bin` is newer than it, or when its recorded architecture
version has fallen behind the locally downloaded trainable's version.

## Capture and persistence

Driven from the connected-device screen: **Load** a model (stages the signed payload plus
the wearer's normalization parameters onto the device), **Start capture** opens a session
and subscribes to the PPG/ML services (each window and each inference result merged by
sequence number, since either can arrive first or be lost), **Stop capture** closes it.
Each sample stores raw PPG, on-device timestamps, receive time, and (when present) the
raw features and int8 score. Stored in Room at `capture.db`.

## On-device preprocessing (Process)

One idempotent pass over a capture group's stored windows (`CapturePipeline.kt`):

- **Features** — any complete window missing an ML result (dropped packet) gets its
  13-feature vector recomputed on the phone, matching the firmware's extractor, and stored
  without a score (running the model just to fill the score would waste computation).
- **Normalization parameters** — the same pass derives the wearer's z-score parameters
  from that group's own windows: per-feature mean/std over the feature vectors, plus one
  mean/std over the raw BVP stream (what on-device training normalizes with).

Samples are stored un-normalized everywhere; whoever feeds a model z-scores first, since
normalization is per wearer and travels with nothing.

**Which capture's parameters get staged.** Preprocessing computes parameters per group,
so one group is nominated as the wearer's reference: processing a group turns its button
into **Pick**; picking clears any previous pick (only one at a time). With nothing picked,
staging falls back to the most recently started processed group. On-device training
always normalizes with the parameters of the group it's training on, ignoring the pick.

The `.ssds` import (a subject export from the backend) carries windows only — sequence
number and faked device timestamps — and derives its normalization parameters through
Process like any recorded group, so imported windows preprocess and train identically to
streamed ones.

## On-device training (Train on capture)

Reached from a model's detail screen. One local epoch (`Trainer.kt`):

1. Loads the trainable model (baked-in weights are the global snapshot), restoring a
   previous epoch's `trained_weights.bin` on top if present.
2. Assembles windows from a capture group and z-scores each with that group's own signal
   parameters — the normalized window is the whole model input; the label is unused (the
   autoencoder is self-supervised).
3. Writes `trained_weights.bin` and the starting snapshot to `base_weights.bin`, which
   marks the quantized artifact outdated. The upload actions then submit `trained − base`
   as the federated update.

Requires the model downloaded and the capture group processed. The screen reports the
run's metrics (reconstruction error before/after via `eval`, mean/last-batch train loss,
windows trained/dropped, and the L2 norm of the update) — metrics live on the screen only,
nothing extra is written to disk.
