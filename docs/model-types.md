# Model types

Every model is a custom `tf.Module` (`backend/ml/models/`) with explicit `eval` / `train`
/ `save` / `restore` signatures, so the same graph is LiteRT-trainable on-device and its
flattened weights can move through FedAvg. Two representations of each model are exported:
a float32 trainable `.tflite` (on-device training, LiteRT) and an int8 `.tflite` (TFLM /
ESP32 inference). Models z-score their own inputs internally in `eval`/`train` (baked
constants); the int8 build is exported from a separate non-normalizing `infer` signature
that expects **already-normalized** input, so the normalization params travel to the
firmware alongside the signed model — see [model-signing.md](model-signing.md).

## Conditioning

Every model is conditioned on a `cond` vector: z-scored demographics plus a causal
*activity context* (trailing-2-minute mean/std of the ACC magnitude). The context is
computed from the raw ACC signal and fed raw, like everything else — the model normalizes
it internally.

## `FeatureMLP` — supervised anomaly classifier (current on-device candidate)

A Dense-only network over a 17-value hand-crafted feature vector, computed from each
non-overlapping 8-second window (512 BVP samples @ 64 Hz + 256 ACC samples @ 32 Hz):

- Per-channel (BVP and ACC): mean, std, min, max, range, RMS, mean-abs-diff.
- BVP-only (spectral): zero-crossing rate, dominant frequency, HR-band (0.7–3.5 Hz) energy
  ratio.

This extraction is implemented identically in three places, so it must stay in sync:
`backend/ml/data.py` (training/calibration), `firmware/main/ml/features.c` (on-device,
esp-dsp FFT), and `application/.../capture/domain/WindowFeatures.kt` (on-device recovery
when a BLE result was lost, JDSP FFT). Features are stored **raw** everywhere; the model
bakes in the z-score constants and normalizes them internally in `eval`/`train`.

Labels for training come from synthetic anomaly injection into the raw BVP signal (five
kinds: spike, amplitude blow-up, band-limited noise, timewarp tachy/brady, afib-like
jittered timewarp), window-aligned so every window is fully clean or fully anomalous.
Being Dense-only, `FeatureMLP` is fully int8-quantizable and is the current ESP32-side
inference model.

## Autoencoder family — `CNNAutoencoder` (focus) / `LSTMAutoencoder` / `GRUAutoencoder`

Reconstruct the BVP channel of a `[BVP, ACC]` window (raw, model-normalized internally)
and use reconstruction MSE as the anomaly score. The encoder sees `[BVP, ACC]` but the
decoder reconstructs BVP only — ACC is exogenous context that explains motion artifacts
without being part of the anomaly score. A small bottleneck plus latent dropout push the
decoder to lean on the `cond` vector to generate the signal *expected for this person at
this activity level*, rather than copying its input.

The **CNN variant is the current focus**: non-recurrent strided convs + FiLM-conditioned
upsampling, which quantizes cleanly for on-device training. LSTM/GRU variants are kept for
comparison and as the teacher in a pseudo-labeling pipeline (`distill_labels.py`) that
trains `FeatureMLP` on autoencoder-derived labels instead of synthetic ones.

Detection also OR's in two cheap rhythm indices (in-band spectral entropy, beat-interval
coefficient-of-variation) alongside reconstruction MSE, since reconstruction error alone is
an integrity detector that's structurally weak on rhythm anomalies like afib.

See `backend/README.md` for training commands, dataset pipeline and the current roadmap.
