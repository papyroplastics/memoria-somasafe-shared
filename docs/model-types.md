# Model types

Every model is a custom `tf.Module` (`backend/ml/models/`) with explicit `eval` / `train`
/ `save` / `restore` signatures, so the same graph is LiteRT-trainable on-device and its
flattened weights can move through FedAvg. Two representations of each model are exported:
a float32 trainable `.tflite` (on-device training, LiteRT) and an int8 `.tflite` (TFLM /
ESP32 inference). Models z-score their own inputs internally in `eval`/`train` (baked
constants); the int8 build is exported from a separate non-normalizing `infer` signature
that expects **already-normalized** input, so the normalization params travel to the
firmware alongside the signed model — see [model-signing.md](model-signing.md).

## `FeatureMLP` — supervised anomaly classifier (current on-device candidate)

A Dense-only network over a 17-value hand-crafted feature vector, computed from each
non-overlapping 8-second window (512 BVP samples @ 64 Hz + 256 ACC samples @ 32 Hz):

- Per-channel (BVP and ACC): mean, std, min, max, range, RMS, mean-abs-diff.
- BVP-only (spectral): zero-crossing rate, dominant frequency, HR-band (0.7–3.5 Hz) energy
  ratio.

This extraction is implemented identically in three places, so it must stay in sync:
`backend/ml/preprocessing.py` (training/calibration), `firmware/main/ml/features.c` (on-device,
esp-dsp FFT), and `application/.../capture/domain/WindowFeatures.kt` (on-device recovery
when a BLE result was lost, JDSP FFT). Features are stored **raw** everywhere; the model
bakes in the z-score constants and normalizes them internally in `eval`/`train`.

Labels for training come from synthetic anomaly injection into the raw BVP signal (five
kinds: amplitude blow-up, band-limited noise, tachycardia, bradycardia, afib-like
jittered timewarp), window-aligned so every window is fully clean or fully anomalous.
Being Dense-only, `FeatureMLP` is fully int8-quantizable and is the current ESP32-side
inference model.

## Autoencoder family — `CNNAutoencoder` (focus) / `LSTMAutoencoder` / `GRUAutoencoder`

Reconstruct a BVP window (raw, model-normalized internally) and use reconstruction MSE as
the anomaly score. The signal is the only input: the autoencoders take BVP and nothing
else. ACC never reaches them — it exists in the pipeline solely as an input to
`FeatureMLP`'s hand-crafted features.

The **CNN variant is the current focus**: non-recurrent strided convs and upsampling,
which quantize cleanly for on-device training. LSTM/GRU variants are kept for comparison
and as the teacher in a knowledge-distillation pipeline (`knowledge_distillation.py`) that
trains `FeatureMLP` on autoencoder-derived soft labels instead of synthetic ones.

Reconstruction MSE is the whole detector. See
[anomalies-and-distillation.md](anomalies-and-distillation.md) for how the score becomes a
decision, how its threshold is calibrated, and how the labels are distilled.

See `backend/README.md` for training commands, dataset pipeline and the current roadmap.
