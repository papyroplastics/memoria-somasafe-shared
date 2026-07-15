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
the anomaly score. Encoder and decoder both see BVP only — ACC as a raw encoder channel
measured as a no-op, so it reaches the model solely through the `cond` vector's activity
context. What makes the error separate anomalies is how tightly the model fits the
clean-BVP manifold: a sharper fit makes off-manifold input miss by relatively more, so
detection improves with latent capacity rather than with a narrow bottleneck.

The **CNN variant is the current focus**: non-recurrent strided convs and upsampling,
which quantize cleanly for on-device training. The `cond` vector enters once, joined to
the code at the bottleneck, so the decoder reconstructs from `[z, cond]` jointly.
LSTM/GRU variants are kept for comparison and as the teacher in a pseudo-labeling pipeline
(`distill_labels.py`) that trains `FeatureMLP` on autoencoder-derived labels instead of
synthetic ones.

Reconstruction MSE is the whole detector. In-band spectral entropy is computed alongside
it as a hand-crafted *baseline* that `distill_eval.py` reports for comparison; it is not
part of the detector and never reaches the distilled labels.

See `backend/README.md` for training commands, dataset pipeline and the current roadmap.
