# Model types

Every model is a custom `tf.Module` (`backend/ml/models/`) with explicit `eval` / `train`
/ `save` / `restore` signatures, so the same graph is LiteRT-trainable on-device and its
flattened weights can move through FedAvg. Two representations of each model are exported:
a float32 trainable `.tflite` (on-device training, LiteRT) and an int8 `.tflite` (TFLM /
ESP32 inference).

**No model normalizes its own input.** Every signature — `eval` and `train` alike, and
therefore the int8 build converted from `eval` — expects input that is already z-scored,
and z-scored *per user*: the client derives the parameters from its own wearer's signal
rather than receiving a population constant with the model. There is consequently no
separate non-normalizing `infer` signature, and nothing about normalization travels in the
signed payload — see [model-signing.md](model-signing.md).

## `FeatureMLP` — supervised anomaly classifier (current on-device candidate)

A Dense-only network over a 13-value hand-crafted feature vector, computed from each
non-overlapping 8-second window (512 BVP samples @ 64 Hz):

- Time-domain: mean, std, min, max, range, RMS, mean-abs-diff.
- Spectral: zero-crossing rate, dominant frequency, HR-band (0.7–3.5 Hz) energy ratio.
- Pulse-band shape (0.5–4.0 Hz = 30–240 bpm): spectral centroid, spectral spread, and the
  log power ratio above the band. The band is deliberately wider than the
  HR-band ratio's so a slowed rhythm still falls inside it. These three say *where* the
  in-band energy sits rather than how much of it there is, which is what separates a
  slowed or accelerated rhythm from a normal one; they are read off the same FFT the two
  features above already need.

This extraction is implemented identically in three places, so it must stay in sync:
`backend/ml/preprocessing.py` (training/calibration), `firmware/main/ml/features.c` (on-device,
esp-dsp FFT), and `application/.../capture/domain/WindowFeatures.kt` (on-device recovery
when a BLE result was lost, JDSP FFT). Features are computed and stored **raw**
everywhere; whoever feeds the model z-scores them first, against that user's own
statistics.

Labels for training come from synthetic anomaly injection into the raw BVP signal (five
kinds: amplitude blow-up, band-limited noise, tachycardia, bradycardia, afib-like
jittered timewarp), window-aligned so every window is fully clean or fully anomalous, and
mixed by the loader rather than stored.
Being Dense-only, `FeatureMLP` is fully int8-quantizable and is the current ESP32-side
inference model.

## Autoencoder family — `FeatureAutoencoder` / `CNNAutoencoder` / `LSTMAutoencoder` / `GRUAutoencoder`

Reconstruct a BVP window (raw, model-normalized internally) and use reconstruction MSE as
the anomaly score. The signal is the only input: the autoencoders take BVP and nothing
else.

Among the three waveform variants the **CNN** one is the reference: non-recurrent strided
convs and upsampling, which quantize cleanly for on-device training. LSTM/GRU variants are
kept for comparison and as the teacher in a knowledge-distillation pipeline
(`knowledge_distillation.py`) that trains `FeatureMLP` on autoencoder-derived soft labels
instead of synthetic ones.

### `FeatureAutoencoder` — feature reconstruction (detector focus)

Same contract — trained on normal windows only, scored by reconstruction error — but what
it reconstructs is the same 13-value feature vector `FeatureMLP` classifies, rather than
the waveform. It is the merge of the project's two other models: `FeatureMLP`'s input,
`CNNAutoencoder`'s unsupervised objective. Since the vector is already computed off-model
everywhere — by the loader offline, by the device on-line — the model needs nothing the
system does not already produce. The trainable part is four dense layers over 13 numbers,
the smallest model in the project.

The reason is that a waveform autoencoder's reconstruction error measures signal
*complexity*, not novelty: it is essentially the high-frequency residual the bottleneck
could not carry, so an anomaly that smooths or slows the signal is reconstructed *better*
than a normal window and lands below the threshold instead of above it. Bradycardia is the
clearest case — a waveform autoencoder ranks it at AUC 0.34 on held-out subjects, i.e.
actively inverted, which pins the detector's ROC near the diagonal no matter how well the
model is trained. Every feature has a bounded normal range, so a departure
in either direction leaves the region the bottleneck learned, and the same score detects
both a rhythm that is too slow and one that is too fast. `anomaly_kinds.py` measures that
per kind for any model: bradycardia goes from AUC 0.35 under `CNNAutoencoder` — inverted —
to 0.97 here.

The three pulse-band shape features exist for this model. Measured on the same held-out
subjects, the autoencoder over the original 17 reaches ROC-AUC 0.66 and leaves bradycardia
at 0.75; adding centroid, spread and the high-band ratio takes it to 0.74 and 0.97, level
with a purpose-built 14-value spectral descriptor that was tried first and dropped in
favour of reusing one vector across both models.

Reconstruction MSE is the whole detector either way. See
[anomalies-and-distillation.md](anomalies-and-distillation.md) for how the score becomes a
decision, how its threshold is calibrated, and how the labels are distilled.

See `backend/README.md` for training commands, dataset pipeline and the current roadmap.
