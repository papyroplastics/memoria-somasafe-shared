# ML training pipeline (backend)

How `backend/ml/` is put together, so a new model or dataset only has to implement a
small, fixed shape. See [model-types.md](model-types.md) for what each architecture is,
and [anomalies-and-distillation.md](anomalies-and-distillation.md) for the anomaly
labelling and distillation case studies.

## Three layers

Training is split so any model can run under any loop:

- **Model** (`TrainableModel`): the graph — `eval` / `train` / `save` / `restore`, plus
  `transfer_from` (copy compatible trainable weights from another instance of the same
  architecture, transferring the overlapping region where a shape differs — used for
  cross-batch-size transfer learning). All state lives in `trainable_variables`, saved
  and restored as one flat float32 buffer, and `train` only mutates the variables and
  returns a dict carrying `loss`. Gradient descent is not part of the base contract:
  `BackpropModel` is the subclass that owns Adam and the tape/apply step, so a subclass
  only writes the forward pass and the loss. A model that trains by accumulating
  statistics rather than by backprop inherits `TrainableModel` directly.
- **DataSource** (`ml/sources/`): a generic 3-method ABC — `subject_ids`, `datapoints`
  (the model-ready array for one subject) and `calibration_data` (an int8 calibration
  sample), plus an optional `labels`. A source is always a single modality + variant
  (e.g. "PPG-DaLiA's clean signal windows" or "its mixed feature vectors, low-activity
  only") with no selector argument, so a new kind of dataset (a different signal type,
  images, text) only has to implement this shape. `ml/dataset_list.py` generates every
  combination a dataset module offers (activity filter x modality x variant) into full
  registry keys.
- **Trainer**: only what is model-specific — `subject_arrays` and `eval_metrics`
  (accuracy for the MLP, reconstruction error for the autoencoders). Two class
  attributes name the dataset entries a model family uses as full registry keys: a
  `training_key` and an optional `calibration_key` for the int8 converter (defaulting to
  `training_key`). Each model module exposes `get_model(data_root, batch_size=None)` and
  `get_trainer(data_root, batch_size=None)`.
- **Loop** (`training.py`): orchestration only — `normal_loop` and `federated_loop`
  (simulated FedAvg, aggregating each round's client deltas with `weighted_average`), over
  generic `evaluate` / `train_epoch` steps. Loops talk only to the `Trainer` interface, so
  a `(model, trainer)` pair works with either and the two are directly comparable.

Both loops split at **subject** granularity: `--eval-subjects` selects which subjects are
held out whole; the centralized loop pools the rest, the federated loop trains those
subjects as separate clients. `scripts.figures.plot_convergence` plots both curves from
the `run.yaml` manifests without retraining, and refuses to overlay runs whose manifests
disagree.

## Datasets and the activity filter

PPG-DaLiA records every subject through eight protocol activities; a per-window activity
id is derived once and every windowing grid indexes off it. A registry key is
`<activity-filter>-<modality>-<variant>` (e.g. `ppg-dalia-low-features-mixed`):

| activity filter | what it serves |
| --- | --- |
| `ppg-dalia` | every window |
| `ppg-dalia-low` | only low-activity windows (sitting, driving, lunch, working) |

`modality` is `signal` or `features`; `variant` is `clean`, `mixed` (synthetic anomaly
mix), or one of the five anomaly kinds. The filter is strict — a window survives only if
every sample in it carries an allowed activity id — and is applied at load time, never on
disk. Every PPG trainer's `training_key` names a `ppg-dalia-low` variant (motion
artefacts during activity dominate the waveform otherwise), while `calibration_key` names
the unfiltered clean variant, since the device runs on the wearer's whole day. Per-subject
normalization is deliberately not filtered — a subject's z-score params come from its
whole clean recording either way.

## What is on disk, and what is not

Only two stages are written out, both by `scripts/system/get_dataset.py`: `clean-signals/S*/`
(raw BVP, ACC magnitude, activity track) and `anomalous-signals/<kind>/S*/` (one
fully-anomalous copy per kind). Everything else — the anomaly mix and its labels, the
20-value feature vectors, every normalization parameter — is derived on the call from
those two, since it's milliseconds of numpy per subject and caching it meant rebuilding
the dataset whenever a definition moved.
`scripts/system/export_subject_data.py` materializes a snapshot of one subject (the
`.ssds` protobuf) for the consumers that can't import the backend — the Android app and
the firmware harness.

## Running training

```bash
uv run -m scripts.system.get_dataset                              # fetch + preprocess (idempotent)
uv run -m scripts.system.train feature-mlp                        # normal loop
uv run -m scripts.system.train feature-mlp --loop federated        # simulated FedAvg
uv run -m scripts.system.train cnn-ae --eval-subjects 11-15         # hold out a subject range
```

Useful flags: `--epochs` / `--rounds` / `--local-epochs` (loop tuning), `--batch-size`
(GPU throughput; on-device default is often 1), `--eval-subjects` (`N`, `n-m`, `i,j,k`, or
`none` to train on everyone and skip evaluation), `--tag <name>` (suffix the run's weights
and results so it doesn't clobber the canonical one), `--load-weights` (resume a tagged
run instead of a fresh init), `--no-tflite` (skip the `.tflite` exports for a run that's
never served). Each run writes `weights.npy`, `trainable.tflite` and `quantized.tflite`
into `shared/gen/models/<model>/`, and a history plot/CSV/`run.yaml` manifest into
`results/<model>/<loop>/`.

`transfer_learn` bridges GPU-trained large-batch weights into the deployable default-batch
model (the batch size is baked into the `.tflite` input signature), fine-tuning for a few
epochs after seeding from the large-batch weights via `transfer_from`.

`export_subject_data.py` packs one subject's windows into the `.ssds` protobuf
(`shared/gen/exports/S<id>.ssds`), read by the app (to skip the UART/ESP/BLE path) and the
firmware harness (`serial_write.py` replays it, `test_model.py` checks the device's
features/score against the stored ones — this consumer needs a gapless export). Flags:
`--missing-samples` / `--missing-features` (drawn independently, drop that fraction of
windows' signal or ML result to model packet loss) and `--clean` (export the anomaly-free
signal instead of the mix).

## Environment

Python `==3.13.*`, TensorFlow `2.21.*`, managed with `uv`. GPU is optional: `uv sync
--extra cuda` swaps in CUDA-enabled TensorFlow; the default install is CPU-only.
