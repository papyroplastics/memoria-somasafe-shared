# Versioning semantics

Several independent things are called "version" across this project. None of them are
ever bumped just because — since there's no real deployment, all of them stay at `1`
until a real, compatibility-breaking change forces a bump (see the root README's note on
pragmatic versioning). This doc is the map of which is which.

| Name | Lives in | Meaning |
|------|----------|---------|
| `BLE_INTERFACE_VERSION` | `firmware/main/ble/host.h`; mirrored as `Firmware.interface_version` (backend) | The whole app↔firmware BLE contract: every service's wire framing, plus the model payload layout (see [ble-protocol.md](ble-protocol.md), [model-signing.md](model-signing.md)). An app build and a firmware build must agree on it to talk to each other at all. |
| `ML_CONTRACT_VERSION` / `contract_version` | `firmware/main/ml/contract.h`; `ModelVersion.contract_version` (backend); rides inside the signed model payload | How a specific model is fed: the norm-param layout (count/order) and its I/O tensor shapes. Independent of `BLE_INTERFACE_VERSION` — the BLE contract can stay put while a new model type ships with a new contract version. See the contract-version table in [model-signing.md](model-signing.md). |
| Model `version` | `ModelVersion.version` (backend), hand-bumped in `ml/model_list.py` | A specific model's compatibility generation: bundles `min_app_version`, `contract_version` and that version's `norm_params`. Only the **latest** version of a model accepts federated submissions and aggregates; older versions are frozen (still served, e.g. for apps that haven't updated, but out of the federated population). |
| `fingerprint` | Derived (`Trainer.arch_fingerprint()`) | A hash of the ordered trainable-variable layout plus the baked normalization params. Not hand-set — it's a tripwire: `seed_db.py` aborts if the fingerprint moved but `version` didn't, so a forgotten version bump can't silently mix incompatible weights into one federated population. A version bump with an unchanged fingerprint is fine (e.g. only `min_app_version` changed). |
| `min_app_version` | `ModelVersion.min_app_version` | The oldest app build that can use this model version. The app checks this against its own version and disables incompatible models client-side. |
| Weights (`weights_id` / `weights_version`) | `GlobalWeights.id` / `.created_at` (backend); `X-Weights-ID` / `X-Weights-Timestamp` headers | A FedAvg round's output: the flat parameter buffer plus the serving artifacts baked from it. Moves independently of (and much more often than) the model `version` — a new weights snapshot under the *same* version just means "re-pull the trainable artifact and keep training"; a new `version` means the local federated state resets. |

## Client-side interaction

The Android app tracks `version`, `fingerprint` and `weights_version` per downloaded
model in `meta.json`. A moved `version` or `fingerprint` invalidates the app's local
federated state (`weights.json`, quantized artifact) — they belonged to the superseded
generation. A newer `weights_version` under the same `version` just means a newer global
snapshot is available to train from. See `application/README.md` ("Weights ride the
trainable artifact") for the exact reset logic.

## Firmware distribution

The OTA path is implemented on both ends. The firmware's BLE OTA service accepts
server-signed images and reports `BLE_INTERFACE_VERSION` plus the running app version
string (from `firmware/version.txt`) through its version characteristic (see
[ble-protocol.md](ble-protocol.md)). The backend stores published builds as `Firmware`
rows: `interface_version` is the BLE contract the build was compiled against, and
`supported_contracts` is the **list** of `ML_CONTRACT_VERSION`s it can run — one image
may support several model contract types. The device performs no version checks itself;
the phone is authoritative.

Publishing is a two-step flow: `firmware/scripts/export_image.py` (`make export-image`)
copies the built image plus a metadata JSON (version string, interface version, contract
list) into `shared/gen/firmware/{version}/`, and the backend seed script scans that
directory, signs each image with the server key (plain ECDSA P-256/SHA-256 over the raw
image bytes — the same signature `firmware/scripts/test_ota.py` produces) and inserts
any version not yet in the database.

Clients use `GET /ota/versions/{interface}` — the builds published for their own
`BLE_INTERFACE_VERSION`, newest first, each carrying its `supported_contracts` — and
`GET /ota/download/{interface}/{version}`, which returns the raw image with the server
signature in the `X-Firmware-Signature` header, forwarded verbatim to the device's OTA
service.
