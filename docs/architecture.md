# Target architecture

SomaSafe is a federated learning system for cardiovascular anomaly detection on PPG
(photoplethysmography) signals. The privacy premise drives the whole design: raw sensor
data never leaves the user's devices. Three tiers, with the ESP32 fully isolated behind
the phone:

- **ESP32 (`firmware/`)** acquires PPG/ACC data, streams it to the paired phone over BLE,
  and runs inference on the current int8 `.tflite` model with TensorFlow Lite Micro
  (Espressif's `esp-tflite-micro` fork). It has no internet connection — the Android
  device is its only external interface.
- **Android app (`application/`)** has two roles: federated participant (trains a local
  model on the streamed data with LiteRT's on-device training extension) and exclusive
  relay between server and ESP32 (downloads new global models and pushes them to the
  device; uploads weight updates). Two model artifacts are maintained per task: a float32
  trainable model for local training and an int8 quantized shell that is relayed to the
  ESP32 for inference.
- **Server (`backend/`)** aggregates client weight updates into new global model versions,
  quantizes them for TFLM, and distributes them. Multiple models can coexist, each with
  its own versioning, round schedule and quantization pipeline.

## Data flow

Training round: the ESP32 streams data to the phone → the phone trains locally → the
phone uploads only the weight delta → the server validates and aggregates the round's
updates → the new global model is quantized and versioned. Distribution: the phone fetches
the latest model version → relays it to the ESP32 over BLE → the ESP32 loads it into TFLM.
Raw signals only ever travel one hop, over BLE.

## Security model

Security is layered per hop rather than end-to-end, since each hop has a different trust
boundary:

- **Model integrity** (server → phone → ESP32): the server signs every distributed
  quantized model; the firmware verifies the signature before loading, so even a
  compromised phone cannot inject models. See [model-signing.md](model-signing.md).
- **Device identity** (ESP32 → server, via the phone): each ESP32 is factory-provisioned
  with an ECDSA P-256 keypair; a user proves control of a physical device with a
  challenge/response signature to unlock rate-limited server endpoints. See
  [device-attestation.md](device-attestation.md).
- **BLE link** (phone ↔ ESP32): LE Secure Connections with passkey entry and MITM
  protection, bonding enabled. See [ble-protocol.md](ble-protocol.md).
- **User auth** (phone ↔ server): stateful bearer tokens over HTTP. See
  [authentication.md](authentication.md).

Federated aggregation itself has basic Byzantine-robustness (submission validation,
z-score outlier filtering) but no stronger scheme (trimmed mean, FLTrust); see each
module's README for what's still on its roadmap.
