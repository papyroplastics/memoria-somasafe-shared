# Model signing

The server signs every distributed quantized model; the ESP32 verifies the
signature against its factory-provisioned server public key (`srv_pub`) before
loading the model. The signature is **transport-independent**: the app
packages the fields for the device however its BLE interface version
dictates, and the firmware verifies over the canonical byte string below.

The server's signing keypair is generated once by `shared/Makefile` (`make setup`, plain
`openssl ecparam`/`ec`, into `shared/gen/`); its public half is baked into every device's
`srv_pub` at factory-provisioning time by `firmware/scripts/gen_factory_nvs.py`, which
also generates each device's own identity keypair — see
[device-attestation.md](device-attestation.md).

## Canonical signed bytes

The model's **contract version** (`u16`, little-endian) followed by the int8 `.tflite`.
Signature: ECDSA P-256 over SHA-256 of those bytes, DER-encoded.

The contract version says how a device must feed the graph — what input tensor shape it
declares, and how many normalization parameters go with it. That makes it exactly the kind
of statement the signature exists to protect: flipping it would make a device feed a model
a layout it was not built for. It comes from `ModelSpec.contract_version` in the backend's
registry and is `0` for every model that is not meant for the ESP32 at all.

| contract | model | input |
|----------|-------|-------|
| `1` | `FeatureMLP` | one int8 `[1, 17]` tensor of z-scored window features; 17 mean + 17 std parameters |

The **normalization parameters are not signed and never travel with the model.** Every
model signature — `eval` and `train` — takes already-normalized input, and the z-scoring is
per wearer: the client derives the parameters from its own wearer's signal, so the server
has nothing to say about them and could not sign them if it wanted to. Two devices running
the same model version normalize differently, which is the point.

## Delivery

The backend serves the tflite as the response body and the rest in headers:

- `X-Model-Signature` — DER signature, base64
- `X-Contract-Version` — the contract version the signature covers

Reference implementation: `backend/ml/payload.py`.

## BLE payload (interface version 1)

The app assembles the delivered fields, plus the wearer's own normalization parameters,
into the payload staged onto the device over the ML client-buffer service
(little-endian):

| field              | type          | signed |
|--------------------|---------------|--------|
| `sig_len`          | `u16`         | no     |
| `sig`              | `u8[sig_len]` | no     |
| `norm_len`         | `u16`         | no     |
| `norm_params`      | `f32[]`       | no     |
| `contract_version` | `u16`         | yes    |
| `tflite`           | `u8[]`        | yes    |

The app derives that block from its own capture store — the per-feature mean/std of a
capture group's raw feature vectors, picked on the Captures tab (see
`application/README.md`); the firmware harness (`scripts/lib/capture.py`) does the same
over a subject export's vectors. The client-supplied normalization block sits **ahead of**
the signed fields on purpose: it leaves `contract_version ‖ tflite` contiguous at the end
of the buffer, so the device verifies the signature in place over
`data + 4 + sig_len + norm_len` without reassembling or copying the payload anywhere.
`norm_params` is `mean[n]` followed by `std[n]`, and `norm_len` must match the length the
device's contract fixes (136 bytes for contract 1) — a mismatch is rejected before the
signature is even checked.

Assembler: `application/.../bluetooth/domain/ModelPayload.kt`.
Verifier: `firmware/main/ml/infer.cc` (`parse_payload`), against
`firmware/main/ml/contract.h`.
