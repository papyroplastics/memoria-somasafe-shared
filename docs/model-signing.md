# Model signing

The server signs every distributed quantized model; the ESP32 verifies the
signature against its factory-provisioned server public key (`srv_pub`) before
loading the model. The signature is **transport-independent**: the app
packages the fields for the device however its BLE interface version
dictates, and the firmware rebuilds the canonical byte string below from the
delivered fields and verifies over it.

The server's signing keypair is generated once with `shared/make_keys.sh`
(plain `openssl ecparam`/`ec`, into `shared/gen/`); its public half is baked
into every device's `srv_pub` at factory-provisioning time by
`firmware/scripts/gen_factory_nvs.py`, which also generates each device's own
identity keypair — see [device-attestation.md](device-attestation.md).

## Canonical signed bytes (little-endian)

| field              | type   | meaning                                                        |
|--------------------|--------|----------------------------------------------------------------|
| `contract_version` | `u16`  | how the model is fed: norm-param layout + I/O signatures       |
| `norm_params`      | `f32[]`| z-score params; count fixed by `contract_version`              |
| `tflite`           | `u8[]` | the int8 model                                                 |

Signature: ECDSA P-256 over SHA-256 of the concatenation, DER-encoded.

The device applies `norm_params` as `(x - mean) / std` before feeding the
(non-normalizing) int8 model.

## Contract versions

| version | model      | norm_params layout                                     |
|---------|------------|--------------------------------------------------------|
| 1       | FeatureMLP | `mean[17]`, `std[17]`                                  |
| 2       | AE family  | signal `mean[2]`, `std[2]`; cond `mean[8]`, `std[8]`   |

## Delivery

The backend serves the tflite as the response body and the remaining fields in
headers, base64 where binary:

- `X-Model-Signature` — DER signature, base64
- `X-Contract-Version` — decimal integer
- `X-Norm-Params` — raw LE float32 buffer, base64

Reference implementation: `backend/ml/payload.py`.

## BLE payload (interface version 1)

The app assembles the delivered fields into the payload staged onto the device
over the ML client-buffer service (little-endian):

| field              | type          |
|--------------------|---------------|
| `sig_len`          | `u16`         |
| `sig`              | `u8[sig_len]` |
| `contract_version` | `u16`         |
| `norm_params`      | `f32[]`       |
| `tflite`           | `u8[]`        |

Everything after the signature header is exactly the canonical signed bytes, so
the firmware verifies the signature over `body = data + 2 + sig_len` directly,
then reads `contract_version` to know the norm-param layout.

Assembler: `application/.../bluetooth/domain/ModelPayload.kt`.
Verifier: `firmware/main/ml/infer.cc` (`parse_payload`).
