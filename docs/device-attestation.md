# Device attestation

A user proves control of a physical ESP32 by having it sign a server-issued challenge
with its factory ECDSA P-256 key. A successful attestation makes the user that device's
registered owner, which is what unlocks the rate-limited model endpoints (downloads,
uploads) for their account. It is a one-time ownership proof, not a per-upload signature:
once a user is the registered owner, their federated updates are trusted like any other
authenticated request — the device does not re-sign each weight upload.

## Factory identity

Each ESP32 is factory-provisioned (`firmware/scripts/gen_factory_nvs.py`) with, in its own
`factory_data` NVS partition (kept apart from the default `nvs` so an erased default
partition never touches the provisioned identity):

- `serial` — device serial number (`SN` + 12 hex chars).
- `dev_priv` — device ECDSA P-256 private key (raw 32-byte scalar).
- `esp_pub` — device ECDSA P-256 public key (65-byte uncompressed point).
- `srv_pub` — the server's ECDSA P-256 public key, used to verify signed models (see
  [model-signing.md](model-signing.md)).

The server's own keypair is generated separately with `shared/make_keys.sh`; its public
half is what gets baked into `srv_pub` at provisioning time, and its private half is what
the backend signs models with (`common/config.py`). The device row (`serial` +
`public_key`) is registered server-side ownerless from the same factory NVS image
(`backend/scripts/seed_db.py`).

## Ownership flow

1. `POST /device/challenge {serial}` → the server stores a random nonce + metadata in
   Redis under a fresh `instance_id`, `DEVICE_CHALLENGE_TTL_SECONDS` (default 300 s) TTL.
2. The client builds the canonical payload — `nonce ‖ instance_id ‖ server_time ‖ user_id
   ‖ serial` (big-endian) — and has the device sign its SHA-256 over its signing
   characteristic (`firmware/main/device/service.c`, `dev_priv`). The signature is DER.
3. `POST /device/attest {instance_id, signature}` consumes the challenge (one-shot
   `GETDEL`), rebuilds the payload, verifies the DER signature against the device's stored
   public key with ECDSA(SHA-256), and on success records the calling user as the owner.

Ownership can change at most once per `DEVICE_ATTEST_COOLDOWN_SECONDS` (default 24 h):
`/device/challenge` is rejected with `429` while a device's `last_attested_at` is within
that window. Only a **successful** attestation sets `last_attested_at`, so failed or
abandoned challenges never consume the window.

`GET /device/owned` returns the serials the calling user currently owns.

## What's not implemented

Per-upload device signing (the device re-signing every weight-update hash so the server
can verify each individual submission came from a live device session) was an earlier
design idea and is not worth building — it adds complexity without a real security
benefit over the one-time ownership proof, given there's no adversarial deployment. Not
on the roadmap.
