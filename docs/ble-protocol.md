# BLE protocol

The ESP32 is a NimBLE GATT peripheral; the Android app is the only client. Everything
below is versioned as a whole by `BLE_INTERFACE_VERSION` (`firmware/main/ble/host.h`,
mirrored as `Firmware.interface_version` in the backend schema) — see
[versioning.md](versioning.md).

## Services

- **PPG service** — notify-only characteristic streaming raw BVP (64 Hz) + ACC (32 Hz)
  samples every second, fragmented to the negotiated MTU.
- **Model-transfer / client buffer** — a generic client-writable buffer (size, write
  position, READY/NOT_READY state characteristic) used both to upload a model payload and
  to upload an arbitrary payload to the device-signing service. A consumer task that
  finishes reading a readied buffer resets it to NOT_READY, releasing the lock.
- **ML results / errors** — notifies inference results (feature vector + int8 score) and
  error codes.
- **Device service** — a read-only serial-number characteristic, plus the signing
  characteristic used for [device attestation](device-attestation.md): the client uploads
  a payload through the client buffer and the firmware ECDSA-signs it with the factory
  device key and notifies the DER signature back.
- **OTA service** — firmware updates over BLE: a read-only version characteristic
  (`u16 BLE_INTERFACE_VERSION`, little-endian, followed by the app version string), a state
  characteristic (read/write/notify) driving the update, and write-only image and
  signature characteristics accepting sequential writes. The client writes state `1` to
  start (writing `1` again aborts and restarts the transfer, `0` aborts back to idle),
  streams the raw app image and its DER ECDSA P-256/SHA-256 signature, then writes `2`
  to finalize: the firmware verifies the image against the factory-provisioned
  `srv_pub`, sets the boot partition, notifies `2` and restarts into the new firmware.
  Any failure aborts the update and notifies `0xFF`. The phone is authoritative — the
  device performs no version checks and there is no rollback.

## Reconstruction layer

Payloads larger than one MTU are reframed by a shared layer
(`firmware/main/ble/notif_transaction.c`, mirrored by
`application/.../bluetooth/domain/NotifTransaction.kt` as `TransactionReassembler`). Each
logical payload (a "transaction") is fragmented across notifications, each carrying a
3-byte header: flags (START/END), a per-transaction id, and a monotonic sequence number —
so the client can reframe the payload without knowing its total length in advance. A
service can add its own framing inside the payload on top of this (e.g. the PPG service
stamps on-device start/end timestamps on the first/last fragment of a window).

## Model payload

The model uploaded over the client buffer is a signed payload, not a bare `.tflite` — see
[model-signing.md](model-signing.md) for the exact byte layout and who assembles/verifies
it.

## Link security

LE Secure Connections (SC) with MITM protection via passkey entry, and bonding, are
implemented (`firmware/main/ble/host.c`, `gap.c`). The ESP32 acts as the passkey
*display* side (`BLE_HS_IO_DISPLAY_ONLY`): it generates a random 6-digit passkey and logs
it (stands in for an on-device screen — the project never manufactures real hardware, so
there is no actual display); the user enters that passkey on the phone to complete
pairing. Once bonded, a repeat pairing attempt from the same peer deletes the old bond and
retries rather than failing. Sensitive characteristics are only accessible over an
encrypted, authenticated link, protecting agains passive sniffer, MITM attacks and 
extenrnal attackers trying to connect directly to de device without direct acces to
it phisically, does not protect agains an attacker who has the device on their hands.

