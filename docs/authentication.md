# Authentication

User accounts are **seeded, not self-registered** — there is no sign-up endpoint or app
screen. `backend/scripts/seed_db.py` (`make seed`) bootstraps a fresh DB with a default
user (`SEED_USER` / `SEED_PASSWORD`, default `somasafe` / `somasafe`); it is idempotent.

## Sessions

Sessions are stateful. Login returns an opaque `access_token` (30 min) + `refresh_token`
(30 days); only their sha256 is stored server-side (`AuthSession` table), so a session can
be revoked instantly. Passwords are argon2-hashed (`pwdlib`).

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/auth/token` | login (form `username`+`password`) → `{access_token, refresh_token, ...}` |
| POST | `/auth/refresh` | `{refresh_token}` → rotated token pair |
| POST | `/auth/logout` | revoke the current session |
| POST | `/auth/logout-all` | revoke every session for the user |
| GET | `/auth/me` | current user |

Requests authenticate with `Authorization: Bearer <access_token>`. An expired access
token (`401`) is refreshed via `/auth/refresh` without re-entering the password.

## Client side (Android app)

The Backend tab gates everything behind sign-in; there is no registration UI. Tokens and
the username are kept in `EncryptedSharedPreferences` (`Auth.kt`/`AuthStore`). Requests
attach the access token and transparently refresh it once on a `401`; logout revokes the
session server-side and clears local tokens.

Being logged in is necessary but not sufficient for the model-download/upload endpoints —
those additionally require the user to be a verified device owner, see
[device-attestation.md](device-attestation.md). Per-model rate limiting on top of auth is
documented in `backend/README.md`.
