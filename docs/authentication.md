# Authentication

User accounts are **seeded, not self-registered** — there is no sign-up endpoint or app
screen. `backend/scripts/seed_db.py` (`make seed`) bootstraps a fresh DB with a default
user (`SEED_USER` / `SEED_PASSWORD`, default `somasafe` / `somasafe`); it is idempotent.

## Sessions

Sessions are stateful. Login returns an opaque `access_token` (30 min) + `refresh_token`
(30 days); only their sha256 is stored server-side, so a session can be revoked instantly.
The two live in different stores, split by how often each is checked: the `access_token`
hash sits in Redis with a TTL matching its own lifetime (`api/lib/session.py`), since it's
looked up on every authed request and a Postgres row (plus a write to bump a last-used
timestamp) per request is pure overhead for a token that expires in minutes anyway; the
`refresh_token` hash stays in Postgres (`AuthSession` table) since it's long-lived and only
touched on login/refresh/logout. Passwords are argon2-hashed (`pwdlib`).

Logging out (or rotating via refresh) revokes the `AuthSession` row and drops the Redis
access-token key; `logout-all` additionally walks a per-user Redis index to drop every
live access token for that user. One deliberate gap: rotating a refresh token does not
retroactively kill the access token it was paired with — that access token simply expires
on its own short TTL, same as it would if the client just kept using it instead of
refreshing early.

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
