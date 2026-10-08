
# Minimal secure aggregation, honest-but-curious server

This document describes a simple protocol for secure aggregation under tight (arguably unrealistic) constraints, based on a more complete algorithm that's robust agains client dropouts but also way more complicated, described in the paper "Practical Secure Aggregation for Privacy-Preserving Machine Learning" K. Bonawit et al. (2017)

## Notation

| Symbol | Meaning |
|---|---|
| `S` | the server (aggregator) |
| `C` | the cohort of one session: an ordered set of `n` clients |
| `u, v` | clients, with a total order (any deterministic one — client id works) |
| `m` | number of scalar model weights (the flattened vector length) |
| `R` | the ring modulus, `2^32` |
| `d_u` | client `u`'s weight delta, `m` floats |
| `W` | the global weights the round (and each of its sessions) is based on |

## Rounds and sessions

A **round** is everything contributed on one snapshot of global weights `W`, exactly as on
the dense paths: it has no row and no state of its own, and it ends when an aggregation
produces the next snapshot. A **session** is one masked-sum cohort inside a round: a few
clients (`SECURE_SESSION_MIN_MEMBERS` to `SECURE_SESSION_MAX_MEMBERS`) whose updates are
summed together, producing a **partial result**, the session's mean delta. Sessions on the
same `W` run one after another as clients arrive, so a round collects as many of them as
clients show up for, and its aggregation trims and averages their partials.

The server learns each session's sum and nothing finer, so a client hides among the `n`
members of its session, not among the whole round: the session size is the privacy level.

## Parties and assumptions

- The server follows the protocol but reads everything it receives.
- Clients follow the protocol. No Byzantine behaviour inside a session; a dropout fails
  only its own session.
- Every client can reach the server. No client can reach another client. All client-to-client information flow is mediated by the server, and there is exactly one such flow: public keys.

## Primitives

Four, all off the shelf:

- **KA** — a Diffie–Hellman key agreement (P-256 ECDH, X25519, whatever both platforms have). `KA.agree(sk_u, pk_v) → shared_uv`, with the defining property `shared_uv = shared_vu`.
- **KDF** — HKDF-SHA256. Turns a shared secret plus a context string into a uniform 32-byte seed.
- **PRG** — a stream cipher used as a keystream generator (AES-256-CTR or ChaCha20, zero nonce). `PRG(seed, m) → m` uniform elements of `Z_R`.
- **Quantizer** — a fixed-point map from `R^m` to `Z_R^m` (below).

No secret sharing. No signatures. No authenticated encryption between clients. No commitments.

## Phase 0 — enrolment

Client `u` generates a long-term KA keypair `(sk_u, pk_u)` and keeps `sk_u` private forever; `sk_u` is never transmitted, never shared, never reconstructed — this is the entire reason no dropout-recovery machinery is needed. It publishes `pk_u` to the server. Because the server is honest, this implementation folds enrolment into the session join rather than a separate persistent registry: when `u` joins a session it sends `pk_u`, and the server **snapshots** it into that session's roster (`SecureSessionMember.ka_public_key`). The snapshot — not a lookup against a mutable key table — is what the masks are derived against, so a client that later rotates its key cannot desync a session already sealed. A client reuses the same long-term keypair across sessions (the freshness that matters comes from the `session_id` salt, below), and no PKI or signatures are needed since the server never impersonates a peer.

## Phase 1 — local training (client, entirely offline)

Client `u` downloads the active weights `W`, trains locally against them, and computes
`d_u = θ_local − W`. Training comes first, so the only part of the protocol a client has to
stay online for is the short join → seal → mask → submit exchange that follows.

## Phase 2 — session setup (server)

The client joins with the id of the weights it trained on
(`POST /model/secure/join/{key}/{weights_id}`); a join on weights that are no longer active
is refused and the client retrains. The join places it in that snapshot's single open
session (opening one if there is none), and the server fixes, and commits to, a
**session descriptor**:

```
session_id    unique, never reused
W             the base global weights
roster        [(u, pk_u) for u in C], in the canonical order
n = |C|       must be >= 3
B             clipping bound (per-coordinate)
S = floor(2^31 / (n * B))    fixed-point scale
R = 2^32
```

The roster must be **identical for every client and fixed before anyone masks**. That is the only synchronisation this protocol needs, and it's the sole reason a session is a first-class object (`SecureSession`). The join that fills a session to `SECURE_SESSION_MAX_MEMBERS` **seals** it on the spot — freezing the roster and fixing `n` and `S` — and the next join opens a new session on the same `W`. A session that stays open for `SECURE_SESSION_OPEN_TIMEOUT_SECONDS` is sealed by the worker's sweep if it has at least `SECURE_SESSION_MIN_MEMBERS`, and failed otherwise. Joins lock the open session row until they commit, so they serialize, and no member can slip in after the seal. Sealed, the session serves its descriptor to every client in `C` (`GET /model/secure/session/{session_id}`).

A user holds **at most one seat per snapshot** (`UNIQUE(base_weights_id, user_id)`): a second join on the same `W` is refused, unless the seat is still in an open session, in which case it only refreshes the key.

`n ≥ 3` because the sum of two values plus one participant's own value reveals the other's. In general, `n − 1` colluding clients always deanonymise the last one — that's inherent to *any* secure aggregation scheme, since the output is a sum. Pick `n` large enough that the aggregate is meaningfully anonymising.

## Phase 3 — masking (client)

With the descriptor in hand, client `u` quantizes and masks the `d_u` it already computed.

**Quantize.** Masking requires exact arithmetic; floats don't cancel. Map each coordinate into the ring:

```
q_u[i] = round(clip(d_u[i], -B, B) * S) mod R
```

`S` is chosen so `|Σ_u Σ true values| < 2^31`, i.e. the signed sum can never wrap. Values in `[2^31, 2^32)` are read back as negative — plain two's-complement in the ring.

**Derive masks.** For every *other* member `v` of the roster:

```
shared_uv = KA.agree(sk_u, pk_v)
seed_uv   = KDF(shared_uv, salt = session_id, info = "secagg-v1")
mask_uv   = PRG(seed_uv, m)          # m elements of Z_R
```

Because `KA.agree` is symmetric, `u` and `v` independently derive **the same** `seed_uv`, hence the same `mask_uv` — without ever exchanging a message. This is the whole trick.

**Mask.** Use the total order to make each pair's mask cancel: the smaller-indexed member adds, the larger subtracts (or vice versa — just be consistent).

```
y_u = q_u
for v in roster, v != u:
    if u < v: y_u = (y_u + mask_uv) mod R
    else:     y_u = (y_u - mask_uv) mod R
```

`y_u` is `m` elements of `Z_R`. The client sends only `y_u`. Nothing else. It never learns anything about any other client, and it never talks to one.

## Phase 4 — session sum (server)

The server collects `y_u` from all `n` members; the sweep dispatches the sum as soon as the last one arrives. With all `n` in hand:

```
z = (Σ_{u in C} y_u) mod R
```

Every pairwise mask appears exactly twice in that sum — once added by one member, once subtracted by the other — so it cancels **exactly**, in the ring. What survives:

```
z = (Σ_u q_u) mod R
```

Dequantize and average:

```
z_signed     = z if z < 2^31 else z - 2^32      # elementwise
partial_mean = (z_signed / S) / n
```

The worker streams the members' vectors into a single accumulator, so memory stays proportional to `m` whatever `n` is. It stores `partial_mean` as the session's partial result (`SecurePartial`) and drops the masked vectors, which are useless once summed.

The session lifecycle is `open → sealed → summing → summed | failed`. Summing claims the session (`sealed → summing`) before doing any work, so a duplicate dispatch is a no-op; a worker lost mid-sum sends the session back to `sealed` to be summed again.

**Dropouts.** If any member is still missing `SECURE_SESSION_SEAL_TIMEOUT_SECONDS` after the seal, the session fails. There is no partial recovery within a session, but a failed session costs little: its sum was never computed, so the masked vectors already received reveal nothing (each is still padded by the missing member's masks) and are deleted along with the session's seats. The members who did submit get their rate limits cleared and can join a fresh session on the same `W` with the same delta, without retraining; the member who dropped keeps its cooldown. A session whose base weights stop being the active ones fails too, since its deltas no longer apply.

## Phase 5 — round aggregation (server)

The round is aggregated by the same task as the dense paths, fired for every model by beat every `FED_AGG_INTERVAL_SECONDS` (see [server-internals.md](server-internals.md)). For a `secure` model it reads the partial results of the summed sessions on the active `W`. If those sessions cover at least `FED_MIN_SUBMISSIONS` members, it combines the partials with the coordinate-wise trimmed mean (`FED_TRIM_RATIO`) and commits

```
W_next = W + trimmed_mean(partial_means)
```

with its serving artifacts, through the same one-child-per-parent guard as the dense path; otherwise it does nothing, and the round stays live until the next tick. It never waits for sessions still in progress: once `W_next` lands, they are on stale weights, and they fail or are ignored. The gateway takes no part in this: it only ever checks that a join or submission targets the active weights.

Each session counts once in the trimmed mean, whatever its size, so a session's weight is not proportional to its members. In exchange, a poisoned update can drag only its own session's mean (always within roughly `[−B, B)`), and trimming drops up to `⌊k · FED_TRIM_RATIO⌋` such sessions out of `k`. With the default ratio of 0.2 that needs at least 5 sessions; below that the trimmed mean is a plain mean of the partials.

## Why it's private

The server's view of a session is `{y_u}` plus the public keys. Two claims:

1. **Each `y_u` alone is uniform.** `y_u = q_u + (masks)`, and each mask is a PRG output on a seed the server cannot derive — deriving `seed_uv` requires `sk_u` or `sk_v`, so under DDH plus PRG security the masks are computationally indistinguishable from uniform in `Z_R^m`. A uniform mask added mod `R` is a one-time pad. `y_u` carries no information about `q_u`.

2. **The set `{y_u}` reveals exactly the sum and nothing more.** The masks are pairwise-antisymmetric, so `Σ y_u = Σ q_u` is fixed, but conditioned on that sum the individual `y_u` are jointly uniform. (This is Lemma 6.1 in the paper — it's the load-bearing statement, and it's proved by induction on `n`.) So the server learns `Σ q_u` and nothing beyond it.

That's the guarantee: **the server learns each session's sum, and only that.** The round's result is computed from those sums, so it reveals nothing more. The one-seat-per-snapshot rule keeps it that way across sessions: if a delta could appear in several sums with different peers, the server could solve the resulting linear system for individual updates.

## What was deleted and why

Worth being able to say precisely, because it's the whole contribution of the design:

| Paper mechanism | Exists to… | Why you don't need it |
|---|---|---|
| Shamir sharing of `sk_u` | let survivors reconstruct a dropout's masks | a missing client fails only its session, and the survivors rejoin another |
| Self-mask `b_u` + its shares | stop the server from unmasking a *slow* client it wrongly declared dead | no recovery round exists, so there is nothing to unmask |
| Unmasking round | collect the shares above | nothing to collect |
| ConsistencyCheck round | stop an *active* server telling different clients different dropout sets | server is honest |
| PKI + signatures | stop an active server Sybil-attacking a client in the key-exchange round | server is honest |

Delete all five and the four-round interactive protocol collapses to: publish a key once, then **one client→server message per session**. No client→client messages at any point.

## The invariants you must not break

These are the ways a correct-looking implementation silently leaks:

- **`session_id` in the KDF salt.** Long-term keys mean `shared_uv` is constant forever. If the seed doesn't change per session, then `y¹_u − y²_u = q¹_u − q²_u` and the server reads the difference of two of your updates in the clear. Fresh seed every session, derived from a value that never repeats.
- **One submission per client per session, enforced structurally.** Two `y_u` under the same masks leak `q¹_u − q²_u` for the same reason. Reject the second, don't merely rate-limit it.
- **One seat per client per snapshot, enforced structurally.** A delta summed in two sessions with different peers gives the server two equations about it; enough of them and individual updates can be solved for.
- **Seats are released only if their session was never summed.** A session that fails before its sum is computed revealed nothing, so its members may contribute again; one that fails during or after summing keeps its seats consumed.
- **No cleartext submission may enter a masked session.** One unmasked `q_v` in the sum breaks nothing mathematically — the total is still correct — but `v` has published its own update, and the masks of its peers no longer sum to zero across the *remaining* set, so the server can subtract `q_v` and continue. Enforce one submission format per round.
- **The roster is frozen at seal.** If two clients disagree about who's in `C`, or about a member's public key, the masks don't cancel and the aggregate is garbage. Snapshot the public keys into the session, don't join to a mutable key table.
- **The order is canonical and shared.** Both sides of every pair must agree on who adds and who subtracts.

## What is unconditionally lost

The server cannot see any individual update, so nothing that inspects an individual update can run: no per-submission norm gate, no z-scored outlier filter, no per-client validity verdict. This is not an implementation gap — secure aggregation and per-client input validation are in direct tension, and the paper leaves range-proofs as future work precisely because the cheapest known construction costs more than the entire protocol. What remains available:

- **Client-side clipping to `B`** bounds any single client's influence on the mean to `B/n` per coordinate — but nothing forces an honest-looking client to clip.
- **Aggregate-level sanity**: check each session's mean is finite and plausible, and reject the *session* if not.
- **Session-level trimming**: the round's trimmed mean runs over whole sessions, so a poisoned update can be dropped only together with its session's honest members.

Byzantine robustness and dropout robustness are the same future-work item, and they arrive together: both require the server to learn something about individuals, or require clients to prove things about themselves in zero knowledge.
