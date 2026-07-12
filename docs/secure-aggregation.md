
# Minimal secure aggregation, honest-but-curious server

This document describes a simple protocol for secure aggregation under tight (arguably unrealistic) constraints, based on a more complete algorithm that's robust agains client dropouts but also way more complicated, described in the paper "Practical Secure Aggregation for Privacy-Preserving Machine Learning" K. Bonawit et al. (2017)

## Notation

| Symbol | Meaning |
|---|---|
| `S` | the server (aggregator) |
| `C` | the cohort for one round: an ordered set of `n` clients |
| `u, v` | clients, with a total order (any deterministic one — client id works) |
| `m` | number of scalar model weights (the flattened vector length) |
| `R` | the ring modulus, `2^32` |
| `d_u` | client `u`'s weight delta, `m` floats |
| `W` | the global weights the round is based on |

## Parties and assumptions

- The server follows the protocol but reads everything it receives.
- Clients follow the protocol. No dropouts, no Byzantine behaviour.
- Every client can reach the server. No client can reach another client. All client-to-client information flow is mediated by the server, and there is exactly one such flow: public keys.

## Primitives

Four, all off the shelf:

- **KA** — a Diffie–Hellman key agreement (P-256 ECDH, X25519, whatever both platforms have). `KA.agree(sk_u, pk_v) → shared_uv`, with the defining property `shared_uv = shared_vu`.
- **KDF** — HKDF-SHA256. Turns a shared secret plus a context string into a uniform 32-byte seed.
- **PRG** — a stream cipher used as a keystream generator (AES-256-CTR or ChaCha20, zero nonce). `PRG(seed, m) → m` uniform elements of `Z_R`.
- **Quantizer** — a fixed-point map from `R^m` to `Z_R^m` (below).

No secret sharing. No signatures. No authenticated encryption between clients. No commitments.

## Phase 0 — enrolment

Client `u` generates a long-term KA keypair `(sk_u, pk_u)` and keeps `sk_u` private forever; `sk_u` is never transmitted, never shared, never reconstructed — this is the entire reason no dropout-recovery machinery is needed. It publishes `pk_u` to the server. Because the server is honest, this implementation folds enrolment into round join rather than a separate persistent registry: when `u` joins a round it sends `pk_u`, and the server **snapshots** it into that round's roster (`SecureRoundMember.ka_public_key`). The snapshot — not a lookup against a mutable key table — is what the masks are derived against, so a client that later rotates its key cannot desync a round already sealed. A client reuses the same long-term keypair across rounds (the freshness that matters comes from the `round_id` salt, below), and no PKI or signatures are needed since the server never impersonates a peer.

## Phase 1 — round setup (server)

The server fixes, and commits to, a **round descriptor**:

```
round_id      unique, never reused
W             the base global weights for this round
roster        [(u, pk_u) for u in C], in the canonical order
n = |C|       must be >= 3
B             clipping bound (per-coordinate)
S = floor(2^31 / (n * B))    fixed-point scale
R = 2^32
```

The roster must be **identical for every client and fixed before anyone masks**. That is the only synchronisation this protocol needs, and it's the sole reason a round is a first-class object rather than an implicit window. The server distributes the descriptor to every client in `C`.

`n ≥ 3` because the sum of two values plus one participant's own value reveals the other's. In general, `n − 1` colluding clients always deanonymise the last one — that's inherent to *any* secure aggregation scheme, since the output is a sum. Pick `n` large enough that the aggregate is meaningfully anonymising.

## Phase 2 — local work (client, entirely offline)

Client `u` receives the descriptor, trains locally against `W`, and computes `d_u = θ_local − W`.

**Quantize.** Masking requires exact arithmetic; floats don't cancel. Map each coordinate into the ring:

```
q_u[i] = round(clip(d_u[i], -B, B) * S) mod R
```

`S` is chosen so `|Σ_u Σ true values| < 2^31`, i.e. the signed sum can never wrap. Values in `[2^31, 2^32)` are read back as negative — plain two's-complement in the ring.

**Derive masks.** For every *other* member `v` of the roster:

```
shared_uv = KA.agree(sk_u, pk_v)
seed_uv   = KDF(shared_uv, salt = round_id, info = "secagg-v1")
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

## Phase 3 — aggregation (server)

The server collects `y_u` from all `n` members. If any member is missing when the round's deadline passes, **the round fails**: discard the submissions, keep the previous global weights, start a new round. There is no partial recovery — that is the accepted scope cut, and the thing dropout robustness would buy you.

With all `n` in hand:

```
z = (Σ_{u in C} y_u) mod R
```

Every pairwise mask appears exactly twice in that sum — once added by one member, once subtracted by the other — so it cancels **exactly**, in the ring. What survives:

```
z = (Σ_u q_u) mod R
```

Dequantize and average:

```
z_signed   = z if z < 2^31 else z - 2^32      # elementwise
mean_delta = (z_signed / S) / n
W_next     = W + mean_delta
```

`mean_delta` is the uniformly-weighted FedAvg mean of the clients' deltas, up to quantization error. It's numerically the same thing your unmasked path already computes; only the route it took differs.

## Why it's private

The server's view is `{y_u}` plus the public keys. Two claims:

1. **Each `y_u` alone is uniform.** `y_u = q_u + (masks)`, and each mask is a PRG output on a seed the server cannot derive — deriving `seed_uv` requires `sk_u` or `sk_v`, so under DDH plus PRG security the masks are computationally indistinguishable from uniform in `Z_R^m`. A uniform mask added mod `R` is a one-time pad. `y_u` carries no information about `q_u`.

2. **The set `{y_u}` reveals exactly the sum and nothing more.** The masks are pairwise-antisymmetric, so `Σ y_u = Σ q_u` is fixed, but conditioned on that sum the individual `y_u` are jointly uniform. (This is Lemma 6.1 in the paper — it's the load-bearing statement, and it's proved by induction on `n`.) So the server learns `Σ q_u` and nothing beyond it.

That's the guarantee: **the server learns the sum, and only the sum.**

## What was deleted and why

Worth being able to say precisely, because it's the whole contribution of the design:

| Paper mechanism | Exists to… | Why you don't need it |
|---|---|---|
| Shamir sharing of `sk_u` | let survivors reconstruct a dropout's masks | no dropouts — a missing client just fails the round |
| Self-mask `b_u` + its shares | stop the server from unmasking a *slow* client it wrongly declared dead | no recovery round exists, so there is nothing to unmask |
| Unmasking round | collect the shares above | nothing to collect |
| ConsistencyCheck round | stop an *active* server telling different clients different dropout sets | server is honest |
| PKI + signatures | stop an active server Sybil-attacking a client in the key-exchange round | server is honest |

Delete all five and the four-round interactive protocol collapses to: publish a key once, then **one client→server message per round**. No client→client messages at any point.

## The invariants you must not break

These are the ways a correct-looking implementation silently leaks:

- **`round_id` in the KDF salt.** Long-term keys mean `shared_uv` is constant forever. If the seed doesn't change per round, then `y¹_u − y²_u = q¹_u − q²_u` and the server reads the difference of two of your updates in the clear. Fresh seed every round, derived from a value that never repeats.
- **One submission per client per round, enforced structurally.** Two `y_u` under the same masks leak `q¹_u − q²_u` for the same reason. Reject the second, don't merely rate-limit it.
- **No cleartext submission may enter a masked round.** One unmasked `q_v` in the sum breaks nothing mathematically — the total is still correct — but `v` has published its own update, and the masks of its peers no longer sum to zero across the *remaining* set, so the server can subtract `q_v` and continue. Enforce one submission format per round.
- **The roster is frozen at seal.** If two clients disagree about who's in `C`, or about a member's public key, the masks don't cancel and the aggregate is garbage. Snapshot the public keys into the round, don't join to a mutable key table.
- **The order is canonical and shared.** Both sides of every pair must agree on who adds and who subtracts.

## What is unconditionally lost

The server cannot see any individual update, so nothing that inspects an individual update can run: no per-submission norm gate, no z-scored outlier filter, no per-client validity verdict. This is not an implementation gap — secure aggregation and per-client input validation are in direct tension, and the paper leaves range-proofs as future work precisely because the cheapest known construction costs more than the entire protocol. What remains available:

- **Client-side clipping to `B`** bounds any single client's influence on the mean to `B/n` per coordinate — but nothing forces an honest-looking client to clip.
- **Aggregate-level sanity**: check the resulting `mean_delta` is finite and its norm is plausible, and reject the *round* if not.

Byzantine robustness and dropout robustness are the same future-work item, and they arrive together: both require the server to learn something about individuals, or require clients to prove things about themselves in zero knowledge.
