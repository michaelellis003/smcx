# 0025. Monotone float32 multinomial prefix rounding

Date: 2026-07-19 | Status: proposed | Supersedes: — | Superseded-by: —

## Context

The all-algorithm baseline found a committed-seed ancestor inversion in
`multinomial` at `N=100,000` on both CPU and Metal. The exponential-spacing
construction is mathematically ordered, but separately rounded outputs of a
parallel float32 prefix sum can decrease locally, violating the documented
nondecreasing-output contract.

## Options considered

- Sort the queries — directly enforces order, but changes the O(N)
  construction to O(N log N).
- Use a sequential prefix sum — preserves the mathematical construction, but
  serializes a large part of the GPU kernel.
- Project the rounded prefix sums through `jnp.maximum.accumulate` — restores
  the invariant in O(N), but its jax-mps 0.10.9 lowering is pathologically
  slow at the campaign scales.
- Project them through `jax.lax.associative_scan(jnp.maximum, ...)` — the same
  O(N) prefix-maximum semantics with a performant lowering on CPU and Metal.

## Decision

We will apply `jax.lax.associative_scan(jnp.maximum, ...)` to the rounded
exponential-spacing prefix sums before normalization. The resulting
nondecreasing queries continue through the shared clipped right-bisect kernel.

## Consequences

The public ordering contract holds at the failing committed key on CPU and
Metal while the construction remains O(N). Multinomial pays one extra
cumulative-max scan. Fixed-key ancestors can change where parallel prefix
rounding previously inverted or collapsed adjacent queries; those outputs did
not satisfy the existing contract. Distributional moment and covariance gates
remain unchanged and must pass on both backends.
