# The reproducibility contract

Every stochastic smcx function takes an explicit JAX PRNG key and
documents how it splits that key. This page states the library-wide
contract in one place; the per-function Notes restate only what is
specific to each entry point.

## Keys are contracts

A function's key-split tree is part of its public behavior. When a
docstring says the root key "splits once into the prior and stage
roots" or "splits by time, then draw", that schedule is frozen by
tests: an implementation change that reroutes a subkey is a breaking
change, not an internal detail. Two consequences follow.

The same key reproduces the same result on the same backend and code
path. Rerunning `bootstrap_filter` with the same key, data, and
configuration returns bitwise-identical output on the same machine and
device. This is what makes stochastic failures diffable: a failing
seed is a permanent reproduction, and smcx's own statistical tests
commit their root keys for exactly this reason.

Independent streams stay independent of adaptive decisions. Adaptive
algorithms (`temper`, `ibis`, and the conditional-resampling filters)
reserve every subkey a stage could need before reading any value, so
whether an earlier stage resampled cannot shift a later stage's random
stream. This branch invariance is also test-frozen.

## What is deliberately not promised

- **Cross-backend equality.** CPU, CUDA, and Metal lower reductions
  and factorizations differently; the same key gives statistically
  identical but not bitwise-identical results across devices.
  Tolerances in smcx's own suite compare backends at a small multiple
  of machine epsilon, never exactly.
- **Cross-version equality.** A release may change a key schedule or
  kernel only as a documented breaking change, but numerical drift
  within tolerances can occur at any dependency bump.
- **Draw-count prefix stability.** Asking for more draws or particles
  reallocates subkeys; the first `n` draws of a larger run do not
  match a smaller run.
- **Code-path stability under fallbacks.** Where a function documents
  alternative numerical paths (for example `posterior_sample`'s
  Cholesky-or-spectral covariance factors), same-key equality holds
  per path; an input that switches paths switches streams.

## Practical guidance

Thread keys with `jax.random.split` and never reuse a key across
calls whose outputs you treat as independent. For replicated
experiments, split one root into per-replicate keys so each replicate
is a complete, independently keyed run; estimate Monte Carlo error
from the replicates rather than trusting any single key. To archive a
result, record the package version, the backend, the precision
configuration, and the root key — together these reproduce the run.
