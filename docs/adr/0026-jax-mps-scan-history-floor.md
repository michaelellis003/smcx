# 0026. jax-mps 0.10.10 floor for scan history

Date: 2026-07-19 | Status: accepted | Supersedes: — | Superseded-by: —

## Context

The all-algorithm campaign measured `store_history=True` at 2.17--2.27
times the history-off Metal runtime and 1.16--1.52 GiB peak device memory
for a 45.8 MiB dense output. jax-mps 0.10.9 lowers
`stablehlo.dynamic_update_slice` through operand-wide gather, mask, and
selection operations, so each stacked `lax.scan` output rewrites its full
accumulator at every step. jax-mps 0.10.10, released 2026-07-18, includes
[PR #219](https://github.com/tillahoffmann/jax-mps/pull/219)'s native
`mlx.slice_update`,
[PR #220](https://github.com/tillahoffmann/jax-mps/pull/220)'s native dynamic
slice, and
[PR #222](https://github.com/tillahoffmann/jax-mps/pull/222)'s shared clamped
start construction.

## Options considered

- Retain 0.10.9 and rewrite smcx history storage -- duplicates a backend fix
  in algorithm code and risks changing the public history contract.
- Pin an unreleased jax-mps commit -- obtains the fix but weakens package and
  lockfile reproducibility.
- Raise the optional Metal floor to 0.10.10 -- consumes the released upstream
  fix without adding project-specific branches.

## Decision

We will require `jax-mps>=0.10.10,<0.11` for the optional `metal` extra. Metal
history performance evidence from 0.10.9 remains historical diagnostic
evidence and is not used for current performance claims. The all-algorithm
Metal profiles and local Metal test gate must run again against the new floor.

## Consequences

Stacked scan histories use the native MLX slice primitives that motivated the
JAX pivot, without a smcx-specific storage path. Metal users can no longer
install jax-mps 0.10.9 through the supported extra. The lockfile changes, the
backend must pass the full local Metal suite, and CPU/Metal comparisons across
the floor are invalid unless rerun under one source and dependency identity.
