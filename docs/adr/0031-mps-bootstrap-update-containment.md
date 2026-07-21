# 0031. MPS host-loop containment for bootstrap chunk updates

Date: 2026-07-21 | Status: accepted | Supersedes: 0026/0028 (MPS
`bootstrap_update` execution only) | Superseded-by: —

## Context

jax-mps 0.10.10 exposes an MLX Metal fence-bookkeeping defect in a compiled
multi-step `lax.scan`: a donated dynamic-offset input is misclassified as a
temporary, so a required producer-fence wait can be lost. The resulting first
history write is corrupt in the standalone trigger oracle, while the same
ordered computation as repeated public `bootstrap_step` calls is exact. The
defect reproduces through public MLX operations and is tracked by
[MLX #3880](https://github.com/ml-explore/mlx/issues/3880); no released smcx
API has reproduced it, but the unreleased chunk update does.

## Options considered

- Wait for a fixed jax-mps wheel — avoids downstream code but blocks the
  checkpoint update indefinitely.
- Carry and update history inside the scan — still uses the affected dynamic
  update machinery and is only a graph-scheduling perturbation.
- Run the update on CPU — avoids Metal but silently discards requested device
  acceleration and transfers checkpoint state.
- Loop over the released one-step path on MPS — accepts one dispatch per
  observation while avoiding the demonstrated multi-step executable.

## Decision

When the checkpoint arrays reside on MPS, `bootstrap_update` will execute an
eager Python loop over the public `bootstrap_step`, using the caller's ordered
keys, emissions, and inputs one observation at a time. It will retain the
resulting records and stack them only after the loop; the MPS path will contain
no multi-observation scan or in-scan history buffer. Conditional chunk
evidence will use the same ordered Neumaier accumulation, and history-on,
final-only, PyTree, input-alignment, validation, and degeneracy semantics
remain those of ADR-0028.

The MPS path will reject an outer `jax.jit`, which would stage the Python loop
back into one multi-step executable and defeat the containment. The path is
selected from the checkpoint arrays' resident platform, is automatic, and has
no user override. CPU and other backends retain the scan implementation.
`bootstrap_filter`, `bootstrap_init`, `bootstrap_step`, their fixed-key
outputs, and ADR-0026's `jax-mps>=0.10.10,<0.11` floor remain unchanged. This
narrowly supersedes ADR-0026's no-project-specific-storage consequence and
ADR-0028's scan-execution choice for MPS chunk updates only. smcx will not
patch, repin, or downgrade MLX or jax-mps for this release.

## Consequences

The new chunk API remains releasable on supported Metal hardware without a
local MLX or jax-mps patch. MPS chunk updates pay per-observation dispatch
overhead, so this is a correctness containment rather than a performance
claim. It covers the demonstrated smcx failure; it does not claim to repair
MLX or arbitrary JAX graphs supplied by users.

[smcx #38](https://github.com/michaelellis003/smcx/issues/38) tracks removal.
Removal requires a released jax-mps wheel containing the fixed MLX revision,
30/30 fresh-process passes for both trigger and control oracles, the targeted
and full jax-mps gates, exact D3 equivalence tests, and all four smcx gates
including the physical-M-series suite without xdist. A separately installed
MLX release is insufficient because jax-mps statically links its pinned MLX.
