# 0028. Streaming bootstrap-filter checkpoints

Date: 2026-07-20 | Status: proposed | Supersedes: — | Superseded-by: —

## Context

The public bootstrap filter is one-shot: its live state and pure scan step are
private, so callers cannot process observations incrementally or resume a run.
Its scan carries normalized log weights and a running evidence sum, while the
resampling decision also consumes the current ESS. A resumable API must retain
that state without changing the existing fixed-key output, explicit-key
policy, input alignment, or structured-state contract.

Chunking introduces a second determinism concern. A step compiled alone and
the same source function lowered inside `lax.scan` can produce different HLO,
so XLA does not provide a backend-independent bitwise guarantee merely because
the Python function and keys match.

## Options considered

- Reuse `ParticleState` alone and recompute everything else — preserves one
  container, but it cannot retain the evidence correction and makes the
  resampling state implicit.
- Extend `ParticleState` — exposes all live values directly, but changes the
  arity of a public container also used by the guided and auxiliary filters.
- Wrap `ParticleState` in a bootstrap-specific checkpoint — adds one narrow
  public type while preserving the existing container and algorithm-specific
  continuation boundaries.
- Run every path as a host loop over the same separately jitted step — gives a
  mechanism for a backend-independent bitwise contract, but replaces one
  compiled time loop with one dispatch per observation.
- Lower one shared pure step inside `lax.scan` and gate equivalence on each
  supported platform — retains the existing compiled-loop architecture, but
  limits exact-equivalence claims to tested compiler/backend combinations.

## Decision

We will add bootstrap-specific `init`, `step`, and chunk-`update` operations.
`bootstrap_init` consumes the first emission and, when input-aware, the first
input. `bootstrap_step` consumes one explicit key, the live checkpoint, the
emission at that time, and its aligned input. `bootstrap_update` consumes an
ordered key array aligned one-for-one with the emissions and optional inputs in
the chunk; it never creates a hidden splitting schedule.

`BootstrapCheckpoint` will be a `NamedTuple` containing only:

- the current `ParticleState`, whose particle PyTree follows ADR-0024 and whose
  log weights are normalized;
- the current ESS read by the next resampling decision; and
- the Neumaier evidence-correction scalar.

The `ParticleState.log_marginal_likelihood` field remains the leading sum in
the ordered accumulator. The compensated cumulative evidence is that value
plus the checkpoint correction. Keeping the correction outside
`ParticleState` avoids changing its public arity and preserves the leading
sum used by the legacy one-shot result. The feature change will correct the
`ParticleState` docstring to identify its log weights as normalized; it will
not change that container's fields.

`BootstrapStepInfo` will contain the ancestor indices, post-weighting ESS,
whether resampling occurred, and the conditional log-evidence increment for
that observation. Initialization reports identity ancestors and
`resampled=False`. A chunk posterior contains only that chunk's histories and
the compensated conditional evidence for that chunk; the returned checkpoint
retains cumulative evidence. Its history-on and final-only forms follow
ADR-0011, and histories from consecutive chunks concatenate in observation
order.

The pure bootstrap step will be shared by direct stepping and scan-based
updates. We choose the scan option: exact equality for repeated steps, any
chunking, and the one-shot composition is a release gate on supported CPU and
physical M-series Metal configurations, but is not promised for an untested
XLA backend/compiler pair. The gate uses array equality with identical ordered
keys and covers both resampling branches. A supported-platform failure blocks
the release rather than being relaxed to a tolerance. This scopes the bitwise
claim explicitly while retaining one compiled loop for a chunk.

`bootstrap_filter` will compose the same initialization and step core. It will
retain its current split schedule, callback order, histories, and fixed-key
arrays exactly. In particular, its `marginal_loglik` remains the leading
ordered sum already returned by the public API; checkpoint operations retain
the correction needed for compensated continuation. The first-emission and
per-time input convention remains ADR-0022.

This contract rolls out for the bootstrap filter first. Auxiliary and guided
filters follow only after this API settles. Native RBPF continuation belongs
to its own algorithm design. Liu–West and SMC² are excluded because their
complete resumable states need separate decisions; an SMC² posterior is not a
checkpoint for its resident inner filters.

## Consequences

Callers can stream observations, retain O(N) live state between chunks, and
join chunk histories without replaying earlier data. Explicit per-observation
keys make chunk boundaries independent of an internal split schedule. The
legacy convenience function remains numerically stable for existing callers,
while checkpoint users retain the additional evidence correction.

The new types and functions expand the public API and therefore require a
feature release. Exact chunking equivalence is a tested support guarantee, not
a theorem about all XLA lowerings. Scan-based updates preserve batching but a
caller that invokes `bootstrap_step` repeatedly accepts per-step dispatch;
debugging or supporting a new backend requires running the equivalence gate.
