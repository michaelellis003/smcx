# 0024. Structured latent-state PyTrees in standard particle filters

Date: 2026-07-19 | Status: accepted (ratified 2026-07-19) |
Supersedes: — | Superseded-by: —

## Context

The standard filters currently require each latent particle to be one
dense vector even though their posterior containers and scan carries are
already JAX PyTrees. This forces structured states such as a nonlinear
state plus per-particle Kalman moments to be packed manually, obstructing
the planned Rao--Blackwellized Dynamax recipe. Parameter-learning
algorithms additionally require Euclidean covariance and proposal
geometry that tree structure alone does not define.

## Options considered

- Keep dense latent vectors — preserves the narrowest contract, but
  leaves structured state packing to every caller.
- Support latent PyTrees with core JAX in the standard filters and
  simulation — broadens the useful model boundary without another
  dependency or a parameter-geometry abstraction.
- Adopt Equinox containers and filtered transformations — adds a model
  framework dependency for behavior JAX already provides.
- Generalize latent states, parameters, and all diagnostics together —
  maximizes uniformity, but conflates structural batching with
  mathematically Euclidean summaries and proposals.

## Decision

We will support a nonempty latent-state PyTree in ``bootstrap_filter``,
``auxiliary_filter``, ``guided_filter``, and ``simulate`` using only core
JAX. A particle cloud has the same fixed tree structure for the whole
run; every dynamic leaf is a JAX array whose leading axis is the
particle axis. Per-particle callbacks receive that structure with
the leading axis removed. Transition and proposal outputs must preserve
the tree structure, trailing leaf shapes, and dtypes. Stored histories
preserve the structure and prepend a time axis to every leaf. A dense
``(N, D)`` array remains a one-leaf PyTree and must retain identical
fixed-key numerical output.

Trajectory reconstruction and posterior prediction will operate leafwise.
Diagnostics that require vector-space arithmetic remain dense-array
APIs and reject structured histories with an explicit error. Liu--West,
tempered SMC, and SMC2 remain dense in this first slice. Liu--West latent
state trees are mechanically possible but deferred to keep the public
change narrow; all three algorithms' parameter clouds remain dense
because their proposal kernels need an explicit Euclidean
parameterization. We will not add Equinox or another PyTree dependency.

## Consequences

Structured Rao--Blackwellized and composite latent states no longer need
manual packing, and user-owned registered PyTree classes remain usable.
Tree structure, leaf shapes, and dtypes cannot change within a compiled
scan; ragged state remains unsupported. More leaves can increase tracing,
compilation, and dispatch overhead, so homogeneous values should stay
packed and CPU/Metal benchmarks must report the cost. Existing dense
summary functions remain simple, while callers with structured state
must select or project a dense leaf before using Euclidean diagnostics.
