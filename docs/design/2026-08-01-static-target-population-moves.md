# Static-target population moves: architecture refinement

*Status: accepted 2026-08-01. This is a narrow amendment to
`docs/design/2026-07-27-workbench-architecture.md`. It supersedes the blanket
wording of §8's fixed-loop sentence and guardrail 1 only for the
population-adaptive static-target case defined below. Every other rule in the
course-keeping guide remains binding.*

## The conflict

The shared `FeynmanKac + run_smc` loop is the only driver for state-space
sequential importance-resampling. Its transition is per-particle,
pre-potential, device-traceable, and returns a standard filter record.

`temper` already has a different demonstrated requirement: after correcting
a static parameter cloud, it fits a proposal to the weighted population in
host float64, selects, and applies a target-invariant move with an acceptance
diagnostic. Host float64 is an established numerical boundary, not an
implementation convenience: it avoids cancellation at ordinary parameter
offsets and uses a guarded factorization policy. Proposed IBIS needs the same
population operation after each data correction.

Neither existing generic loop is a suitable owner. Extending `run_smc` would
add post-potential cloud hooks and arbitrary diagnostic channels to the public
state-space FK record. `run_particle_filter` owns whole steps, but its callback
runs inside `lax.scan`, where the NumPy factor cannot execute; it also discards
algorithm-specific carry and exposes no extra trace channel. A JAX covariance
rewrite, `pure_callback`, public host execution mode, or auxiliary-particle
encoding would each replace or weaken an established contract to serve this
case.

## Refined loop rule

1. State-space sequential importance-resampling remains
   `FeynmanKac + run_smc`. Named filters may not add a driver.
2. A dense static-target sampler whose invariant move adapts to a weighted
   population may use a host shell only through the one private shared
   static-SMC stage core defined here.
3. The exception applies only when the decision issue identifies a required
   host numerical boundary. Conditional resampling, a desired Python API, or
   implementation ease is not enough.
4. A fixed data schedule still pre-splits every stage key before value-based
   decisions. Only the stage orchestration is host-side; likelihood,
   prefix-target, and mutation kernels remain pure and compiled.
5. The public wrapper must disclose that its outer function is not jittable or
   vmappable. No platform branch may enter the stage or its consumers.

This is not permission for another named filter loop, a general host mode, or
a monolithic static-sampler engine.

## One static-SMC stage

Before IBIS is exported, existing `temper` and new `ibis` both use one private
stage contract. Given a resident cloud, normalized log weights, a vector of
target increments, two already-disjoint stage keys, a selection rule,
resampler, and target-specific move callback, the stage exclusively owns:

- log-domain correction and degeneracy gating;
- the evidence increment and Neumaier carry;
- pre-selection ESS and the selection decision;
- resampler invocation, int32/range validation, aligned PyTree gathering, and
  the uniform-weight reset;
- move dispatch only on selection, structural acceptance validation, and a
  zero acceptance value otherwise; and
- post-stage ESS plus the complete stage diagnostic record.

The move receives both the corrected weighted population and gathered seeds,
so the default can fit the existing covariance before selection without
duplicating stage order. `temper` retains its released root/stage key tree;
IBIS freezes its new pre-split tree. Both supply resampling and mutation keys
unconditionally, and the shared stage never advances a key inside a branch.

Thin shells retain only irreducibly different work:

- `temper` solves an adaptive next temperature, constructs
  `delta * log_likelihood`, and stops at one;
- `ibis` reads the next fixed observation, constructs its exact likelihood
  increment and prefix target, and stops after T rows; and
- each assembles its distinct public result record.

Temperature bisection, data-prefix evaluation, target caches, and result
containers are not generalized. SMC²'s resident inner filters and PMMH replay
also remain outside this core.

## Compatibility and gates

The `temper` rewire must preserve every released fixed-key result under its
existing CPU fixture before IBIS is public. The shared stage gets direct
uncompiled and compiled-equivalence tests for correction/evidence; forced and
skipped selection tests; resampler validation; and a key-consumption test.
IBIS adds its own branch-invariant pre-split schedule, prefix-value/gradient,
history, statistical, float32, and physical-Metal gates.

The refinement adds no dependency and no public generic-core surface. It can
be revisited only through another dated decision if a device-native
population factor is shown to preserve the accepted numerical domain or a
second execution model demonstrates a broader public abstraction.
