# Filter evidence test ownership

*Status: accepted. Date: 2026-07-27.*

This is a point-in-time design snapshot. The focused decision and its
implementation are tracked in
[issue #242](https://github.com/michaelellis003/smcx/issues/242).

## Context

[Issue #111](https://github.com/michaelellis003/smcx/issues/111) identified
ordinary sequential float32 accumulation in the bootstrap, auxiliary,
guided, and Liu--West filters. The error was observable in their public
`marginal_loglik` after long series even though every reported evidence
increment remained correct.
[PR #194](https://github.com/michaelellis003/smcx/pull/194) carried a
Neumaier total-and-correction pair through each filter scan and added one
10,000-stage parameterized regression across the four public filters.

[PR #204](https://github.com/michaelellis003/smcx/pull/204) later made the
necessary Metal containment decision: the supported MPS path executes a
sequence of one-step scans while other backends retain one full scan. The
10,000-stage fixture therefore requested 9,999 one-step scans in each MPS
case. In [CI run 30251755598][hosted-run], the four cases consumed 80.01,
114.24, 109.09, and 136.95 seconds on the hosted Metal leg, or 440.29
runner-seconds in total.

The preceding
[CI test-ownership snapshot](2026-07-26-ci-test-ownership.md) requires
permanent tests to own observable package behavior or a numerical invariant
and says that expensive tests should be redesigned without discarding a
unique scientific contract.

## Minimal defect evidence

The float32 spacing at `2**24` is two. In the exact three-term sequence
`[2**24, 1, -(2**24)]`, ordinary float32 addition rounds away the unit, then
cancels the two large terms and returns zero. A compensation term retains
the unit and returns exactly one.

Three terms are minimal for an exact public float32 assertion. After only
the first two terms, the mathematical result cannot be represented as a
float32 scalar; the third term cancels the large magnitude and leaves the
representable unit. As recorded in
[issue #242](https://github.com/michaelellis003/smcx/issues/242), all four
filters returned zero for this sequence at pre-fix commit
[`7890794`][pre-fix]
and return exactly one with the compensated implementation. Their public
increment traces remain exactly equal to the three inputs.

## Decision

- Retain one parameterized public integration test covering the bootstrap,
  auxiliary, guided, and Liu--West filters.
- Replace the 10,000 repeated inputs with the term-minimal float32
  cancellation sequence.
- Assert both parts of the contract exactly: the per-stage evidence
  increments are unchanged and `marginal_loglik` is `1.0`.
- Remove the approximate higher-precision summation oracle. This exact
  sequence needs neither a tolerance nor an external arithmetic oracle.

## Out of scope

This decision does not change the compensated-summation implementation,
public API, fixed-key numerical behavior, CI matrix, release controls, or
Metal containment. It does not alter the separate evidence tests for the
generic runner, checkpointed filtering, Kalman filters, SMC2, or tempering.
It also makes no general performance claim; repository benchmark policy
continues to require dated results and environment metadata for such claims.

## Consequences

The regression still fails if any one-shot filter reverts to ordinary
float32 evidence accumulation, and it continues to protect the exact public
increment trace for all four filter families. Its cost no longer scales with
an arbitrary 10,000-stage fixture. The removed long-horizon input was
historical defect-discovery evidence, not a distinct package contract.

[hosted-run]:
  https://github.com/michaelellis003/smcx/actions/runs/30251755598
[pre-fix]:
  https://github.com/michaelellis003/smcx/commit/7890794
