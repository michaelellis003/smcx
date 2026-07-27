# CI test ownership and bounded runtime

*Status: accepted. Date: 2026-07-26.*

This is a point-in-time design snapshot. The focused decision and its
implementation are tracked in
[issue #231](https://github.com/michaelellis003/smcx/issues/231).

## Context

The suite grew from 680 to 834 tests during the issue-resolution campaign.
A pull-request run remained close to 22 minutes because hosted Metal was
already the critical path, but total runner work grew from about 68 to 89
minutes. Release-bearing changes repeat the suite for the pull request,
squash merge, and generated release commit.

Most of the CPU increase was concentrated in diagnostics and tempering. One
tempering regression requested an accepted target near the float32 boundary,
then asserted that the resulting schedule had exactly 1,356 stages. It cost
50.82 seconds on macOS CPU, 85.50 seconds on Python 3.13, and 126.27 seconds
on Python 3.11 in
[PR #230 CI](https://github.com/michaelellis003/smcx/actions/runs/30227166812)
while checking an internal numerical schedule rather than a promised result.

CI setup is not the runtime bottleneck, but an unpinned `uv` installer also
caused an unrelated transient failure while resolving its latest release.

## Decision

- Permanent tests continue to own observable `smcx` behavior, numerical
  invariants, or supported integrations. Cost alone is not grounds to remove
  a unique scientific contract.
- The exact 1,356-stage assertion is replaced by a public budget-boundary
  contract: a small `max_stages` value fails, a larger caller-selected value
  succeeds, temperatures increase monotonically, and the final temperature
  is one.
- `uv` is pinned to version 0.11.32 in the shared CI setup action. Dependency
  caches, the Python/platform matrix, coverage enforcement, hosted Metal
  handling, and the authoritative local physical-Metal gate are unchanged.
- Slow stochastic or numerical tests are retained when they provide unique,
  error-calibrated evidence. Redundant validation studies and implementation
  artifacts should instead be reduced, redesigned around a derived estimator
  error, or moved to a dated research snapshot.
- Persistent JAX compilation caching and `pytest-xdist` remain separate
  decisions. A compiled cache is trusted executable state, and xdist adds a
  dependency and needs coverage and determinism validation. Neither is
  required for this fix.
- Release-workflow duplication is a separate control-flow decision. This
  change does not weaken any release or platform gate.

## Consequences

The replacement preserves the meaningful raised-budget behavior while taking
seconds rather than minutes. It removes the three measured per-leg costs
above from each full workflow, and coverage execution also benefits, without
reducing contract or platform coverage. Pinning the installer makes setup
reproducible but is not presented as a test-runtime optimization.

Future CI reductions must identify the contract owner before changing a test.
Moment tests retain the repository's derived five-estimator-standard-error
tolerance. Distributional tests retain a committed seed and documented
significance level. Backend-specific numerical tests remain on every backend
needed to expose the fault.
