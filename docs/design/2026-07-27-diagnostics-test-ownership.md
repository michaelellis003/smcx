# Diagnostics test ownership

*Status: accepted. Date: 2026-07-27.*

This point-in-time snapshot extends, but does not modify,
[the 2026-07-26 CI test-ownership decision](2026-07-26-ci-test-ownership.md).
The focused follow-up and implementation are tracked in
[issue #241](https://github.com/michaelellis003/smcx/issues/241).

## Context

The earlier audit retained scientific tests that own distinct observable
contracts and identified diagnostics tests for separate review. Timing from
[PR #230](https://github.com/michaelellis003/smcx/actions/runs/30227166812)
showed the replicate calibration for `log_ml_variance` taking 56.24 seconds
on Python 3.11, 51.64 seconds on Python 3.13, and 38.56 seconds on macOS CPU.
The generic `tail_ess` JIT smoke took another 15.19, 15.69, and 8.16 seconds
on those legs. The hosted Metal failure in that run surfaced during the
replicate calibration; this is an association, not evidence that the test
caused the backend hang.

Cost is not the ownership problem. Issue #157 made multinomial calibration a
documented contract, so the replicate comparison owns unique integration
evidence. Its original 40-filter ratio band from one-third to three was not a
valid permanent gate: it had neither the repository's derived
five-estimator-standard-error tolerance nor a documented significance level.
At its committed seed, the mean single-run estimate was 0.2738343920, the
empirical log-marginal-likelihood variance was 0.2758219552, and their ratio
was 0.9927940357.

The replacement runs 256 filters with 600 particles and 50 observations in
one JIT-vectorized call. On an M3 Pro with macOS 26.2, Python 3.14.0, and JAX
0.10.2, all 256 estimates were finite on CPU and physical Metal. The CPU
estimate-to-variance ratio was 0.965509686797, with log ratio -0.035099144231
and jackknife standard error 0.097015625466. Metal produced ratio
0.985856122728, log ratio -0.014244855181, and standard error 0.092414212856.
Both are inside their five-SE gates, whose widths are required to detect a
factor-two scale error.

The JIT smoke used a random 500-particle filter and asserted only that its
compiled output was finite. Exact lower-edge, uniform-median,
directional-upper-edge, singleton-tail, and float16-overflow regressions
already execute `tail_ess` eagerly and under JIT.

The same review found two nearby tail-ESS tests that did not state valid
contracts. Tail ESS can exceed global ESS when many equally weighted tail
particles surround one dominant central particle. A random 4,000-particle
test also used an ad hoc 15% band for the uniform-tail result. Shape and bound
checks paid for complete 1,000-particle filters when small deterministic
posteriors exercise the same public result.

This follows the [Scientific Python testing guidance][sp]:
performance matters because faster tests can run in more CI environments.
It also follows [Gandy and Scott's Monte Carlo testing framework][gandy-scott],
which makes false-rejection control an explicit part of statistical testing,
and [pytest's flaky-test guidance][pytest-flaky],
which explains why an unreliable CI signal wastes investigation time.

## Decision

- Replace the sequential 40-replicate calibration with 256 filters evaluated
  by one JIT-vectorized call. Test the log ratio of the mean single-run
  estimate to replicated log-ML variance against five delete-one jackknife
  standard errors. This preserves #157's multinomial calibration contract
  with an explicit error derivation.
- Keep the deterministic NumPy restatement as the formula oracle.
  Coalescence, lag exactness, and argument validation retain their contracts.
- Remove the generic `tail_ess` JIT smoke. Keep the exact eager/JIT numerical
  regressions.
- Replace the false global-ESS ordering with an exact counterexample: two
  equally weighted particles in each 5% tail around one particle holding 90%
  mass have tail ESS two and global ESS 16/13.
- Merge the random uniform-tail, shape, and bound tests into one exact
  contract: each eighth of the existing 32-particle uniform clouds has tail
  ESS four, producing one value for each of three retained rows.
- Do not use slow markers, looser bands, or platform-specific selection.
  Those choices would retain redundant or statistically uncalibrated work.

## Consequences

The suite has three fewer tests. It replaces 185.48 seconds of measured work
across the two originally reviewed tests on the three standard hosted legs,
before coverage and Metal, with a vectorized calibration and small exact
contracts. That total is pre-change attributable time, not a promised
workflow wall-time reduction; post-change CI remains the measurement.

Public APIs, numerical implementation, coverage enforcement, the Python and
platform matrix, hosted Metal containment, and the authoritative physical
Metal gate are unchanged. Persistent compilation caching, parallel pytest,
matrix selection, and release-flow changes remain separate decisions.

[sp]: https://learn.scientific-python.org/development/guides/pytest/
[gandy-scott]: https://arxiv.org/abs/2001.06465
[pytest-flaky]: https://docs.pytest.org/en/stable/explanation/flaky.html
