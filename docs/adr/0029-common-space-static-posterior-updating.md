# 0029. Common-space static posterior updating

Date: 2026-07-20 | Status: proposed | Supersedes: — | Superseded-by: —

## Context

Weighted draws from an old static posterior can be corrected for new data,
but they are not a live particle-filter checkpoint. Direct importance
correction needs only the new-data likelihood ratio; creating particles at
new locations needs the old target density in the particles' coordinate
measure. Reusing `temper` would obscure that distinction because `temper`
starts from prior draws and reports full-evidence semantics.

## Options considered

- Offer direct reweighting only — needs the least information, but cannot
  repair a depleted cloud.
- Resample imported particles without an old target — returns duplicates and
  creates no support, despite looking like rejuvenation.
- Fit a density to imported particles — permits moves, but silently replaces
  the old posterior with a new modeling approximation.
- Reuse `temper` or expose arbitrary mutation kernels — conflates evidence
  contracts and commits to abstractions before a second kernel needs them;
  BlackJAX's ongoing
  [factory-stack unwind](https://github.com/blackjax-devs/blackjax/issues/774)
  is the cautionary precedent.
- Add a separate direct-correction API with target-backed adaptive RWM
  bridging — requires an extra callable exactly when new support is needed.

## Decision

We will add a separate `update_static_posterior` path for a fixed-dimensional
dense unconstrained state `u`, initially shaped `(N, d)`. It accepts valid
normalized equal or nonuniform log weights and a per-particle
`log_increment_fn`; a user-owned codec must flatten any PyTree source. Its
conceptual signature is:

```python
update_static_posterior(
    key,
    particles,
    log_weights,
    log_increment_fn,
    *,
    log_base_target_fn=None,
    ...,
)
```

A dedicated NamedTuple will return particles, normalized log weights,
`conditional_log_evidence`, and the diagnostics below. It will not reuse
`TemperedPosterior`, whose evidence and equal-weight guarantees are different.

For direct correction, with normalized old weights and
`ell[i] = log_increment_fn(u[i])`, the updater computes

```text
a[i] = log_weights[i] + ell[i]
conditional_log_evidence = logsumexp(a)
log_weights_new[i] = a[i] - logsumexp(a).
```

This path returns the original support with new normalized weights. Poor ESS
or Pareto-k is reported rather than treated as permission to manufacture new
support. Resampling alone can only duplicate that support and must never be
described as rejuvenation.

Supplying `log_base_target_fn` opts into adaptive resample-move bridging when
the direct candidate violates the configured ESS target. The bridge targets

```text
pi_phi(u) proportional to
    gamma_old(u) * exp(phi * ell(u)),    0 <= phi <= 1.
```

The base target may be unnormalized: its additive constant cancels from RWM
ratios. It must nevertheless evaluate `gamma_old` in the same coordinate
measure as the imported particles.

Each stage reweights, records its normalizer ratio, resamples when required,
and applies random-walk Metropolis moves invariant to `pi_phi`. The first
slice has no generic mutation-kernel interface. Any explicit request for
bridging or rejuvenation without `log_base_target_fn` raises, because an
acceptance ratio needs target values away from the imported particles
([Dai et al., 2022](https://arxiv.org/abs/2007.11936)).

Coordinates are part of that contract. If `theta = decode(u)`, the Jacobian
cancels from the incremental ratio,

```text
pi_new^U(u) / pi_old^U(u) = L_new(decode(u)),
```

but the move target must include it:

```text
log gamma_old^U(u) = log gamma_old^Theta(decode(u))
                     + log |det d theta / d u|.
```

Diagnostics will expose the updated cloud's pre-resampling ESS and maximum
normalized weight, the bridge schedule, move acceptance, and Pareto-k fitted
to the normalized *increment* weights. The implementation will reuse the GPD
fit in `diagnostics.py`; for `S = N` imported particles, it reports the PSIS
threshold `min(1 - 1/log10(S), 0.7)` from
[Vehtari et al. (2024)](https://jmlr.org/papers/v25/19-556.html). Exceeding the
threshold never raises or suppresses the direct result. An all-`-inf`
increment remains a degenerate-weight error.

`conditional_log_evidence` always estimates only `Z_new / Z_old`. Bridge
stage increments accumulate to the same quantity. A full old-plus-new value
may be formed only from an explicitly supplied old evidence estimate and
inherits every qualification of that estimate. The ratio estimate and its
log also inherit finite-sample and source bias; taking the log introduces the
usual Jensen bias even when an unlogged ratio estimator is unbiased.

Source claims will be labeled in the API documentation and examples; smcx
cannot infer or certify provenance:

- Exact independent old-posterior draws have only finite-Monte-Carlo error;
  valid SMC or importance draws retain their supplied nonuniform weights.
- MCMC draws retain finite-chain bias, autocorrelation, divergences, and
  convergence uncertainty. Post-update particle ESS and Pareto-k do not
  replace source-chain ESS, R-hat, or warmup checks.
- Uniform VI draws updated only by `L_new` target `q(u) L_new(u)`, not the
  original Bayesian posterior. Importance correction requires evaluable `q`,
  the old target, and absolute continuity; evidence remains limited when the
  required normalizers are unavailable.
- Arbitrary empirical draws are an approximate warm start. The closest
  published precedent shortens a later SMC schedule using an earlier weighted
  cloud, but does not close the theory gap for imported finite or correlated
  draws ([Generalized Posterior Calibration via SMC,
  arXiv:2404.16528](https://arxiv.org/abs/2404.16528)).

`temper` keeps its prior-sample, prior-density, full-evidence behavior and
bitwise fixed-key outputs. A private shared engine is allowed only under a
frozen-output regression gate. Production work on this updater will not run
concurrently with the tempering-accuracy study.

## Required analytic oracles

For an old Gaussian target `N(m0, C0)` and a new linear-Gaussian batch
`y | theta ~ N(H theta, R)`, tests will use

```text
C1^-1 = C0^-1 + H^T R^-1 H
m1 = C1 (C0^-1 m0 + H^T R^-1 y)
log(Z_new / Z_old) = log N(y; H m0, R + H C0 H^T).
```

The implementation PR must add these RED tests before production code:

1. One stacked batch and two sequential batches agree on weighted mean,
   covariance, and the analytic posterior; sequential conditional evidence
   increments sum to the stacked predictive log density.
2. Particles drawn from an offset Gaussian proposal `q`, with normalized
   `p_old / q` weights, recover the same posterior and evidence despite
   deliberately skewed nonuniform weights.
3. A positive variance `v = exp(u)` uses an inverse-gamma old target and
   zero-mean Gaussian data. Forced resample-move bridging matches the analytic
   inverse-gamma posterior only when `log p_V(exp(u)) + u` supplies the
   Jacobian; an otherwise identical no-`+u` target must show detectable bias.
4. Requesting moves or bridging without a base target raises.
5. A committed large or shifted Gaussian batch makes direct correction
   degenerate enough to trigger adaptive bridging, whose final moments and
   conditional evidence still meet the analytic oracle.
6. Committed benign and mismatched Gaussian updates place Pareto-k below and
   above its sample-size threshold respectively; the latter is returned, not
   raised.
7. For Gaussian VI proposal `q`, uniform imported draws match the explicitly
   labeled approximate target `q L_new`; weights proportional to `p_old / q`
   instead recover the Bayesian target `p_old L_new`.
8. Frozen fixed-key `temper` outputs remain bitwise unchanged.
9. An increment that is `-inf` on the entire cloud raises
   `DegenerateWeightsError`.

Stochastic oracle checks use committed seeds and tolerances equal to five
times a derived estimator standard error, with the derivation beside each
assertion.

## Consequences

The direct path is useful for any common-space weighted cloud and never
pretends that resampling repairs support. Target-backed bridging can recover
from weight collapse, at the cost of requiring a correct density, transform,
and RWM budget. The initial dense/RWM scope leaves constrained PyTrees and
additional move kernels to later evidence-backed ADRs, while the separate API
preserves `temper` and live-filter continuation as distinct contracts.
