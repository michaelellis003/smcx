# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
r"""Diagnostic utilities for particle filter posteriors.

Posterior summaries (Vehtari: *report posterior summaries with
uncertainty*; McElreath: *always report intervals, not just means*):

- :func:`weighted_mean` — weighted posterior mean at each time step
- :func:`weighted_variance` — weighted posterior variance
- :func:`weighted_quantile` — weighted quantiles for credible
  intervals
- :func:`param_weighted_mean` — weighted parameter mean (Liu-West)
- :func:`param_weighted_quantile` — weighted parameter quantiles

Computational faithfulness (Vehtari: *can we trust the computation?*):

- :func:`particle_diversity` — fraction of unique particles per step
- :func:`log_ml_increments` — per-step evidence contributions
- :func:`pareto_k_diagnostic` — per-step Pareto-k reliability
- :func:`tail_ess` — ESS for tail quantiles
- :func:`diagnose` — summary diagnostics with warnings

Model comparison:

- :func:`log_bayes_factor` — log Bayes factor between two models
- :func:`replicated_log_ml` — Monte Carlo variability of log-ML
- :func:`cumulative_log_score` — running predictive log-score

Posterior predictive checks:

- :func:`posterior_predictive_sample` — one-step-ahead predictions

Scoring rules:

- :func:`crps` — Continuous Ranked Probability Score

All functions are pure, stateless, operate on arrays from
:class:`~smcjax.containers.ParticleFilterPosterior` or
:class:`~smcjax.containers.LiuWestPosterior`, and are
JIT-compatible.
"""

import math
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Array, Float, Int

from smcjax._utils import _weighted_quantile_1d
from smcjax.containers import LiuWestPosterior, ParticleFilterResult
from smcjax.types import PRNGKeyT, Scalar
from smcjax.weights import normalize

_PRIOR_STRENGTH = 10
"""Effective sample size of the weakly informative prior on k."""

_PRIOR_K = 0.5
"""Prior mean for k.  Vehtari et al. (2024) use 0.5."""

_EXC_FLOOR = 1e-100
"""Exceedance floor below which the tail is treated as degenerate."""


def _weighted_mean_field(
    log_weights: Float[Array, 'ntime num_particles'],
    field: Float[Array, 'ntime num_particles dim'],
) -> Float[Array, 'ntime dim']:
    """Compute weighted mean of a (ntime, N, D) field.

    Args:
        log_weights: Log weights, shape ``(ntime, num_particles)``.
        field: Values to average, shape ``(ntime, num_particles, D)``.

    Returns:
        Weighted means, shape ``(ntime, D)``.
    """
    weights = vmap(normalize)(log_weights)
    return jnp.einsum('tn,tnd->td', weights, field)


def _weighted_quantile_field(
    log_weights: Float[Array, 'ntime num_particles'],
    field: Float[Array, 'ntime num_particles dim'],
    q: Float[Array, ' num_quantiles'],
) -> Float[Array, 'ntime num_quantiles dim']:
    """Compute weighted quantiles of a (ntime, N, D) field.

    Args:
        log_weights: Log weights, shape ``(ntime, num_particles)``.
        field: Values, shape ``(ntime, num_particles, D)``.
        q: Quantile levels in [0, 1].

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, D)``.
    """
    weights = vmap(normalize)(log_weights)

    def _quantile_one_time(
        field_t: Float[Array, 'num_particles dim'],
        weights_t: Float[Array, ' num_particles'],
    ) -> Float[Array, 'num_quantiles dim']:
        """Compute quantiles for one time step, all dims."""
        return vmap(_weighted_quantile_1d, in_axes=(1, None, None))(
            field_t, weights_t, q
        ).T

    return vmap(_quantile_one_time)(field, weights)


def weighted_mean(
    posterior: ParticleFilterResult,
) -> Float[Array, 'ntime state_dim']:
    r"""Compute the weighted mean of particles at each time step.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Weighted means, shape ``(ntime, state_dim)``.
    """
    return _weighted_mean_field(
        posterior.filtered_log_weights,
        posterior.filtered_particles,
    )


def weighted_variance(
    posterior: ParticleFilterResult,
) -> Float[Array, 'ntime state_dim']:
    r"""Compute the weighted variance of particles at each time step.

    Uses the formula :math:`V = \sum_i w_i (x_i - \mu)^2` where
    :math:`\mu` is the weighted mean.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Weighted variances, shape ``(ntime, state_dim)``.
    """
    means = weighted_mean(posterior)
    deviations = posterior.filtered_particles - means[:, None, :]
    return _weighted_mean_field(
        posterior.filtered_log_weights,
        deviations**2,
    )


def weighted_quantile(
    posterior: ParticleFilterResult,
    q: Float[Array, ' num_quantiles'],
) -> Float[Array, 'ntime num_quantiles state_dim']:
    r"""Compute weighted quantiles of particles at each time step.

    Uses a sorted resampling approach for JIT compatibility:
    sorts particles, computes cumulative weights, and interpolates.

    Args:
        posterior: Particle filter posterior output.
        q: Quantile levels in [0, 1], e.g. ``jnp.array([0.025, 0.975])``
            for a 95% credible interval.

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, state_dim)``.
    """
    return _weighted_quantile_field(
        posterior.filtered_log_weights,
        posterior.filtered_particles,
        q,
    )


def log_ml_increments(
    posterior: ParticleFilterResult,
) -> Float[Array, ' ntime']:
    r"""Extract per-step log marginal likelihood increments.

    The marginal log-likelihood can be decomposed as:

    .. math::

        \log p(y_{1:T}) = \sum_{t=1}^T
            \log p(y_t \mid y_{1:t-1})

    This function returns the individual increments, which diagnose
    which observations are hardest for the model.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Per-step evidence increments, shape ``(ntime,)``.  These sum
        to ``posterior.marginal_loglik``.
    """
    return posterior.log_evidence_increments


def particle_diversity(
    posterior: ParticleFilterResult,
) -> Float[Array, ' ntime']:
    r"""Compute the fraction of unique particles at each time step.

    Particle diversity measures path degeneracy: a value near 1 means
    most particles are distinct, while near 0 means heavy duplication
    after resampling.

    Uses an indicator-based method (not ``jnp.unique``) for JIT
    compatibility: counts the fraction of particles that differ from
    their predecessor in the sorted order.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Diversity fraction in [0, 1] at each time step,
        shape ``(ntime,)``.
    """
    ancestors = posterior.ancestors  # (ntime, num_particles)
    num_particles = ancestors.shape[1]

    def _diversity_one_step(
        anc: Int[Array, ' num_particles'],
    ) -> Float[Array, '']:
        """Count fraction of unique ancestors at one time step."""
        sorted_anc = jnp.sort(anc)
        # First element is always unique; subsequent are unique if
        # different from predecessor
        is_unique = jnp.concatenate(
            [
                jnp.array([True]),
                sorted_anc[1:] != sorted_anc[:-1],
            ]
        )
        return jnp.sum(is_unique) / num_particles

    return vmap(_diversity_one_step)(ancestors)


def log_bayes_factor(
    log_ml_1: Scalar,
    log_ml_2: Scalar,
) -> Scalar:
    r"""Compute the log Bayes factor between two models.

    .. math::

        \log BF_{12} = \log p(y_{1:T} \mid M_1)
                     - \log p(y_{1:T} \mid M_2)

    Positive values favour model 1; negative values favour model 2.

    .. warning::

        Marginal likelihoods are sensitive to the prior in ways that
        the posterior is not.  Bayes factors evaluate priors, not
        posteriors (Gelman, 2023).  With weakly informative priors the
        marginal likelihood is dominated by prior tails that have
        little effect on posterior inference, so a Bayes factor can
        reverse sign under prior changes that leave the posterior
        essentially unchanged.  Consider complementing Bayes factors
        with predictive comparisons (e.g. cumulative log-scores from
        :func:`cumulative_log_score` or CRPS from :func:`crps`) and
        use :func:`replicated_log_ml` to quantify Monte Carlo
        variability.

    Args:
        log_ml_1: Log marginal likelihood of model 1.
        log_ml_2: Log marginal likelihood of model 2.

    Returns:
        Scalar log Bayes factor.
    """
    return jnp.asarray(log_ml_1) - jnp.asarray(log_ml_2)


def replicated_log_ml(
    key: PRNGKeyT,
    filter_fn: Callable[[PRNGKeyT], Scalar],
    num_replicates: int,
) -> Float[Array, ' num_replicates']:
    r"""Run a particle filter multiple times to assess log-ML variability.

    Uses :func:`jax.vmap` over PRNG keys for efficient parallel
    evaluation.  The resulting distribution of log-ML estimates
    quantifies Monte Carlo uncertainty in the evidence.

    Args:
        key: JAX PRNG key.
        filter_fn: Function ``(key) -> scalar`` that runs a particle
            filter and returns the marginal log-likelihood.
        num_replicates: Number of independent filter runs.

    Returns:
        Array of log-ML estimates, shape ``(num_replicates,)``.
    """
    keys = jr.split(key, num_replicates)
    return jnp.asarray(vmap(filter_fn)(keys))


def param_weighted_mean(
    posterior: LiuWestPosterior,
) -> Float[Array, 'ntime param_dim']:
    r"""Compute the weighted mean of parameter particles at each step.

    Args:
        posterior: Liu-West filter posterior output.

    Returns:
        Weighted parameter means, shape ``(ntime, param_dim)``.
    """
    return _weighted_mean_field(
        posterior.filtered_log_weights,
        posterior.filtered_params,
    )


def param_weighted_quantile(
    posterior: LiuWestPosterior,
    q: Float[Array, ' num_quantiles'],
) -> Float[Array, 'ntime num_quantiles param_dim']:
    r"""Compute weighted quantiles of parameter particles at each step.

    Args:
        posterior: Liu-West filter posterior output.
        q: Quantile levels in [0, 1], e.g. ``jnp.array([0.025, 0.975])``
            for a 95% credible interval.

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, param_dim)``.
    """
    return _weighted_quantile_field(
        posterior.filtered_log_weights,
        posterior.filtered_params,
        q,
    )


# --- Posterior predictive checks -------------------------------------------


def posterior_predictive_sample(
    key: PRNGKeyT,
    posterior: ParticleFilterResult,
    transition_sampler: Callable,
    emission_sampler: Callable,
    num_samples: int | None = None,
) -> Float[Array, 'ntime num_samples emission_dim']:
    r"""Draw one-step-ahead posterior predictive samples.

    At each time step :math:`t`, we:

    1. Resample particle indices from the normalised weights.
    2. Propagate each resampled state through ``transition_sampler``.
    3. Draw an emission from ``emission_sampler``.

    This gives iid samples from the posterior predictive
    :math:`p(y_{t+1} \mid y_{1:t})`, which can be compared with
    the actual observation :math:`y_{t+1}` for posterior predictive
    checking (Gelman et al., 2013, ch. 6).

    Args:
        key: JAX PRNG key.
        posterior: Particle filter posterior output.
        transition_sampler: Function ``(key, state) -> state``.
        emission_sampler: Function ``(key, state) -> emission``.
        num_samples: Number of predictive draws per time step.
            Defaults to the number of particles.

    Returns:
        Predictive samples, shape
        ``(ntime, num_samples, emission_dim)``.
    """
    ntime, n_particles = posterior.filtered_log_weights.shape
    if num_samples is None:
        num_samples = n_particles

    def _sample_one_step(
        log_weights_t: Float[Array, ' num_particles'],
        particles_t: Float[Array, 'num_particles state_dim'],
        step_key: PRNGKeyT,
    ) -> Float[Array, 'num_samples emission_dim']:
        """Draw predictive samples at one time step."""
        k1, k2, k3 = jr.split(step_key, 3)
        weights = jnp.exp(log_weights_t - jnp.max(log_weights_t))
        weights = weights / jnp.sum(weights)
        indices = jr.choice(k1, n_particles, shape=(num_samples,), p=weights)
        resampled = particles_t[indices]
        # Propagate through transition
        trans_keys = jr.split(k2, num_samples)
        propagated = vmap(transition_sampler)(trans_keys, resampled)
        # Draw emissions
        emit_keys = jr.split(k3, num_samples)
        return vmap(emission_sampler)(emit_keys, propagated)

    step_keys = jr.split(key, ntime)
    return vmap(_sample_one_step)(
        posterior.filtered_log_weights,
        posterior.filtered_particles,
        step_keys,
    )


def crps(
    predictions: Float[Array, ' num_samples'],
    observation: Scalar,
) -> Scalar:
    r"""Compute the Continuous Ranked Probability Score.

    CRPS is a proper scoring rule for probabilistic forecasts:

    .. math::

        \text{CRPS} = \mathbb{E}|Y - y|
                     - \tfrac{1}{2}\,\mathbb{E}|Y - Y'|

    where :math:`Y, Y'` are iid predictive samples and :math:`y`
    is the observation.

    Args:
        predictions: iid samples from the predictive distribution.
        observation: Observed scalar value.

    Returns:
        Scalar CRPS (lower is better, zero for perfect prediction).
    """
    obs = jnp.asarray(observation)
    n = predictions.shape[0]
    # E|Y - y|
    term1 = jnp.mean(jnp.abs(predictions - obs))
    # E|Y - Y'| via sort-based O(N log N) identity:
    #   E|Y-Y'| = (2 / N^2) * sum_i (2i - N + 1) * Y_{(i)}
    y_sorted = jnp.sort(predictions)
    i = jnp.arange(n, dtype=predictions.dtype)
    term2 = 2.0 * jnp.sum((2.0 * i - n + 1.0) * y_sorted) / (n * n)
    return jnp.asarray(term1 - 0.5 * term2)


# --- Pareto-k diagnostic ---------------------------------------------------


def _fit_generalized_pareto(
    x: Float[Array, ' m'],
) -> Float[Array, '']:
    r"""Fit GPD shape k via Zhang & Stephens (2009) with Vehtari prior.

    Implements the profile-likelihood estimator of Zhang and Stephens
    (2009) with Bayesian model averaging over a grid of candidate
    shape parameters, then applies the weakly informative prior from
    Vehtari, Simpson, Gelman, Yao, and Gabry (2024) that shrinks
    :math:`\hat{k}` toward 0.5.

    Matches the algorithm in NumPyro's ``_fit_generalized_pareto_impl``
    and ArviZ's ``gpdfitnew``.

    Args:
        x: Positive exceedances (sorted ascending), length m.

    Returns:
        Estimated GPD shape parameter k.
    """
    m = x.shape[0]
    num_candidates = 30 + math.isqrt(m)
    prior = 3  # Zhang-Stephens grid prior

    x_star = x[-1]  # max (sorted ascending)
    # First-quartile value for the grid denominator
    x_q25 = x[max(0, m // 4 - 1)]

    # Zhang-Stephens candidate b values:
    #   b_i = 1/x_star + (1 - sqrt(m / i)) / (prior * x_q25)
    # for i = 0.5, 1.5, ..., num_candidates - 0.5
    i = jnp.arange(1, num_candidates + 1, dtype=jnp.float64) - 0.5
    b_grid = 1.0 / x_star + (1.0 - jnp.sqrt(m / i)) / (prior * x_q25)

    # Profile log-likelihood for each candidate:
    #   k_i = mean(log1p(-b_i * x))
    #   L_i = m * (log(-b_i / k_i) - k_i - 1)
    log_terms = jnp.log1p(-b_grid[:, None] * x[None, :])
    k_grid = jnp.mean(log_terms, axis=1)

    log_lik = m * (jnp.log(-b_grid / k_grid) - k_grid - 1.0)

    # Replace NaN/Inf with -inf before computing weights
    log_lik = jnp.where(jnp.isfinite(log_lik), log_lik, -jnp.inf)

    # Posterior weights via softmax (matches NumPyro's pairwise formulation)
    log_lik_max = jnp.max(log_lik)
    w = jnp.exp(log_lik - log_lik_max)
    w_sum = jnp.sum(w)
    w = w / (w_sum + _EXC_FLOOR)

    # Zero out negligible weights for numerical stability
    eps_threshold = 10.0 * jnp.finfo(jnp.float64).eps
    w = jnp.where(w >= eps_threshold, w, 0.0)
    w = w / (jnp.sum(w) + _EXC_FLOOR)

    # Posterior mean for b, then derive k
    b_hat = jnp.sum(w * b_grid)
    k_hat = jnp.mean(jnp.log1p(-b_hat * x))

    # Vehtari et al. (2024) prior regularisation:
    #   k_final = k * m/(m+a) + a * 0.5/(m+a)
    a = jnp.float64(_PRIOR_STRENGTH)
    k_reg = k_hat * m / (m + a) + a * _PRIOR_K / (m + a)
    return jnp.asarray(k_reg)


def _fit_pareto_k(
    log_weights: Float[Array, ' num_particles'],
) -> Float[Array, '']:
    r"""Fit the shape parameter k of a generalised Pareto distribution.

    Extracts the upper tail of the importance weights and fits a GPD
    via the Zhang and Stephens (2009) profile-likelihood estimator
    with the Vehtari et al. (2024) weakly informative prior.

    The tail sample is the top
    ``M = ceil(min(0.2 * N, 3 * sqrt(N)))`` order statistics,
    following ArviZ and NumPyro conventions.

    Args:
        log_weights: Normalised log importance weights at one step.

    Returns:
        Estimated shape parameter k.
    """
    n = log_weights.shape[0]
    # Tail count: min(20% of N, 3*sqrt(N)), at least 10.
    m = max(10, math.ceil(min(0.2 * n, 3.0 * math.sqrt(n))))
    m = min(m, n - 1)

    # Sort ascending; tail is the last m elements.
    log_w_sorted = jnp.sort(log_weights)
    # Cutoff: the element just below the tail
    cutoff = log_w_sorted[n - m - 1]
    # Exceedances in the original weight scale:
    #   exp(log_w_tail) - exp(cutoff)
    # This matches ArviZ/NumPyro: exceedance above the threshold
    # weight, not above the threshold log-weight.
    log_tail = log_w_sorted[n - m :]
    exceedances = jnp.exp(log_tail) - jnp.exp(cutoff)
    # Ensure non-negative (numerical noise)
    exceedances = jnp.maximum(exceedances, 0.0)

    # When all weights are equal the exceedances are zero and the
    # GPD fit is degenerate.  Return the prior mean in that case.
    max_exc = jnp.max(exceedances)
    prior_mean = _PRIOR_K * _PRIOR_STRENGTH / (m + _PRIOR_STRENGTH)
    return jnp.asarray(
        jnp.where(
            max_exc > _EXC_FLOOR,
            _fit_generalized_pareto(exceedances),
            prior_mean,
        )
    )


def pareto_k_diagnostic(
    posterior: ParticleFilterResult,
) -> Float[Array, ' ntime']:
    r"""Compute the Pareto-k diagnostic at each time step.

    Fits a generalised Pareto distribution (GPD) to the upper tail
    of the importance weights using the Zhang and Stephens (2009)
    profile-likelihood estimator with the weakly informative prior
    from Vehtari, Simpson, Gelman, Yao, and Gabry (2024).

    The shape parameter :math:`\hat{k}` indicates reliability:

    - :math:`\hat{k} < 0.5`: good, finite variance of the IS estimate
    - :math:`0.5 \le \hat{k} < 0.7`: marginal
    - :math:`0.7 \le \hat{k} < 1.0`: unreliable (infinite variance)
    - :math:`\hat{k} \ge 1.0`: very unreliable (infinite mean)

    The tail size is ``ceil(min(0.2 * N, 3 * sqrt(N)))`` order
    statistics, matching the conventions of ArviZ and NumPyro.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Per-step Pareto-k estimates, shape ``(ntime,)``.
    """
    return vmap(_fit_pareto_k)(posterior.filtered_log_weights)


# --- Tail-ESS --------------------------------------------------------------


def tail_ess(
    posterior: ParticleFilterResult,
    q: float = 0.05,
) -> Float[Array, ' ntime']:
    r"""Compute tail effective sample size at each time step.

    Tail-ESS measures how well the weighted particle approximation
    represents the tails of the distribution (Vehtari, Gelman,
    Simpson, Carpenter, and Burkner, 2020).  We compute the ESS
    for the indicator function :math:`I(w_i \ge w_{(q)})`, i.e.
    how well the largest weights are distributed.

    Specifically, for normalised weights :math:`w_i` we compute
    the ESS of the weights truncated below the :math:`(1-q)`
    quantile:

    .. math::

        \text{tail-ESS} = \frac{
            \bigl(\sum_{i : w_i \ge c} w_i \bigr)^2
        }{
            \sum_{i : w_i \ge c} w_i^2
        }

    where :math:`c` is the :math:`(1-q)` quantile of the weights.

    Args:
        posterior: Particle filter posterior output.
        q: Tail probability.  Default 0.05, so we examine the top
            5% weight mass.

    Returns:
        Tail-ESS at each time step, shape ``(ntime,)``.
    """

    def _tail_ess_one_step(
        log_weights: Float[Array, ' num_particles'],
    ) -> Float[Array, '']:
        """Compute tail-ESS for one time step."""
        weights = jnp.exp(log_weights - jnp.max(log_weights))
        weights = weights / jnp.sum(weights)
        # Threshold: (1-q) quantile of weights
        sorted_w = jnp.sort(weights)
        cum_w = jnp.cumsum(sorted_w)
        threshold = sorted_w[jnp.searchsorted(cum_w, 1.0 - q)]
        # Tail weights
        tail_mask = weights >= threshold
        tail_w = jnp.where(tail_mask, weights, 0.0)
        sum_w = jnp.sum(tail_w)
        sum_w2 = jnp.sum(tail_w**2)
        return jnp.asarray(
            jnp.where(
                sum_w2 > 0.0,
                sum_w**2 / sum_w2,
                jnp.float64(0.0),
            )
        )

    return vmap(_tail_ess_one_step)(posterior.filtered_log_weights)


# --- Cumulative log-score ---------------------------------------------------


def cumulative_log_score(
    posterior: ParticleFilterResult,
) -> Float[Array, ' ntime']:
    r"""Compute the cumulative one-step-ahead predictive log-score.

    The log-evidence increments :math:`\log p(y_t \mid y_{1:t-1})`
    are already one-step-ahead predictive log-densities.  This
    function returns their running sum:

    .. math::

        S_t = \sum_{s=1}^{t} \log p(y_s \mid y_{1:s-1})

    so that :math:`S_T` equals the total marginal log-likelihood.
    Comparing :math:`S_t` across models gives a time-resolved
    predictive comparison that, unlike Bayes factors, is less
    sensitive to the prior.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Cumulative log-scores, shape ``(ntime,)``.
    """
    return jnp.cumsum(posterior.log_evidence_increments)


# --- Diagnostic summary ----------------------------------------------------


def diagnose(
    posterior: ParticleFilterResult,
    ess_threshold: float = 0.1,
    diversity_threshold: float = 0.1,
    pareto_k_threshold: float = 0.7,
) -> dict[str, Any]:
    r"""Summarise filter health and flag potential problems.

    Runs a battery of diagnostics and returns a dictionary with
    scalar summaries and a list of plain-text warnings.  The
    thresholds are configurable; the defaults flag situations
    where the particle approximation is likely unreliable.

    Args:
        posterior: Particle filter posterior output.
        ess_threshold: Fraction of N below which ESS triggers a
            warning.
        diversity_threshold: Diversity below which a warning is
            triggered.
        pareto_k_threshold: Pareto-k above which a warning is
            triggered.

    Returns:
        Dictionary with keys:

        - ``min_ess``: minimum ESS across all time steps
        - ``min_diversity``: minimum particle diversity
        - ``max_pareto_k``: maximum Pareto-k across time steps
        - ``min_tail_ess``: minimum tail-ESS across time steps
        - ``ess_below_threshold``: count of steps where
          ESS < ``ess_threshold * N``
        - ``warnings``: list of diagnostic warning strings
    """
    n = posterior.filtered_particles.shape[1]
    ess_vals = posterior.ess
    diversity = particle_diversity(posterior)
    k_hat = pareto_k_diagnostic(posterior)
    t_ess = tail_ess(posterior)

    min_ess = float(jnp.min(ess_vals))
    min_div = float(jnp.min(diversity))
    max_k = float(jnp.max(k_hat))
    min_t_ess = float(jnp.min(t_ess))
    ess_count = int(jnp.sum(ess_vals < ess_threshold * n))

    warnings: list[str] = []
    if min_ess < ess_threshold * n:
        warnings.append(
            f'ESS dropped below {ess_threshold:.0%} of N '
            f'at {ess_count} step(s) (min ESS = {min_ess:.1f})'
        )
    if min_div < diversity_threshold:
        warnings.append(
            f'Particle diversity fell below '
            f'{diversity_threshold:.0%} (min = {min_div:.3f})'
        )
    if max_k > pareto_k_threshold:
        warnings.append(
            f'Pareto-k exceeded {pareto_k_threshold} '
            f'(max k = {max_k:.3f}); importance weights '
            f'have infinite variance at some steps'
        )

    return {
        'min_ess': min_ess,
        'min_diversity': min_div,
        'max_pareto_k': max_k,
        'min_tail_ess': min_t_ess,
        'ess_below_threshold': ess_count,
        'warnings': warnings,
    }
