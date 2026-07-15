# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

#
# Ported to MLX from smcjax (https://github.com/michaelellis003/smcjax,
# frozen @ e93d527), Apache-2.0. Modified: MLX arrays; tail_ess
# reimplemented as a quantile-based tail diagnostic (the smcjax
# version measured weight concentration while citing Vehtari et al. —
# panel finding, design §8); Pareto-k documentation corrected
# (variance is infinite for all k >= 0.5; 0.7 is the practical
# reliability threshold) and `diagnose` adopts the sample-size-
# dependent threshold min(1 - 1/log10(N), 0.7); the GPD fit runs in
# numpy float64 host-side (sanctioned diagnostics escape hatch,
# ADR-0003).

r"""Diagnostic utilities for particle filter posteriors.

Posterior summaries: :func:`weighted_mean`, :func:`weighted_variance`,
:func:`weighted_quantile`, :func:`param_weighted_mean`,
:func:`param_weighted_quantile`. Computational faithfulness:
:func:`particle_diversity`, :func:`log_ml_increments`,
:func:`pareto_k_diagnostic`, :func:`tail_ess`, :func:`diagnose`.
Model comparison: :func:`log_bayes_factor`, :func:`replicated_log_ml`,
:func:`cumulative_log_score`. Predictive checks:
:func:`posterior_predictive_sample`, :func:`crps`.

ESS (stored on the posterior) is a resampling trigger, not a
convergence certificate (Elvira, Martino & Robert 2022); the
diagnostics here exist because weights can look healthy while the
approximation is not. All functions are pure and operate on any
:class:`~smcx.containers.ParticleFilterResult`.
"""

import math
from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
from jaxtyping import Float

from smcx.containers import LiuWestPosterior, ParticleFilterResult
from smcx.resampling import _searchsorted_take, multinomial
from smcx.types import KeyT, Scalar
from smcx.weights import normalize

_PRIOR_STRENGTH = 10
_PRIOR_K = 0.5
_EXC_FLOOR = 1e-100


# --- weighted summaries ----------------------------------------------


def _weighted_mean_field(log_weights: mx.array, field: mx.array) -> mx.array:
    # Elementwise-multiply + pairwise-summed reduction: never a
    # matmul/dot against the weight vector (~1000x accuracy gap,
    # docs/research/numerical-methods.md).
    w = mx.vmap(normalize)(log_weights)
    return mx.sum(w[:, :, None] * field, axis=1)


def weighted_mean(
    posterior: ParticleFilterResult,
) -> Float[mx.array, "ntime state_dim"]:
    """Weighted posterior mean of the particles at each time step."""
    return _weighted_mean_field(
        posterior.filtered_log_weights, posterior.filtered_particles
    )


def weighted_variance(
    posterior: ParticleFilterResult,
) -> Float[mx.array, "ntime state_dim"]:
    """Weighted posterior variance at each time step (two-pass)."""
    w = mx.vmap(normalize)(posterior.filtered_log_weights)
    mean = mx.sum(w[:, :, None] * posterior.filtered_particles, axis=1)
    centered = posterior.filtered_particles - mean[:, None, :]
    return mx.sum(w[:, :, None] * centered * centered, axis=1)


def _weighted_quantile_1d(
    values: mx.array, weights: mx.array, q: mx.array
) -> mx.array:
    """Midpoint-CDF weighted quantiles for one 1-D vector."""
    order = mx.argsort(values)
    v = mx.take(values, order)
    w = mx.take(weights, order)
    cum = mx.cumsum(w)
    # Midpoint CDF: centre each particle's mass in its interval,
    # normalized so the axis is [0, 1].
    mid = (mx.concatenate([mx.zeros((1,)), cum[:-1]]) + cum) / (
        2.0 * mx.maximum(cum[-1], 1e-30)
    )
    # Linear interpolation of (mid, v) at q via right-bisect. The
    # take-chain searchsorted, NOT the Metal kernel: this helper runs
    # under vmap, where CustomKernel has no rule (ADR-0009 carve-out),
    # and q is tiny so kernel choice is irrelevant.
    n = v.shape[0]
    hi = mx.clip(_searchsorted_take(mid, q), 1, n - 1)
    lo = hi - 1
    x0, x1 = mx.take(mid, lo), mx.take(mid, hi)
    y0, y1 = mx.take(v, lo), mx.take(v, hi)
    t = mx.clip((q - x0) / mx.maximum(x1 - x0, 1e-12), 0.0, 1.0)
    return y0 + t * (y1 - y0)


def _weighted_quantile_field(
    log_weights: mx.array, field: mx.array, q: mx.array
) -> mx.array:
    # Plain loops over (time, dim): nested mx.vmap corrupts the
    # take-based bisect in the 1-D helper (flatten semantics under
    # stacked batch dims — observed nondeterministically wrong).
    ntime, _, dim = field.shape
    out = []
    for t in range(ntime):
        w_t = normalize(log_weights[t])
        cols = [
            _weighted_quantile_1d(field[t, :, d], w_t, q) for d in range(dim)
        ]
        out.append(mx.stack(cols, axis=1))
    return mx.stack(out)


def weighted_quantile(
    posterior: ParticleFilterResult,
    q: Float[mx.array, " num_quantiles"],
) -> Float[mx.array, "ntime num_quantiles state_dim"]:
    """Weighted quantiles (midpoint-CDF interpolation) per step.

    Args:
        posterior: Filter output.
        q: Quantile levels in [0, 1], e.g. ``mx.array([0.025, 0.975])``.

    Returns:
        Quantiles, shape ``(ntime, num_quantiles, state_dim)``.
    """
    return _weighted_quantile_field(
        posterior.filtered_log_weights, posterior.filtered_particles, q
    )


def param_weighted_mean(
    posterior: LiuWestPosterior,
) -> Float[mx.array, "ntime param_dim"]:
    """Weighted mean of the Liu-West parameter particles per step."""
    return _weighted_mean_field(
        posterior.filtered_log_weights, posterior.filtered_params
    )


def param_weighted_quantile(
    posterior: LiuWestPosterior,
    q: Float[mx.array, " num_quantiles"],
) -> Float[mx.array, "ntime num_quantiles param_dim"]:
    """Weighted quantiles of the Liu-West parameter particles."""
    return _weighted_quantile_field(
        posterior.filtered_log_weights, posterior.filtered_params, q
    )


# --- computational faithfulness ---------------------------------------


def particle_diversity(
    posterior: ParticleFilterResult,
) -> Float[mx.array, " ntime"]:
    """Fraction of unique ancestors per step (path-degeneracy gauge).

    Near 1: distinct lineages survive; near 0: heavy duplication.
    Sort-based (compile-safe), no ``unique``.
    """
    ancestors = posterior.ancestors
    n = ancestors.shape[1]

    def one_step(anc):
        s = mx.sort(anc)
        distinct = mx.sum(mx.not_equal(s[1:], s[:-1]).astype(mx.float32)) + 1.0
        return distinct / n

    return mx.vmap(one_step)(ancestors)


def log_ml_increments(
    posterior: ParticleFilterResult,
) -> Float[mx.array, " ntime"]:
    """Per-step evidence increments (sums to ``marginal_loglik``)."""
    return posterior.log_evidence_increments


def cumulative_log_score(
    posterior: ParticleFilterResult,
) -> Float[mx.array, " ntime"]:
    """Running one-step-ahead predictive log-score.

    Cumulative sum of the evidence increments; the final entry equals
    the marginal log-likelihood. Less prior-sensitive than Bayes
    factors for model comparison.
    """
    return mx.cumsum(posterior.log_evidence_increments)


def log_bayes_factor(log_ml_1: Scalar, log_ml_2: Scalar) -> Float[mx.array, ""]:
    """Log Bayes factor of model 1 over model 2.

    Note both inputs are log-Zhat values: each is downward-biased by
    ~Var/2 (Jensen), and the biases differ with MC variance — use
    :func:`replicated_log_ml` to check the comparison is resolved.
    """
    return mx.array(log_ml_1) - mx.array(log_ml_2)


def replicated_log_ml(
    key: KeyT,
    filter_fn: Callable[[KeyT], Scalar],
    num_replicates: int,
) -> Float[mx.array, " num_replicates"]:
    """Independent filter replicates of the log-ML estimate.

    A sequential Python loop (the filter shell is not vmappable);
    the spread quantifies Monte Carlo uncertainty in the evidence —
    target Var(log Zhat) ~ 1 when tuning N (Pitt et al. 2012).
    """
    keys = mx.random.split(key, num_replicates)
    return mx.stack([
        mx.array(filter_fn(keys[r])) for r in range(num_replicates)
    ])


# --- posterior predictive ---------------------------------------------


def posterior_predictive_sample(
    key: KeyT,
    posterior: ParticleFilterResult,
    transition_sampler: Callable,
    emission_sampler: Callable,
    num_samples: int | None = None,
) -> Float[mx.array, "ntime num_samples emission_dim"]:
    """One-step-ahead posterior predictive draws per time step.

    Resamples states from the step-t weighted cloud, propagates, and
    emits — iid draws from ``p(y_{t+1} | y_{1:t})`` for predictive
    checking (Gelman et al. 2013, ch. 6).
    """
    ntime, n_particles = posterior.filtered_log_weights.shape
    m = n_particles if num_samples is None else num_samples
    out = []
    step_keys = mx.random.split(key, ntime)
    for t in range(ntime):
        k1, k2, k3 = mx.random.split(step_keys[t], 3)
        w = normalize(posterior.filtered_log_weights[t])
        idx = multinomial(k1, w, m)
        states = mx.take(posterior.filtered_particles[t], idx, axis=0)
        tkeys = mx.random.split(k2, m)
        propagated = mx.vmap(transition_sampler)(tkeys, states)
        ekeys = mx.random.split(k3, m)
        out.append(mx.vmap(emission_sampler)(ekeys, propagated))
    return mx.stack(out)


def crps(
    predictions: Float[mx.array, " num_samples"],
    observation: Scalar,
) -> Float[mx.array, ""]:
    """Continuous Ranked Probability Score (lower is better).

    ``CRPS = E|Y - y| - E|Y - Y'|/2`` via the sort-based
    O(N log N) identity; a proper scoring rule, zero for a perfect
    point prediction.
    """
    obs = mx.array(observation)
    n = predictions.shape[0]
    term1 = mx.mean(mx.abs(predictions - obs))
    y_sorted = mx.sort(predictions)
    i = mx.arange(n).astype(predictions.dtype)
    term2 = 2.0 * mx.sum((2.0 * i - n + 1.0) * y_sorted) / (n * n)
    return term1 - 0.5 * term2


# --- Pareto-k (numpy f64 host-side; ADR-0003 diagnostics hatch) -------


def _fit_generalized_pareto(x: np.ndarray) -> float:
    """Zhang & Stephens (2009) GPD shape fit + Vehtari (2024) prior.

    Matches NumPyro's ``_fit_generalized_pareto_impl`` / ArviZ
    ``gpdfitnew`` (both Apache-2.0 conventions; reimplemented from
    the papers — never port from avehtari/PSIS, which is GPL-3).
    """
    m = x.shape[0]
    num_candidates = 30 + math.isqrt(m)
    prior = 3
    x_star = x[-1]
    x_q25 = x[max(0, m // 4 - 1)]
    i = np.arange(1, num_candidates + 1, dtype=np.float64) - 0.5
    b_grid = 1.0 / x_star + (1.0 - np.sqrt(m / i)) / (prior * x_q25)
    with np.errstate(all="ignore"):
        k_grid = np.mean(np.log1p(-b_grid[:, None] * x[None, :]), axis=1)
        log_lik = m * (np.log(-b_grid / k_grid) - k_grid - 1.0)
    log_lik = np.where(np.isfinite(log_lik), log_lik, -np.inf)
    w = np.exp(log_lik - log_lik.max())
    w = w / (w.sum() + _EXC_FLOOR)
    w = np.where(w >= 10.0 * np.finfo(np.float64).eps, w, 0.0)
    w = w / (w.sum() + _EXC_FLOOR)
    b_hat = float(np.sum(w * b_grid))
    k_hat = float(np.mean(np.log1p(-b_hat * x)))
    a = float(_PRIOR_STRENGTH)
    return k_hat * m / (m + a) + a * _PRIOR_K / (m + a)


def _fit_pareto_k(log_weights: np.ndarray) -> float:
    n = log_weights.shape[0]
    m = max(10, math.ceil(min(0.2 * n, 3.0 * math.sqrt(n))))
    m = min(m, n - 1)
    lw = np.sort(log_weights.astype(np.float64))
    cutoff = lw[n - m - 1]
    exceedances = np.maximum(np.exp(lw[n - m :]) - np.exp(cutoff), 0.0)
    prior_mean = _PRIOR_K * _PRIOR_STRENGTH / (m + _PRIOR_STRENGTH)
    if exceedances.max() <= _EXC_FLOOR:
        return prior_mean
    return _fit_generalized_pareto(exceedances)


def pareto_k_diagnostic(
    posterior: ParticleFilterResult,
) -> Float[mx.array, " ntime"]:
    """Per-step Pareto-k reliability of the importance weights.

    GPD tail fit (top ``min(0.2N, 3*sqrt(N))`` order statistics,
    ArviZ/NumPyro convention). Interpretation (Vehtari et al. 2024):
    the importance-ratio variance is infinite for ALL ``k >= 0.5``;
    estimates remain practically reliable up to ``k < 0.7`` (PSIS
    convergence-rate results), so 0.7 — or the sample-size-dependent
    ``min(1 - 1/log10(N), 0.7)`` — is the action threshold, not the
    infinite-variance boundary. Computed host-side in float64
    (diagnostics escape hatch, ADR-0003); not compile-safe.
    """
    lw = np.array(posterior.filtered_log_weights, dtype=np.float64)
    return mx.array([_fit_pareto_k(lw[t]) for t in range(lw.shape[0])])


# --- tail-ESS (corrected semantics; design §8) ------------------------


def tail_ess(
    posterior: ParticleFilterResult,
    q: float = 0.05,
) -> Float[mx.array, " ntime"]:
    """Effective particles supporting the distribution tails.

    For each step, each state dimension, and each tail, restrict the
    normalized weights to the particles beyond the weighted q / 1-q
    quantile and compute ``(sum w)^2 / sum w^2`` — the effective
    number of particles estimating that tail. Returns the minimum
    over dimensions and both tails (in the spirit of the quantile
    tail-ESS of Vehtari, Gelman, Simpson, Carpenter & Burkner 2021).

    Uniform weights give ~``q * N`` (the tail only ever holds a q
    fraction of the mass); compare against ``q * N``, not ``N``.

    Note: this replaces the smcjax quantity of the same name, which
    measured top-weight-mass concentration and did not examine
    particle values (panel finding; ADR-0010 allows the fix).
    """
    qs = mx.array([q, 1.0 - q])
    ntime, _, dim = posterior.filtered_particles.shape

    def n_eff(tw):
        s2 = mx.sum(tw * tw)
        return mx.where(s2 > 0.0, mx.sum(tw) ** 2 / mx.maximum(s2, 1e-30), 0.0)

    # Plain Python loops over the small time/dim axes: nested mx.vmap
    # corrupts the take-based bisect inside the quantile helper
    # (flatten semantics under stacked batch dims), and diagnostics
    # are not hot-loop code.
    out = []
    for t in range(ntime):
        w = normalize(posterior.filtered_log_weights[t])
        per_dim = []
        for d in range(dim):
            vals = posterior.filtered_particles[t, :, d]
            edges = _weighted_quantile_1d(vals, w, qs)
            lo = mx.where(vals <= edges[0], w, 0.0)
            hi = mx.where(vals >= edges[1], w, 0.0)
            per_dim.append(mx.minimum(n_eff(lo), n_eff(hi)))
        out.append(mx.min(mx.stack(per_dim)))
    return mx.stack(out)


# --- summary -----------------------------------------------------------


def diagnose(
    posterior: ParticleFilterResult,
    ess_threshold: float = 0.1,
    diversity_threshold: float = 0.1,
    pareto_k_threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize filter health and flag problems (not compile-safe).

    Args:
        posterior: Filter output.
        ess_threshold: Warn when ESS < this fraction of N.
        diversity_threshold: Warn below this diversity.
        pareto_k_threshold: Warn above this k. Default None uses the
            sample-size-dependent ``min(1 - 1/log10(N), 0.7)``
            (Vehtari et al. 2024).

    Returns:
        Dict with ``min_ess``, ``min_diversity``, ``max_pareto_k``,
        ``min_tail_ess``, ``ess_below_threshold``, ``warnings``.
    """
    n = posterior.filtered_particles.shape[1]
    if pareto_k_threshold is None:
        pareto_k_threshold = min(1.0 - 1.0 / math.log10(max(n, 11)), 0.7)
    diversity = particle_diversity(posterior)
    k_hat = pareto_k_diagnostic(posterior)
    t_ess = tail_ess(posterior)

    min_ess = float(mx.min(posterior.ess).item())
    min_div = float(mx.min(diversity).item())
    max_k = float(mx.max(k_hat).item())
    min_t_ess = float(mx.min(t_ess).item())
    ess_count = int(mx.sum(posterior.ess < ess_threshold * n).item())

    warnings: list[str] = []
    if min_ess < ess_threshold * n:
        warnings.append(
            f"ESS dropped below {ess_threshold:.0%} of N at "
            f"{ess_count} step(s) (min ESS = {min_ess:.1f})"
        )
    if min_div < diversity_threshold:
        warnings.append(
            f"Particle diversity fell below {diversity_threshold:.0%} "
            f"(min = {min_div:.3f})"
        )
    if max_k > pareto_k_threshold:
        warnings.append(
            f"Pareto-k exceeded {pareto_k_threshold:.2f} "
            f"(max k = {max_k:.3f}); importance-weight estimates are "
            f"practically unreliable at some steps (variance is "
            f"already infinite for k >= 0.5)"
        )

    return {
        "min_ess": min_ess,
        "min_diversity": min_div,
        "max_pareto_k": max_k,
        "min_tail_ess": min_t_ess,
        "ess_below_threshold": ess_count,
        "warnings": warnings,
    }
