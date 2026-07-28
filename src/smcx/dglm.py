# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Dynamic generalized linear model filtering (West, Harrison & Migon).

Exponential-family observations over a linear state evolution carried
by moments only. Each step matches a conjugate prior to the two
moments of the linear predictor $\lambda_t = F' x_t$, updates it
exactly on the observation, and feeds the posterior moments back to
the state by linear Bayes estimation:

$$
m_t = a_t + R_t F\,\frac{f_t^* - f_t}{q_t}, \qquad
C_t = R_t - R_t F F' R_t\,\frac{1 - q_t^*/q_t}{q_t}.
$$

The recursion is sequential, deterministic, and closed-form, and it
is approximate in exactly three places: the state prior is specified
by moments alone, the conjugate form is an assumption matched to two
moments, and the state feedback is the optimal *linear* Bayes
estimate (no error bound). Conditional on the matched conjugate
prior, the one-step forecast density, the conjugate update, and the
posterior moments are exact. For the normal family both
approximations vanish and the recursion is exactly the Kalman filter
[West, Harrison & Migon, 1985, sec. 2.2] — the primary test gate.
This filter sits between `smcx.dlm_filter` (exact) and the particle
filters (asymptotically exact), which are the natural accuracy check.

Forecast densities and special functions come from `jax.scipy`; smcx
implements only the inverse moment-matching solves.

References:
    West, M., Harrison, P. J., and Migon, H. S. (1985). Dynamic
    generalized linear models and Bayesian forecasting. Journal of
    the American Statistical Association, 80(389), 73-83.
    https://doi.org/10.1080/01621459.1985.10477131
    West, M., and Harrison, J. (1997). Bayesian Forecasting and
    Dynamic Models, second edition, chapter 14.
    https://doi.org/10.1007/b98971
"""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax import core, lax
from jax.scipy.special import digamma, polygamma
from jax.scipy.stats import betabinom, nbinom
from jaxtyping import Array, Shaped

from smcx._numerics import _neumaier_add
from smcx._utils import _canonicalize_emissions
from smcx.containers import DGLMFilterPosterior
from smcx.dlm import _validate_positive_scalar
from smcx.kalman import _check_covariance, _check_float_array
from smcx.types import EmissionSequence, Scalar

_NEWTON_ITERATIONS = 12
_NEWTON_ITERATIONS_2D = 25


class DGLMFamily(NamedTuple):
    r"""Observation-family contract for `dglm_filter`.

    Four pure callables over scalars, in the notation of West,
    Harrison & Migon (1985). ``alpha`` and ``beta`` are the conjugate
    parameters of the family; for the normal family they degenerate
    to the moments themselves.

    Attributes:
        match_moments: ``(f, q) -> (alpha, beta)`` — conjugate prior
            parameters matched to the prior mean and variance of the
            linear predictor.
        log_forecast: ``(y, alpha, beta) -> log p(y)`` — the exact
            one-step forecast log density under the matched prior.
        update: ``(y, alpha, beta) -> (alpha_star, beta_star)`` — the
            exact conjugate posterior parameters.
        posterior_moments: ``(alpha_star, beta_star) -> (f_star,
            q_star)`` — posterior mean and variance of the linear
            predictor, fed back to the state by linear Bayes.
    """

    match_moments: Callable[..., tuple[Scalar, Scalar]]
    log_forecast: Callable[..., Scalar]
    update: Callable[..., tuple[Scalar, Scalar]]
    posterior_moments: Callable[..., tuple[Scalar, Scalar]]


def _inverse_trigamma(value: Scalar) -> Scalar:
    r"""Solve $\psi'(\alpha) = q$ for $\alpha > 0$.

    Log-space Newton with a fixed iteration count so the solve is
    jit-, vmap-, and grad-compatible. The initializer
    $\alpha_0 = 1/q + 1/2$ comes from the expansion
    $\psi'(\alpha) = 1/\alpha + 1/(2\alpha^2) + O(\alpha^{-3})$ and
    the iteration converges quadratically; twelve steps reach f64
    machine precision across the tested domain $q \in [10^{-4}, 20]$.
    """
    log_alpha = jnp.log(1.0 / value + 0.5)
    for _ in range(_NEWTON_ITERATIONS):
        alpha = jnp.exp(log_alpha)
        residual = polygamma(1, alpha) - value
        slope = polygamma(2, alpha) * alpha
        log_alpha = log_alpha - residual / slope
    return jnp.exp(log_alpha)


def poisson() -> DGLMFamily:
    r"""Poisson observations with a log link.

    Conjugate gamma on the rate: the exact moment match solves
    $f = \psi(\alpha) - \log\beta$, $q = \psi'(\alpha)$
    (one inverse-trigamma Newton solve; $\beta$ then closed-form),
    the one-step forecast is the exact negative binomial
    $\mathrm{Nb}(\alpha, \beta/(1+\beta))$, and the conjugate update
    is $(\alpha + y, \beta + 1)$ [West, Harrison & Migon, 1985,
    sec. 5]. Emissions are nonnegative counts.
    """

    def match_moments(forecast_mean, forecast_variance):
        alpha = _inverse_trigamma(forecast_variance)
        beta = jnp.exp(digamma(alpha) - forecast_mean)
        return alpha, beta

    def log_forecast(emission, alpha, beta):
        return nbinom.logpmf(emission, alpha, beta / (1.0 + beta))

    def update(emission, alpha, beta):
        return alpha + emission, beta + 1.0

    def posterior_moments(alpha, beta):
        return digamma(alpha) - jnp.log(beta), polygamma(1, alpha)

    return DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )


def _match_beta_moments(
    mean: Scalar, variance: Scalar
) -> tuple[Scalar, Scalar]:
    r"""Solve the beta moment-matching equations for the logit link.

    The system is $\psi(\alpha) - \psi(\beta) = f$ with
    $\psi'(\alpha) + \psi'(\beta) = q$.

    Two-dimensional log-space Newton with a fixed iteration count,
    initialized at WHM's mode/curvature closed forms
    $\alpha_0 = (1 + e^{f})/q$, $\beta_0 = (1 + e^{-f})/q$ — the
    $q \to 0$ asymptote of the exact system.
    """
    log_alpha = jnp.log((1.0 + jnp.exp(mean)) / variance)
    log_beta = jnp.log((1.0 + jnp.exp(-mean)) / variance)
    for _ in range(_NEWTON_ITERATIONS_2D):
        alpha, beta = jnp.exp(log_alpha), jnp.exp(log_beta)
        residual_mean = digamma(alpha) - digamma(beta) - mean
        residual_var = polygamma(1, alpha) + polygamma(1, beta) - variance
        j11 = polygamma(1, alpha) * alpha
        j12 = -polygamma(1, beta) * beta
        j21 = polygamma(2, alpha) * alpha
        j22 = polygamma(2, beta) * beta
        determinant = j11 * j22 - j12 * j21
        log_alpha = (
            log_alpha - (j22 * residual_mean - j12 * residual_var) / determinant
        )
        log_beta = (
            log_beta - (-j21 * residual_mean + j11 * residual_var) / determinant
        )
    return jnp.exp(log_alpha), jnp.exp(log_beta)


def binomial(*, trials: int) -> DGLMFamily:
    r"""Binomial observations with a logit link and known trials.

    Conjugate beta on the success probability: the exact moment match
    solves $f = \psi(\alpha) - \psi(\beta)$,
    $q = \psi'(\alpha) + \psi'(\beta)$ (2-D Newton), the one-step
    forecast is the exact beta-binomial, and the conjugate update is
    $(\alpha + y, \beta + n - y)$ [West, Harrison & Migon, 1985,
    sec. 5]. Emissions are counts in $\{0, \dots, n\}$; ``trials``
    is a static known $n \ge 1$ shared across steps.
    """
    if int(trials) < 1:
        raise ValueError(f"trials must be a positive integer; got {trials}")

    def match_moments(forecast_mean, forecast_variance):
        return _match_beta_moments(forecast_mean, forecast_variance)

    def log_forecast(emission, alpha, beta):
        return betabinom.logpmf(emission, trials, alpha, beta)

    def update(emission, alpha, beta):
        return alpha + emission, beta + trials - emission

    def posterior_moments(alpha, beta):
        return (
            digamma(alpha) - digamma(beta),
            polygamma(1, alpha) + polygamma(1, beta),
        )

    return DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )


def bernoulli() -> DGLMFamily:
    r"""Bernoulli observations with a logit link.

    The single-trial case of `smcx.binomial`; the one-step forecast
    is the exact beta-Bernoulli,
    $\Pr[y_t = 1] = \alpha_t/(\alpha_t + \beta_t)$. Emissions are in
    $\{0, 1\}$.
    """
    return binomial(trials=1)


class _DGLMCarry(NamedTuple):
    """Moment state carried through the scan."""

    mean: Shaped[Array, " state_dim"]
    covariance: Shaped[Array, "state_dim state_dim"]
    marginal_loglik: Shaped[Array, ""]
    log_evidence_compensation: Shaped[Array, ""]


def dglm_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_covariance: Shaped[Array, "*initial_cov_shape"],
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    emissions: EmissionSequence,
    *,
    family: DGLMFamily,
    transition_covariance: Shaped[Array, "*evolution_shape"] | None = None,
    discount: Scalar | None = None,
    dispersion_discount: Scalar = 1.0,
) -> DGLMFilterPosterior:
    r"""Run the WHM dynamic generalized linear model filter.

    Args:
        initial_mean: Prior mean $m_0$, shape ``(state_dim,)``.
        initial_covariance: Prior covariance $C_0$; positive
            semidefinite, shape ``(state_dim, state_dim)``. Moments
            only — no distributional form is assumed.
        transition_matrix: State evolution matrix $G$.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``; the linear predictor is $F' x_t$.
        emissions: Univariate observations shaped ``(ntime,)`` or
            ``(ntime, 1)``; the family documents its domain (counts
            for `smcx.poisson`).
        family: `DGLMFamily` record of the four conjugate callables.
            Required; there is no default because a mismatched family
            fails silently, not loudly.
        transition_covariance: Evolution covariance $W$ — static
            ``(state_dim, state_dim)`` or timed
            ``(ntime - 1, state_dim, state_dim)``. Supply exactly one
            of this and ``discount``.
        discount: Discount factor $\delta \in (0, 1]$ specifying
            $R_t = G C_{t-1} G' / \delta$.
        dispersion_discount: Berry and West's random-effects discount
            $\rho \in (0, 1]$: the linear-predictor variance is
            inflated to $q_t/\rho$ before moment matching, adding
            unpredictable extra-dispersion each step. The default 1
            recovers the plain DGLM.

    Returns:
        `smcx.containers.DGLMFilterPosterior`. ``filtered_means`` and
        ``filtered_covariances`` are linear-Bayes moment summaries,
        not parameters of a posterior distribution; distributional
        statements flow through the conjugate parameters.
        ``marginal_loglik`` is the exact evidence of the sequentially
        specified approximating model (the sum of closed-form
        forecast log densities under exact moment matching).

    Raises:
        ValueError: Malformed arrays or domains, multivariate
            emissions, or an evolution specification that is not
            exactly one of the two forms.
    """
    if (transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of transition_covariance and discount"
        )
    if discount is not None:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
    dispersion_value = _validate_positive_scalar(
        dispersion_discount, "dispersion_discount"
    )
    if not isinstance(dispersion_value, core.Tracer) and (
        float(dispersion_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"dispersion_discount must be in (0, 1]; got {dispersion_value}"
        )

    _check_float_array(initial_mean, "initial_mean")
    dtype = initial_mean.dtype
    state_dim = initial_mean.shape[0] if initial_mean.ndim == 1 else 0
    if state_dim == 0:
        raise ValueError("initial_mean must have shape (state_dim,)")

    emissions = _canonicalize_emissions(emissions)
    if emissions.ndim != 2 or emissions.shape[1] != 1:
        raise ValueError(
            "dglm_filter is univariate: emissions must have shape "
            f"(ntime,) or (ntime, 1); got {emissions.shape}"
        )
    emission_values = emissions.astype(dtype)
    num_timesteps = emissions.shape[0]

    for name, value, expected in (
        ("initial_covariance", initial_covariance, (state_dim, state_dim)),
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    _check_covariance(
        initial_covariance, "initial_covariance", positive_definite=False
    )
    if transition_covariance is not None:
        value = transition_covariance
        _check_float_array(value, "transition_covariance", dtype)
        if value.shape not in (
            (state_dim, state_dim),
            (num_timesteps - 1, state_dim, state_dim),
        ):
            raise ValueError(
                "transition_covariance must have shape "
                f"({state_dim}, {state_dim}) or "
                f"({num_timesteps - 1}, {state_dim}, {state_dim}); "
                f"got {value.shape}"
            )
        _check_covariance(
            value, "transition_covariance", positive_definite=False
        )
        if value.ndim == 2:
            timed_evolution = jnp.broadcast_to(
                value, (num_timesteps - 1, state_dim, state_dim)
            )
        else:
            timed_evolution = value
    else:
        timed_evolution = jnp.zeros(
            (num_timesteps - 1, state_dim, state_dim), dtype=dtype
        )

    use_discount = discount is not None
    inverse_discount = (
        1.0 / jnp.asarray(discount_value, dtype=dtype)
        if use_discount
        else jnp.asarray(1.0, dtype=dtype)
    )

    inverse_dispersion = 1.0 / jnp.asarray(dispersion_value, dtype=dtype)

    def _update(carry: _DGLMCarry, prior_mean, prior_cov, emission_t):
        forecast_mean = observation_vector @ prior_mean
        forecast_variance = (
            observation_vector @ prior_cov @ observation_vector
        ) * inverse_dispersion
        alpha, beta = family.match_moments(forecast_mean, forecast_variance)
        log_density = jnp.asarray(
            family.log_forecast(emission_t[0], alpha, beta), dtype=dtype
        )
        alpha_post, beta_post = family.update(emission_t[0], alpha, beta)
        post_mean, post_variance = family.posterior_moments(
            alpha_post, beta_post
        )
        gain = prior_cov @ observation_vector / forecast_variance
        new_mean = prior_mean + gain * (post_mean - forecast_mean)
        new_cov = prior_cov - jnp.outer(gain, gain) * (
            forecast_variance - post_variance
        )
        loglik, compensation = _neumaier_add(
            carry.marginal_loglik,
            carry.log_evidence_compensation,
            log_density,
        )
        new_carry = _DGLMCarry(new_mean, new_cov, loglik, compensation)
        return new_carry, (
            new_mean,
            new_cov,
            jnp.asarray(alpha_post, dtype=dtype),
            jnp.asarray(beta_post, dtype=dtype),
            log_density,
        )

    def _step(carry: _DGLMCarry, args):
        emission_t, evolution_t = args
        prior_mean = transition_matrix @ carry.mean
        propagated = (
            transition_matrix @ carry.covariance
        ) @ transition_matrix.T
        if use_discount:
            prior_cov = propagated * inverse_discount
        else:
            prior_cov = propagated + evolution_t
        return _update(carry, prior_mean, prior_cov, emission_t)

    init = _DGLMCarry(
        initial_mean.astype(dtype),
        initial_covariance.astype(dtype),
        jnp.zeros((), dtype=dtype),
        jnp.zeros((), dtype=dtype),
    )
    # Library convention: emissions[0] conditions the prior directly;
    # evolution (or discounting) applies between observations.
    carry_0, first = _update(
        init, init.mean, init.covariance, emission_values[0]
    )
    final, rest = lax.scan(
        _step, carry_0, (emission_values[1:], timed_evolution)
    )
    outputs = tuple(
        jnp.concatenate([first_leaf[None], rest_leaf])
        for first_leaf, rest_leaf in zip(first, rest, strict=True)
    )
    means, covariances, alphas, betas, increments = outputs

    return DGLMFilterPosterior(
        filtered_means=means,
        filtered_covariances=covariances,
        conjugate_alphas=alphas,
        conjugate_betas=betas,
        marginal_loglik=final.marginal_loglik + final.log_evidence_compensation,
        log_evidence_increments=increments,
    )
