# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Conjugate dynamic linear model filtering with an unknown variance.

The one setting where a static parameter admits exact sequential
Bayesian learning: a linear-Gaussian model whose single unknown
observational variance $V$ scales every covariance,

$$
x_t = G x_{t-1} + w_t, \quad w_t \sim \mathcal N(0, V \tilde W_t),
\qquad
y_t = F' x_t + v_t, \quad v_t \sim \mathcal N(0, V),
$$

with $x_0 \mid V \sim \mathcal N(m_0, V \tilde C_0)$ and an
Inverse-Gamma prior on $V$. The joint posterior is
Normal--Inverse-Gamma at every step, the recursion carries
$(m_t, \tilde C_t, n_t, S_t)$ in closed form, and one-step forecasts
are exact Student-$t$ — so the returned marginal likelihood is exact
[West and Harrison, 1997, ch. 4]. The evolution covariance may be
given explicitly or by a discount factor
($\tilde R_t = G \tilde C_{t-1} G' / \delta$); a discount specifies
the model implicitly rather than estimating a variance. Learning
several free covariances breaks the conjugacy — that is sequential
Monte Carlo's territory, not this filter's.

References:
    West, M., and Harrison, J. (1997). Bayesian Forecasting and
    Dynamic Models, second edition, chapter 4.
    https://doi.org/10.1007/b98971
"""

import math
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsIndex, TypeAlias

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import core, debug_infs, debug_nans, lax
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Shaped

from smcx._numerics import _neumaier_add
from smcx._utils import _canonicalize_emissions, _positive_integer
from smcx.containers import (
    DLMFilterPosterior,
    DLMForecast,
    DLMForecastPaths,
    DLMSmootherPosterior,
)
from smcx.kalman import (
    _backward_pass,
    _canonicalize_filter_covariances,
    _check_covariance,
    _check_float_array,
    _check_sampling_factors,
    _sampling_covariance_factor,
    _sanitize_missing,
    _symmetrize,
    _time_matrix,
    _validate_emission_rows,
)
from smcx.types import GaussianEmissionSequence, PRNGKeyT, Scalar

if TYPE_CHECKING:
    _CountArgument: TypeAlias = SupportsIndex
else:
    _CountArgument: TypeAlias = Any


class _DLMCarry(NamedTuple):
    """Scale-free conjugate state carried through the scan."""

    mean: Float[Array, " state_dim"]
    scale_free_covariance: Float[Array, "state_dim state_dim"]
    shape: Float[Array, ""]
    scale: Float[Array, ""]
    marginal_loglik: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]


def _validate_positive_scalar(value: object, name: str) -> object:
    """Require a positive scalar and validate tracer structure eagerly."""
    if isinstance(value, core.Tracer):
        if value.ndim != 0 or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(f"{name} must be a floating scalar")
        return value
    try:
        scalar = float(value)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive scalar") from error
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {scalar}")
    return scalar


def _validate_dlm_filter_posterior(
    posterior: DLMFilterPosterior,
) -> tuple[int, int, jnp.dtype, Float[Array, "ntime state_dim state_dim"]]:
    """Validate a DLM record and canonicalize its scale-free covariances."""
    means = posterior.filtered_means
    if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
        raise ValueError("filtered_means must have shape (T, d) with T, d > 0")
    num_timesteps, state_dim = means.shape
    dtype = means.dtype
    _check_float_array(means, "filtered_means")
    names = (
        "filtered_scale_free_covariances",
        "scale_shapes",
        "scale_estimates",
        "log_evidence_increments",
    )
    shapes = ((num_timesteps, state_dim, state_dim), *((num_timesteps,),) * 3)
    for name, shape in zip(names, shapes, strict=True):
        value = getattr(posterior, name)
        _check_float_array(value, name, dtype)
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
    for name in ("scale_shapes", "scale_estimates"):
        value = getattr(posterior, name)
        if not isinstance(value, core.Tracer):
            concrete = np.asarray(value)
            if not np.all(np.isfinite(concrete) & (concrete > 0.0)):
                raise ValueError(f"{name} must contain finite positive values")
    marginal = jnp.asarray(posterior.marginal_loglik)
    if marginal.ndim != 0:
        raise ValueError("marginal_loglik must be scalar")
    _check_float_array(marginal, "marginal_loglik", dtype)
    name = names[0]
    covariance = _canonicalize_filter_covariances(
        getattr(posterior, name), name
    )
    return num_timesteps, state_dim, dtype, covariance


def dlm_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_scale_free_covariance: Shaped[Array, "*initial_cov_shape"],
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    emissions: GaussianEmissionSequence,
    *,
    scale_free_transition_covariance: (
        Shaped[Array, "*evolution_shape"] | None
    ) = None,
    discount: Scalar | None = None,
    prior_shape: Scalar = 1.0,
    prior_scale: Scalar = 1.0,
    variance_discount: Scalar = 1.0,
) -> DLMFilterPosterior:
    r"""Run the conjugate unknown-variance DLM filter.

    Args:
        initial_mean: Prior mean $m_0$, shape ``(state_dim,)``.
        initial_scale_free_covariance: Prior covariance divided by the
            unknown variance, $\tilde C_0 = C_0 / V$; positive
            semidefinite, shape ``(state_dim, state_dim)``.
        transition_matrix: State evolution matrix $G$.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``, or a time-varying history with shape
            ``(ntime, state_dim)`` (dynamic regression: row ``t`` is
            the covariate vector at ``t``); the observation mean is
            $F_t' x_t$.
        emissions: Univariate observations shaped ``(ntime,)`` or
            ``(ntime, 1)``. A NaN entry marks that datum missing: the
            evolution (including state and variance discounts) still
            applies, the update is skipped, ``(n_t, S_t)`` gain no
            observation information, and ``log_evidence_increments``
            carries an exact zero there (ADR-0034). Infinite entries
            are rejected eagerly.
        scale_free_transition_covariance: Evolution covariance divided
            by the unknown variance, $\tilde W = W / V$ — static
            ``(state_dim, state_dim)`` or timed
            ``(ntime - 1, state_dim, state_dim)``. Supply exactly one of
            this and ``discount``.
        discount: Discount factor $\delta \in (0, 1]$ specifying
            $\tilde R_t = G \tilde C_{t-1} G' / \delta$. A discount
            states a model whose evolution noise is a fixed fraction
            of the current uncertainty; it does not estimate a
            variance.
        prior_shape: Inverse-Gamma prior degrees of freedom $n_0 > 0$.
        prior_scale: Inverse-Gamma prior point estimate $S_0 > 0$ of
            the unknown variance.
        variance_discount: Variance discount $\delta_V \in (0, 1]$.
            One keeps the variance constant and learns it exactly;
            below one, each evolution discounts the accumulated
            degrees of freedom, $n_t = \delta_V n_{t-1} + 1$ — the
            book's ordering — which is exact inference under the
            implied beta-gamma multiplicative random walk on the
            precision [West and Harrison, 1997, sec. 10.8].

    Returns:
        `smcx.containers.DLMFilterPosterior` with the scale-free
        filtered moments, the $(n_t, S_t)$ traces, and the exact
        Student-$t$ marginal likelihood.
        ``scale_estimates[:, None, None] *
        filtered_scale_free_covariances`` is the Student-$t$ scale
        matrix $S_t \tilde C_t$ of the filtered state marginal; the
        filtered covariance is $n_t / (n_t - 2)\, S_t \tilde C_t$
        and exists only for $n_t > 2$.

    Raises:
        ValueError: Malformed arrays or domains, multivariate
            emissions, or an evolution specification that is not
            exactly one of the two forms. Value checks run eagerly
            and are skipped for traced arrays.
    """
    if (scale_free_transition_covariance is None) == (discount is None):
        raise ValueError("supply exactly one evolution covariance or discount")
    if discount is not None:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
    shape_0 = _validate_positive_scalar(prior_shape, "prior_shape")
    scale_0 = _validate_positive_scalar(prior_scale, "prior_scale")
    variance_discount_value = _validate_positive_scalar(
        variance_discount, "variance_discount"
    )
    if not isinstance(variance_discount_value, core.Tracer) and (
        float(variance_discount_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"variance_discount must be in (0, 1]; got "
            f"{variance_discount_value}"
        )

    emissions = _canonicalize_emissions(emissions)
    if emissions.ndim != 2 or emissions.shape[1] != 1:
        raise ValueError(
            "dlm_filter is univariate: emissions must have shape "
            f"(ntime,) or (ntime, 1); got {emissions.shape}"
        )
    _check_float_array(emissions, "emissions")
    _validate_emission_rows(emissions)
    dtype = emissions.dtype
    num_timesteps = emissions.shape[0]

    state_dim = initial_mean.shape[0] if initial_mean.ndim == 1 else 0
    if state_dim == 0:
        raise ValueError("initial_mean must have shape (state_dim,)")
    for name, value, expected in (
        (
            "initial_scale_free_covariance",
            initial_scale_free_covariance,
            (state_dim, state_dim),
        ),
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    _check_float_array(observation_vector, "observation_vector", dtype)
    if observation_vector.shape == (state_dim,):
        timed_observation = jnp.broadcast_to(
            observation_vector, (num_timesteps, state_dim)
        )
    elif observation_vector.shape == (num_timesteps, state_dim):
        timed_observation = observation_vector
    else:
        raise ValueError(
            f"observation_vector must have shape ({state_dim},) or "
            f"({num_timesteps}, {state_dim}); got "
            f"{observation_vector.shape}"
        )
    _check_covariance(
        initial_scale_free_covariance,
        "initial_scale_free_covariance",
        positive_definite=False,
    )
    if scale_free_transition_covariance is not None:
        value = scale_free_transition_covariance
        _check_float_array(value, "scale_free_transition_covariance", dtype)
        if value.shape not in (
            (state_dim, state_dim),
            (num_timesteps - 1, state_dim, state_dim),
        ):
            raise ValueError(
                "scale_free_transition_covariance must have shape "
                f"({state_dim}, {state_dim}) or "
                f"({num_timesteps - 1}, {state_dim}, {state_dim}); "
                f"got {value.shape}"
            )
        _check_covariance(
            value,
            "scale_free_transition_covariance",
            positive_definite=False,
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

    log_pi = jnp.asarray(math.log(math.pi), dtype=dtype)
    variance_discount_array = jnp.asarray(variance_discount_value, dtype=dtype)
    use_discount = discount is not None
    inverse_discount = (
        1.0 / jnp.asarray(discount_value, dtype=dtype)
        if use_discount
        else jnp.asarray(1.0, dtype=dtype)
    )

    def _update(
        carry: _DLMCarry,
        prior_mean,
        prior_cov,
        emission_t,
        dof,
        observation_vector,
    ):
        # An all-NaN emission marks the datum missing (ADR-0034): the
        # evolution (including both discounts, applied by the caller of
        # this update through prior_cov and dof) stands, the update is
        # the identity, and the increment is exactly zero. The zero-fill
        # keeps the unselected branch finite for gradients.
        missing, safe_emission = _sanitize_missing(emission_t)
        forecast = observation_vector @ prior_mean
        forecast_scale_free = (
            observation_vector @ prior_cov @ observation_vector + 1.0
        )
        residual = safe_emission[0] - forecast
        gain = prior_cov @ observation_vector / forecast_scale_free
        # The forecast scale ``carry.scale * forecast_scale_free`` is
        # never materialized: for a representable scale near the dtype
        # maximum the product overflows while the log density and the
        # updated scale stay representable (follow-up review R2). Its
        # logarithm separates, and the whitening divides by the two
        # factors in turn. Whiten before squaring: the raw square
        # overflows float32 for residuals past ~1.8e19 while the
        # Student-t log density (heavy tails) stays representable
        # (#281). Barriers stop XLA from reassociating the divisions
        # into the overflowing product, which would also break the
        # eager/JIT numerical-equivalence contract.
        log_forecast_scale = jnp.log(carry.scale) + jnp.log(forecast_scale_free)
        # One factor at a time: dof * forecast_scale_free itself can
        # overflow while every result and each per-factor quotient is
        # representable (follow-up review R2).
        whitened = lax.optimization_barrier(
            ((residual / jnp.sqrt(forecast_scale_free)) / jnp.sqrt(dof))
            / jnp.sqrt(carry.scale)
        )
        squared = whitened * whitened
        abs_whitened = jnp.abs(whitened)
        large = abs_whitened > 1e15
        safe_abs = jnp.where(large, abs_whitened, jnp.ones_like(abs_whitened))
        log1p_term = jnp.where(
            large, 2.0 * jnp.log(safe_abs), jnp.log1p(squared)
        )
        log_density = (
            gammaln((dof + 1.0) / 2.0)
            - gammaln(dof / 2.0)
            - 0.5 * (jnp.log(dof) + log_pi + log_forecast_scale)
            - (dof + 1.0) / 2.0 * log1p_term
        )
        new_mean = jnp.where(missing, prior_mean, prior_mean + gain * residual)
        new_cov = jnp.where(
            missing,
            prior_cov,
            prior_cov - jnp.outer(gain, gain) * forecast_scale_free,
        )
        new_shape = jnp.where(missing, dof, dof + 1.0)
        half_width = lax.optimization_barrier(
            (residual / jnp.sqrt(forecast_scale_free)) / jnp.sqrt(dof)
        )
        # Bounded ratio before multiplication: ``dof * (...)`` overflows
        # about ``dof``-fold before the divided result does (R2).
        ratio = dof / new_shape
        new_scale = jnp.where(
            missing,
            carry.scale,
            lax.optimization_barrier(ratio * carry.scale)
            + lax.optimization_barrier(ratio * (half_width * half_width)),
        )
        log_density = jnp.where(
            missing, jnp.zeros_like(log_density), log_density
        )
        loglik, compensation = _neumaier_add(
            carry.marginal_loglik,
            carry.log_evidence_compensation,
            log_density,
        )
        new_carry = _DLMCarry(
            new_mean, new_cov, new_shape, new_scale, loglik, compensation
        )
        return new_carry, (
            new_mean,
            new_cov,
            new_shape,
            new_scale,
            log_density,
        )

    def _step(carry: _DLMCarry, args):
        emission_t, evolution_t, observation_t = args
        prior_mean = transition_matrix @ carry.mean
        propagated = (
            transition_matrix @ carry.scale_free_covariance
        ) @ transition_matrix.T
        if use_discount:
            prior_cov = propagated * inverse_discount
        else:
            prior_cov = propagated + evolution_t
        # Variance discounting acts at the evolution, before the new
        # observation raises the degrees of freedom by one (W&H's
        # ordering, n_t = delta_V n_{t-1} + 1).
        dof = variance_discount_array * carry.shape
        return _update(
            carry, prior_mean, prior_cov, emission_t, dof, observation_t
        )

    init = _DLMCarry(
        initial_mean.astype(dtype),
        initial_scale_free_covariance.astype(dtype),
        jnp.asarray(shape_0, dtype=dtype),
        jnp.asarray(scale_0, dtype=dtype),
        jnp.zeros((), dtype=dtype),
        jnp.zeros((), dtype=dtype),
    )
    # Library convention: emissions[0] conditions the prior directly;
    # evolution (or discounting) applies between observations.
    carry_0, first = _update(
        init,
        init.mean,
        init.scale_free_covariance,
        emissions[0],
        init.shape,
        timed_observation[0],
    )
    final, rest = lax.scan(
        _step,
        carry_0,
        (emissions[1:], timed_evolution, timed_observation[1:]),
    )
    outputs = tuple(
        jnp.concatenate([first_leaf[None], rest_leaf])
        for first_leaf, rest_leaf in zip(first, rest, strict=True)
    )
    means, covariances, shapes, scales, increments = outputs

    return DLMFilterPosterior(
        filtered_means=means,
        filtered_scale_free_covariances=covariances,
        scale_shapes=shapes,
        scale_estimates=scales,
        marginal_loglik=final.marginal_loglik + final.log_evidence_compensation,
        log_evidence_increments=increments,
    )


def dlm_smoother(
    filtered_posterior: DLMFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_shape"],
    *,
    scale_free_transition_covariance: (
        Shaped[Array, "*evolution_shape"] | None
    ) = None,
    discount: Scalar | None = None,
) -> DLMSmootherPosterior:
    r"""Run constant-common-variance DLM retrospective analysis.

    Reconstructed scale-free moments feed the shared Joseph-form smoother:
    $x_t\mid V,y_{1:T}\sim N(m_t^s,V\widetilde C_t^s)$ conditionally and
    $x_t\mid y_{1:T}\sim T_{n_T}[m_t^s,S_T\widetilde C_t^s]$ marginally.

    Args:
        filtered_posterior: Output of `smcx.dlm_filter` under a constant
            common variance (``variance_discount=1``).
        transition_matrix: Static state evolution matrix $G$, shape
            ``(state_dim, state_dim)``.
        scale_free_transition_covariance: Evolution covariance divided by
            the common variance, static ``(state_dim, state_dim)`` or timed
            ``(ntime - 1, state_dim, state_dim)``. Supply exactly one of this
            and ``discount``.
        discount: State discount $\delta\in(0,1]$ used by the filter. Supply
            exactly one of this and ``scale_free_transition_covariance``.

    Returns:
        `smcx.containers.DLMSmootherPosterior`. Here
        $S_T\widetilde C_t^s$ is the Student-t scale matrix, not its covariance;
        for $n_T>2$ covariance is $n_T/(n_T-2)$ times it; smcx never forms it.

    Raises:
        ValueError: Invalid record/evolution shape, dtype, or covariance.

    Note:
        Resupply the filter's $G$, $\widetilde W$, or $\delta$ and a record
        made with ``variance_discount=1``. The record cannot verify these
        facts; a mismatch can silently describe the wrong marginals.
        The result stores a canonical symmetric filtering covariance history;
        the caller's record is unchanged. Positive-time priors must be
        positive definite because the kernel factors them. Filtered
        covariances may be semidefinite, and $T=1$ takes no factorization.
        A traced factorization failure can produce NaNs or a JAX debug error.
        This covariance-form Joseph update is not a square-root method.
        Reconstructed noise can have roundoff-sized negative modes, and an
        ill-conditioned transition can amplify error by roughly its squared
        condition number per step, losing positive semidefiniteness.

    References:
        West, M., and Harrison, J. (1997). Bayesian Forecasting and Dynamic
        Models, second edition, sections 4.5 and 4.8.
        https://doi.org/10.1007/b98971
    """
    if (scale_free_transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of scale_free_transition_covariance "
            "and discount"
        )
    num_timesteps, state_dim, dtype, canonical_covariances = (
        _validate_dlm_filter_posterior(filtered_posterior)
    )
    _check_float_array(transition_matrix, "transition_matrix", dtype)
    actual = transition_matrix.shape
    expected = (state_dim, state_dim)
    if actual != expected:
        raise ValueError(f"transition_matrix shape {actual} != {expected}")

    if scale_free_transition_covariance is not None:
        evolution = scale_free_transition_covariance
        _check_float_array(evolution, "scale_free_transition_covariance", dtype)
        timed_evolution = _time_matrix(
            evolution,
            num_timesteps - 1,
            state_dim,
            state_dim,
            "scale_free_transition_covariance",
        )
        _check_covariance(
            evolution,
            "scale_free_transition_covariance",
            positive_definite=False,
        )
    else:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(f"discount outside (0, 1]: {discount_value}")
        inverse_discount = 1.0 / jnp.asarray(discount_value, dtype=dtype)

    propagated = (
        transition_matrix @ canonical_covariances[:-1]
    ) @ transition_matrix.T
    if scale_free_transition_covariance is None:
        next_predicted_covariances = _symmetrize(propagated * inverse_discount)
    else:
        next_predicted_covariances = _symmetrize(propagated + timed_evolution)
    _check_covariance(
        next_predicted_covariances,
        "reconstructed_scale_free_predicted_covariances",
        positive_definite=True,
    )
    predicted_means = jnp.concatenate((
        filtered_posterior.filtered_means[:1],
        filtered_posterior.filtered_means[:-1] @ transition_matrix.T,
    ))
    predicted_covariances = jnp.concatenate((
        canonical_covariances[:1],
        next_predicted_covariances,
    ))
    smoothed_means, smoothed_covariances = _backward_pass(
        filtered_posterior.filtered_means,
        canonical_covariances,
        predicted_means,
        predicted_covariances,
        jnp.broadcast_to(
            transition_matrix, (num_timesteps - 1, state_dim, state_dim)
        ),
    )
    return DLMSmootherPosterior(
        filtered_means=filtered_posterior.filtered_means,
        filtered_scale_free_covariances=canonical_covariances,
        scale_shapes=filtered_posterior.scale_shapes,
        scale_estimates=filtered_posterior.scale_estimates,
        marginal_loglik=filtered_posterior.marginal_loglik,
        log_evidence_increments=filtered_posterior.log_evidence_increments,
        smoothed_means=smoothed_means,
        smoothed_scale_free_covariances=smoothed_covariances,
    )


def dlm_forecast(
    filtered_posterior: DLMFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    *,
    num_steps: _CountArgument,
    scale_free_transition_covariance: (
        Shaped[Array, "*evolution_shape"] | None
    ) = None,
    discount: Scalar | None = None,
    variance_discount: Scalar = 1.0,
) -> DLMForecast:
    r"""Iterate k-step DLM forecast distributions from the frontier.

    Starting at the terminal conjugate state $(m_T, C_T, n_T, S_T)$,
    each horizon applies the West--Harrison prediction recursions
    (1997, section 4.4): $a(k) = G\,a(k-1)$ and
    $R(k) = G\,R(k-1)\,G^\top + W_{T+k}$ in scale-free form, with the
    univariate Student-t observation forecast
    $y_{T+k} \mid y_{1:T} \sim \mathrm{T}_{n(k)}
    \bigl(F^\top a(k),\, S_T\,(F^\top R(k)\,F + 1)\bigr)$.

    Under a ``discount`` specification the evolution variance is
    frozen at its frontier value
    $W = G\,C_T\,G^\top (1 - \delta)/\delta$ for every horizon, the
    practical strategy of West and Harrison, section 6.3.3. Filtering
    through a run of all-NaN emissions instead reapplies the discount
    at each data-free step (ADR-0034's time-driven decay), so with a
    discount the two agree at horizon one and diverge beyond it; with
    an explicit evolution covariance they agree at every horizon.

    Args:
        filtered_posterior: Output of `smcx.dlm_filter`. The forecast
            cannot verify that the resupplied pieces below match the
            filter run.
        transition_matrix: Static state evolution matrix $G$, shape
            ``(state_dim, state_dim)``.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``.
        num_steps: Positive integer forecast horizon. This controls an
            output shape and must be closed over or marked static
            through an outer ``jax.jit`` boundary.
        scale_free_transition_covariance: Static ``(state_dim,
            state_dim)`` or per-horizon ``(num_steps, state_dim,
            state_dim)`` evolution covariance divided by the unknown
            observation variance. Supply exactly one of this and
            ``discount``.
        discount: State discount $\delta \in (0, 1]$ used by the
            filter.
        variance_discount: Variance discount $\delta_V \in (0, 1]$
            used by the filter. Each horizon decays the Student-t
            degrees of freedom once, continuing the filter's
            time-driven evolution; with the default constant-variance
            model the degrees of freedom stay $n_T$ at every horizon.

    Returns:
        Per-horizon state and Student-t observation forecast moments.

    Raises:
        ValueError: The posterior or a model piece has an invalid
            shape, dtype, count, or domain, or the evolution
            specification is not exactly one of covariance and
            discount.

    Note:
        No data enter a forecast: the conjugate pair $(n, S)$ is never
        updated, only decayed by ``variance_discount``, and the
        horizon-one forecast density evaluated at a realized emission
        reproduces the filter's evidence increment.
    """
    if (scale_free_transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of scale_free_transition_covariance "
            "and discount"
        )
    num_steps = _positive_integer(num_steps, name="num_steps")
    _, state_dim, dtype, canonical_covariances = _validate_dlm_filter_posterior(
        filtered_posterior
    )
    for name, value, expected in (
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    use_discount = discount is not None
    if use_discount:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
        inverse_discount = 1.0 / jnp.asarray(discount_value, dtype=dtype)
    else:
        value = scale_free_transition_covariance
        assert value is not None  # XOR-checked above
        _check_float_array(value, "scale_free_transition_covariance", dtype)
        timed_evolution = _time_matrix(
            value,
            num_steps,
            state_dim,
            state_dim,
            "scale_free_transition_covariance",
        )
        _check_covariance(
            value,
            "scale_free_transition_covariance",
            positive_definite=False,
        )
    variance_discount_value = _validate_positive_scalar(
        variance_discount, "variance_discount"
    )
    if not isinstance(variance_discount_value, core.Tracer) and (
        float(variance_discount_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"variance_discount must be in (0, 1]; got "
            f"{variance_discount_value}"
        )
    variance_discount_array = jnp.asarray(variance_discount_value, dtype=dtype)

    del canonical_covariances
    mean = filtered_posterior.filtered_means[-1]
    # The raw stored covariance IS the filter's carry; symmetrizing it
    # here would break the exact agreement with a continued filter run.
    covariance = filtered_posterior.filtered_scale_free_covariances[-1]
    shape = filtered_posterior.scale_shapes[-1]
    scale = filtered_posterior.scale_estimates[-1]

    propagated_frontier = (transition_matrix @ covariance) @ transition_matrix.T
    if use_discount:
        first_covariance = propagated_frontier * inverse_discount
        frontier_w = propagated_frontier * (inverse_discount - 1.0)
        rest_evolution = jnp.broadcast_to(
            frontier_w, (num_steps - 1, state_dim, state_dim)
        )
    else:
        first_covariance = propagated_frontier + timed_evolution[0]
        rest_evolution = timed_evolution[1:]
    first_mean = transition_matrix @ mean
    first_shape = variance_discount_array * shape

    def _horizon(carry, evolution_k):
        mean_k, covariance_k, shape_k = carry
        next_mean = transition_matrix @ mean_k
        next_covariance = (
            transition_matrix @ covariance_k
        ) @ transition_matrix.T + evolution_k
        next_shape = variance_discount_array * shape_k
        return (next_mean, next_covariance, next_shape), (
            next_mean,
            next_covariance,
            next_shape,
        )

    _, rest = lax.scan(
        _horizon,
        (first_mean, first_covariance, first_shape),
        rest_evolution,
    )
    state_means = jnp.concatenate((first_mean[None], rest[0]))
    state_covariances = jnp.concatenate((first_covariance[None], rest[1]))
    shapes = jnp.concatenate((first_shape[None], rest[2]))
    observation_means = state_means @ observation_vector
    observation_scale_free = (
        jnp.einsum(
            "d,kde,e->k",
            observation_vector,
            state_covariances,
            observation_vector,
        )
        + 1.0
    )
    return DLMForecast(
        state_means=state_means,
        state_scale_free_covariances=state_covariances,
        observation_means=observation_means,
        observation_scales=scale * observation_scale_free,
        scale_shapes=shapes,
        scale_estimates=jnp.broadcast_to(scale, (num_steps,)),
    )


def dlm_forecast_sample(
    key: PRNGKeyT,
    filtered_posterior: DLMFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    *,
    num_steps: _CountArgument,
    num_draws: _CountArgument,
    scale_free_transition_covariance: (
        Shaped[Array, "*evolution_shape"] | None
    ) = None,
    discount: Scalar | None = None,
    variance_discount: Scalar = 1.0,
) -> DLMForecastPaths:
    r"""Draw exact joint DLM forecast paths from the frontier.

    Each path draws one observational precision from the terminal
    Gamma posterior, a terminal state from the correspondingly scaled
    Gaussian, and then iterates the state equation with per-horizon
    evolution noise and emission noise, all scaled by that path's
    variance. Sharing the variance draw across a path is what makes
    the joint law multivariate Student-t; horizon slices reproduce
    `dlm_forecast`'s marginals (ADR-0036, issue #415).

    Under a ``discount`` specification the evolution noise uses the
    frozen frontier variance of `dlm_forecast` at every horizon.
    Under a ``variance_discount`` below one, each horizon applies one
    beta-gamma shock to the precision (the walk whose marginals carry
    the discounted degrees of freedom), so the per-horizon Student-t
    marginals again match the closed forms.

    Args:
        key: JAX PRNG key. Split once into a precision key, a
            terminal-state key, and one key per horizon; each horizon
            key splits into a variance-shock key, a transition-noise
            key, and an emission-noise key.
        filtered_posterior: Output of `smcx.dlm_filter`. The forecast
            cannot verify that the resupplied pieces below match the
            filter run.
        transition_matrix: Static state evolution matrix $G$, shape
            ``(state_dim, state_dim)``.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``.
        num_steps: Positive integer forecast horizon. This controls an
            output shape and must be closed over or marked static
            through an outer ``jax.jit`` boundary.
        num_draws: Positive integer path count; also an output shape.
        scale_free_transition_covariance: Static ``(state_dim,
            state_dim)`` or per-horizon ``(num_steps, state_dim,
            state_dim)`` evolution covariance divided by the unknown
            variance. Supply exactly one of this and ``discount``.
        discount: State discount $\delta \in (0, 1]$ used by the
            filter.
        variance_discount: Variance discount $\delta_V \in (0, 1]$
            used by the filter.

    Returns:
        Draw-major state and univariate emission trajectories.

    Raises:
        ValueError: The posterior or a model piece has an invalid
            shape, dtype, count, or domain, the evolution
            specification is not exactly one of covariance and
            discount, or a concrete covariance is not factorable on
            the active backend.

    Note:
        Scale-free covariances factor through ordinary Cholesky with
        the same bounded semidefinite spectral fallback as
        `posterior_sample`; a zero frontier evolution variance
        (``discount=1``) factors to zero noise.
    """
    if (scale_free_transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of scale_free_transition_covariance "
            "and discount"
        )
    num_steps = _positive_integer(num_steps, name="num_steps")
    num_draws = _positive_integer(num_draws, name="num_draws")
    _, state_dim, dtype, canonical_covariances = _validate_dlm_filter_posterior(
        filtered_posterior
    )
    del canonical_covariances
    for name, value, expected in (
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    use_discount = discount is not None
    if use_discount:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
        inverse_discount = 1.0 / jnp.asarray(discount_value, dtype=dtype)
    else:
        value = scale_free_transition_covariance
        assert value is not None  # XOR-checked above
        _check_float_array(value, "scale_free_transition_covariance", dtype)
        timed_evolution = _time_matrix(
            value,
            num_steps,
            state_dim,
            state_dim,
            "scale_free_transition_covariance",
        )
        _check_covariance(
            value,
            "scale_free_transition_covariance",
            positive_definite=False,
        )
    variance_discount_value = _validate_positive_scalar(
        variance_discount, "variance_discount"
    )
    if not isinstance(variance_discount_value, core.Tracer) and (
        float(variance_discount_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"variance_discount must be in (0, 1]; got "
            f"{variance_discount_value}"
        )
    variance_discount_array = jnp.asarray(variance_discount_value, dtype=dtype)

    mean = filtered_posterior.filtered_means[-1]
    # The raw stored covariance IS the filter's carry; symmetrizing it
    # here would break coherence with a continued filter run.
    covariance = filtered_posterior.filtered_scale_free_covariances[-1]
    shape = filtered_posterior.scale_shapes[-1]
    scale = filtered_posterior.scale_estimates[-1]

    propagated_frontier = (transition_matrix @ covariance) @ transition_matrix.T
    if use_discount:
        frontier_w = propagated_frontier * (inverse_discount - 1.0)
        noise_covariances = jnp.broadcast_to(
            frontier_w, (num_steps, state_dim, state_dim)
        )
    else:
        noise_covariances = timed_evolution
    with debug_nans(False), debug_infs(False):
        terminal_factor = _sampling_covariance_factor(_symmetrize(covariance))
        noise_factors = lax.map(_sampling_covariance_factor, noise_covariances)
    _check_sampling_factors(terminal_factor[None])
    _check_sampling_factors(noise_factors)

    keys = jr.split(key, num_steps + 2)
    precision = jr.gamma(keys[0], shape / 2.0, (num_draws,), dtype=dtype) / (
        shape * scale / 2.0
    )
    root_variance = jnp.sqrt(1.0 / precision)
    terminal_noise = jr.normal(keys[1], (num_draws, state_dim), dtype=dtype)
    states_0 = mean + root_variance[:, None] * (
        terminal_noise @ terminal_factor.T
    )

    def _path_step(carry, args):
        states, precision_k, dof_k = carry
        step_key, noise_factor = args
        shock_key, state_key, emission_key = jr.split(step_key, 3)
        # One beta-gamma shock per horizon: the walk whose precision
        # marginal is Gamma with the discounted degrees of freedom.
        # The double-where keeps the delta_V = 1 branch NaN-free.
        beta_b = (1.0 - variance_discount_array) * dof_k / 2.0
        degenerate = beta_b <= 0.0
        safe_b = jnp.where(degenerate, jnp.ones_like(beta_b), beta_b)
        shock = jr.beta(
            shock_key,
            variance_discount_array * dof_k / 2.0,
            safe_b,
            (num_draws,),
            dtype=dtype,
        )
        shock = jnp.where(degenerate, jnp.ones_like(shock), shock)
        next_precision = precision_k * shock / variance_discount_array
        next_dof = variance_discount_array * dof_k
        root = jnp.sqrt(1.0 / next_precision)
        state_noise = jr.normal(state_key, (num_draws, state_dim), dtype=dtype)
        next_states = states @ transition_matrix.T + root[:, None] * (
            state_noise @ noise_factor.T
        )
        emission_noise = jr.normal(emission_key, (num_draws,), dtype=dtype)
        emissions = next_states @ observation_vector + root * emission_noise
        return (next_states, next_precision, next_dof), (
            next_states,
            emissions,
        )

    _, (state_paths, emission_paths) = lax.scan(
        _path_step,
        (states_0, precision, shape),
        (keys[2:], noise_factors),
    )
    return DLMForecastPaths(
        state_paths=jnp.swapaxes(state_paths, 0, 1),
        emission_paths=jnp.swapaxes(emission_paths, 0, 1),
    )
