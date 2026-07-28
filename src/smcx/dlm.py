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
from typing import NamedTuple

import jax.numpy as jnp
from jax import core, lax
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Shaped

from smcx._numerics import _neumaier_add
from smcx._utils import _canonicalize_emissions
from smcx.containers import DLMFilterPosterior
from smcx.kalman import _check_covariance, _check_float_array
from smcx.types import GaussianEmissionSequence, Scalar


class _DLMCarry(NamedTuple):
    """Scale-free conjugate state carried through the scan."""

    mean: Float[Array, " state_dim"]
    scale_free_covariance: Float[Array, "state_dim state_dim"]
    shape: Float[Array, ""]
    scale: Float[Array, ""]
    marginal_loglik: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]


def _validate_positive_scalar(value: object, name: str) -> object:
    """Require a finite, strictly positive scalar; pass tracers through."""
    if isinstance(value, core.Tracer):
        return value
    try:
        scalar = float(value)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive scalar") from error
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {scalar}")
    return scalar


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
            ``(state_dim,)``; the observation mean is $F' x_t$.
        emissions: Univariate observations shaped ``(ntime,)`` or
            ``(ntime, 1)``.
        scale_free_transition_covariance: Evolution covariance divided
            by the unknown variance, $\tilde W = W / V$ — static
            ``(state_dim, state_dim)`` or timed
            ``(ntime, state_dim, state_dim)``. Supply exactly one of
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
        Student-$t$ marginal likelihood. The scaled covariance is
        ``scale_estimates[:, None, None] *
        filtered_scale_free_covariances``.

    Raises:
        ValueError: Malformed arrays or domains, multivariate
            emissions, or an evolution specification that is not
            exactly one of the two forms.
    """
    if (scale_free_transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of scale_free_transition_covariance "
            "and discount"
        )
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
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
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

    def _update(carry: _DLMCarry, prior_mean, prior_cov, emission_t, dof):
        forecast = observation_vector @ prior_mean
        forecast_scale_free = (
            observation_vector @ prior_cov @ observation_vector + 1.0
        )
        residual = emission_t[0] - forecast
        gain = prior_cov @ observation_vector / forecast_scale_free
        forecast_scale = carry.scale * forecast_scale_free
        log_density = (
            gammaln((dof + 1.0) / 2.0)
            - gammaln(dof / 2.0)
            - 0.5 * (jnp.log(dof) + log_pi + jnp.log(forecast_scale))
            - (dof + 1.0)
            / 2.0
            * jnp.log1p(residual * residual / (dof * forecast_scale))
        )
        new_mean = prior_mean + gain * residual
        new_cov = prior_cov - jnp.outer(gain, gain) * forecast_scale_free
        new_shape = dof + 1.0
        new_scale = (
            carry.scale
            * (dof + residual * residual / (forecast_scale_free * carry.scale))
            / new_shape
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
        emission_t, evolution_t = args
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
        return _update(carry, prior_mean, prior_cov, emission_t, dof)

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
        init, init.mean, init.scale_free_covariance, emissions[0], init.shape
    )
    final, rest = lax.scan(_step, carry_0, (emissions[1:], timed_evolution))
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
