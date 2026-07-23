# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Exact Gaussian inference for linear state-space models.

The filter implements the covariance-form recursion of Kalman (1960)
with a Joseph covariance update. Dynamax and statsmodels are independent
numerical validation references; no implementation code is copied from
either project.

References:
    Kalman, R. E. (1960). A New Approach to Linear Filtering and
    Prediction Problems. https://doi.org/10.1115/1.3662552
"""

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Float

from smcx.containers import GaussianFilterPosterior
from smcx.types import InputSequence


class _FilterState(NamedTuple):
    """Filtered moments and cumulative evidence carried through the scan."""

    mean: Float[Array, " state_dim"]
    covariance: Float[Array, "state_dim state_dim"]
    marginal_loglik: Float[Array, ""]


class _FilterStepOutput(NamedTuple):
    """Predicted and filtered values emitted by one filter step."""

    predicted_mean: Float[Array, " state_dim"]
    predicted_covariance: Float[Array, "state_dim state_dim"]
    filtered_mean: Float[Array, " state_dim"]
    filtered_covariance: Float[Array, "state_dim state_dim"]
    log_evidence_increment: Float[Array, ""]


def _symmetrize(
    covariance: Float[Array, "state_dim state_dim"],
) -> Float[Array, "state_dim state_dim"]:
    """Remove roundoff asymmetry from a covariance matrix."""
    return 0.5 * (covariance + covariance.T)


def _condition(
    predicted_mean: Float[Array, " state_dim"],
    predicted_covariance: Float[Array, "state_dim state_dim"],
    observation_matrix: Float[Array, "observation_dim state_dim"],
    observation_covariance: Float[Array, "observation_dim observation_dim"],
    observation: Float[Array, " observation_dim"],
    observation_bias: Float[Array, " observation_dim"],
) -> tuple[
    Float[Array, " state_dim"],
    Float[Array, "state_dim state_dim"],
    Float[Array, ""],
]:
    """Condition one Gaussian prior on one linear-Gaussian observation."""
    residual = observation - observation_matrix @ predicted_mean
    residual = residual - observation_bias
    innovation_covariance = _symmetrize(
        observation_matrix @ predicted_covariance @ observation_matrix.T
        + observation_covariance
    )
    innovation_cholesky = jnp.linalg.cholesky(innovation_covariance)
    covariance_observation = predicted_covariance @ observation_matrix.T
    lower_solution = solve_triangular(
        innovation_cholesky,
        covariance_observation.T,
        lower=True,
    )
    gain = solve_triangular(
        innovation_cholesky.T,
        lower_solution,
        lower=False,
    ).T
    filtered_mean = predicted_mean + gain @ residual
    identity = jnp.eye(predicted_mean.shape[0], dtype=predicted_mean.dtype)
    residual_operator = identity - gain @ observation_matrix
    filtered_covariance = _symmetrize(
        residual_operator @ predicted_covariance @ residual_operator.T
        + gain @ observation_covariance @ gain.T
    )
    whitened_residual = solve_triangular(
        innovation_cholesky,
        residual,
        lower=True,
    )
    observation_dim = observation.shape[0]
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=observation.dtype)
    log_evidence_increment = -0.5 * (
        observation_dim * log_two_pi
        + 2.0 * jnp.log(jnp.diag(innovation_cholesky)).sum()
        + whitened_residual @ whitened_residual
    )
    return filtered_mean, filtered_covariance, log_evidence_increment


def kalman_filter(
    initial_mean: Float[Array, " state_dim"],
    initial_covariance: Float[Array, "state_dim state_dim"],
    transition_matrix: Float[Array, "state_dim state_dim"],
    transition_covariance: Float[Array, "state_dim state_dim"],
    observation_matrix: Float[Array, "observation_dim state_dim"],
    observation_covariance: Float[Array, "observation_dim observation_dim"],
    emissions: Float[Array, "ntime observation_dim"],
    *,
    transition_bias: Float[Array, " state_dim"] | None = None,
    observation_bias: Float[Array, " observation_dim"] | None = None,
    transition_input_matrix: Float[Array, "state_dim input_dim"] | None = None,
    observation_input_matrix: Float[Array, "observation_dim input_dim"]
    | None = None,
    inputs: InputSequence | None = None,
) -> GaussianFilterPosterior:
    r"""Run an exact Kalman filter.

    The model is

    .. math::

        x_0 &\sim N(m_0, P_0),\\
        x_t &= F x_{t-1} + b + B u_t + q_t,\\
        y_t &= H x_t + d + D u_t + r_t.

    This first implementation accepts static operators. Time-varying
    operators and control terms are added by the same public contract.

    Args:
        initial_mean: Prior mean for ``x[0]``, shape ``(state_dim,)``.
        initial_covariance: Prior covariance for ``x[0]``.
        transition_matrix: Static transition matrix.
        transition_covariance: Static transition-noise covariance.
        observation_matrix: Static observation matrix.
        observation_covariance: Static observation-noise covariance.
        emissions: Observations with shape ``(ntime, observation_dim)``.
        transition_bias: Optional affine transition term.
        observation_bias: Optional affine observation term.
        transition_input_matrix: Optional transition control matrix.
        observation_input_matrix: Optional observation control matrix.
        inputs: Optional time-aligned controls.

    Returns:
        Predicted and filtered Gaussian moments and exact log evidence.
    """
    del transition_input_matrix, observation_input_matrix, inputs
    transition_bias_arr = (
        jnp.zeros_like(initial_mean)
        if transition_bias is None
        else transition_bias
    )
    observation_bias_arr = (
        jnp.zeros(
            observation_matrix.shape[0],
            dtype=initial_mean.dtype,
        )
        if observation_bias is None
        else observation_bias
    )
    filtered_mean_0, filtered_covariance_0, increment_0 = _condition(
        initial_mean,
        initial_covariance,
        observation_matrix,
        observation_covariance,
        emissions[0],
        observation_bias_arr,
    )
    state_0 = _FilterState(
        filtered_mean_0,
        filtered_covariance_0,
        increment_0,
    )

    def _step(
        state: _FilterState,
        observation: Float[Array, " observation_dim"],
    ) -> tuple[_FilterState, _FilterStepOutput]:
        predicted_mean = transition_matrix @ state.mean + transition_bias_arr
        predicted_covariance = _symmetrize(
            transition_matrix @ state.covariance @ transition_matrix.T
            + transition_covariance
        )
        filtered_mean, filtered_covariance, increment = _condition(
            predicted_mean,
            predicted_covariance,
            observation_matrix,
            observation_covariance,
            observation,
            observation_bias_arr,
        )
        next_state = _FilterState(
            filtered_mean,
            filtered_covariance,
            state.marginal_loglik + increment,
        )
        output = _FilterStepOutput(
            predicted_mean,
            predicted_covariance,
            filtered_mean,
            filtered_covariance,
            increment,
        )
        return next_state, output

    final_state, rest = lax.scan(_step, state_0, emissions[1:])
    predicted_means = jnp.concatenate((
        initial_mean[None],
        rest.predicted_mean,
    ))
    predicted_covariances = jnp.concatenate((
        initial_covariance[None],
        rest.predicted_covariance,
    ))
    filtered_means = jnp.concatenate((
        filtered_mean_0[None],
        rest.filtered_mean,
    ))
    filtered_covariances = jnp.concatenate((
        filtered_covariance_0[None],
        rest.filtered_covariance,
    ))
    increments = jnp.concatenate((
        increment_0[None],
        rest.log_evidence_increment,
    ))
    return GaussianFilterPosterior(
        final_state.marginal_loglik,
        predicted_means,
        predicted_covariances,
        filtered_means,
        filtered_covariances,
        increments,
    )
