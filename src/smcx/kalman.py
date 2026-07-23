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
from jaxtyping import Array, Float, Shaped

from smcx._utils import _canonicalize_inputs
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


def _check_float_array(
    value: Shaped[Array, "*shape"],
    name: str,
    dtype: jnp.dtype | None = None,
) -> None:
    """Validate one public dense-array dtype."""
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError(f"{name} must have a floating dtype")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(
            f"all arrays must have dtype {dtype}; got {name}={value.dtype}"
        )


def _time_matrix(
    value: Shaped[Array, "*shape"],
    length: int,
    rows: int,
    columns: int,
    name: str,
) -> Float[Array, "ntime rows columns"]:
    """Validate and broadcast a static or time-varying matrix."""
    static_shape = (rows, columns)
    timed_shape = (length, rows, columns)
    if value.shape == static_shape:
        return jnp.broadcast_to(value, timed_shape)
    if value.shape != timed_shape:
        raise ValueError(
            f"{name} must have shape {static_shape} or {timed_shape}; "
            f"got {value.shape}"
        )
    return value


def _time_vector(
    value: Shaped[Array, "*shape"] | None,
    length: int,
    size: int,
    dtype: jnp.dtype,
    name: str,
) -> Float[Array, "ntime size"]:
    """Validate and broadcast an optional static or time-varying vector."""
    if value is None:
        return jnp.zeros((length, size), dtype=dtype)
    static_shape = (size,)
    timed_shape = (length, size)
    if value.shape == static_shape:
        return jnp.broadcast_to(value, timed_shape)
    if value.shape != timed_shape:
        raise ValueError(
            f"{name} must have shape {static_shape} or {timed_shape}; "
            f"got {value.shape}"
        )
    return value


def kalman_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_covariance: Shaped[Array, "*initial_covariance_shape"],
    transition_matrix: Shaped[Array, "*transition_matrix_shape"],
    transition_covariance: Shaped[Array, "*transition_covariance_shape"],
    observation_matrix: Shaped[Array, "*observation_matrix_shape"],
    observation_covariance: Shaped[Array, "*observation_covariance_shape"],
    emissions: Shaped[Array, "*emissions_shape"],
    *,
    transition_bias: Shaped[Array, "*transition_bias_shape"] | None = None,
    observation_bias: Shaped[Array, "*observation_bias_shape"] | None = None,
    transition_input_matrix: Shaped[Array, "*transition_input_matrix_shape"]
    | None = None,
    observation_input_matrix: Shaped[Array, "*observation_input_matrix_shape"]
    | None = None,
    inputs: InputSequence | None = None,
) -> GaussianFilterPosterior:
    r"""Run an exact Kalman filter.

    The model is

    .. math::

        x_0 &\sim N(m_0, P_0),\\
        x_t &= F x_{t-1} + b + B u_t + q_t,\\
        y_t &= H x_t + d + D u_t + r_t.

    Args:
        initial_mean: Prior mean for ``x[0]``, shape ``(state_dim,)``.
        initial_covariance: Prior covariance for ``x[0]``.
        transition_matrix: Static ``(state_dim, state_dim)`` matrix or
            time-varying array with leading length ``ntime - 1``.
        transition_covariance: Static or time-varying transition-noise
            covariance.
        observation_matrix: Static ``(observation_dim, state_dim)`` matrix
            or time-varying array with leading length ``ntime``.
        observation_covariance: Static or time-varying observation-noise
            covariance.
        emissions: Observations with shape ``(ntime, observation_dim)``.
        transition_bias: Optional static or length ``ntime - 1`` affine
            transition term.
        observation_bias: Optional static or length ``ntime`` affine
            observation term.
        transition_input_matrix: Optional static or length ``ntime - 1``
            transition control matrix.
        observation_input_matrix: Optional static or length ``ntime``
            observation control matrix.
        inputs: Optional controls with shape ``(ntime,)`` or
            ``(ntime, input_dim)``. ``inputs[t]`` reaches observation
            ``t`` and the transition into ``t``; ``inputs[0]`` does not
            alter the supplied prior.

    Returns:
        Predicted and filtered Gaussian moments and exact log evidence.

    Raises:
        ValueError: An array has an invalid shape or dtype, timed terms
            are misaligned, or control matrices are supplied without inputs.
    """
    if initial_mean.ndim != 1 or initial_mean.shape[0] == 0:
        raise ValueError("initial_mean must have shape (state_dim,) with d > 0")
    if emissions.ndim != 2 or emissions.shape[0] == 0:
        raise ValueError(
            "emissions must have shape (T, observation_dim) with T > 0"
        )
    num_timesteps, observation_dim = emissions.shape
    state_dim = initial_mean.shape[0]
    if observation_dim == 0:
        raise ValueError("emissions must have observation_dim > 0")
    dtype = initial_mean.dtype
    named_arrays = (
        ("initial_mean", initial_mean),
        ("initial_covariance", initial_covariance),
        ("transition_matrix", transition_matrix),
        ("transition_covariance", transition_covariance),
        ("observation_matrix", observation_matrix),
        ("observation_covariance", observation_covariance),
        ("emissions", emissions),
    )
    for name, value in named_arrays:
        expected_dtype = None if name == "initial_mean" else dtype
        _check_float_array(value, name, expected_dtype)
    optional_arrays = (
        ("transition_bias", transition_bias),
        ("observation_bias", observation_bias),
        ("transition_input_matrix", transition_input_matrix),
        ("observation_input_matrix", observation_input_matrix),
    )
    for name, value in optional_arrays:
        if value is not None:
            _check_float_array(value, name, dtype)

    num_transitions = num_timesteps - 1
    transition_matrices = _time_matrix(
        transition_matrix,
        num_transitions,
        state_dim,
        state_dim,
        "transition_matrix",
    )
    transition_covariances = _time_matrix(
        transition_covariance,
        num_transitions,
        state_dim,
        state_dim,
        "transition_covariance",
    )
    observation_matrices = _time_matrix(
        observation_matrix,
        num_timesteps,
        observation_dim,
        state_dim,
        "observation_matrix",
    )
    observation_covariances = _time_matrix(
        observation_covariance,
        num_timesteps,
        observation_dim,
        observation_dim,
        "observation_covariance",
    )
    transition_biases = _time_vector(
        transition_bias,
        num_transitions,
        state_dim,
        dtype,
        "transition_bias",
    )
    observation_biases = _time_vector(
        observation_bias,
        num_timesteps,
        observation_dim,
        dtype,
        "observation_bias",
    )
    if inputs is None:
        if (
            transition_input_matrix is not None
            or observation_input_matrix is not None
        ):
            raise ValueError("input matrices require inputs")
    else:
        _check_float_array(inputs, "inputs", dtype)
        inputs = _canonicalize_inputs(inputs, num_timesteps)
        input_dim = inputs.shape[1]
        if transition_input_matrix is not None:
            transition_controls = _time_matrix(
                transition_input_matrix,
                num_transitions,
                state_dim,
                input_dim,
                "transition_input_matrix",
            )
            transition_biases = transition_biases + jnp.einsum(
                "tdu,tu->td", transition_controls, inputs[1:]
            )
        if observation_input_matrix is not None:
            observation_controls = _time_matrix(
                observation_input_matrix,
                num_timesteps,
                observation_dim,
                input_dim,
                "observation_input_matrix",
            )
            observation_biases = observation_biases + jnp.einsum(
                "tdu,tu->td", observation_controls, inputs
            )

    if initial_covariance.shape != (state_dim, state_dim):
        raise ValueError(
            "initial_covariance must have shape "
            f"({state_dim}, {state_dim}); got {initial_covariance.shape}"
        )
    filtered_mean_0, filtered_covariance_0, increment_0 = _condition(
        initial_mean,
        initial_covariance,
        observation_matrices[0],
        observation_covariances[0],
        emissions[0],
        observation_biases[0],
    )
    state_0 = _FilterState(
        filtered_mean_0,
        filtered_covariance_0,
        increment_0,
    )

    def _step(
        state: _FilterState,
        args: tuple[Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[_FilterState, _FilterStepOutput]:
        (
            observation,
            transition,
            transition_noise,
            transition_offset,
            observation_operator,
            observation_noise,
            observation_offset,
        ) = args
        predicted_mean = transition @ state.mean + transition_offset
        predicted_covariance = _symmetrize(
            transition @ state.covariance @ transition.T + transition_noise
        )
        filtered_mean, filtered_covariance, increment = _condition(
            predicted_mean,
            predicted_covariance,
            observation_operator,
            observation_noise,
            observation,
            observation_offset,
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

    scan_inputs = (
        emissions[1:],
        transition_matrices,
        transition_covariances,
        transition_biases,
        observation_matrices[1:],
        observation_covariances[1:],
        observation_biases[1:],
    )
    final_state, rest = lax.scan(_step, state_0, scan_inputs)
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
