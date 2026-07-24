# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Gaussian inference for linear and nonlinear state-space models.

The linear filter implements the exact covariance-form recursion of Kalman
(1960) with a Joseph covariance update. The smoother implements the backward
recursion of Rauch, Tung, and Striebel (1965). The nonlinear filters use
either Schmidt's (1966) first-order approximation or Julier's (2002) scaled
unscented transform for additive Gaussian noise.

statsmodels and Dynamax validate the linear methods. Stone Soup and Dynamax
provide independent numerical comparisons for the nonlinear filters. They are
not implementation lineage; no code was copied or translated.

References:
    Kalman, R. E. (1960). A New Approach to Linear Filtering and
    Prediction Problems. https://doi.org/10.1115/1.3662552
    Rauch, H. E., Tung, F., and Striebel, C. T. (1965). Maximum
    Likelihood Estimates of Linear Dynamic Systems.
    https://doi.org/10.2514/3.3166
    Schmidt, S. F. (1966). Application of State-Space Methods to
    Navigation Problems.
    https://doi.org/10.1016/B978-1-4831-6716-9.50011-4
    Särkkä, S., and Svensson, L. (2023). Bayesian Filtering and
    Smoothing, second edition, chapter 7.
    https://doi.org/10.1017/9781108917407
    Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur
    Summation endlicher Summen.
    https://doi.org/10.1002/zamm.19740540106
"""

import math
from typing import NamedTuple, cast

import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Float, Shaped

from smcx._utils import _canonicalize_inputs
from smcx.containers import GaussianFilterPosterior, GaussianSmootherPosterior
from smcx.types import (
    InputSequence,
    ObservationJacobianFn,
    ObservationJacobianFnWithInput,
    ObservationMeanFn,
    ObservationMeanFnWithInput,
    TransitionJacobianFn,
    TransitionJacobianFnWithInput,
    TransitionMeanFn,
    TransitionMeanFnWithInput,
)


class _FilterState(NamedTuple):
    """Filtered moments and cumulative evidence carried through the scan."""

    mean: Float[Array, " state_dim"]
    covariance: Float[Array, "state_dim state_dim"]
    marginal_loglik: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]


class _FilterStepOutput(NamedTuple):
    """Predicted and filtered values emitted by one filter step."""

    predicted_mean: Float[Array, " state_dim"]
    predicted_covariance: Float[Array, "state_dim state_dim"]
    filtered_mean: Float[Array, " state_dim"]
    filtered_covariance: Float[Array, "state_dim state_dim"]
    log_evidence_increment: Float[Array, ""]


class _NonlinearFilterStepInput(NamedTuple):
    """Arrays consumed by one nonlinear Gaussian filter step."""

    emission: Float[Array, " observation_dim"]
    transition_covariance: Float[Array, "state_dim state_dim"]
    observation_covariance: Float[Array, "observation_dim observation_dim"]


class _NonlinearFilterStepInputWithInput(NamedTuple):
    """Arrays and input consumed by one input-aware nonlinear step."""

    emission: Float[Array, " observation_dim"]
    transition_covariance: Float[Array, "state_dim state_dim"]
    observation_covariance: Float[Array, "observation_dim observation_dim"]
    input_t: Float[Array, " input_dim"]


class _SmootherState(NamedTuple):
    """Smoothed moments carried backward through the model."""

    mean: Float[Array, " state_dim"]
    covariance: Float[Array, "state_dim state_dim"]


class _ScaledUnscentedRule(NamedTuple):
    """Stable coefficients for one symmetric scaled unscented rule."""

    sigma_scale: Float[Array, ""]
    off_center_weight: Float[Array, ""]
    covariance_rank_one_weight: Float[Array, ""]


class _NonlinearFilterSetup(NamedTuple):
    """Validated dimensions and broadcast covariances for nonlinear filters."""

    num_timesteps: int
    observation_dim: int
    state_dim: int
    dtype: jnp.dtype
    transition_covariances: Float[Array, "num_transitions state_dim state_dim"]
    observation_covariances: Float[
        Array, "ntime observation_dim observation_dim"
    ]


def _symmetrize(
    covariance: Float[Array, "state_dim state_dim"],
) -> Float[Array, "state_dim state_dim"]:
    """Remove roundoff asymmetry from a covariance matrix."""
    return 0.5 * (covariance + covariance.T)


def _neumaier_add(
    total: Float[Array, ""],
    compensation: Float[Array, ""],
    value: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Add one evidence increment with Neumaier compensation."""
    updated = total + value
    compensation = compensation + jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, compensation


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
    return _condition_from_residual(
        predicted_mean,
        predicted_covariance,
        observation_matrix,
        observation_covariance,
        residual,
    )


def _condition_from_residual(
    predicted_mean: Float[Array, " state_dim"],
    predicted_covariance: Float[Array, "state_dim state_dim"],
    observation_matrix: Float[Array, "observation_dim state_dim"],
    observation_covariance: Float[Array, "observation_dim observation_dim"],
    residual: Float[Array, " observation_dim"],
) -> tuple[
    Float[Array, " state_dim"],
    Float[Array, "state_dim state_dim"],
    Float[Array, ""],
]:
    """Condition one Gaussian prior using a supplied innovation residual."""
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
    observation_dim = residual.shape[0]
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=residual.dtype)
    log_evidence_increment = -0.5 * (
        observation_dim * log_two_pi
        + 2.0 * jnp.log(jnp.diag(innovation_cholesky)).sum()
        + whitened_residual @ whitened_residual
    )
    return filtered_mean, filtered_covariance, log_evidence_increment


def _filter_step(
    state: _FilterState,
    args: tuple[Array, Array, Array, Array, Array, Array, Array],
) -> tuple[_FilterState, _FilterStepOutput]:
    """Apply one pure Kalman predict-and-condition step."""
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
    evidence, compensation = _neumaier_add(
        state.marginal_loglik,
        state.log_evidence_compensation,
        increment,
    )
    next_state = _FilterState(
        filtered_mean,
        filtered_covariance,
        evidence,
        compensation,
    )
    output = _FilterStepOutput(
        predicted_mean,
        predicted_covariance,
        filtered_mean,
        filtered_covariance,
        increment,
    )
    return next_state, output


def _check_float_array(
    value: Shaped[Array, "*shape"],
    name: str,
    dtype: jnp.dtype | None = None,
) -> None:
    """Validate one public dense-array dtype."""
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError(f"{name} must have a floating dtype")
    supported = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
    if value.dtype not in supported:
        raise ValueError(
            f"{name} must have dtype float32 or float64; got {value.dtype}"
        )
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


def _check_callback_array(
    value: object,
    expected_shape: tuple[int, ...],
    name: str,
    dtype: jnp.dtype,
) -> None:
    """Validate a nonlinear Gaussian callback result."""
    if not isinstance(value, Array):
        raise ValueError(f"{name} must return a JAX array")
    _check_float_array(value, name, dtype)
    if value.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {value.shape}"
        )


def _prepare_nonlinear_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_covariance: Shaped[Array, "*initial_covariance_shape"],
    transition_covariance: Shaped[Array, "*transition_covariance_shape"],
    observation_covariance: Shaped[Array, "*observation_covariance_shape"],
    emissions: Shaped[Array, "*emissions_shape"],
) -> _NonlinearFilterSetup:
    """Validate arrays shared by the extended and unscented filters."""
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
        ("transition_covariance", transition_covariance),
        ("observation_covariance", observation_covariance),
        ("emissions", emissions),
    )
    for name, value in named_arrays:
        expected_dtype = None if name == "initial_mean" else dtype
        _check_float_array(value, name, expected_dtype)
    if initial_covariance.shape != (state_dim, state_dim):
        raise ValueError(
            "initial_covariance must have shape "
            f"({state_dim}, {state_dim}); got {initial_covariance.shape}"
        )
    transition_covariances = _time_matrix(
        transition_covariance,
        num_timesteps - 1,
        state_dim,
        state_dim,
        "transition_covariance",
    )
    observation_covariances = _time_matrix(
        observation_covariance,
        num_timesteps,
        observation_dim,
        observation_dim,
        "observation_covariance",
    )
    return _NonlinearFilterSetup(
        num_timesteps,
        observation_dim,
        state_dim,
        dtype,
        transition_covariances,
        observation_covariances,
    )


def _scaled_unscented_rule(
    state_dim: int,
    dtype: jnp.dtype,
    alpha: float,
    beta: float,
    kappa: float,
) -> _ScaledUnscentedRule:
    """Validate and construct a symmetric scaled unscented rule."""
    parameters = (("alpha", alpha), ("beta", beta), ("kappa", kappa))
    for name, value in parameters:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if alpha <= 0.0:
        raise ValueError("alpha must be greater than zero")
    if state_dim + kappa <= 0.0:
        raise ValueError("state_dim + kappa must be greater than zero")
    alpha_squared = alpha * alpha
    if alpha_squared * kappa + state_dim * beta < 0.0:
        raise ValueError(
            "alpha**2 * kappa + state_dim * beta must be nonnegative"
        )
    dtype_info = jnp.finfo(dtype)
    scale_squared = alpha_squared * (state_dim + kappa)
    if not math.isfinite(scale_squared) or scale_squared < float(
        dtype_info.smallest_subnormal
    ):
        raise ValueError(
            f"alpha, beta, and kappa produce non-finite weights in {dtype}"
        )
    off_center_weight = 0.5 / scale_squared
    central_mean_weight = (scale_squared - state_dim) / scale_squared
    central_covariance_weight = central_mean_weight + 1.0 - alpha_squared + beta
    correction = beta - alpha_squared
    covariance_rank_one_weight = (
        off_center_weight * off_center_weight * correction
        if correction
        else 0.0
    )
    derived = (
        scale_squared,
        off_center_weight,
        central_mean_weight,
        central_covariance_weight,
        covariance_rank_one_weight,
    )
    if any(
        not math.isfinite(value) or abs(value) > float(dtype_info.max)
        for value in derived
    ):
        raise ValueError(
            f"alpha, beta, and kappa produce non-finite weights in {dtype}"
        )
    return _ScaledUnscentedRule(
        jnp.asarray(math.sqrt(scale_squared), dtype=dtype),
        jnp.asarray(off_center_weight, dtype=dtype),
        jnp.asarray(covariance_rank_one_weight, dtype=dtype),
    )


def _sigma_points(
    mean: Float[Array, " state_dim"],
    covariance: Float[Array, "state_dim state_dim"],
    rule: _ScaledUnscentedRule,
) -> Float[Array, "num_sigma state_dim"]:
    """Generate center-first, column-oriented symmetric sigma points."""
    factor = jnp.linalg.cholesky(_symmetrize(covariance))
    offsets = (rule.sigma_scale * factor).T
    return jnp.concatenate((
        mean[None],
        mean[None] + offsets,
        mean[None] - offsets,
    ))


def _unscented_moments(
    values: Float[Array, "num_sigma value_dim"],
    rule: _ScaledUnscentedRule,
) -> tuple[
    Float[Array, " value_dim"],
    Float[Array, "value_dim value_dim"],
]:
    """Evaluate scaled-rule moments with center-relative arithmetic."""
    center = values[0]
    num_pairs = (values.shape[0] - 1) // 2
    positive_deltas = values[1 : num_pairs + 1] - center
    negative_deltas = values[num_pairs + 1 :] - center
    delta_sum = (positive_deltas + negative_deltas).sum(axis=0)
    mean = center + rule.off_center_weight * delta_sum
    covariance = rule.off_center_weight * (
        jnp.einsum(
            "ij,ik->jk",
            positive_deltas,
            positive_deltas,
        )
        + jnp.einsum(
            "ij,ik->jk",
            negative_deltas,
            negative_deltas,
        )
    ) + rule.covariance_rank_one_weight * jnp.outer(delta_sum, delta_sum)
    return mean, _symmetrize(covariance)


def _unscented_cross_covariance(
    state_points: Float[Array, "num_sigma state_dim"],
    transformed_points: Float[Array, "num_sigma value_dim"],
    rule: _ScaledUnscentedRule,
) -> Float[Array, "state_dim value_dim"]:
    """Form a paired cross covariance without subtractive centering."""
    state_dim = state_points.shape[1]
    offsets = 0.5 * (
        state_points[1 : state_dim + 1] - state_points[state_dim + 1 :]
    )
    transformed_differences = (
        transformed_points[1 : state_dim + 1]
        - transformed_points[state_dim + 1 :]
    )
    return rule.off_center_weight * offsets.T @ transformed_differences


def _unscented_condition(
    predicted_mean: Float[Array, " state_dim"],
    predicted_covariance: Float[Array, "state_dim state_dim"],
    observation_mean_fn: ObservationMeanFn | ObservationMeanFnWithInput,
    observation_covariance: Float[Array, "observation_dim observation_dim"],
    observation: Float[Array, " observation_dim"],
    rule: _ScaledUnscentedRule,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[
    Float[Array, " state_dim"],
    Float[Array, "state_dim state_dim"],
    Float[Array, ""],
]:
    """Condition one Gaussian using a scaled unscented observation."""
    state_points = _sigma_points(
        predicted_mean,
        predicted_covariance,
        rule,
    )
    if input_t is None:
        observation_fn = cast(ObservationMeanFn, observation_mean_fn)
        observation_points = vmap(observation_fn)(state_points)
    else:
        observation_fn_u = cast(
            ObservationMeanFnWithInput,
            observation_mean_fn,
        )
        observation_points = vmap(
            observation_fn_u,
            in_axes=(0, None),
        )(state_points, input_t)
    _check_callback_array(
        observation_points,
        (state_points.shape[0], observation.shape[0]),
        "observation_mean_fn output",
        predicted_mean.dtype,
    )
    observation_mean, transformed_covariance = _unscented_moments(
        observation_points,
        rule,
    )
    cross_covariance = _unscented_cross_covariance(
        state_points,
        observation_points,
        rule,
    )
    innovation_covariance = _symmetrize(
        transformed_covariance + observation_covariance
    )
    innovation_cholesky = jnp.linalg.cholesky(innovation_covariance)
    lower_solution = solve_triangular(
        innovation_cholesky,
        cross_covariance.T,
        lower=True,
    )
    gain = solve_triangular(
        innovation_cholesky.T,
        lower_solution,
        lower=False,
    ).T
    residual = observation - observation_mean
    filtered_mean = predicted_mean + gain @ residual
    corrected_points = state_points - observation_points @ gain.T
    _, residual_covariance = _unscented_moments(corrected_points, rule)
    filtered_covariance = _symmetrize(
        residual_covariance + gain @ observation_covariance @ gain.T
    )
    whitened_residual = solve_triangular(
        innovation_cholesky,
        residual,
        lower=True,
    )
    observation_dim = residual.shape[0]
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=residual.dtype)
    log_evidence_increment = -0.5 * (
        observation_dim * log_two_pi
        + 2.0 * jnp.log(jnp.diag(innovation_cholesky)).sum()
        + whitened_residual @ whitened_residual
    )
    return filtered_mean, filtered_covariance, log_evidence_increment


def _unscented_filter_step(
    state: _FilterState,
    args: _NonlinearFilterStepInput,
    transition_mean_fn: TransitionMeanFn | TransitionMeanFnWithInput,
    observation_mean_fn: ObservationMeanFn | ObservationMeanFnWithInput,
    rule: _ScaledUnscentedRule,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[_FilterState, _FilterStepOutput]:
    """Apply one pure unscented Kalman predict-and-condition step."""
    state_points = _sigma_points(state.mean, state.covariance, rule)
    if input_t is None:
        transition_fn = cast(TransitionMeanFn, transition_mean_fn)
        transition_points = vmap(transition_fn)(state_points)
    else:
        transition_fn_u = cast(
            TransitionMeanFnWithInput,
            transition_mean_fn,
        )
        transition_points = vmap(
            transition_fn_u,
            in_axes=(0, None),
        )(state_points, input_t)
    _check_callback_array(
        transition_points,
        state_points.shape,
        "transition_mean_fn output",
        state.mean.dtype,
    )
    predicted_mean, transformed_covariance = _unscented_moments(
        transition_points,
        rule,
    )
    predicted_covariance = _symmetrize(
        transformed_covariance + args.transition_covariance
    )
    filtered_mean, filtered_covariance, increment = _unscented_condition(
        predicted_mean,
        predicted_covariance,
        observation_mean_fn,
        args.observation_covariance,
        args.emission,
        rule,
        input_t,
    )
    evidence, compensation = _neumaier_add(
        state.marginal_loglik,
        state.log_evidence_compensation,
        increment,
    )
    next_state = _FilterState(
        filtered_mean,
        filtered_covariance,
        evidence,
        compensation,
    )
    output = _FilterStepOutput(
        predicted_mean,
        predicted_covariance,
        filtered_mean,
        filtered_covariance,
        increment,
    )
    return next_state, output


def _extended_filter_step(
    state: _FilterState,
    args: _NonlinearFilterStepInput,
    transition_mean_fn: TransitionMeanFn | TransitionMeanFnWithInput,
    transition_jacobian_fn: (
        TransitionJacobianFn | TransitionJacobianFnWithInput
    ),
    observation_mean_fn: ObservationMeanFn | ObservationMeanFnWithInput,
    observation_jacobian_fn: (
        ObservationJacobianFn | ObservationJacobianFnWithInput
    ),
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[_FilterState, _FilterStepOutput]:
    """Apply one pure extended Kalman predict-and-condition step."""
    state_dim = state.mean.shape[0]
    observation_dim = args.emission.shape[0]
    dtype = state.mean.dtype
    if input_t is None:
        transition_fn = cast(TransitionMeanFn, transition_mean_fn)
        transition_jacobian_function = cast(
            TransitionJacobianFn,
            transition_jacobian_fn,
        )
        observation_fn = cast(ObservationMeanFn, observation_mean_fn)
        observation_jacobian_function = cast(
            ObservationJacobianFn,
            observation_jacobian_fn,
        )
        transition_mean = transition_fn(state.mean)
        transition_jacobian = transition_jacobian_function(state.mean)
    else:
        transition_fn_u = cast(
            TransitionMeanFnWithInput,
            transition_mean_fn,
        )
        transition_jacobian_function_u = cast(
            TransitionJacobianFnWithInput,
            transition_jacobian_fn,
        )
        observation_fn_u = cast(
            ObservationMeanFnWithInput,
            observation_mean_fn,
        )
        observation_jacobian_function_u = cast(
            ObservationJacobianFnWithInput,
            observation_jacobian_fn,
        )
        transition_mean = transition_fn_u(state.mean, input_t)
        transition_jacobian = transition_jacobian_function_u(
            state.mean,
            input_t,
        )
    _check_callback_array(
        transition_mean,
        (state_dim,),
        "transition_mean_fn output",
        dtype,
    )
    _check_callback_array(
        transition_jacobian,
        (state_dim, state_dim),
        "transition_jacobian_fn output",
        dtype,
    )
    predicted_covariance = _symmetrize(
        transition_jacobian @ state.covariance @ transition_jacobian.T
        + args.transition_covariance
    )
    if input_t is None:
        observation_mean = observation_fn(transition_mean)
        observation_jacobian = observation_jacobian_function(transition_mean)
    else:
        observation_mean = observation_fn_u(transition_mean, input_t)
        observation_jacobian = observation_jacobian_function_u(
            transition_mean,
            input_t,
        )
    _check_callback_array(
        observation_mean,
        (observation_dim,),
        "observation_mean_fn output",
        dtype,
    )
    _check_callback_array(
        observation_jacobian,
        (observation_dim, state_dim),
        "observation_jacobian_fn output",
        dtype,
    )
    filtered_mean, filtered_covariance, increment = _condition_from_residual(
        transition_mean,
        predicted_covariance,
        observation_jacobian,
        args.observation_covariance,
        args.emission - observation_mean,
    )
    evidence, compensation = _neumaier_add(
        state.marginal_loglik,
        state.log_evidence_compensation,
        increment,
    )
    next_state = _FilterState(
        filtered_mean,
        filtered_covariance,
        evidence,
        compensation,
    )
    output = _FilterStepOutput(
        transition_mean,
        predicted_covariance,
        filtered_mean,
        filtered_covariance,
        increment,
    )
    return next_state, output


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

    Note:
        Arrays must use float32 or float64. Covariances must be finite,
        symmetric, and positive definite. Missing observations are not
        supported.
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
        jnp.zeros_like(increment_0),
    )

    scan_inputs = (
        emissions[1:],
        transition_matrices,
        transition_covariances,
        transition_biases,
        observation_matrices[1:],
        observation_covariances[1:],
        observation_biases[1:],
    )
    final_state, rest = lax.scan(_filter_step, state_0, scan_inputs)
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
        final_state.marginal_loglik + final_state.log_evidence_compensation,
        predicted_means,
        predicted_covariances,
        filtered_means,
        filtered_covariances,
        increments,
    )


def extended_kalman_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_covariance: Shaped[Array, "*initial_covariance_shape"],
    transition_mean_fn: TransitionMeanFn | TransitionMeanFnWithInput,
    transition_jacobian_fn: (
        TransitionJacobianFn | TransitionJacobianFnWithInput
    ),
    transition_covariance: Shaped[Array, "*transition_covariance_shape"],
    observation_mean_fn: ObservationMeanFn | ObservationMeanFnWithInput,
    observation_jacobian_fn: (
        ObservationJacobianFn | ObservationJacobianFnWithInput
    ),
    observation_covariance: Shaped[Array, "*observation_covariance_shape"],
    emissions: Shaped[Array, "*emissions_shape"],
    *,
    inputs: InputSequence | None = None,
) -> GaussianFilterPosterior:
    r"""Run a first-order extended Kalman filter.

    The model has nonlinear conditional means and additive Gaussian noise:

    .. math::

        x_0 &\sim N(m_0, P_0),\\
        x_t &= f(x_{t-1}, u_t) + q_t,\\
        y_t &= h(x_t, u_t) + r_t.

    Jacobian callbacks are explicit and use output-by-input orientation.
    At each positive time, ``f`` and its Jacobian are evaluated at the
    preceding filtered mean; ``h`` and its Jacobian are evaluated at the
    predicted mean. Callers may supply analytic callbacks or create them
    explicitly with ``jax.jacfwd``.

    Args:
        initial_mean: Prior mean for ``x[0]``, shape ``(state_dim,)``.
        initial_covariance: Prior covariance for ``x[0]``.
        transition_mean_fn: ``state -> state_mean`` or, when inputs are
            supplied, ``(state, input_t) -> state_mean``.
        transition_jacobian_fn: State Jacobian of ``transition_mean_fn``
            with the same input signature.
        transition_covariance: Static transition covariance or a timed
            array with leading length ``ntime - 1``.
        observation_mean_fn: ``state -> observation_mean`` or, when inputs
            are supplied, ``(state, input_t) -> observation_mean``.
        observation_jacobian_fn: State Jacobian of
            ``observation_mean_fn`` with the same input signature.
        observation_covariance: Static observation covariance or a timed
            array with leading length ``ntime``.
        emissions: Observations with shape ``(ntime, observation_dim)``.
        inputs: Optional exogenous inputs with shape ``(ntime, input_dim)``
            or ``(ntime,)``. Input ``t`` reaches observation ``t`` and the
            transition into ``t``; input zero does not alter the prior.

    Returns:
        Approximate Gaussian filtering moments. ``marginal_loglik`` and
        ``log_evidence_increments`` contain linearized Gaussian innovation
        log densities, not the exact nonlinear-model marginal likelihood.

    Raises:
        ValueError: An array or callback output has an invalid shape or
            dtype.

    Note:
        Arrays must share a float32 or float64 dtype. Covariances must be
        finite, symmetric, and positive definite. Missing observations are
        not supported.

    References:
        Schmidt, S. F. (1966). Application of State-Space Methods to
        Navigation Problems.
        https://doi.org/10.1016/B978-1-4831-6716-9.50011-4
        Särkkä, S., and Svensson, L. (2023). Bayesian Filtering and
        Smoothing, second edition, chapter 7.
        https://doi.org/10.1017/9781108917407
    """
    setup = _prepare_nonlinear_filter(
        initial_mean,
        initial_covariance,
        transition_covariance,
        observation_covariance,
        emissions,
    )
    num_timesteps = setup.num_timesteps
    observation_dim = setup.observation_dim
    state_dim = setup.state_dim
    dtype = setup.dtype
    transition_covariances = setup.transition_covariances
    observation_covariances = setup.observation_covariances

    if inputs is None:
        inputs_arr = None
        observation_fn = cast(ObservationMeanFn, observation_mean_fn)
        observation_jacobian_function = cast(
            ObservationJacobianFn,
            observation_jacobian_fn,
        )
        observation_mean_0 = observation_fn(initial_mean)
        observation_jacobian_0 = observation_jacobian_function(initial_mean)
    else:
        _check_float_array(inputs, "inputs", dtype)
        inputs_arr = _canonicalize_inputs(inputs, num_timesteps)
        observation_fn_u = cast(
            ObservationMeanFnWithInput,
            observation_mean_fn,
        )
        observation_jacobian_function_u = cast(
            ObservationJacobianFnWithInput,
            observation_jacobian_fn,
        )
        observation_mean_0 = observation_fn_u(initial_mean, inputs_arr[0])
        observation_jacobian_0 = observation_jacobian_function_u(
            initial_mean,
            inputs_arr[0],
        )
    _check_callback_array(
        observation_mean_0,
        (observation_dim,),
        "observation_mean_fn output",
        dtype,
    )
    _check_callback_array(
        observation_jacobian_0,
        (observation_dim, state_dim),
        "observation_jacobian_fn output",
        dtype,
    )
    filtered_mean_0, filtered_covariance_0, increment_0 = (
        _condition_from_residual(
            initial_mean,
            initial_covariance,
            observation_jacobian_0,
            observation_covariances[0],
            emissions[0] - observation_mean_0,
        )
    )
    state_0 = _FilterState(
        filtered_mean_0,
        filtered_covariance_0,
        increment_0,
        jnp.zeros_like(increment_0),
    )

    if inputs_arr is None:
        scan_inputs = _NonlinearFilterStepInput(
            emissions[1:],
            transition_covariances,
            observation_covariances[1:],
        )

        def _step(
            state: _FilterState,
            args: _NonlinearFilterStepInput,
        ) -> tuple[_FilterState, _FilterStepOutput]:
            return _extended_filter_step(
                state,
                args,
                transition_mean_fn,
                transition_jacobian_fn,
                observation_mean_fn,
                observation_jacobian_fn,
            )

        final_state, rest = lax.scan(_step, state_0, scan_inputs)
    else:
        scan_inputs_u = _NonlinearFilterStepInputWithInput(
            emissions[1:],
            transition_covariances,
            observation_covariances[1:],
            inputs_arr[1:],
        )

        def _step_with_input(
            state: _FilterState,
            args: _NonlinearFilterStepInputWithInput,
        ) -> tuple[_FilterState, _FilterStepOutput]:
            step_args = _NonlinearFilterStepInput(
                args.emission,
                args.transition_covariance,
                args.observation_covariance,
            )
            return _extended_filter_step(
                state,
                step_args,
                transition_mean_fn,
                transition_jacobian_fn,
                observation_mean_fn,
                observation_jacobian_fn,
                args.input_t,
            )

        final_state, rest = lax.scan(
            _step_with_input,
            state_0,
            scan_inputs_u,
        )
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
        final_state.marginal_loglik + final_state.log_evidence_compensation,
        predicted_means,
        predicted_covariances,
        filtered_means,
        filtered_covariances,
        increments,
    )


def unscented_kalman_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_covariance: Shaped[Array, "*initial_covariance_shape"],
    transition_mean_fn: TransitionMeanFn | TransitionMeanFnWithInput,
    transition_covariance: Shaped[Array, "*transition_covariance_shape"],
    observation_mean_fn: ObservationMeanFn | ObservationMeanFnWithInput,
    observation_covariance: Shaped[Array, "*observation_covariance_shape"],
    emissions: Shaped[Array, "*emissions_shape"],
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    inputs: InputSequence | None = None,
) -> GaussianFilterPosterior:
    r"""Run a scaled unscented Kalman filter.

    The model has nonlinear conditional means and additive Gaussian noise:

    .. math::

        x_0 &\sim N(m_0, P_0),\\
        x_t &= f(x_{t-1}, u_t) + q_t,\\
        y_t &= h(x_t, u_t) + r_t.

    The symmetric ``2d + 1`` rule defaults to
    ``(alpha, beta, kappa) = (1, 2, 0)``. Its default covariance weights
    are nonnegative. Center-relative arithmetic and a residual-sigma update
    avoid a subtractive covariance update; positive semidefiniteness remains
    subject to floating-point roundoff.

    Args:
        initial_mean: Prior mean for ``x[0]``, shape ``(state_dim,)``.
        initial_covariance: Prior covariance for ``x[0]``.
        transition_mean_fn: ``state -> state_mean`` or, when inputs are
            supplied, ``(state, input_t) -> state_mean``.
        transition_covariance: Static transition covariance or a timed
            array with leading length ``ntime - 1``.
        observation_mean_fn: ``state -> observation_mean`` or, when inputs
            are supplied, ``(state, input_t) -> observation_mean``.
        observation_covariance: Static observation covariance or a timed
            array with leading length ``ntime``.
        emissions: Observations with shape ``(ntime, observation_dim)``.
        alpha: Positive sigma-point spread parameter.
        beta: Covariance correction parameter.
        kappa: Secondary sigma-point scaling parameter.
        inputs: Optional exogenous inputs with shape ``(ntime, input_dim)``
            or ``(ntime,)``. Input ``t`` reaches observation ``t`` and the
            transition into ``t``; input zero does not alter the prior.

    Returns:
        Approximate Gaussian filtering moments. ``marginal_loglik`` and
        ``log_evidence_increments`` contain unscented Gaussian innovation
        log densities, not the exact nonlinear-model marginal likelihood.

    Raises:
        ValueError: An array, callback output, or scaled-rule parameter is
            invalid.

    Note:
        Arrays must share a float32 or float64 dtype. Covariances must be
        finite, symmetric, and positive definite. Missing observations are
        not supported. Smaller ``alpha`` values may improve local quadrature
        but are more cancellation-prone in float32.

    References:
        Julier, S. J. (2002). The Scaled Unscented Transformation.
        https://doi.org/10.1109/ACC.2002.1025369
        Särkkä, S., and Svensson, L. (2023). Bayesian Filtering and
        Smoothing, second edition, chapter 8.
        https://doi.org/10.1017/9781108917407
    """
    setup = _prepare_nonlinear_filter(
        initial_mean,
        initial_covariance,
        transition_covariance,
        observation_covariance,
        emissions,
    )
    rule = _scaled_unscented_rule(
        setup.state_dim,
        setup.dtype,
        alpha,
        beta,
        kappa,
    )
    if inputs is None:
        inputs_arr = None
        input_0 = None
    else:
        _check_float_array(inputs, "inputs", setup.dtype)
        inputs_arr = _canonicalize_inputs(inputs, setup.num_timesteps)
        input_0 = inputs_arr[0]
    filtered_mean_0, filtered_covariance_0, increment_0 = _unscented_condition(
        initial_mean,
        initial_covariance,
        observation_mean_fn,
        setup.observation_covariances[0],
        emissions[0],
        rule,
        input_0,
    )
    state_0 = _FilterState(
        filtered_mean_0,
        filtered_covariance_0,
        increment_0,
        jnp.zeros_like(increment_0),
    )
    if inputs_arr is None:
        scan_inputs = _NonlinearFilterStepInput(
            emissions[1:],
            setup.transition_covariances,
            setup.observation_covariances[1:],
        )

        def _step(
            state: _FilterState,
            args: _NonlinearFilterStepInput,
        ) -> tuple[_FilterState, _FilterStepOutput]:
            return _unscented_filter_step(
                state,
                args,
                transition_mean_fn,
                observation_mean_fn,
                rule,
            )

        final_state, rest = lax.scan(_step, state_0, scan_inputs)
    else:
        scan_inputs_u = _NonlinearFilterStepInputWithInput(
            emissions[1:],
            setup.transition_covariances,
            setup.observation_covariances[1:],
            inputs_arr[1:],
        )

        def _step_with_input(
            state: _FilterState,
            args: _NonlinearFilterStepInputWithInput,
        ) -> tuple[_FilterState, _FilterStepOutput]:
            step_args = _NonlinearFilterStepInput(
                args.emission,
                args.transition_covariance,
                args.observation_covariance,
            )
            return _unscented_filter_step(
                state,
                step_args,
                transition_mean_fn,
                observation_mean_fn,
                rule,
                args.input_t,
            )

        final_state, rest = lax.scan(
            _step_with_input,
            state_0,
            scan_inputs_u,
        )
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
        final_state.marginal_loglik + final_state.log_evidence_compensation,
        predicted_means,
        predicted_covariances,
        filtered_means,
        filtered_covariances,
        increments,
    )


def _validate_filter_posterior(
    posterior: GaussianFilterPosterior,
) -> tuple[int, int, jnp.dtype]:
    """Validate the moment shapes needed by the public smoother."""
    means = posterior.filtered_means
    if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
        raise ValueError(
            "filtered_means must have shape (T, state_dim) with positive axes"
        )
    num_timesteps, state_dim = means.shape
    expected_shapes = (
        ("predicted_means", posterior.predicted_means, means.shape),
        (
            "predicted_covariances",
            posterior.predicted_covariances,
            (num_timesteps, state_dim, state_dim),
        ),
        (
            "filtered_covariances",
            posterior.filtered_covariances,
            (num_timesteps, state_dim, state_dim),
        ),
        (
            "log_evidence_increments",
            posterior.log_evidence_increments,
            (num_timesteps,),
        ),
    )
    dtype = means.dtype
    _check_float_array(means, "filtered_means")
    for name, value, shape in expected_shapes:
        _check_float_array(value, name, dtype)
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}; got {value.shape}"
            )
    marginal = jnp.asarray(posterior.marginal_loglik)
    if marginal.ndim != 0:
        raise ValueError("marginal_loglik must be scalar")
    _check_float_array(marginal, "marginal_loglik", dtype)
    return num_timesteps, state_dim, dtype


def _rts_step(
    next_state: _SmootherState,
    args: tuple[Array, Array, Array, Array, Array],
) -> tuple[_SmootherState, _SmootherState]:
    """Apply one uncompiled Rauch--Tung--Striebel backward step."""
    (
        filtered_mean,
        filtered_covariance,
        next_predicted_mean,
        next_predicted_covariance,
        transition_matrix,
    ) = args
    cross_covariance = filtered_covariance @ transition_matrix.T
    predicted_cholesky = jnp.linalg.cholesky(next_predicted_covariance)
    lower_solution = solve_triangular(
        predicted_cholesky,
        cross_covariance.T,
        lower=True,
    )
    gain = solve_triangular(
        predicted_cholesky.T,
        lower_solution,
        lower=False,
    ).T
    smoothed_mean = filtered_mean + gain @ (
        next_state.mean - next_predicted_mean
    )
    smoothed_covariance = _symmetrize(
        filtered_covariance
        + gain @ (next_state.covariance - next_predicted_covariance) @ gain.T
    )
    state = _SmootherState(smoothed_mean, smoothed_covariance)
    return state, state


def rts_smoother(
    filtered_posterior: GaussianFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_matrix_shape"],
) -> GaussianSmootherPosterior:
    r"""Run a Rauch--Tung--Striebel backward smoother.

    This stage consumes only the forward pass's stored moments and the
    transition operators. A caller may therefore construct a compatible
    :class:`GaussianFilterPosterior` with a custom filtering method and
    reuse this smoother independently.

    Args:
        filtered_posterior: Forward-pass Gaussian moments.
        transition_matrix: Static ``(state_dim, state_dim)`` matrix or
            time-varying array with leading length ``ntime - 1``. Entry
            ``i`` maps state ``i`` to state ``i + 1``.

    Returns:
        The retained forward-pass fields plus smoothed means and
        covariances.

    Raises:
        ValueError: The posterior or transition array has an invalid shape
            or dtype.

    Note:
        Predicted covariances must be finite, symmetric, and positive
        definite.
    """
    num_timesteps, state_dim, dtype = _validate_filter_posterior(
        filtered_posterior
    )
    _check_float_array(transition_matrix, "transition_matrix", dtype)
    transition_matrices = _time_matrix(
        transition_matrix,
        num_timesteps - 1,
        state_dim,
        state_dim,
        "transition_matrix",
    )
    last_state = _SmootherState(
        filtered_posterior.filtered_means[-1],
        filtered_posterior.filtered_covariances[-1],
    )
    scan_inputs = (
        filtered_posterior.filtered_means[:-1],
        filtered_posterior.filtered_covariances[:-1],
        filtered_posterior.predicted_means[1:],
        filtered_posterior.predicted_covariances[1:],
        transition_matrices,
    )
    _, earlier_states = lax.scan(
        _rts_step,
        last_state,
        scan_inputs,
        reverse=True,
    )
    smoothed_means = jnp.concatenate((
        earlier_states.mean,
        last_state.mean[None],
    ))
    smoothed_covariances = jnp.concatenate((
        earlier_states.covariance,
        last_state.covariance[None],
    ))
    return GaussianSmootherPosterior(
        *filtered_posterior,
        smoothed_means,
        smoothed_covariances,
    )
