# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for scaled unscented Kalman filtering and smoothing."""

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import solve_triangular

import smcx
import smcx.kalman as kalman_module
from tests import _unscented_kalman_reference as nonlinear_reference
from tests._gaussian_smoothing_reference import dense_joint_moments


def _identity(state):
    return state


def _square(state):
    return state**2


def _minimal_float32_ukf(**parameters):
    zero = jnp.zeros(1, dtype=jnp.float32)
    one = jnp.eye(1, dtype=jnp.float32)
    return smcx.unscented_kalman_filter(
        zero,
        one,
        _identity,
        one,
        _identity,
        one,
        zero[None],
        **parameters,
    )


def _assert_roundoff_close(actual, expected, *, ulps=512):
    """Compare one well-conditioned result within an f32/f64 budget."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected, dtype=actual_array.dtype)
    scale = max(1.0, float(np.max(np.abs(expected_array))))
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=ulps * np.finfo(actual_array.dtype).eps * scale,
    )


def test_scaled_unscented_moments_recover_correlated_gaussian():
    """Column-oriented sigma points retain a correlated Gaussian."""
    mean = jnp.array([1.0, -2.0])
    covariance = jnp.array([[0.5, 0.2], [0.2, 0.4]])
    rule = kalman_module._scaled_unscented_rule(
        2,
        mean.dtype,
        1.0,
        2.0,
        0.0,
    )

    points = kalman_module._sigma_points(mean, covariance, rule)
    recovered_mean, recovered_covariance = kalman_module._unscented_moments(
        points, rule
    )

    _assert_roundoff_close(recovered_mean, mean)
    _assert_roundoff_close(recovered_covariance, covariance)


def test_unscented_core_reduces_to_one_linear_filter_step():
    """The pure scaled transform and condition match exact linear algebra."""
    initial_mean = jnp.array([0.2, -0.1])
    initial_covariance = jnp.array([[0.5, 0.03], [0.03, 0.4]])
    transition_matrix = jnp.array([[0.85, 0.1], [-0.05, 0.9]])
    transition_bias = jnp.array([0.02, -0.03])
    transition_covariance = jnp.array([[0.08, 0.01], [0.01, 0.06]])
    observation_matrix = jnp.array([[1.0, -0.2]])
    observation_bias = jnp.array([0.04])
    observation_covariance = jnp.array([[0.3]])
    emissions = jnp.array([[0.1], [-0.2]])

    def transition_mean(state):
        return transition_matrix @ state + transition_bias

    def observation_mean(state):
        return observation_matrix @ state + observation_bias

    exact = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        transition_matrix,
        transition_covariance,
        observation_matrix,
        observation_covariance,
        emissions,
        transition_bias=transition_bias,
        observation_bias=observation_bias,
    )
    rule = kalman_module._scaled_unscented_rule(
        2,
        initial_mean.dtype,
        1.0,
        2.0,
        0.0,
    )
    filtered_mean, filtered_covariance, increment = (
        kalman_module._unscented_condition(
            initial_mean,
            initial_covariance,
            observation_mean,
            observation_covariance,
            emissions[0],
            rule,
        )
    )
    state = kalman_module._FilterState(
        filtered_mean,
        filtered_covariance,
        increment,
        jnp.zeros_like(increment),
    )
    final_state, output = kalman_module._unscented_filter_step(
        state,
        kalman_module._NonlinearFilterStepInput(
            emissions[1],
            transition_covariance,
            observation_covariance,
        ),
        transition_mean,
        observation_mean,
        rule,
    )

    _assert_roundoff_close(filtered_mean, exact.filtered_means[0])
    _assert_roundoff_close(
        filtered_covariance,
        exact.filtered_covariances[0],
    )
    _assert_roundoff_close(increment, exact.log_evidence_increments[0])
    _assert_roundoff_close(output.predicted_mean, exact.predicted_means[1])
    _assert_roundoff_close(
        output.predicted_covariance,
        exact.predicted_covariances[1],
    )
    _assert_roundoff_close(output.filtered_mean, exact.filtered_means[1])
    _assert_roundoff_close(
        output.filtered_covariance,
        exact.filtered_covariances[1],
    )
    _assert_roundoff_close(
        final_state.marginal_loglik,
        exact.marginal_loglik,
    )


def test_input_aware_unscented_core_matches_linear_controls():
    """The pure core applies controls at the destination time."""
    initial_mean = jnp.array([0.1, -0.2])
    initial_covariance = jnp.array([[0.6, 0.04], [0.04, 0.45]])
    transition_matrix = jnp.array([[0.9, 0.05], [-0.1, 0.8]])
    transition_bias = jnp.array([0.03, -0.02])
    transition_input_matrix = jnp.array([[0.4], [0.1]])
    transition_covariance = jnp.array([[0.07, 0.01], [0.01, 0.05]])
    observation_matrix = jnp.array([[1.0, 0.25]])
    observation_bias = jnp.array([-0.05])
    observation_input_matrix = jnp.array([[0.2]])
    observation_covariance = jnp.array([[0.25]])
    emissions = jnp.array([[20.0], [-0.1]])
    inputs = jnp.array([100.0, 0.2])

    def transition_mean(state, input_t):
        return (
            transition_matrix @ state
            + transition_bias
            + transition_input_matrix @ input_t
        )

    def observation_mean(state, input_t):
        return (
            observation_matrix @ state
            + observation_bias
            + observation_input_matrix @ input_t
        )

    exact = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        transition_matrix,
        transition_covariance,
        observation_matrix,
        observation_covariance,
        emissions,
        transition_bias=transition_bias,
        observation_bias=observation_bias,
        transition_input_matrix=transition_input_matrix,
        observation_input_matrix=observation_input_matrix,
        inputs=inputs,
    )
    rule = kalman_module._scaled_unscented_rule(
        2,
        initial_mean.dtype,
        1.0,
        2.0,
        0.0,
    )
    filtered_mean, filtered_covariance, increment = (
        kalman_module._unscented_condition(
            initial_mean,
            initial_covariance,
            observation_mean,
            observation_covariance,
            emissions[0],
            rule,
            inputs[0, None],
        )
    )
    state = kalman_module._FilterState(
        filtered_mean,
        filtered_covariance,
        increment,
        jnp.zeros_like(increment),
    )
    final_state, output = kalman_module._unscented_filter_step(
        state,
        kalman_module._NonlinearFilterStepInput(
            emissions[1],
            transition_covariance,
            observation_covariance,
        ),
        transition_mean,
        observation_mean,
        rule,
        inputs[1, None],
    )

    _assert_roundoff_close(filtered_mean, exact.filtered_means[0])
    _assert_roundoff_close(
        filtered_covariance,
        exact.filtered_covariances[0],
    )
    _assert_roundoff_close(increment, exact.log_evidence_increments[0])
    _assert_roundoff_close(output.predicted_mean, exact.predicted_means[1])
    _assert_roundoff_close(
        output.predicted_covariance,
        exact.predicted_covariances[1],
    )
    _assert_roundoff_close(output.filtered_mean, exact.filtered_means[1])
    _assert_roundoff_close(
        output.filtered_covariance,
        exact.filtered_covariances[1],
    )
    _assert_roundoff_close(
        final_state.marginal_loglik,
        exact.marginal_loglik,
    )
    public = smcx.unscented_kalman_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_covariance,
        observation_mean,
        observation_covariance,
        emissions,
        inputs=inputs,
    )
    for actual, expected in zip(public, exact, strict=True):
        _assert_roundoff_close(actual, expected)


def test_unscented_kalman_reduces_to_linear_filter():
    """Affine mean callbacks reproduce every exact-filter field."""
    initial_mean = jnp.array([0.2, -0.1])
    initial_covariance = jnp.array([[0.5, 0.03], [0.03, 0.4]])
    transition_matrix = jnp.array([[0.85, 0.1], [-0.05, 0.9]])
    transition_bias = jnp.array([0.02, -0.03])
    transition_covariance = jnp.array([
        [[0.08, 0.01], [0.01, 0.06]],
        [[0.07, 0.00], [0.00, 0.05]],
        [[0.09, -0.01], [-0.01, 0.08]],
    ])
    observation_matrix = jnp.array([[1.0, -0.2]])
    observation_bias = jnp.array([0.04])
    observation_covariance = jnp.array([
        [[0.3]],
        [[0.2]],
        [[0.4]],
        [[0.25]],
    ])
    emissions = jnp.array([[0.1], [-0.2], [0.3], [0.05]])

    def transition_mean(state):
        return transition_matrix @ state + transition_bias

    def observation_mean(state):
        return observation_matrix @ state + observation_bias

    exact = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        transition_matrix,
        transition_covariance,
        observation_matrix,
        observation_covariance,
        emissions,
        transition_bias=transition_bias,
        observation_bias=observation_bias,
    )
    unscented = smcx.unscented_kalman_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_covariance,
        observation_mean,
        observation_covariance,
        emissions,
    )

    for actual, expected in zip(unscented, exact, strict=True):
        _assert_roundoff_close(actual, expected)

    compiled = jax.jit(
        lambda observations: smcx.unscented_kalman_filter(
            initial_mean,
            initial_covariance,
            transition_mean,
            transition_covariance,
            observation_mean,
            observation_covariance,
            observations,
        )
    )(emissions)
    for actual, expected in zip(compiled, unscented, strict=True):
        _assert_roundoff_close(actual, expected)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"alpha": 0.0}, "alpha must be greater"),
        ({"beta": math.inf}, "beta must be finite"),
        ({"kappa": -1.0}, "state_dim.*kappa"),
        ({"beta": -1.0}, "must be nonnegative"),
        ({"alpha": 1e-30}, "non-finite weights"),
        ({"alpha": 1e-200}, "non-finite weights"),
    ],
)
def test_unscented_filter_rejects_invalid_rule(parameters, message):
    """Invalid scaled-rule parameters use the public exception."""
    with pytest.raises(ValueError, match=message):
        _minimal_float32_ukf(**parameters)


@pytest.mark.parametrize(
    ("argument", "scalar", "constraint"),
    [
        ("initial_covariance", 0.0, "be positive definite"),
        ("transition_covariance", -1.0, "be positive semidefinite"),
        ("observation_covariance", 0.0, "be positive definite"),
        ("initial_covariance", jnp.nan, "contain only finite values"),
    ],
)
def test_unscented_filter_rejects_invalid_covariance(
    argument,
    scalar,
    constraint,
):
    """Factored covariances must be PD; the transition covariance PSD."""
    zero = jnp.zeros(1, dtype=jnp.float32)
    covariance = jnp.eye(1, dtype=jnp.float32)
    model: dict[str, Any] = {
        "initial_mean": zero,
        "initial_covariance": covariance,
        "transition_mean_fn": _identity,
        "transition_covariance": covariance,
        "observation_mean_fn": _identity,
        "observation_covariance": covariance,
        "emissions": zero[None],
    }
    model[argument] = jnp.array([[scalar]], dtype=jnp.float32)

    with pytest.raises(ValueError, match=f"{argument} must {constraint}"):
        smcx.unscented_kalman_filter(**model)


def test_unscented_accepts_singular_transition_covariance():
    """A rank-deficient transition covariance is a valid model.

    Noise enters only the second coordinate (constant-velocity shape).
    The UKF never factors the transition covariance itself — only the
    prior and the predicted covariance, which the propagated sigma
    spread keeps positive definite here — so the filter must accept
    the model and, with linear means, match the exact Kalman filter.
    """
    initial_mean = jnp.array([0.2, -0.1])
    initial_covariance = jnp.array([[0.5, 0.03], [0.03, 0.4]])
    transition_matrix = jnp.array([[1.0, 0.1], [0.0, 0.9]])
    transition_covariance = jnp.array([[0.0, 0.0], [0.0, 0.25]])
    observation_matrix = jnp.array([[1.0, 0.0]])
    observation_covariance = jnp.array([[0.3]])
    emissions = jnp.array([[0.1], [-0.2], [0.05]])

    exact = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        transition_matrix,
        transition_covariance,
        observation_matrix,
        observation_covariance,
        emissions,
    )
    unscented = smcx.unscented_kalman_filter(
        initial_mean,
        initial_covariance,
        lambda state: transition_matrix @ state,
        transition_covariance,
        lambda state: observation_matrix @ state,
        observation_covariance,
        emissions,
    )

    for actual, expected in zip(unscented, exact, strict=True):
        _assert_roundoff_close(actual, expected)


def test_unscented_guard_boundary_keeps_covariance_psd():
    """The filter/smoother PSD-guarantee boundary is safe, not a bug.

    With ``s = alpha**2 * (d + kappa)``, the accepted domain
    ``alpha**2 * kappa + d * beta >= 0`` equals
    ``s + d * (beta - alpha**2) >= 0``, which by Cauchy-Schwarz over
    the paired sigma deltas is exactly the condition for the paired
    moment to dominate the negative rank-one correction. The boundary
    configuration below has ``beta < alpha**2`` (a subtractive
    rank-one weight) yet must be accepted and must keep every filtered and
    smoothed covariance PSD up to roundoff.
    """
    posterior = smcx.unscented_kalman_filter(
        jnp.array([0.1], dtype=jnp.float32),
        jnp.array([[0.2]], dtype=jnp.float32),
        lambda state: state**3,
        jnp.array([[0.05]], dtype=jnp.float32),
        _square,
        jnp.array([[0.5]], dtype=jnp.float32),
        jnp.array([[0.3], [0.2]], dtype=jnp.float32),
        alpha=2.0,
        beta=0.0,
        kappa=0.0,
    )

    smoothed = smcx.gaussian_smoother(
        posterior,
        lambda state: state**3,
        method=smcx.unscented(alpha=2.0, beta=0.0, kappa=0.0),
    )
    mean = posterior.filtered_means[0, 0]
    covariance = posterior.filtered_covariances[0, 0, 0]
    # For d=1, alpha=2, and kappa=0, the cubic sigma cross moment is
    # D = P * (3*m**2 + 4*P). The analytic/public means were exact on CPU
    # and MPS; 16 f32 eps covers this scalar solve's operation depth.
    cross_covariance = covariance * (3.0 * mean**2 + 4.0 * covariance)
    gain = cross_covariance / posterior.predicted_covariances[1, 0, 0]
    expected_mean = mean + gain * (
        posterior.filtered_means[1, 0] - posterior.predicted_means[1, 0]
    )
    _assert_roundoff_close(
        smoothed.smoothed_means[0, 0], expected_mean, ulps=16
    )
    covariances = jnp.concatenate((
        posterior.filtered_covariances,
        smoothed.smoothed_covariances,
    ))
    eigenvalues = np.linalg.eigvalsh(np.asarray(covariances))
    floor = -8 * np.finfo(np.float32).eps
    assert np.all(np.isfinite(eigenvalues))
    assert np.all(eigenvalues >= floor)
    diagonals = np.diagonal(
        np.asarray(smoothed.smoothed_covariances),
        axis1=-2,
        axis2=-1,
    )
    assert np.all(diagonals >= 0.0)


def test_unscented_smoother_uses_sigma_cross_covariance_directly():
    """The public gain uses D rather than recovering it through A_eff."""
    dtype = jnp.float32
    cosine = jnp.sqrt(jnp.asarray(2.0 / 3.0, dtype))
    sine = -jnp.sqrt(jnp.asarray(1.0 / 3.0, dtype))
    rotation = jnp.array([[cosine, -sine], [sine, cosine]], dtype)
    filtered_covariance = rotation @ jnp.diag(jnp.array([1.0, 1e-4], dtype))
    filtered_covariance = filtered_covariance @ rotation.T
    filtered_covariance = (filtered_covariance + filtered_covariance.T) / 2
    filtered_mean = jnp.array([-0.07525469, -0.146573], dtype)
    transition = jnp.array(
        [[93.36103, -29.875969], [104.618095, -19.394943]],
        dtype,
    )
    offsets = (
        jnp.sqrt(jnp.asarray(2.0, dtype))
        * jnp.linalg.cholesky(filtered_covariance)
    ).T
    center = transition @ filtered_mean
    state_positive = filtered_mean + offsets
    state_negative = filtered_mean - offsets
    positive = jax.vmap(lambda state: transition @ state)(state_positive)
    negative = jax.vmap(lambda state: transition @ state)(state_negative)
    delta_sum = ((positive - center) + (negative - center)).sum(axis=0)
    state_offsets = 0.5 * (state_positive - state_negative)
    cross_covariance = 0.25 * state_offsets.T @ (positive - negative)
    transformed_covariance = 0.25 * (
        (positive - center).T @ (positive - center)
        + (negative - center).T @ (negative - center)
    ) + 0.0625 * jnp.outer(delta_sum, delta_sum)
    predicted_covariance = transformed_covariance + 0.01 * jnp.eye(
        2, dtype=dtype
    )
    predicted_covariance = (predicted_covariance + predicted_covariance.T) / 2
    predicted_mean = center + 0.25 * delta_sum
    residual = jnp.array([-7.1869713, 6.953233], dtype)
    posterior = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0, dtype),
        jnp.stack((filtered_mean, predicted_mean)),
        jnp.stack((filtered_covariance, predicted_covariance)),
        jnp.stack((filtered_mean, predicted_mean + residual)),
        jnp.stack((filtered_covariance, 0.5 * predicted_covariance)),
        jnp.zeros(2, dtype),
    )

    smoothed = smcx.gaussian_smoother(
        posterior,
        lambda state: transition @ state,
        method=smcx.unscented(),
    )

    factor = jnp.linalg.cholesky(predicted_covariance)
    lower_solution = solve_triangular(factor, cross_covariance.T, lower=True)
    gain = solve_triangular(factor.T, lower_solution, lower=False).T
    expected = filtered_mean + gain @ residual
    scale = max(1.0, float(np.abs(expected).max()))
    # The A_eff-to-D reconstruction mutant misses by approximately 1.36e-3
    # on CPU and 4.39e-3 on MPS; this budget is at most 1.53e-5.
    tolerance = float(128 * np.finfo(np.float32).eps * scale)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_means[0]),
        np.asarray(expected),
        rtol=0.0,
        atol=tolerance,
    )


def test_unscented_smoother_matches_dense_linear_oracle_with_inputs():
    """G7 and destination-time inputs hold for a varying linear model."""
    initial_mean = jnp.array([0.2, -0.1])
    initial_covariance = jnp.array([[0.7, 0.08], [0.08, 0.5]])
    base_transition = jnp.array([[0.9, 0.1], [-0.05, 0.85]])
    input_slope = jnp.array([[0.03, -0.02], [0.01, 0.02]])
    transition_covariance = jnp.array([[0.08, 0.01], [0.01, 0.06]])
    observation_matrix = jnp.array([[1.0, 0.2]])
    observation_covariance = jnp.array([[0.3]])
    emissions = jnp.array([[0.1], [-0.2], [0.3], [0.05]])
    inputs = jnp.array([[40.0], [-0.4], [0.6], [0.2]])

    def transition_mean(state, input_t):
        matrix = base_transition + input_t[0] * input_slope
        return matrix @ state

    def observation_mean(state, _input_t):
        return observation_matrix @ state

    posterior = smcx.gaussian_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_covariance,
        observation_mean,
        observation_covariance,
        emissions,
        method=smcx.unscented(),
        inputs=inputs,
    )
    smoothed = smcx.gaussian_smoother(
        posterior,
        transition_mean,
        method=smcx.unscented(),
        inputs=inputs,
    )

    transitions = np.stack([
        np.asarray(base_transition + input_t[0] * input_slope)
        for input_t in inputs[1:]
    ])
    expected_means, expected_covariances, _ = dense_joint_moments(
        np.asarray(initial_mean),
        np.asarray(initial_covariance),
        transitions,
        np.broadcast_to(np.asarray(transition_covariance), (3, 2, 2)),
        np.zeros((3, 2)),
        np.broadcast_to(np.asarray(observation_matrix), (4, 1, 2)),
        np.broadcast_to(np.asarray(observation_covariance), (4, 1, 1)),
        np.zeros((4, 1)),
        np.asarray(emissions),
    )
    # The dense joint has condition number below 68.7; observed error is
    # at most 3.75 scaled eps on CPU and MPS.
    _assert_roundoff_close(smoothed.smoothed_means, expected_means, ulps=64)
    _assert_roundoff_close(
        smoothed.smoothed_covariances, expected_covariances, ulps=64
    )


def test_unscented_smoother_exposes_mismatched_record_precondition():
    """G8: an EKF record has no Unscented covariance guarantee."""
    posterior = smcx.extended_kalman_filter(
        jnp.array([0.2]),
        jnp.array([[0.5]]),
        lambda state: state**3,
        lambda state: (3.0 * state**2)[None],
        jnp.array([[0.001]]),
        _identity,
        lambda _state: jnp.ones((1, 1)),
        jnp.array([[0.2]]),
        jnp.array([[0.1], [0.15]]),
    )

    smoothed = smcx.gaussian_smoother(
        posterior,
        lambda state: state**3,
        method=smcx.unscented(),
    )
    mean = posterior.filtered_means[0, 0]
    covariance = posterior.filtered_covariances[0, 0, 0]
    effective_transition = covariance + 3.0 * mean**2
    reconstructed_noise = (
        posterior.predicted_covariances[1, 0, 0]
        - effective_transition**2 * covariance
    )
    scale = max(
        1.0,
        float(abs(posterior.predicted_covariances[1, 0, 0])),
        float(abs(effective_transition**2 * covariance)),
    )
    assert float(reconstructed_noise) < (
        -128 * np.finfo(np.asarray(reconstructed_noise).dtype).eps * scale
    )

    expected = smcx.rts_smoother(
        posterior,
        effective_transition.reshape(1, 1),
    )
    assert np.all(np.isfinite(np.asarray(smoothed.smoothed_covariances)))
    # The scalar reduction differs by at most one eps on CPU and MPS.
    _assert_roundoff_close(
        smoothed.smoothed_means, expected.smoothed_means, ulps=32
    )
    _assert_roundoff_close(
        smoothed.smoothed_covariances, expected.smoothed_covariances, ulps=32
    )


@pytest.mark.parametrize(
    ("transition_mean", "message"),
    [
        (lambda _state: jnp.zeros(2), r"must have shape \(1,\)"),
        (lambda _state: jnp.zeros(1, dtype=jnp.float16), "must have dtype"),
        (lambda _state: np.zeros(1), "must return a JAX array"),
    ],
)
@pytest.mark.parametrize("with_input", [False, True], ids=["plain", "input"])
def test_unscented_smoother_rejects_invalid_transition_output(
    transition_mean,
    message,
    with_input,
):
    """Both callback arities report structural errors uniformly."""
    posterior = smcx.unscented_kalman_filter(
        jnp.zeros(1),
        jnp.eye(1),
        _identity,
        jnp.eye(1),
        _identity,
        jnp.eye(1),
        jnp.zeros((2, 1)),
    )

    def callback(state, *_input_t):
        return transition_mean(state)

    inputs = jnp.zeros((2, 1)) if with_input else None

    with pytest.raises(ValueError, match=message):
        smcx.gaussian_smoother(
            posterior,
            callback,
            method=smcx.unscented(),
            inputs=inputs,
        )


def test_unscented_nondefault_rule_matches_scalar_oracle():
    """A valid negative central weight retains analytic scalar moments."""
    posterior = smcx.unscented_kalman_filter(
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([[1.0]], dtype=jnp.float32),
        _identity,
        jnp.array([[0.1]], dtype=jnp.float32),
        _square,
        jnp.array([[0.75]], dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        alpha=0.5,
        beta=0.0,
        kappa=1.0,
    )
    expected_logpdf = -0.5 * (math.log(2.0 * math.pi) + 1.0)
    # Four f32 eps covers the scalar transform and log-density operation depth.
    budget = float(4 * np.finfo(np.float32).eps)

    np.testing.assert_allclose(
        posterior.filtered_means,
        [[0.0]],
        rtol=0.0,
        atol=budget,
    )
    np.testing.assert_allclose(
        posterior.filtered_covariances,
        [[[1.0]]],
        rtol=0.0,
        atol=budget,
    )
    np.testing.assert_allclose(
        [posterior.marginal_loglik, posterior.log_evidence_increments[0]],
        expected_logpdf,
        rtol=0.0,
        atol=budget,
    )


def test_unscented_filter_regenerates_points_after_process_noise():
    """Process noise reaches a nonlinear observation transform."""
    posterior = smcx.unscented_kalman_filter(
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([[1.0]], dtype=jnp.float32),
        lambda state: jnp.zeros_like(state),
        jnp.array([[1.0]], dtype=jnp.float32),
        _square,
        jnp.array([[1.0]], dtype=jnp.float32),
        jnp.zeros((2, 1), dtype=jnp.float32),
    )
    expected = -0.5 * (math.log(6.0 * math.pi) + 1.0 / 3.0)
    # Four f32 eps covers both scalar transform and log-density evaluations.
    budget = float(4 * np.finfo(np.float32).eps)

    np.testing.assert_allclose(
        posterior.predicted_covariances[1],
        [[1.0]],
        rtol=0.0,
        atol=budget,
    )
    np.testing.assert_allclose(
        posterior.log_evidence_increments,
        [expected, expected],
        rtol=0.0,
        atol=budget,
    )


def test_unscented_float32_update_is_psd_and_accurate():
    """Residual-sigma conditioning survives a Metal cancellation case."""
    dtype = jnp.float32
    covariance = jnp.array(
        [[0.9975046, 0.04986679], [0.04986679, 0.00349542]],
        dtype=dtype,
    )
    observation_matrix = jnp.array(
        [[9.9755106, -0.0069942847], [0.69942844, 0.099755101]],
        dtype=dtype,
    )

    def observation_mean(state):
        return observation_matrix @ state

    posterior = smcx.unscented_kalman_filter(
        jnp.zeros(2, dtype=dtype),
        covariance,
        _identity,
        jnp.eye(2, dtype=dtype),
        observation_mean,
        1e-8 * jnp.eye(2, dtype=dtype),
        jnp.zeros((1, 2), dtype=dtype),
    )
    actual = np.asarray(posterior.filtered_covariances[0], dtype=np.float64)
    expected = np.array([
        [9.999998754e-11, 5.046806022e-15],
        [5.046806022e-15, 9.990034666e-7],
    ])
    eps = np.finfo(np.float32).eps
    # CPU/MPS errors are <=5.19e-8; 2*eps/3 retains 53% forward-error margin.
    accuracy_budget = float(2 * eps / 3)
    # Sixteen ulps at the posterior scale covers covariance eigensolver error.
    psd_budget = float(16 * eps * np.linalg.norm(expected, ord=2))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=accuracy_budget,
    )
    assert np.linalg.eigvalsh((actual + actual.T) / 2).min() >= -psd_budget


def test_unscented_filter_and_smoother_match_independent_reference():
    """Every moment matches the Stone Soup UKF and smoother oracle."""
    reference = nonlinear_reference

    def transition_mean(state):
        return jnp.stack((
            0.82 * state[0] + 0.18 * state[1] + 0.05 * jnp.sin(state[0]),
            -0.12 * state[0] + 0.90 * state[1] + 0.04 * state[0] * state[1],
        ))

    def observation_mean(state):
        return jnp.stack((
            state[0] + 0.10 * state[1] ** 2,
            0.65 * state[1] + 0.12 * jnp.sin(state[0]),
        ))

    posterior = smcx.unscented_kalman_filter(
        jnp.asarray(reference.INITIAL_MEAN),
        jnp.asarray(reference.INITIAL_COVARIANCE),
        transition_mean,
        jnp.asarray(reference.TRANSITION_COVARIANCE),
        observation_mean,
        jnp.asarray(reference.OBSERVATION_COVARIANCE),
        jnp.asarray(reference.EMISSIONS),
        alpha=reference.ALPHA,
        beta=reference.BETA,
        kappa=reference.KAPPA,
    )
    method = smcx.unscented(
        alpha=reference.ALPHA,
        beta=reference.BETA,
        kappa=reference.KAPPA,
    )

    def smooth(record):
        return smcx.gaussian_smoother(record, transition_mean, method=method)

    smoothed = smooth(posterior)
    compiled = jax.jit(smooth)(posterior)
    expected_fields = (
        reference.MARGINAL_LOG_LIKELIHOOD,
        reference.PREDICTED_MEANS,
        reference.PREDICTED_COVARIANCES,
        reference.FILTERED_MEANS,
        reference.FILTERED_COVARIANCES,
        reference.LOG_EVIDENCE_INCREMENTS,
        reference.SMOOTHED_MEANS,
        reference.SMOOTHED_COVARIANCES,
    )

    # Five 2x2 steps have innovation condition number below 2.62, while
    # positive-time predicted and nonterminal filtered covariances are below
    # 1.50. The established 256*eps budget covers CPU and Metal depth.
    for actual, expected in zip(smoothed, expected_fields, strict=True):
        _assert_roundoff_close(actual, expected, ulps=256)

    # Compilation adds no substantive error to this well-conditioned record.
    for actual, expected in zip(compiled, smoothed, strict=True):
        _assert_roundoff_close(actual, expected, ulps=32)
