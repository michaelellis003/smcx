# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the scaled unscented numerical core."""

import jax.numpy as jnp
import numpy as np

import smcx
import smcx.kalman as kalman_module


def _assert_roundoff_close(actual, expected):
    """Compare one well-conditioned result within an f32/f64 budget."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected, dtype=actual_array.dtype)
    scale = max(1.0, float(np.max(np.abs(expected_array))))
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=512 * np.finfo(actual_array.dtype).eps * scale,
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
