# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for scaled unscented Kalman filtering."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
import smcx.kalman as kalman_module
from tests import _unscented_kalman_reference as nonlinear_reference


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
    budget = 4 * np.finfo(np.float32).eps

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
    budget = 4 * np.finfo(np.float32).eps

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
    accuracy_budget = 2 * eps / 3
    # Sixteen ulps at the posterior scale covers covariance eigensolver error.
    psd_budget = 16 * eps * np.linalg.norm(expected, ord=2)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=accuracy_budget,
    )
    assert np.linalg.eigvalsh((actual + actual.T) / 2).min() >= -psd_budget


def test_unscented_kalman_matches_independent_nonlinear_reference():
    """Every posterior field matches the Stone Soup UKF oracle."""
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
    expected_fields = (
        reference.MARGINAL_LOG_LIKELIHOOD,
        reference.PREDICTED_MEANS,
        reference.PREDICTED_COVARIANCES,
        reference.FILTERED_MEANS,
        reference.FILTERED_COVARIANCES,
        reference.LOG_EVIDENCE_INCREMENTS,
    )

    # Five 2x2 steps have innovation condition number below 2.62.
    # 256*eps*scale covers the observed operation depth on CPU and Metal.
    for actual, expected in zip(posterior, expected_fields, strict=True):
        _assert_roundoff_close(actual, expected, ulps=256)
