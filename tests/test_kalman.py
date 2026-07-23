# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact linear-Gaussian filtering and smoothing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
from tests import _kalman_reference as multivariate_reference
from tests._lgssm_reference import EXACT_LOG_LIKELIHOOD, REFERENCE_TIMES
from tests._lgssm_reference import FILTERED_MEANS as EXACT_FILTERED_MEANS
from tests._lgssm_reference import FILTERED_VARIANCES as EXACT_FILTERED_VARS


def _assert_roundoff_close(actual, expected):
    """Compare the small, well-conditioned reference within f32/f64 error."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    # The fixture has five 2x2 recurrences and covariance condition numbers
    # below 2.7. A 64*eps*scale forward-error budget covers their matrix
    # products and triangular solves while remaining dtype-honest.
    scale = max(1.0, float(np.max(np.abs(expected_array))))
    atol = 64 * np.finfo(actual_array.dtype).eps * scale
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
    )


def test_kalman_filter_matches_frozen_dynamax_reference(
    lgssm_params, lgssm_data
):
    """The exact filter reproduces independently generated moments."""
    _, emissions = lgssm_data
    posterior = smcx.kalman_filter(
        lgssm_params["initial_mean"],
        lgssm_params["initial_cov"],
        lgssm_params["dynamics_weights"],
        lgssm_params["dynamics_cov"],
        lgssm_params["emissions_weights"],
        lgssm_params["emissions_cov"],
        emissions,
    )

    is_f64 = posterior.filtered_means.dtype == jnp.float64
    # Dynamax's PSD solve adds 1e-9 jitter. Against the unjittered
    # covariance-form recurrence this shifts the 50-step f64 log evidence
    # by 2.3e-9 and selected variances by at most 5.1e-10. The 5e-9
    # absolute gate admits that known oracle-policy difference; 2e-5 is
    # the explicit f32/Metal arithmetic budget.
    atol = 5e-9 if is_f64 else 2e-5
    np.testing.assert_allclose(
        posterior.marginal_loglik,
        EXACT_LOG_LIKELIHOOD,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.filtered_means[REFERENCE_TIMES, 0],
        EXACT_FILTERED_MEANS,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.filtered_covariances[REFERENCE_TIMES, 0, 0],
        EXACT_FILTERED_VARS,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.predicted_means[0],
        lgssm_params["initial_mean"],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.predicted_means[1:, 0],
        0.9 * posterior.filtered_means[:-1, 0],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.log_evidence_increments.sum(),
        posterior.marginal_loglik,
        rtol=0.0,
        atol=atol,
    )


def test_time_varying_terms_and_controls_preserve_input_alignment():
    """Timed terms compose with controls applied to the destination state."""
    emissions = jnp.array([[0.2], [-0.1], [0.4], [0.7]])
    inputs = jnp.array([100.0, 2.0, -1.0, 0.5])
    initial_mean = jnp.array([0.0])
    initial_covariance = jnp.array([[1.0]])
    transition_matrix = jnp.array([[0.9]])
    transition_covariance = jnp.array([[0.25]])
    observation_matrix = jnp.array([[1.0]])
    observation_covariance = jnp.array([[1.0]])

    controlled = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        jnp.broadcast_to(transition_matrix, (3, 1, 1)),
        jnp.broadcast_to(transition_covariance, (3, 1, 1)),
        jnp.broadcast_to(observation_matrix, (4, 1, 1)),
        jnp.broadcast_to(observation_covariance, (4, 1, 1)),
        emissions + inputs[:, None],
        transition_bias=jnp.array([[0.1], [0.2], [0.3]]),
        observation_bias=jnp.zeros((4, 1)),
        transition_input_matrix=jnp.array([[0.7]]),
        observation_input_matrix=jnp.array([[1.0]]),
        inputs=inputs,
    )
    no_observation_control = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        transition_matrix,
        transition_covariance,
        observation_matrix,
        observation_covariance,
        emissions,
        transition_bias=jnp.array([0.0]),
    )

    # Shifting y[t] and its observation mean by the same D @ u[t] leaves
    # every innovation unchanged. The conspicuous unused input[0] catches
    # accidental outgoing-transition alignment.
    # Subtracting the shifted f32 observation costs about 1.6e-6 here;
    # 2e-5 is the package's explicit Metal arithmetic budget.
    atol = 1e-12 if controlled.filtered_means.dtype == jnp.float64 else 2e-5
    np.testing.assert_allclose(
        controlled.filtered_means[0],
        no_observation_control.filtered_means[0],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        controlled.predicted_means[1:, 0],
        0.9 * controlled.filtered_means[:-1, 0]
        + jnp.array([0.1, 0.2, 0.3])
        + 0.7 * inputs[1:],
        rtol=0.0,
        atol=atol,
    )


def test_kalman_filter_compiled_matches_eager():
    """Compilation preserves every field of the exact filter result."""
    args = (
        jnp.array([0.0, 0.2]),
        jnp.array([[1.0, 0.1], [0.1, 0.8]]),
        jnp.array([[0.9, 0.2], [0.0, 0.7]]),
        jnp.array([[0.2, 0.03], [0.03, 0.1]]),
        jnp.array([[1.0, -0.2]]),
        jnp.array([[0.4]]),
        jnp.array([[0.3], [-0.5], [0.1]]),
    )

    eager = smcx.kalman_filter(*args)
    compiled = jax.jit(smcx.kalman_filter)(*args)

    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(compiled_value, eager_value)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("initial_covariance", jnp.eye(2), "initial_covariance"),
        ("transition_matrix", jnp.ones((3, 1, 1)), "transition_matrix"),
        ("observation_matrix", jnp.ones((2, 2)), "observation_matrix"),
        ("emissions", jnp.empty((0, 1)), "emissions"),
    ],
)
def test_kalman_filter_rejects_misaligned_shapes(argument, value, message):
    """Malformed dense models fail at the public Python boundary."""
    model = {
        "initial_mean": jnp.zeros(1),
        "initial_covariance": jnp.eye(1),
        "transition_matrix": jnp.eye(1),
        "transition_covariance": jnp.eye(1),
        "observation_matrix": jnp.eye(1),
        "observation_covariance": jnp.eye(1),
        "emissions": jnp.zeros((3, 1)),
    }
    model[argument] = value

    with pytest.raises(ValueError, match=message):
        smcx.kalman_filter(**model)


def test_kalman_filter_rejects_input_matrix_without_inputs():
    """A control operator cannot silently behave as an affine zero."""
    with pytest.raises(ValueError, match="input matrices require inputs"):
        smcx.kalman_filter(
            jnp.zeros(1),
            jnp.eye(1),
            jnp.eye(1),
            jnp.eye(1),
            jnp.eye(1),
            jnp.eye(1),
            jnp.zeros((2, 1)),
            transition_input_matrix=jnp.eye(1),
        )


def test_multivariate_filter_and_smoother_match_independent_references():
    """Timed affine results match statsmodels and cross-checked Dynamax."""
    reference = multivariate_reference
    posterior = smcx.kalman_filter(
        jnp.asarray(reference.INITIAL_MEAN),
        jnp.asarray(reference.INITIAL_COVARIANCE),
        jnp.asarray(reference.TRANSITION_MATRIX),
        jnp.asarray(reference.TRANSITION_COVARIANCE),
        jnp.asarray(reference.OBSERVATION_MATRIX),
        jnp.asarray(reference.OBSERVATION_COVARIANCE),
        jnp.asarray(reference.EMISSIONS),
        transition_bias=jnp.asarray(reference.TRANSITION_BIAS),
        observation_bias=jnp.asarray(reference.OBSERVATION_BIAS),
        transition_input_matrix=jnp.asarray(reference.TRANSITION_INPUT_MATRIX),
        observation_input_matrix=jnp.asarray(
            reference.OBSERVATION_INPUT_MATRIX
        ),
        inputs=jnp.asarray(reference.INPUTS),
    )

    _assert_roundoff_close(posterior.predicted_means, reference.PREDICTED_MEANS)
    _assert_roundoff_close(
        posterior.predicted_covariances,
        reference.PREDICTED_COVARIANCES,
    )
    _assert_roundoff_close(posterior.filtered_means, reference.FILTERED_MEANS)
    _assert_roundoff_close(
        posterior.filtered_covariances,
        reference.FILTERED_COVARIANCES,
    )
    _assert_roundoff_close(
        posterior.log_evidence_increments,
        reference.LOG_EVIDENCE_INCREMENTS,
    )
    _assert_roundoff_close(
        posterior.marginal_loglik,
        reference.MARGINAL_LOG_LIKELIHOOD,
    )

    smoothed = smcx.rts_smoother(
        posterior,
        jnp.asarray(reference.TRANSITION_MATRIX),
    )
    _assert_roundoff_close(smoothed.smoothed_means, reference.SMOOTHED_MEANS)
    _assert_roundoff_close(
        smoothed.smoothed_covariances,
        reference.SMOOTHED_COVARIANCES,
    )


def test_rts_smoother_compiled_matches_eager_and_supports_one_step():
    """The independent backward stage compiles and handles an empty scan."""
    filtered = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        jnp.array([[0.9]]),
        jnp.array([[0.25]]),
        jnp.array([[1.0]]),
        jnp.array([[0.5]]),
        jnp.array([[0.2]]),
    )

    eager = smcx.rts_smoother(filtered, jnp.array([[0.9]]))
    compiled = jax.jit(smcx.rts_smoother)(
        filtered,
        jnp.array([[0.9]]),
    )

    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(compiled_value, eager_value)
    np.testing.assert_array_equal(
        eager.smoothed_means,
        filtered.filtered_means,
    )
    np.testing.assert_array_equal(
        eager.smoothed_covariances,
        filtered.filtered_covariances,
    )


def test_rts_smoother_rejects_misaligned_transition_history():
    """The public smoother validates a researcher's supplied operators."""
    filtered = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        jnp.array([[0.9]]),
        jnp.array([[0.25]]),
        jnp.array([[1.0]]),
        jnp.array([[0.5]]),
        jnp.zeros((3, 1)),
    )

    with pytest.raises(ValueError, match="transition_matrix"):
        smcx.rts_smoother(filtered, jnp.ones((3, 1, 1)))
