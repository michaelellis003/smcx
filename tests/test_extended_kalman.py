# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for extended Kalman filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
import smcx.kalman as kalman_module
from tests import _extended_kalman_reference as nonlinear_reference


def _assert_posterior_close(actual, expected, *, ulps=64):
    """Compare well-conditioned filter fields within an f32/f64 budget."""
    for actual_field, expected_field in zip(actual, expected, strict=True):
        actual_array = np.asarray(actual_field)
        expected_array = np.asarray(expected_field, dtype=actual_array.dtype)
        scale = max(1.0, float(np.max(np.abs(expected_array))))
        atol = ulps * np.finfo(actual_array.dtype).eps * scale
        np.testing.assert_allclose(
            actual_array,
            expected_array,
            rtol=0.0,
            atol=atol,
        )


def _nonlinear_transition_mean(state):
    return jnp.stack((
        0.82 * state[0] + 0.18 * state[1] + 0.05 * jnp.sin(state[0]),
        -0.12 * state[0] + 0.90 * state[1] + 0.04 * state[0] * state[1],
    ))


def _nonlinear_transition_jacobian(state):
    return jnp.array([
        [0.82 + 0.05 * jnp.cos(state[0]), 0.18],
        [-0.12 + 0.04 * state[1], 0.90 + 0.04 * state[0]],
    ])


def _nonlinear_observation_mean(state):
    return jnp.stack((
        state[0] + 0.10 * state[1] ** 2,
        0.65 * state[1] + 0.12 * jnp.sin(state[0]),
    ))


def _nonlinear_observation_jacobian(state):
    return jnp.array([
        [1.0, 0.20 * state[1]],
        [0.12 * jnp.cos(state[0]), 0.65],
    ])


def test_extended_kalman_reduces_to_linear_filter():
    """Linear mean callbacks reproduce every exact-filter field."""
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
    observation_covariance = jnp.array([[[0.3]], [[0.2]], [[0.4]], [[0.25]]])
    emissions = jnp.array([[0.1], [-0.2], [0.3], [0.05]])

    def transition_mean(state):
        return transition_matrix @ state + transition_bias

    def transition_jacobian(_state):
        return transition_matrix

    def observation_mean(state):
        return observation_matrix @ state + observation_bias

    def observation_jacobian(_state):
        return observation_matrix

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
    extended = smcx.extended_kalman_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_jacobian,
        transition_covariance,
        observation_mean,
        observation_jacobian,
        observation_covariance,
        emissions,
    )

    _assert_posterior_close(extended, exact)


def test_extended_kalman_input_callbacks_match_linear_controls():
    """Input-aware callbacks use the destination-time package convention."""
    initial_mean = jnp.array([0.1, -0.2])
    initial_covariance = jnp.array([[0.6, 0.04], [0.04, 0.45]])
    transition_matrix = jnp.array([[0.9, 0.05], [-0.1, 0.8]])
    transition_bias = jnp.array([0.03, -0.02])
    transition_input_matrix = jnp.array([[0.4, -0.2], [0.1, 0.3]])
    transition_covariance = jnp.array([[0.07, 0.01], [0.01, 0.05]])
    observation_matrix = jnp.array([[1.0, 0.25]])
    observation_bias = jnp.array([-0.05])
    observation_input_matrix = jnp.array([[0.2, -0.1]])
    observation_covariance = jnp.array([[0.25]])
    emissions = jnp.array([[20.0], [-0.1], [0.35], [0.2]])
    inputs = jnp.array([[100.0, -50.0], [0.2, 0.3], [-0.4, 0.1], [0.5, -0.2]])

    def transition_mean(state, input_t):
        return (
            transition_matrix @ state
            + transition_bias
            + transition_input_matrix @ input_t
        )

    def transition_jacobian(_state, _input_t):
        return transition_matrix

    def observation_mean(state, input_t):
        return (
            observation_matrix @ state
            + observation_bias
            + observation_input_matrix @ input_t
        )

    def observation_jacobian(_state, _input_t):
        return observation_matrix

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
    extended = smcx.extended_kalman_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_jacobian,
        transition_covariance,
        observation_mean,
        observation_jacobian,
        observation_covariance,
        emissions,
        inputs=inputs,
    )

    _assert_posterior_close(extended, exact)


def test_extended_kalman_matches_independent_nonlinear_reference():
    """Every posterior field matches Stone Soup's Joseph-form EKF."""
    reference = nonlinear_reference

    posterior = smcx.extended_kalman_filter(
        jnp.asarray(reference.INITIAL_MEAN),
        jnp.asarray(reference.INITIAL_COVARIANCE),
        _nonlinear_transition_mean,
        _nonlinear_transition_jacobian,
        jnp.asarray(reference.TRANSITION_COVARIANCE),
        _nonlinear_observation_mean,
        _nonlinear_observation_jacobian,
        jnp.asarray(reference.OBSERVATION_COVARIANCE),
        jnp.asarray(reference.EMISSIONS),
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
    _assert_posterior_close(posterior, expected_fields, ulps=256)


def test_extended_kalman_rejects_non_array_callback_output():
    """Callback structural failures use the public validation exception."""

    def transition_mean(_state):
        return [0.0]

    def transition_jacobian(_state):
        return jnp.eye(1)

    def observation_mean(state):
        return state

    def observation_jacobian(_state):
        return jnp.eye(1)

    with pytest.raises(
        ValueError,
        match="transition_mean_fn output must return a JAX array",
    ):
        smcx.extended_kalman_filter(
            jnp.zeros(1),
            jnp.eye(1),
            transition_mean,
            transition_jacobian,
            jnp.eye(1),
            observation_mean,
            observation_jacobian,
            jnp.eye(1),
            jnp.zeros((2, 1)),
        )


def test_explicit_and_autodiff_jacobians_match_under_compilation():
    """Caller-owned autodiff and analytic Jacobians are interchangeable."""
    reference = nonlinear_reference
    arrays = (
        jnp.asarray(reference.INITIAL_MEAN),
        jnp.asarray(reference.INITIAL_COVARIANCE),
        jnp.asarray(reference.TRANSITION_COVARIANCE),
        jnp.asarray(reference.OBSERVATION_COVARIANCE),
        jnp.asarray(reference.EMISSIONS),
    )
    analytic = smcx.extended_kalman_filter(
        arrays[0],
        arrays[1],
        _nonlinear_transition_mean,
        _nonlinear_transition_jacobian,
        arrays[2],
        _nonlinear_observation_mean,
        _nonlinear_observation_jacobian,
        arrays[3],
        arrays[4],
    )
    autodiff_transition = jax.jacfwd(_nonlinear_transition_mean)
    autodiff_observation = jax.jacfwd(_nonlinear_observation_mean)

    def run(
        initial_mean,
        initial_covariance,
        transition_covariance,
        observation_covariance,
        emissions,
    ):
        return smcx.extended_kalman_filter(
            initial_mean,
            initial_covariance,
            _nonlinear_transition_mean,
            autodiff_transition,
            transition_covariance,
            _nonlinear_observation_mean,
            autodiff_observation,
            observation_covariance,
            emissions,
        )

    autodiff = run(*arrays)
    compiled = jax.jit(run)(*arrays)

    _assert_posterior_close(autodiff, analytic, ulps=256)
    _assert_posterior_close(compiled, autodiff)


def test_one_step_rank_one_inputs_and_timed_covariances():
    """A one-step model conditions the prior and canonicalizes scalar inputs."""
    inputs = jnp.array([0.5])

    def transition_mean(state, input_t):
        return state + input_t[0]

    def transition_jacobian(_state, _input_t):
        return jnp.eye(1)

    def observation_mean(state, input_t):
        return state + input_t[0]

    def observation_jacobian(_state, _input_t):
        return jnp.eye(1)

    extended = smcx.extended_kalman_filter(
        jnp.zeros(1),
        jnp.eye(1),
        transition_mean,
        transition_jacobian,
        jnp.empty((0, 1, 1)),
        observation_mean,
        observation_jacobian,
        jnp.ones((1, 1, 1)),
        jnp.array([[0.25]]),
        inputs=inputs,
    )
    exact = smcx.kalman_filter(
        jnp.zeros(1),
        jnp.eye(1),
        jnp.eye(1),
        jnp.empty((0, 1, 1)),
        jnp.eye(1),
        jnp.ones((1, 1, 1)),
        jnp.array([[0.25]]),
        observation_input_matrix=jnp.eye(1),
        inputs=inputs,
    )

    _assert_posterior_close(extended, exact)


def test_uncompiled_extended_step_matches_public_scan():
    """The pure EKF step agrees with the second public filtering step."""
    reference = nonlinear_reference
    initial_mean = jnp.asarray(reference.INITIAL_MEAN)
    initial_covariance = jnp.asarray(reference.INITIAL_COVARIANCE)
    transition_covariance = jnp.asarray(reference.TRANSITION_COVARIANCE)
    observation_covariance = jnp.asarray(reference.OBSERVATION_COVARIANCE)
    emissions = jnp.asarray(reference.EMISSIONS[:2])
    first = smcx.extended_kalman_filter(
        initial_mean,
        initial_covariance,
        _nonlinear_transition_mean,
        _nonlinear_transition_jacobian,
        transition_covariance,
        _nonlinear_observation_mean,
        _nonlinear_observation_jacobian,
        observation_covariance,
        emissions[:1],
    )
    full = smcx.extended_kalman_filter(
        initial_mean,
        initial_covariance,
        _nonlinear_transition_mean,
        _nonlinear_transition_jacobian,
        transition_covariance,
        _nonlinear_observation_mean,
        _nonlinear_observation_jacobian,
        observation_covariance,
        emissions,
    )
    evidence = jnp.asarray(first.marginal_loglik)
    state = kalman_module._FilterState(
        first.filtered_means[0],
        first.filtered_covariances[0],
        evidence,
        jnp.zeros_like(evidence),
    )
    next_state, output = kalman_module._extended_filter_step(
        state,
        kalman_module._ExtendedFilterStepInput(
            emissions[1],
            transition_covariance,
            observation_covariance,
        ),
        _nonlinear_transition_mean,
        _nonlinear_transition_jacobian,
        _nonlinear_observation_mean,
        _nonlinear_observation_jacobian,
    )

    expected_output = tuple(field[1] for field in full[1:])
    _assert_posterior_close(output, expected_output, ulps=256)
    _assert_posterior_close(
        (next_state.marginal_loglik + next_state.log_evidence_compensation,),
        (full.marginal_loglik,),
        ulps=256,
    )


def _valid_extended_model():
    def transition_mean(state):
        return state

    def transition_jacobian(_state):
        return jnp.eye(1)

    def observation_mean(state):
        return state

    def observation_jacobian(_state):
        return jnp.eye(1)

    return {
        "initial_mean": jnp.zeros(1),
        "initial_covariance": jnp.eye(1),
        "transition_mean_fn": transition_mean,
        "transition_jacobian_fn": transition_jacobian,
        "transition_covariance": jnp.eye(1),
        "observation_mean_fn": observation_mean,
        "observation_jacobian_fn": observation_jacobian,
        "observation_covariance": jnp.eye(1),
        "emissions": jnp.zeros((2, 1)),
    }


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("initial_mean", jnp.zeros((1, 1)), "initial_mean"),
        ("initial_covariance", jnp.eye(2), "initial_covariance"),
        ("transition_covariance", jnp.ones((2, 1, 1)), "transition_covariance"),
        (
            "observation_covariance",
            jnp.ones((3, 1, 1)),
            "observation_covariance",
        ),
        ("emissions", jnp.empty((0, 1)), "emissions"),
        ("inputs", jnp.zeros((3, 1)), "inputs"),
    ],
)
def test_extended_kalman_rejects_misaligned_arrays(argument, value, message):
    """Malformed dense models fail at the public Python boundary."""
    model = _valid_extended_model()
    model[argument] = value

    with pytest.raises(ValueError, match=message):
        smcx.extended_kalman_filter(**model)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_extended_kalman_rejects_low_precision(dtype):
    """Unsupported Cholesky dtypes fail cleanly at the public boundary."""
    model = _valid_extended_model()
    for name, value in model.items():
        if isinstance(value, jax.Array):
            model[name] = value.astype(dtype)

    with pytest.raises(ValueError, match="float32 or float64"):
        smcx.extended_kalman_filter(**model)


@pytest.mark.parametrize(
    ("callback", "replacement", "message"),
    [
        (
            "transition_mean_fn",
            lambda state: jnp.zeros(2, dtype=state.dtype),
            "transition_mean_fn output",
        ),
        (
            "transition_jacobian_fn",
            lambda state: jnp.zeros((1, 2), dtype=state.dtype),
            "transition_jacobian_fn output",
        ),
        (
            "observation_mean_fn",
            lambda state: jnp.zeros(2, dtype=state.dtype),
            "observation_mean_fn output",
        ),
        (
            "observation_jacobian_fn",
            lambda state: jnp.zeros((2, 1), dtype=state.dtype),
            "observation_jacobian_fn output",
        ),
        (
            "observation_mean_fn",
            lambda state: state.astype(jnp.float16),
            "float32 or float64",
        ),
    ],
)
def test_extended_kalman_rejects_malformed_callback_output(
    callback,
    replacement,
    message,
):
    """Mean and Jacobian callbacks preserve shape and dtype contracts."""
    model = _valid_extended_model()
    model[callback] = replacement

    with pytest.raises(ValueError, match=message):
        smcx.extended_kalman_filter(**model)
