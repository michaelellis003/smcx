# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact linear-Gaussian filtering and smoothing."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
import smcx.kalman as kalman_module
from tests import _kalman_reference as multivariate_reference
from tests._gaussian_smoothing_reference import dense_joint_marginals
from tests._kalman import kalman_1d
from tests._lgssm_reference import EXACT_LOG_LIKELIHOOD, REFERENCE_TIMES
from tests._lgssm_reference import FILTERED_MEANS as EXACT_FILTERED_MEANS
from tests._lgssm_reference import FILTERED_VARIANCES as EXACT_FILTERED_VARS


def _assert_roundoff_close(actual, expected):
    """Compare the small, well-conditioned reference within f32/f64 error."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    # The fixtures have either 50 stable scalar steps, five 2x2 steps with
    # covariance condition numbers below 2.7, or a 10x10 joint-precision
    # solve with condition number below 15. A 64*eps*scale forward-error
    # budget covers their reductions, triangular solves, and dense solve.
    scale = max(1.0, float(np.max(np.abs(expected_array))))
    atol = 64 * np.finfo(actual_array.dtype).eps * scale
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=atol,
    )


def test_gaussian_filters_accept_scalar_observation_sequences():
    """All Gaussian filters canonicalize one scalar observation per row."""
    mean = jnp.zeros(1)
    covariance = jnp.eye(1)
    emissions = jnp.array([0.1, -0.2])

    linear = smcx.kalman_filter(
        mean,
        covariance,
        covariance,
        covariance,
        covariance,
        covariance,
        emissions,
    )
    extended = smcx.extended_kalman_filter(
        mean,
        covariance,
        lambda state: state,
        lambda _state: covariance,
        covariance,
        lambda state: state,
        lambda _state: covariance,
        covariance,
        emissions,
    )
    unscented = smcx.unscented_kalman_filter(
        mean,
        covariance,
        lambda state: state,
        covariance,
        lambda state: state,
        covariance,
        emissions,
    )

    for posterior in (linear, extended, unscented):
        assert posterior.log_evidence_increments.shape == (2,)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "emissions",
            jnp.ones(2, dtype=jnp.int32),
            "emissions must have a floating dtype",
        ),
        (
            "emissions",
            jnp.ones((2, 1, 1)),
            r"shape \(T,\) or \(T, emission_dim\)",
        ),
        (
            "inputs",
            jnp.ones(2, dtype=jnp.int32),
            "inputs must have a floating dtype",
        ),
        (
            "inputs",
            jnp.empty((2, 0)),
            "input_dim >= 1",
        ),
    ],
)
def test_gaussian_data_validation_raises_plain_value_errors(
    field,
    value,
    message,
):
    """Gaussian shape and dtype violations use the public error type."""
    model = {
        "initial_mean": jnp.zeros(1),
        "initial_covariance": jnp.eye(1),
        "transition_matrix": jnp.eye(1),
        "transition_covariance": jnp.eye(1),
        "observation_matrix": jnp.eye(1),
        "observation_covariance": jnp.eye(1),
        "emissions": jnp.zeros((2, 1)),
    }
    model[field] = value

    with pytest.raises(ValueError, match=message):
        smcx.kalman_filter(**model)


def test_kalman_filter_rejects_numpy_emissions_with_value_error():
    """The public validator owns the non-JAX observation error."""
    covariance = jnp.eye(1)

    with pytest.raises(ValueError, match="emissions must be a JAX array"):
        smcx.kalman_filter(
            jnp.zeros(1),
            covariance,
            covariance,
            covariance,
            covariance,
            covariance,
            np.zeros((2, 1)),  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    "filter_name",
    ["kalman_filter", "extended_kalman_filter", "unscented_kalman_filter"],
)
def test_gaussian_filters_reject_non_jax_inputs_with_value_error(filter_name):
    """Every Gaussian input boundary owns the non-JAX structural error."""
    mean = jnp.zeros(1)
    covariance = jnp.eye(1)
    emissions = jnp.zeros(2)
    middle_arguments = {
        "kalman_filter": (covariance, covariance, covariance),
        "extended_kalman_filter": (
            lambda state, _input: state,
            lambda _state, _input: covariance,
            covariance,
            lambda state, _input: state,
            lambda _state, _input: covariance,
        ),
        "unscented_kalman_filter": (
            lambda state, _input: state,
            covariance,
            lambda state, _input: state,
        ),
    }[filter_name]

    with pytest.raises(ValueError, match="inputs must be a JAX array"):
        getattr(smcx, filter_name)(
            mean,
            covariance,
            *middle_arguments,
            covariance,
            emissions,
            inputs=[[0.0], [1.0]],
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


def test_kalman_filter_reduces_to_scalar_numpy_oracle(lgssm_params, lgssm_data):
    """Every scalar filtered moment follows the independent recurrence."""
    _, emissions = lgssm_data
    exact_loglik, exact_means, exact_variances = kalman_1d(
        np.asarray(emissions[:, 0]),
        a=0.9,
        q=0.25,
        r=1.0,
        m0=0.0,
        p0=1.0,
    )
    posterior = smcx.kalman_filter(
        lgssm_params["initial_mean"],
        lgssm_params["initial_cov"],
        lgssm_params["dynamics_weights"],
        lgssm_params["dynamics_cov"],
        lgssm_params["emissions_weights"],
        lgssm_params["emissions_cov"],
        emissions,
    )

    _assert_roundoff_close(posterior.marginal_loglik, exact_loglik)
    _assert_roundoff_close(posterior.filtered_means[:, 0], exact_means)
    _assert_roundoff_close(
        posterior.filtered_covariances[:, 0, 0],
        exact_variances,
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


def test_scan_steps_uncompiled_match_public_two_step_run():
    """Pure forward and backward steps agree with their public scans."""
    transition = jnp.array([[0.9]])
    transition_covariance = jnp.array([[0.25]])
    observation = jnp.array([[1.0]])
    observation_covariance = jnp.array([[0.5]])
    emissions = jnp.array([[0.2], [-0.1]])
    first = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        transition,
        transition_covariance,
        observation,
        observation_covariance,
        emissions[:1],
    )
    full = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        transition,
        transition_covariance,
        observation,
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
    next_state, output = kalman_module._filter_step(
        state,
        (
            emissions[1],
            transition,
            transition_covariance,
            jnp.zeros(1),
            observation,
            observation_covariance,
            jnp.zeros(1),
        ),
    )

    step_rtol = 32 * float(jnp.finfo(full.filtered_means.dtype).eps)
    np.testing.assert_allclose(
        output.filtered_mean, full.filtered_means[1], rtol=step_rtol
    )
    np.testing.assert_allclose(
        output.filtered_covariance,
        full.filtered_covariances[1],
        rtol=step_rtol,
    )
    np.testing.assert_allclose(
        next_state.marginal_loglik + next_state.log_evidence_compensation,
        full.marginal_loglik,
    )

    smoothed = smcx.rts_smoother(full, transition)
    _, direct = kalman_module._rts_step(
        kalman_module._SmootherState(
            smoothed.smoothed_means[1],
            smoothed.smoothed_covariances[1],
        ),
        (
            full.filtered_means[0],
            full.filtered_covariances[0],
            full.predicted_means[1],
            full.predicted_covariances[1],
            transition,
            None,
        ),
    )
    np.testing.assert_allclose(direct.mean, smoothed.smoothed_means[0])
    np.testing.assert_allclose(
        direct.covariance,
        smoothed.smoothed_covariances[0],
    )


def test_rts_step_uses_a_prepared_cross_covariance_directly():
    """The shared kernel does not reconstruct a producer-supplied D."""
    next_state = kalman_module._SmootherState(
        jnp.array([2.0]),
        jnp.array([[1.0]]),
    )

    _, state = kalman_module._rts_step(
        next_state,
        (
            jnp.array([0.0]),
            jnp.array([[2.0]]),
            jnp.array([0.0]),
            jnp.array([[4.0]]),
            jnp.array([[0.5]]),
            jnp.array([[1.5]]),
        ),
    )

    # D / Pbar = 3/8. Recomputing P A' would instead give a gain of 1/4.
    np.testing.assert_array_equal(state.mean, jnp.array([0.75]))
    np.testing.assert_array_equal(state.covariance, jnp.array([[1.953125]]))


def _valid_linear_model() -> dict[str, jax.Array]:
    """Return a valid two-dimensional linear model."""
    return {
        "initial_mean": jnp.zeros(2),
        "initial_covariance": jnp.eye(2),
        "transition_matrix": jnp.eye(2),
        "transition_covariance": jnp.eye(2),
        "observation_matrix": jnp.eye(2),
        "observation_covariance": jnp.eye(2),
        "emissions": jnp.zeros((2, 2)),
    }


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


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (
            "initial_covariance",
            jnp.nextafter(jnp.zeros((2, 2)), 1.0).at[1, 1].set(0.0),
            "initial_covariance must not contain nonzero subnormal values",
        ),
        (
            "transition_covariance",
            jnp.array([[1.0e20, 0.0], [1.0, 1.0]]),
            "transition_covariance must be symmetric",
        ),
        (
            "observation_covariance",
            jnp.array([[1.0, 0.0], [0.0, jnp.inf]]),
            "observation_covariance must contain only finite values",
        ),
        (
            "transition_covariance",
            jnp.array([
                [1.0, jnp.finfo(jnp.asarray(0.0).dtype).max],
                [-jnp.finfo(jnp.asarray(0.0).dtype).max, 1.0],
            ]),
            "transition_covariance must be symmetric",
        ),
    ],
)
def test_kalman_filter_rejects_invalid_covariance(argument, value, message):
    """Concrete covariance values obey the public Gaussian domain."""
    model = _valid_linear_model()
    model[argument] = value

    with pytest.raises(ValueError, match=message):
        smcx.kalman_filter(**model)


def test_kalman_filter_accepts_semidefinite_state_covariances():
    """Deterministic state components remain a supported linear model."""
    model = _valid_linear_model()
    model["initial_covariance"] = jnp.diag(jnp.array([1.0, 0.0]))
    model["transition_covariance"] = jnp.zeros((2, 2))

    posterior = smcx.kalman_filter(**model)

    assert jnp.all(jnp.isfinite(posterior.filtered_covariances))


def test_kalman_filter_checks_each_timed_covariance_scale():
    """A large valid slice cannot hide an indefinite small slice."""
    model = _valid_linear_model()
    model["emissions"] = jnp.zeros((3, 2))
    model["transition_covariance"] = jnp.array([
        [[1.0e20, 0.0], [0.0, 1.0e20]],
        [[1.0, 2.0], [2.0, 1.0]],
    ])

    with pytest.raises(
        ValueError,
        match="transition_covariance must be positive semidefinite",
    ):
        smcx.kalman_filter(**model)


def test_kalman_filter_skips_covariance_value_checks_when_traced():
    """JIT retains shape checks without concretizing covariance values."""
    model = _valid_linear_model()
    del model["initial_covariance"]

    @jax.jit
    def run(initial_covariance):
        return smcx.kalman_filter(
            **model,
            initial_covariance=initial_covariance,
        )

    posterior = run(jnp.diag(jnp.array([1.0, -0.1])))

    assert posterior.filtered_covariances.shape == (2, 2, 2)


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


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_kalman_filter_rejects_unsupported_low_precision(dtype):
    """Unsupported Cholesky dtypes fail cleanly at the public boundary."""
    with pytest.raises(ValueError, match="float32 or float64"):
        smcx.kalman_filter(
            jnp.zeros(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            jnp.zeros((1, 1), dtype=dtype),
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


def test_rts_smoother_matches_dense_joint_precision_oracle():
    """RTS marginals agree with a derivation-independent Gaussian solve."""
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
    smoothed = smcx.rts_smoother(
        posterior,
        jnp.asarray(reference.TRANSITION_MATRIX),
    )

    transition_offsets = reference.TRANSITION_BIAS + (
        reference.INPUTS[1:] @ reference.TRANSITION_INPUT_MATRIX.T
    )
    observation_offsets = reference.OBSERVATION_BIAS + (
        reference.INPUTS @ reference.OBSERVATION_INPUT_MATRIX.T
    )
    expected_means, expected_covariances = dense_joint_marginals(
        reference.INITIAL_MEAN,
        reference.INITIAL_COVARIANCE,
        reference.TRANSITION_MATRIX,
        reference.TRANSITION_COVARIANCE,
        transition_offsets,
        reference.OBSERVATION_MATRIX,
        reference.OBSERVATION_COVARIANCE,
        observation_offsets,
        reference.EMISSIONS,
    )

    # The 10x10 joint precision has condition number below 15. The existing
    # 64-eps roundoff budget covers both its dense solve and the RTS scan.
    _assert_roundoff_close(smoothed.smoothed_means, expected_means)
    _assert_roundoff_close(
        smoothed.smoothed_covariances,
        expected_covariances,
    )


def test_rts_smoother_compiled_matches_eager():
    """Compilation preserves a nonempty RTS backward scan."""
    filtered = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        jnp.array([[0.9]]),
        jnp.array([[0.25]]),
        jnp.array([[1.0]]),
        jnp.array([[0.5]]),
        jnp.array([[0.2], [-0.1], [0.4]]),
    )

    eager = smcx.rts_smoother(filtered, jnp.array([[0.9]]))
    compiled = jax.jit(smcx.rts_smoother)(
        filtered,
        jnp.array([[0.9]]),
    )

    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(compiled_value, eager_value)


def test_taylor_smoother_constant_jacobian_is_bitwise_rts():
    """A constant Taylor linearization runs the identical backward pass."""
    transition = jnp.array([[0.9]])
    filtered = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        transition,
        jnp.array([[0.25]]),
        jnp.array([[1.0]]),
        jnp.array([[0.5]]),
        jnp.array([[0.2], [-0.1], [0.4]]),
    )

    def unused_callback(*_args):
        raise AssertionError("Taylor smoothing must not call this callback")

    actual = smcx.gaussian_smoother(
        filtered,
        unused_callback,
        method=smcx.taylor_order1(
            lambda _state: transition,
            unused_callback,
        ),
    )
    expected = smcx.rts_smoother(filtered, transition)

    for actual_field, expected_field in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_field, expected_field)


def test_gaussian_smoothers_vmap_match_independent_runs():
    """Batching either public smoother preserves independent posteriors."""
    transition = jnp.array([[0.9]])
    emissions = (
        jnp.array([[0.2], [-0.1], [0.4]]),
        jnp.array([[-0.3], [0.5], [0.1]]),
    )
    filtered = tuple(
        smcx.kalman_filter(
            jnp.array([0.0]),
            jnp.array([[1.0]]),
            transition,
            jnp.array([[0.25]]),
            jnp.array([[1.0]]),
            jnp.array([[0.5]]),
            batch,
        )
        for batch in emissions
    )
    batched = jax.tree.map(
        lambda *fields: jnp.stack(fields),
        *filtered,
    )

    rts = jax.vmap(smcx.rts_smoother, in_axes=(0, None))(
        batched,
        transition,
    )

    def smooth(posterior):
        return smcx.gaussian_smoother(
            posterior,
            lambda state: transition @ state,
            method=smcx.taylor_order1(
                lambda _state: transition,
                lambda _state: jnp.ones((1, 1)),
            ),
        )

    taylor = jax.vmap(smooth)(batched)
    eps = float(jnp.finfo(rts.smoothed_means.dtype).eps)
    for actual in (rts, taylor):
        assert actual.smoothed_means.shape == (2, 3, 1)
        assert actual.smoothed_covariances.shape == (2, 3, 1, 1)
        for index, posterior in enumerate(filtered):
            expected = smcx.rts_smoother(posterior, transition)
            np.testing.assert_allclose(
                actual.smoothed_means[index],
                expected.smoothed_means,
                rtol=32 * eps,
                atol=32 * eps,
            )
            np.testing.assert_allclose(
                actual.smoothed_covariances[index],
                expected.smoothed_covariances,
                rtol=32 * eps,
                atol=32 * eps,
            )


def test_gaussian_smoother_gradients_match_scalar_recursion():
    """Both smoothers differentiate mean and covariance through the scan."""
    filtered = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0),
        jnp.array([[0.0], [0.1]]),
        jnp.array([[[1.0]], [[1.1]]]),
        jnp.array([[0.2], [0.7]]),
        jnp.array([[[0.8]], [[0.4]]]),
        jnp.zeros(2),
    )

    def rts_objective(coefficient):
        smoothed = smcx.rts_smoother(filtered, coefficient[None, None])
        return (
            smoothed.smoothed_means[0, 0]
            + 0.25 * smoothed.smoothed_covariances[0, 0, 0]
        )

    def taylor_objective(coefficient):
        smoothed = smcx.gaussian_smoother(
            filtered,
            lambda state: state,
            method=smcx.taylor_order1(
                lambda _state: coefficient[None, None],
                lambda _state: jnp.ones((1, 1)),
            ),
        )
        return (
            smoothed.smoothed_means[0, 0]
            + 0.25 * smoothed.smoothed_covariances[0, 0, 0]
        )

    coefficient = jnp.asarray(0.6)
    mean_derivative = 0.8 / 1.1 * (0.7 - 0.1)
    covariance_derivative = 2.0 * 0.8**2 * 0.6 * (0.4 - 1.1) / 1.1**2
    expected = jnp.asarray(
        mean_derivative + 0.25 * covariance_derivative,
        dtype=coefficient.dtype,
    )

    eps = float(jnp.finfo(coefficient.dtype).eps)
    for objective in (rts_objective, taylor_objective):
        gradient_fn = jax.grad(objective)
        gradients = (
            gradient_fn(coefficient),
            jax.jit(gradient_fn)(coefficient),
        )
        for actual in gradients:
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=32 * eps,
                atol=32 * eps,
            )


def test_gaussian_smoothers_one_step_are_filter_identity():
    """A one-step model has no backward transition or callback to apply."""
    filtered = smcx.kalman_filter(
        jnp.array([0.0]),
        jnp.array([[1.0]]),
        jnp.array([[0.9]]),
        jnp.array([[0.25]]),
        jnp.array([[1.0]]),
        jnp.array([[0.5]]),
        jnp.array([[0.2]]),
    )
    rts = smcx.rts_smoother(filtered, jnp.array([[0.9]]))

    def unused_callback(*_args):
        raise AssertionError("A one-step smoother must not call callbacks")

    taylor = smcx.gaussian_smoother(
        filtered,
        unused_callback,
        method=smcx.taylor_order1(unused_callback, unused_callback),
        inputs=jnp.array([0.5]),
    )
    for smoothed in (rts, taylor):
        np.testing.assert_array_equal(
            smoothed.smoothed_means,
            filtered.filtered_means,
        )
        np.testing.assert_array_equal(
            smoothed.smoothed_covariances,
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


def _valid_filter_posterior() -> smcx.GaussianFilterPosterior:
    """Return a valid two-time filter posterior."""
    means = jnp.zeros((2, 2))
    covariances = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    return smcx.GaussianFilterPosterior(
        jnp.asarray(0.0),
        means,
        covariances,
        means,
        covariances,
        jnp.zeros(2),
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("method", "Taylor-order-one strategy"),
        ("unscented", "Taylor-order-one strategy"),
        ("input_length", "leading dimension T=2"),
        ("input_dtype", "inputs must have a floating dtype"),
        ("jacobian_shape", r"must have shape \(2, 2\)"),
        ("jacobian_dtype", "must have dtype float32"),
        ("jacobian_type", "must return a JAX array"),
    ],
)
def test_gaussian_smoother_rejects_invalid_new_arguments(case, message):
    """The nonlinear smoother owns its strategy, input, and callback errors."""
    posterior = _valid_filter_posterior()

    def observation_jacobian(state):
        return jnp.eye(state.shape[0], dtype=state.dtype)

    method: object = smcx.taylor_order1(
        lambda state: jnp.eye(state.shape[0], dtype=state.dtype),
        observation_jacobian,
    )
    inputs = None
    if case == "method":
        method = "ekf"
    elif case == "unscented":
        method = smcx.unscented()
    elif case == "input_length":
        inputs = jnp.zeros(3)
    elif case == "input_dtype":
        inputs = jnp.zeros(2, dtype=jnp.int32)
    elif case == "jacobian_shape":
        method = smcx.taylor_order1(
            lambda _state: jnp.ones((1, 1)),
            observation_jacobian,
        )
    elif case == "jacobian_dtype":
        method = smcx.taylor_order1(
            lambda _state: jnp.eye(2, dtype=jnp.float16),
            observation_jacobian,
        )
    else:
        method = smcx.taylor_order1(
            lambda _state: np.eye(2),  # ty: ignore[invalid-argument-type]
            observation_jacobian,
        )

    with pytest.raises(ValueError, match=message):
        smcx.gaussian_smoother(
            posterior,
            lambda state: state,
            method=method,  # ty: ignore[invalid-argument-type]
            inputs=inputs,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "filtered_covariances",
            jnp.array([
                [[1.0, 0.0], [0.0, -0.1]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]),
            "filtered_covariances must be positive semidefinite",
        ),
        (
            "predicted_covariances",
            jnp.array([
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.5], [0.0, 1.0]],
            ]),
            "predicted_covariances\\[1:\\] must be symmetric",
        ),
        (
            "predicted_covariances",
            jnp.array([
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, jnp.nan]],
            ]),
            "predicted_covariances\\[1:\\] must contain only finite values",
        ),
        (
            "predicted_covariances",
            jnp.array([
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]),
            "predicted_covariances\\[1:\\] must be positive definite",
        ),
    ],
)
def test_rts_smoother_rejects_invalid_covariance(field, value, message):
    """Caller-constructed moments obey the smoother's factorization domain."""
    posterior = _valid_filter_posterior()._replace(**{field: value})

    with pytest.raises(ValueError, match=message):
        smcx.rts_smoother(posterior, jnp.eye(2))


def test_rts_smoother_accepts_semidefinite_unfactored_covariances():
    """Only positive-time predictions require strict definiteness."""
    posterior = _valid_filter_posterior()._replace(
        predicted_covariances=jnp.array([
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]),
        filtered_covariances=jnp.zeros((2, 2, 2)),
    )

    smoothed = smcx.rts_smoother(posterior, jnp.eye(2))

    assert jnp.all(jnp.isfinite(smoothed.smoothed_covariances))


def test_joseph_update_preserves_float32_covariance_psd():
    """The public update remains PSD under cancellation in subtractive form."""
    posterior = smcx.kalman_filter(
        jnp.zeros(2, dtype=jnp.float32),
        jnp.array(
            [[350_000.0, -1_145_000.0], [-1_145_000.0, 3_749_000.0]],
            dtype=jnp.float32,
        ),
        jnp.eye(2, dtype=jnp.float32),
        jnp.eye(2, dtype=jnp.float32),
        jnp.array(
            [[0.45, 0.91], [-0.79, 0.95]],
            dtype=jnp.float32,
        ),
        jnp.diag(jnp.array([0.1, 0.4], dtype=jnp.float32)),
        jnp.zeros((1, 2), dtype=jnp.float32),
    )

    covariance = np.asarray(posterior.filtered_covariances[0])
    # For this fixture, the subtractive P-KSK' update has minimum
    # eigenvalue below -0.65 on CPU and Metal, while Joseph gives >0.09.
    assert np.linalg.eigvalsh(covariance).min() >= 0.0


def test_float32_evidence_avoids_long_horizon_accumulation_drift():
    """Long, unequal increments do not accumulate naive-sum drift."""
    num_timesteps = 10_000
    dtype = jnp.float32
    posterior = smcx.kalman_filter(
        jnp.array([1_000.0], dtype=dtype),
        jnp.array([[1.0]], dtype=dtype),
        jnp.array([[1.0]], dtype=dtype),
        jnp.array([[0.01]], dtype=dtype),
        jnp.array([[1.0]], dtype=dtype),
        jnp.array([[0.1]], dtype=dtype),
        jnp.zeros((num_timesteps, 1), dtype=dtype),
    )

    accurate = math.fsum(
        np.asarray(posterior.log_evidence_increments, dtype=np.float64)
    )
    # Returning the compensated f32 pair costs at most one final rounding;
    # two ulps admits backend variation. Naive accumulation misses by >123.
    tolerance = 2 * abs(np.spacing(np.float32(accurate)))
    assert abs(float(posterior.marginal_loglik) - accurate) <= tolerance


class TestGaussianFilterStrategies:
    """gaussian_filter dispatches exactly to the named filters."""

    @staticmethod
    def _nonlinear_model():
        def transition_mean(state):
            return jnp.array([
                0.9 * state[0] + 0.1 * jnp.sin(state[1]),
                0.8 * state[1],
            ])

        def observation_mean(state):
            return jnp.array([state[0] + 0.05 * state[1] ** 2])

        arrays = dict(
            initial_mean=jnp.zeros(2),
            initial_covariance=jnp.eye(2),
            transition_covariance=0.1 * jnp.eye(2),
            observation_covariance=jnp.array([[0.3]]),
            emissions=jnp.array([[0.2], [-0.1], [0.4]]),
        )
        return transition_mean, observation_mean, arrays

    def test_taylor_order1_equals_extended_filter(self):
        transition_mean, observation_mean, arrays = self._nonlinear_model()
        transition_jacobian = jax.jacfwd(transition_mean)
        observation_jacobian = jax.jacfwd(observation_mean)

        via_strategy = smcx.gaussian_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            method=smcx.taylor_order1(
                transition_jacobian, observation_jacobian
            ),
        )
        direct = smcx.extended_kalman_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            transition_jacobian,
            arrays["transition_covariance"],
            observation_mean,
            observation_jacobian,
            arrays["observation_covariance"],
            arrays["emissions"],
        )

        for strategy_field, direct_field in zip(
            via_strategy, direct, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(strategy_field), np.asarray(direct_field)
            )

    def test_unscented_equals_unscented_filter(self):
        transition_mean, observation_mean, arrays = self._nonlinear_model()

        via_strategy = smcx.gaussian_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            method=smcx.unscented(0.9, 2.0, 0.5),
        )
        direct = smcx.unscented_kalman_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            alpha=0.9,
            beta=2.0,
            kappa=0.5,
        )

        for strategy_field, direct_field in zip(
            via_strategy, direct, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(strategy_field), np.asarray(direct_field)
            )

    def test_strategies_are_exchangeable_on_one_model(self):
        """The swap point works: same model, two rules, finite results."""
        transition_mean, observation_mean, arrays = self._nonlinear_model()
        taylor = smcx.gaussian_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            method=smcx.taylor_order1(
                jax.jacfwd(transition_mean), jax.jacfwd(observation_mean)
            ),
        )
        sigma = smcx.gaussian_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            method=smcx.unscented(),
        )

        assert np.all(np.isfinite(np.asarray(taylor.filtered_means)))
        assert np.all(np.isfinite(np.asarray(sigma.filtered_means)))

    def test_rejects_non_strategy_method(self):
        transition_mean, observation_mean, arrays = self._nonlinear_model()
        with pytest.raises(ValueError, match="linearization strategy"):
            smcx.gaussian_filter(
                arrays["initial_mean"],
                arrays["initial_covariance"],
                transition_mean,
                arrays["transition_covariance"],
                observation_mean,
                arrays["observation_covariance"],
                arrays["emissions"],
                method="ekf",  # ty: ignore[invalid-argument-type]
            )
