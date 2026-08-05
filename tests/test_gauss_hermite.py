# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""gauss_hermite strategy gates: quadrature accuracy and bounds (#382)."""

import jax.numpy as jnp
import numpy as np
import pytest

import smcx

M0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y = jnp.asarray([[0.3, -0.1], [0.6, 0.2], [-0.4, 0.9]])


def _linear_run(method):
    return smcx.gaussian_filter(
        M0,
        P0,
        lambda state: A @ state,
        Q,
        lambda state: H @ state,
        R,
        Y,
        method=method,
    )


def test_gauss_hermite_is_exact_on_a_linear_model():
    """Any order integrates linear models exactly."""
    exact = smcx.kalman_filter(M0, P0, A, Q, H, R, Y)
    via_quadrature = _linear_run(smcx.gauss_hermite(order=3))
    eps = float(jnp.finfo(via_quadrature.filtered_means.dtype).eps)
    np.testing.assert_allclose(
        np.asarray(via_quadrature.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=1e3 * eps,
        atol=1e2 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(via_quadrature.marginal_loglik),
        np.asarray(exact.marginal_loglik),
        rtol=1e3 * eps,
    )


def test_order_three_matches_the_unscented_oracle_in_one_dimension():
    """In 1-d, GH(3) equals the unscented rule at alpha=1, kappa=2."""
    m0 = jnp.asarray([0.2])
    p0 = jnp.asarray([[0.8]])
    q = jnp.asarray([[0.2]])
    r = jnp.asarray([[0.4]])
    emissions = jnp.asarray([[0.4], [-0.3], [0.7]])

    def run(method):
        return smcx.gaussian_filter(
            m0,
            p0,
            lambda state: jnp.sin(state),
            q,
            lambda state: state + 0.1 * state**2,
            r,
            emissions,
            method=method,
        )

    via_quadrature = run(smcx.gauss_hermite(order=3))
    via_unscented = run(smcx.unscented(alpha=1.0, beta=0.0, kappa=2.0))
    eps = float(jnp.finfo(via_quadrature.filtered_means.dtype).eps)
    np.testing.assert_allclose(
        np.asarray(via_quadrature.filtered_means),
        np.asarray(via_unscented.filtered_means),
        rtol=1e4 * eps,
        atol=1e3 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(via_quadrature.log_evidence_increments),
        np.asarray(via_unscented.log_evidence_increments),
        rtol=1e4 * eps,
        atol=1e3 * eps,
    )


def test_quartic_moment_needs_the_higher_order():
    """GH(3) integrates a quartic transition exactly; GH(2) cannot."""
    m0 = jnp.asarray([0.3])
    p0 = jnp.asarray([[0.5]])
    q = jnp.asarray([[0.1]])
    r = jnp.asarray([[0.4]])
    emissions = jnp.asarray([[0.2], [0.1]])

    def run(order):
        return smcx.gaussian_filter(
            m0,
            p0,
            lambda state: state**4,
            q,
            lambda state: state,
            r,
            emissions,
            method=smcx.gauss_hermite(order=order),
        )

    # E[x^4] under the time-0 filtered Gaussian, read from the stored
    # moments so the oracle conditions exactly as the filter does.
    filtered = run(3)
    mean_1 = float(filtered.filtered_means[0, 0])
    var_1 = float(filtered.filtered_covariances[0, 0, 0])
    analytic = mean_1**4 + 6.0 * mean_1**2 * var_1 + 3.0 * var_1**2
    predicted = float(run(3).predicted_means[1, 0])
    assert abs(predicted - analytic) < 1e-6 * max(1.0, abs(analytic))
    low_order = float(run(2).predicted_means[1, 0])
    assert abs(low_order - analytic) > 1e-3


def test_smoother_composes_and_reduces_on_the_linear_model():
    """The smoother path under gauss_hermite matches the linear RTS."""
    filtered = _linear_run(smcx.gauss_hermite(order=3))
    smoothed = smcx.gaussian_smoother(
        filtered,
        lambda state: A @ state,
        method=smcx.gauss_hermite(order=3),
    )
    exact = smcx.rts_smoother(smcx.kalman_filter(M0, P0, A, Q, H, R, Y), A)
    eps = float(jnp.finfo(smoothed.smoothed_means.dtype).eps)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_means),
        np.asarray(exact.smoothed_means),
        rtol=1e4 * eps,
        atol=1e3 * eps,
    )


def _build_with(order):
    return smcx.gauss_hermite(order=order)


def test_order_boundary_matrix():
    """The order shares the positive-integer contract."""
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="order"):
            _build_with(bad)


def test_point_count_ceiling_is_enforced():
    """An exponential point count is rejected with the count named."""
    state_dim = 7
    with pytest.raises(ValueError, match="sigma points"):
        smcx.gaussian_filter(
            jnp.zeros(state_dim),
            jnp.eye(state_dim),
            lambda state: state,
            0.1 * jnp.eye(state_dim),
            lambda state: state[:1],
            jnp.asarray([[0.4]]),
            jnp.asarray([[0.2]]),
            method=smcx.gauss_hermite(order=10),
        )


def test_wrong_method_error_names_the_factories():
    """The dispatch error lists every strategy factory."""
    with pytest.raises(ValueError, match="gauss_hermite"):
        _linear_run(object())


def test_input_aware_callbacks_flow_through_filter_and_smoother():
    """Input-aware mean callbacks receive each step's input."""
    inputs = jnp.asarray([[0.5], [-0.5], [0.25]])

    def transition_u(state, input_t):
        return A @ state + jnp.concatenate([input_t, input_t])

    def observation_u(state, input_t):
        return H @ state + jnp.concatenate([input_t, input_t])

    filtered = smcx.gaussian_filter(
        M0,
        P0,
        transition_u,
        Q,
        observation_u,
        R,
        Y,
        method=smcx.gauss_hermite(order=3),
        inputs=inputs,
    )
    exact = smcx.kalman_filter(
        M0,
        P0,
        A,
        Q,
        H,
        R,
        Y,
        transition_input_matrix=jnp.asarray([[1.0], [1.0]]),
        observation_input_matrix=jnp.asarray([[1.0], [1.0]]),
        inputs=inputs,
    )
    eps = float(jnp.finfo(filtered.filtered_means.dtype).eps)
    np.testing.assert_allclose(
        np.asarray(filtered.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=1e4 * eps,
        atol=1e3 * eps,
    )
    smoothed = smcx.gaussian_smoother(
        filtered,
        transition_u,
        method=smcx.gauss_hermite(order=3),
        inputs=inputs,
    )
    exact_smoothed = smcx.rts_smoother(exact, A)
    np.testing.assert_allclose(
        np.asarray(smoothed.smoothed_means),
        np.asarray(exact_smoothed.smoothed_means),
        rtol=1e4 * eps,
        atol=1e3 * eps,
    )


def test_bad_callback_shape_is_rejected():
    """A mis-shaped mean callback output raises the documented error."""
    with pytest.raises(ValueError, match="transition_mean_fn output"):
        smcx.gaussian_filter(
            M0,
            P0,
            lambda state: jnp.concatenate([state, state]),
            Q,
            lambda state: H @ state,
            R,
            Y,
            method=smcx.gauss_hermite(order=2),
        )


def test_smoother_input_requires_matching_history():
    """A one-step record leaves no transitions to linearize."""
    filtered = smcx.gaussian_filter(
        M0,
        P0,
        lambda state: A @ state,
        Q,
        lambda state: H @ state,
        R,
        Y[:1],
        method=smcx.gauss_hermite(order=2),
    )
    smoothed = smcx.gaussian_smoother(
        filtered,
        lambda state: A @ state,
        method=smcx.gauss_hermite(order=2),
    )
    np.testing.assert_array_equal(
        np.asarray(smoothed.smoothed_means),
        np.asarray(filtered.filtered_means),
    )
