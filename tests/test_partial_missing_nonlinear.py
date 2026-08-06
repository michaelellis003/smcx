# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Partially missing rows across the nonlinear family (#433 slice 2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx

MU0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y_PARTIAL = (
    jnp
    .asarray([
        [0.3, -0.1],
        [0.6, 0.2],
        [-0.4, 0.9],
        [0.1, 0.05],
    ])
    .at[1, 0]
    .set(jnp.nan)
    .at[3, 1]
    .set(jnp.nan)
)


def _linear_transition(state):
    return A @ state


def _linear_observation(state):
    return H @ state


def _run_ekf(emissions):
    return smcx.extended_kalman_filter(
        MU0,
        P0,
        _linear_transition,
        lambda state: A,
        Q,
        _linear_observation,
        lambda state: H,
        R,
        emissions,
    )


def _run_ukf(emissions):
    return smcx.unscented_kalman_filter(
        MU0,
        P0,
        _linear_transition,
        Q,
        _linear_observation,
        R,
        emissions,
    )


def _run_gh(emissions):
    return smcx.gaussian_filter(
        MU0,
        P0,
        _linear_transition,
        Q,
        _linear_observation,
        R,
        emissions,
        method=smcx.gauss_hermite(order=3),
    )


_RUNNERS = {
    "ekf": _run_ekf,
    "ukf": _run_ukf,
    "gauss_hermite": _run_gh,
}


def _rtol(array):
    return 1e-9 if array.dtype == jnp.float64 else 5e-4


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_linear_reduction_matches_the_masked_kalman_filter(name):
    """On linear callbacks every method reproduces the masked update."""
    approximate = _RUNNERS[name](Y_PARTIAL)
    exact = smcx.kalman_filter(MU0, P0, A, Q, H, R, Y_PARTIAL)
    rtol = _rtol(approximate.filtered_means)
    np.testing.assert_allclose(
        np.asarray(approximate.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(approximate.filtered_covariances),
        np.asarray(exact.filtered_covariances),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(approximate.log_evidence_increments),
        np.asarray(exact.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_all_nan_rows_keep_the_identity_contract(name):
    """The entirely missing row still stores the prediction exactly."""
    emissions = Y_PARTIAL.at[2].set(jnp.nan)
    posterior = _RUNNERS[name](emissions)
    np.testing.assert_array_equal(
        np.asarray(posterior.filtered_means[2]),
        np.asarray(posterior.predicted_means[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.log_evidence_increments[2]), 0.0
    )


def test_nonlinear_callbacks_run_finite_with_partial_rows():
    """A genuinely nonlinear model filters partial rows finitely."""

    def transition(state):
        return jnp.array([
            0.9 * state[0] + 0.1 * jnp.sin(state[1]),
            0.8 * state[1],
        ])

    def observation(state):
        return jnp.array([state[0] + 0.05 * state[1] ** 2, state[1]])

    posterior = smcx.unscented_kalman_filter(
        MU0, P0, transition, Q, observation, R, Y_PARTIAL
    )
    assert np.all(np.isfinite(np.asarray(posterior.filtered_means)))
    assert np.all(np.isfinite(np.asarray(posterior.log_evidence_increments)))


def test_gradients_stay_finite_through_partial_rows():
    """UKF gradients survive masked components (double-where)."""

    def loss(transition_matrix):
        posterior = smcx.unscented_kalman_filter(
            MU0,
            P0,
            lambda state: transition_matrix @ state,
            Q,
            _linear_observation,
            R,
            Y_PARTIAL,
        )
        return posterior.marginal_loglik

    gradient = jax.grad(loss)(A)
    assert np.all(np.isfinite(np.asarray(gradient)))


@pytest.mark.parametrize("name", sorted(_RUNNERS))
def test_infinite_entries_are_still_rejected(name):
    """Infinities remain never-meaningful across the family."""
    with pytest.raises(ValueError, match="finite"):
        _RUNNERS[name](Y_PARTIAL.at[0, 0].set(jnp.inf))
