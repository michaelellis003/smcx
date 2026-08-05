# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""kalman_forecast_sample gates: coherence with the closed forms (#415)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

MU0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y = jnp.asarray([
    [0.3, -0.1],
    [0.6, 0.2],
    [-0.4, 0.9],
    [0.1, 0.05],
])
B_TRANS = jnp.asarray([0.1, -0.2])
B_OBS = jnp.asarray([0.05, 0.15])
G_TRANS = jnp.asarray([[0.4], [0.7]])
G_OBS = jnp.asarray([[0.2], [-0.3]])

MODEL = smcx.LinearGaussianModel(
    initial_mean=MU0,
    initial_covariance=P0,
    transition_matrix=A,
    transition_covariance=Q,
    observation_matrix=H,
    observation_covariance=R,
)

FILTERED = smcx.kalman_filter(MODEL, Y)
NUM_STEPS = 3
NUM_DRAWS = 20_000
PATHS = smcx.kalman_forecast_sample(
    jr.key(7), FILTERED, MODEL, num_steps=NUM_STEPS, num_draws=NUM_DRAWS
)
CLOSED = smcx.kalman_forecast(FILTERED, MODEL, num_steps=NUM_STEPS)


def test_path_shapes():
    """Draw-major state and emission paths with the documented shapes."""
    assert PATHS.state_paths.shape == (NUM_DRAWS, NUM_STEPS, 2)
    assert PATHS.emission_paths.shape == (NUM_DRAWS, NUM_STEPS, 2)


def test_marginal_moments_match_the_closed_form_within_se():
    """Per-horizon path marginals reproduce the #381 closed forms."""
    states = np.asarray(PATHS.state_paths, dtype=np.float64)
    emissions = np.asarray(PATHS.emission_paths, dtype=np.float64)
    for k in range(NUM_STEPS):
        state_mean = np.asarray(CLOSED.state_means[k], dtype=np.float64)
        state_cov = np.asarray(CLOSED.state_covariances[k], dtype=np.float64)
        se_mean = np.sqrt(np.diag(state_cov) / NUM_DRAWS)
        np.testing.assert_array_less(
            np.abs(states[:, k].mean(axis=0) - state_mean), 6.0 * se_mean
        )
        sample_cov = np.cov(states[:, k].T)
        se_var = np.diag(state_cov) * np.sqrt(2.0 / NUM_DRAWS)
        np.testing.assert_array_less(
            np.abs(np.diag(sample_cov) - np.diag(state_cov)), 6.0 * se_var
        )
        obs_mean = np.asarray(CLOSED.observation_means[k], dtype=np.float64)
        obs_cov = np.asarray(
            CLOSED.observation_covariances[k], dtype=np.float64
        )
        np.testing.assert_array_less(
            np.abs(emissions[:, k].mean(axis=0) - obs_mean),
            6.0 * np.sqrt(np.diag(obs_cov) / NUM_DRAWS),
        )


def test_cross_horizon_covariance_matches_the_closed_form():
    """Empirical Cov(x_{T+1}, x_{T+2}) equals G R(1) within SE bands."""
    states = np.asarray(PATHS.state_paths, dtype=np.float64)
    first = states[:, 0] - states[:, 0].mean(axis=0)
    second = states[:, 1] - states[:, 1].mean(axis=0)
    empirical = first.T @ second / (NUM_DRAWS - 1)
    expected = np.asarray(A, dtype=np.float64) @ np.asarray(
        CLOSED.state_covariances[0], dtype=np.float64
    )
    r1 = np.diag(np.asarray(CLOSED.state_covariances[0], dtype=np.float64))
    r2 = np.diag(np.asarray(CLOSED.state_covariances[1], dtype=np.float64))
    se = np.sqrt(np.outer(r2, r1) / NUM_DRAWS)
    np.testing.assert_array_less(np.abs(empirical.T - expected), 8.0 * se)


def test_record_path_matches_array_path_bitwise():
    """Record and loose-array path draws are identical at a fixed key."""
    from_record = smcx.kalman_forecast_sample(
        jr.key(3), FILTERED, MODEL, num_steps=2, num_draws=16
    )
    from_arrays = smcx.kalman_forecast_sample(
        jr.key(3), FILTERED, A, Q, H, R, num_steps=2, num_draws=16
    )
    for left, right in zip(
        jax.tree.leaves(from_record), jax.tree.leaves(from_arrays), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_biases_shift_the_paths_by_the_closed_form_means():
    """Same key: bias-model paths differ by exactly the mean shift."""
    future_inputs = jnp.asarray([[0.5], [-1.0]])
    base = smcx.kalman_forecast_sample(
        jr.key(11), FILTERED, A, Q, H, R, num_steps=2, num_draws=32
    )
    shifted = smcx.kalman_forecast_sample(
        jr.key(11),
        FILTERED,
        A,
        Q,
        H,
        R,
        num_steps=2,
        num_draws=32,
        transition_bias=B_TRANS,
        observation_bias=B_OBS,
        transition_input_matrix=G_TRANS,
        observation_input_matrix=G_OBS,
        future_inputs=future_inputs,
    )
    closed_base = smcx.kalman_forecast(FILTERED, A, Q, H, R, num_steps=2)
    closed_shift = smcx.kalman_forecast(
        FILTERED,
        A,
        Q,
        H,
        R,
        num_steps=2,
        transition_bias=B_TRANS,
        observation_bias=B_OBS,
        transition_input_matrix=G_TRANS,
        observation_input_matrix=G_OBS,
        future_inputs=future_inputs,
    )
    dtype = np.asarray(base.state_paths).dtype
    rtol = 1e-10 if dtype == np.float64 else 1e-4
    np.testing.assert_allclose(
        np.asarray(shifted.state_paths - base.state_paths),
        np.broadcast_to(
            np.asarray(closed_shift.state_means - closed_base.state_means),
            base.state_paths.shape,
        ),
        rtol=rtol,
        atol=1e-12 if dtype == np.float64 else 1e-5,
    )


def _sample_with(num_steps=1, num_draws=8):
    return smcx.kalman_forecast_sample(
        jr.key(0), FILTERED, MODEL, num_steps=num_steps, num_draws=num_draws
    )


def test_count_boundary_matrix():
    """Both counts share the positive-integer contract."""
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _sample_with(num_steps=bad)
        with pytest.raises(ValueError, match="num_draws"):
            _sample_with(num_draws=bad)


def test_record_with_loose_model_array_is_rejected():
    """The shared resolver names this function in its error."""
    with pytest.raises(ValueError, match="kalman_forecast_sample"):
        smcx.kalman_forecast_sample(
            jr.key(0),
            FILTERED,
            MODEL,
            transition_covariance=Q,
            num_steps=1,
            num_draws=8,
        )
