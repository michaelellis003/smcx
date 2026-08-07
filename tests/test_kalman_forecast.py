# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""kalman_forecast gates: composition, oracles, and boundaries (#381)."""

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


def test_forecast_matches_the_nan_padded_filter_bitwise():
    """k-step state forecasts equal an all-NaN-padded run's predictions."""
    num_steps = 3
    padding = jnp.full((num_steps, Y.shape[1]), jnp.nan)
    padded = smcx.kalman_filter(MODEL, jnp.concatenate((Y, padding)))
    forecast = smcx.kalman_forecast(FILTERED, MODEL, num_steps=num_steps)
    np.testing.assert_array_equal(
        np.asarray(forecast.state_means),
        np.asarray(padded.predicted_means[Y.shape[0] :]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_covariances),
        np.asarray(padded.predicted_covariances[Y.shape[0] :]),
    )


def test_one_step_observation_forecast_matches_the_evidence_increment():
    """The k=1 forecast density at the held-out row is the increment."""
    held_out = smcx.kalman_filter(MODEL, Y[:-1])
    forecast = smcx.kalman_forecast(held_out, MODEL, num_steps=1)
    residual = np.asarray(Y[-1] - forecast.observation_means[0])
    covariance = np.asarray(forecast.observation_covariances[0])
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, residual)
    dim = residual.shape[0]
    log_density = -0.5 * (
        dim * np.log(2.0 * np.pi)
        + 2.0 * np.sum(np.log(np.diag(chol)))
        + whitened @ whitened
    )
    increment = np.asarray(
        smcx.kalman_filter(MODEL, Y).log_evidence_increments[-1]
    )
    rtol = 1e-9 if increment.dtype == np.float64 else 1e-5
    np.testing.assert_allclose(log_density, increment, rtol=rtol)


def test_record_path_matches_array_path_bitwise():
    """Record and loose-array forecasts are identical."""
    from_record = smcx.kalman_forecast(FILTERED, MODEL, num_steps=2)
    from_arrays = smcx.kalman_forecast(FILTERED, A, Q, H, R, num_steps=2)
    for left, right in zip(
        jax.tree.leaves(from_record), jax.tree.leaves(from_arrays), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_two_step_oracle_with_biases_and_inputs():
    """A hand-composed two-step forecast with biases and inputs agrees."""
    future_inputs = jnp.asarray([[0.5], [-1.0]])
    forecast = smcx.kalman_forecast(
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
    rtol = 1e-12 if FILTERED.filtered_means.dtype == jnp.float64 else 1e-5
    mean = np.asarray(FILTERED.filtered_means[-1])
    covariance = np.asarray(FILTERED.filtered_covariances[-1])
    a_np, q_np = np.asarray(A), np.asarray(Q)
    h_np, r_np = np.asarray(H), np.asarray(R)
    for step in range(2):
        input_t = np.asarray(future_inputs[step])
        mean = a_np @ mean + np.asarray(B_TRANS) + np.asarray(G_TRANS) @ input_t
        covariance = a_np @ covariance @ a_np.T + q_np
        covariance = 0.5 * (covariance + covariance.T)
        observation_mean = (
            h_np @ mean + np.asarray(B_OBS) + np.asarray(G_OBS) @ input_t
        )
        observation_covariance = h_np @ covariance @ h_np.T + r_np
        np.testing.assert_allclose(
            np.asarray(forecast.state_means[step]), mean, rtol=rtol
        )
        np.testing.assert_allclose(
            np.asarray(forecast.observation_means[step]),
            observation_mean,
            rtol=rtol,
        )
        np.testing.assert_allclose(
            np.asarray(forecast.observation_covariances[step]),
            0.5 * (observation_covariance + observation_covariance.T),
            rtol=rtol,
        )


def test_time_varying_future_operators():
    """Per-horizon operators with leading length num_steps are accepted."""
    a_future = jnp.stack([A, A * 0.9])
    forecast = smcx.kalman_forecast(FILTERED, a_future, Q, H, R, num_steps=2)
    static = smcx.kalman_forecast(FILTERED, A, Q, H, R, num_steps=2)
    np.testing.assert_array_equal(
        np.asarray(forecast.state_means[0]), np.asarray(static.state_means[0])
    )
    assert not np.array_equal(
        np.asarray(forecast.state_means[1]), np.asarray(static.state_means[1])
    )


def test_gradient_through_a_record_leaf():
    """Differentiating a model leaf matches the loose-array gradient."""

    def loss_record(a):
        model = MODEL._replace(transition_matrix=a)
        return jnp.sum(
            smcx.kalman_forecast(FILTERED, model, num_steps=2).state_means
        )

    def loss_arrays(a):
        return jnp.sum(
            smcx.kalman_forecast(FILTERED, a, Q, H, R, num_steps=2).state_means
        )

    np.testing.assert_array_equal(
        np.asarray(jax.grad(loss_record)(A)),
        np.asarray(jax.grad(loss_arrays)(A)),
    )


def _forecast_with_count(num_steps):
    return smcx.kalman_forecast(FILTERED, MODEL, num_steps=num_steps)


def test_num_steps_boundary_matrix():
    """Count validation matches the shared positive-integer contract."""
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _forecast_with_count(bad)


def test_record_with_loose_model_array_is_rejected():
    """The record and loose model arrays cannot be mixed."""
    with pytest.raises(ValueError, match="record only"):
        smcx.kalman_forecast(
            FILTERED, MODEL, transition_covariance=Q, num_steps=1
        )


def test_missing_loose_array_is_rejected():
    """Omitting a model array without a record raises at the boundary."""
    with pytest.raises(ValueError, match="LinearGaussianModel"):
        smcx.kalman_forecast(FILTERED, A, Q, H, num_steps=1)


def test_input_matrices_require_future_inputs():
    """Input matrices without future inputs get the documented error."""
    with pytest.raises(ValueError, match="future_inputs"):
        smcx.kalman_forecast(
            FILTERED,
            A,
            Q,
            H,
            R,
            num_steps=1,
            transition_input_matrix=G_TRANS,
        )


def test_future_inputs_length_must_match_num_steps():
    """A future-input history shorter than the horizon is rejected."""
    with pytest.raises(ValueError, match="future_inputs"):
        smcx.kalman_forecast(
            FILTERED,
            A,
            Q,
            H,
            R,
            num_steps=3,
            transition_input_matrix=G_TRANS,
            observation_input_matrix=G_OBS,
            future_inputs=jnp.asarray([[0.5], [-1.0]]),
        )


def test_time_varying_record_is_rejected_with_the_future_operator_hint():
    """A timed record cannot supply future operators (P2-2)."""
    num_timesteps = 4
    model = smcx.LinearGaussianModel(
        initial_mean=jnp.zeros(1),
        initial_covariance=jnp.eye(1),
        transition_matrix=jnp.eye(1),
        transition_covariance=0.1 * jnp.eye(1),
        observation_matrix=jnp.ones((num_timesteps, 1, 1)),
        observation_covariance=jnp.eye(1),
    )
    posterior = smcx.kalman_filter(model, jnp.zeros((num_timesteps, 1)))
    with pytest.raises(ValueError, match="time-varying"):
        smcx.kalman_forecast(posterior, model, num_steps=2)
    with pytest.raises(ValueError, match="time-varying"):
        smcx.kalman_forecast_sample(
            jr.key(0), posterior, model, num_steps=2, num_draws=3
        )
