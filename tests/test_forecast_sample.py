# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""forecast_sample and param_forecast_sample gates (#414)."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

RHO = 0.9
Q_SD = 0.4
R_SD = 0.6
Y = jnp.asarray([[0.3], [0.7], [-0.2], [0.5], [0.1], [0.9], [0.4], [-0.1]])
NUM_PARTICLES = 4096


def _initial_sampler(key, n):
    return jr.normal(key, (n, 1))


def _transition_sampler(key, state):
    return RHO * state + Q_SD * jr.normal(key, state.shape)


def _emission_sampler(key, state):
    return state + R_SD * jr.normal(key, state.shape)


def _log_observation_fn(emission, state):
    z = (emission[0] - state[0]) / R_SD
    return -0.5 * z * z


POSTERIOR = smcx.bootstrap_filter(
    jr.key(0),
    _initial_sampler,
    _transition_sampler,
    _log_observation_fn,
    Y,
    NUM_PARTICLES,
)
NUM_STEPS = 3
NUM_DRAWS = 20_000
PATHS = smcx.forecast_sample(
    jr.key(31),
    POSTERIOR,
    _transition_sampler,
    _emission_sampler,
    num_steps=NUM_STEPS,
    num_draws=NUM_DRAWS,
)


def test_path_shapes():
    """Draw-major state and emission paths with the documented shapes."""
    assert PATHS.state_paths.shape == (NUM_DRAWS, NUM_STEPS, 1)
    assert PATHS.emission_paths.shape == (NUM_DRAWS, NUM_STEPS, 1)


def test_linear_gaussian_reduction_within_se():
    """Path marginals agree with kalman_forecast on the same model."""
    exact = smcx.kalman_filter(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[RHO]]),
        jnp.asarray([[Q_SD**2]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[R_SD**2]]),
        Y,
    )
    closed = smcx.kalman_forecast(
        exact,
        jnp.asarray([[RHO]]),
        jnp.asarray([[Q_SD**2]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([[R_SD**2]]),
        num_steps=NUM_STEPS,
    )
    states = np.asarray(PATHS.state_paths, dtype=np.float64)
    emissions = np.asarray(PATHS.emission_paths, dtype=np.float64)
    for k in range(NUM_STEPS):
        state_var = float(closed.state_covariances[k, 0, 0])
        # Monte Carlo error from the draws plus the particle frontier.
        band = 8.0 * np.sqrt(state_var / NUM_DRAWS) + 8.0 * np.sqrt(
            state_var / NUM_PARTICLES
        )
        assert (
            abs(states[:, k, 0].mean() - float(closed.state_means[k, 0])) < band
        )
        obs_var = float(closed.observation_covariances[k, 0, 0])
        band = 8.0 * np.sqrt(obs_var / NUM_DRAWS) + 8.0 * np.sqrt(
            obs_var / NUM_PARTICLES
        )
        assert (
            abs(
                emissions[:, k, 0].mean()
                - float(closed.observation_means[k, 0])
            )
            < band
        )


def test_first_horizon_agrees_with_posterior_predictive_sample():
    """The k=1 emission marginal matches the final predictive row."""
    predictive = smcx.posterior_predictive_sample(
        jr.key(37),
        POSTERIOR,
        _transition_sampler,
        _emission_sampler,
        num_samples=NUM_DRAWS,
    )
    reference = np.asarray(predictive[-1], dtype=np.float64)[:, 0]
    first = np.asarray(PATHS.emission_paths, dtype=np.float64)[:, 0, 0]
    pooled_se = np.sqrt(
        reference.var(ddof=1) / reference.shape[0]
        + first.var(ddof=1) / first.shape[0]
    )
    assert abs(first.mean() - reference.mean()) < 6.0 * pooled_se


def test_input_aware_callbacks_consume_future_inputs():
    """Input-aware callbacks receive one future input per horizon."""
    future_inputs = jnp.asarray([[0.5], [-0.5]])

    def transition_u(key, state, input_t):
        return RHO * state + input_t + Q_SD * jr.normal(key, state.shape)

    def emission_u(key, state, input_t):
        return state + input_t + R_SD * jr.normal(key, state.shape)

    paths = smcx.forecast_sample(
        jr.key(41),
        POSTERIOR,
        transition_u,
        emission_u,
        num_steps=2,
        num_draws=64,
        future_inputs=future_inputs,
    )
    assert paths.state_paths.shape == (64, 2, 1)


def test_liu_west_posterior_is_redirected():
    """A Liu-West posterior gets the documented redirect error."""
    lw = smcx.liu_west_filter(
        jr.key(2),
        _initial_sampler,
        lambda key, state, params: (
            RHO * state + Q_SD * jr.normal(key, state.shape)
        ),
        lambda emission, state, params: (
            -0.5 * ((emission[0] - state[0]) / R_SD) ** 2
        ),
        lambda emission, state, params: (
            -0.5 * ((emission[0] - state[0]) / R_SD) ** 2
        ),
        lambda key, n: jr.normal(key, (n, 1)),
        Y,
        256,
    )
    with pytest.raises(ValueError, match="param_forecast_sample"):
        smcx.forecast_sample(
            jr.key(0),
            lw,
            _transition_sampler,
            _emission_sampler,
            num_steps=1,
            num_draws=8,
        )


def _sample_with(num_steps=1, num_draws=8):
    return smcx.forecast_sample(
        jr.key(0),
        POSTERIOR,
        _transition_sampler,
        _emission_sampler,
        num_steps=num_steps,
        num_draws=num_draws,
    )


def test_count_boundary_matrix():
    """Both counts share the positive-integer contract."""
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _sample_with(num_steps=bad)
        with pytest.raises(ValueError, match="num_draws"):
            _sample_with(num_draws=bad)


def test_future_inputs_length_must_match_num_steps():
    """A future-input history shorter than the horizon is rejected."""
    with pytest.raises(ValueError, match="future_inputs"):
        smcx.forecast_sample(
            jr.key(0),
            POSTERIOR,
            _transition_sampler,
            _emission_sampler,
            num_steps=3,
            num_draws=8,
            future_inputs=jnp.asarray([[0.5], [-0.5]]),
        )


class TestParamForecastSample:
    """Liu-West joint state and parameter forecast paths."""

    def _fit(self):
        return smcx.liu_west_filter(
            jr.key(3),
            _initial_sampler,
            lambda key, state, params: (
                params[0] * state + Q_SD * jr.normal(key, state.shape)
            ),
            lambda emission, state, params: (
                -0.5 * ((emission[0] - state[0]) / R_SD) ** 2
            ),
            lambda emission, state, params: (
                -0.5 * ((emission[0] - state[0]) / R_SD) ** 2
            ),
            lambda key, n: 0.8 + 0.1 * jr.normal(key, (n, 1)),
            Y,
            1024,
        )

    def test_parameters_ride_their_paths(self):
        """Each draw's parameters are constant along its trajectory."""
        posterior = self._fit()
        paths = smcx.param_forecast_sample(
            jr.key(43),
            posterior,
            lambda key, state, params: (
                params[0] * state + Q_SD * jr.normal(key, state.shape)
            ),
            lambda key, state, params: (
                state + R_SD * jr.normal(key, state.shape)
            ),
            num_steps=2,
            num_draws=128,
        )
        assert paths.state_paths.shape == (128, 2, 1)
        assert paths.emission_paths.shape == (128, 2, 1)
        assert paths.parameter_draws.shape == (128, 1)

    def test_count_boundary(self):
        """The param variant shares the positive-integer contract."""
        posterior = self._fit()
        with pytest.raises(ValueError, match="num_steps"):
            smcx.param_forecast_sample(
                jr.key(0),
                posterior,
                lambda key, state, params: state,
                lambda key, state, params: state,
                num_steps=0,
                num_draws=8,
            )
