# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""smoothing_weights gates: RTS oracle, FFBS agreement, limits (#387)."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

RHO, Q_SD, R_SD = 0.9, 0.4, 0.6
NUM_PARTICLES = 1024


def _simulate(key, num_timesteps=25):
    return smcx.simulate(
        key,
        lambda k: jr.normal(k, (1,)),
        lambda k, state: RHO * state + Q_SD * jr.normal(k, (1,)),
        lambda k, state: state + R_SD * jr.normal(k, (1,)),
        num_timesteps=num_timesteps,
    )


def _filter(key, observations):
    return smcx.bootstrap_filter(
        key,
        lambda k, n: jr.normal(k, (n, 1)),
        lambda k, state: RHO * state + Q_SD * jr.normal(k, (1,)),
        lambda emission, state: -0.5 * ((emission[0] - state[0]) / R_SD) ** 2,
        observations,
        NUM_PARTICLES,
    )


def _log_transition(state, prev_state, params, input_t):
    del params, input_t
    return -0.5 * ((state[0] - RHO * prev_state[0]) / Q_SD) ** 2


def _exact_smoother(observations):
    filtered = smcx.kalman_filter(
        jnp.zeros(1),
        jnp.eye(1),
        jnp.asarray([[RHO]]),
        jnp.asarray([[Q_SD**2]]),
        jnp.eye(1),
        jnp.asarray([[R_SD**2]]),
        observations,
    )
    return smcx.rts_smoother(filtered, jnp.asarray([[RHO]]))


def test_lgssm_smoothed_moments_match_rts_within_bands():
    """Weighted smoothing-cloud moments track the exact RTS answer."""
    _, observations = _simulate(jr.key(0))
    posterior = _filter(jr.key(1), observations)
    log_smooth = smcx.smoothing_weights(posterior, _log_transition, None)
    weights = np.exp(np.asarray(log_smooth, dtype=np.float64))
    particles = np.asarray(posterior.filtered_particles, dtype=np.float64)[
        :, :, 0
    ]
    means = (weights * particles).sum(axis=1)
    second = (weights * particles**2).sum(axis=1)
    variances = second - means**2
    exact = _exact_smoother(observations)
    exact_means = np.asarray(exact.smoothed_means, dtype=np.float64)[:, 0]
    exact_vars = np.asarray(exact.smoothed_covariances, dtype=np.float64)[
        :, 0, 0
    ]
    ess = 1.0 / (weights**2).sum(axis=1)
    se = np.sqrt(exact_vars / ess)
    np.testing.assert_array_less(np.abs(means - exact_means), 8.0 * se)
    np.testing.assert_array_less(
        np.abs(variances - exact_vars), 8.0 * exact_vars
    )


def test_agrees_with_backward_simulation_moments():
    """FFBS draw moments and smoothing-weight moments coincide."""
    _, observations = _simulate(jr.key(2), num_timesteps=12)
    posterior = _filter(jr.key(3), observations)
    log_smooth = smcx.smoothing_weights(posterior, _log_transition, None)
    weights = np.exp(np.asarray(log_smooth, dtype=np.float64))
    particles = np.asarray(posterior.filtered_particles, dtype=np.float64)[
        :, :, 0
    ]
    weighted_means = (weights * particles).sum(axis=1)
    draws = smcx.backward_simulation(
        jr.key(4), posterior, _log_transition, None, num_draws=4096
    )
    draw_values = np.asarray(draws.smoothed_trajectories, dtype=np.float64)[
        :, :, 0
    ]
    draw_means = draw_values.mean(axis=0)
    draw_se = draw_values.std(axis=0, ddof=1) / np.sqrt(draw_values.shape[0])
    np.testing.assert_array_less(
        np.abs(weighted_means - draw_means), 8.0 * draw_se + 1e-8
    )


class _Record(NamedTuple):
    marginal_loglik: Any
    filtered_particles: Any
    filtered_log_weights: Any
    ancestors: Any
    ess: Any
    log_evidence_increments: Any


def _record(particles, log_weights):
    ntime, num_particles = log_weights.shape
    return _Record(
        marginal_loglik=jnp.zeros(()),
        filtered_particles=particles,
        filtered_log_weights=log_weights,
        ancestors=jnp.zeros((ntime, num_particles), dtype=jnp.int32),
        ess=jnp.full((ntime,), float(num_particles)),
        log_evidence_increments=jnp.zeros((ntime,)),
    )


def test_unique_parent_limit_collapses_to_the_genealogy():
    """With an identity-parent kernel every row equals the terminal row."""
    particles = jnp.tile(jnp.arange(3.0)[None, :, None], (4, 1, 1))
    log_weights = jnp.log(
        jnp.asarray([
            [0.2, 0.3, 0.5],
            [0.3, 0.4, 0.3],
            [0.25, 0.5, 0.25],
            [0.1, 0.2, 0.7],
        ])
    )

    def identity_kernel(state, prev_state, params, input_t):
        del params, input_t
        return jnp.where(jnp.abs(state[0] - prev_state[0]) < 0.5, 0.0, -jnp.inf)

    log_smooth = smcx.smoothing_weights(
        _record(particles, log_weights), identity_kernel, None
    )
    expected = np.tile(np.asarray(log_weights[-1]), (4, 1))
    np.testing.assert_allclose(
        np.asarray(log_smooth), expected, rtol=1e-6, atol=1e-6
    )


def test_single_particle_rows_are_exactly_zero():
    """N = 1: the sole particle carries all mass at every time."""
    particles = jnp.asarray([[[1.0]], [[2.0]], [[4.0]]])
    log_smooth = smcx.smoothing_weights(
        _record(particles, jnp.zeros((3, 1))), _log_transition, None
    )
    np.testing.assert_array_equal(np.asarray(log_smooth), 0.0)


def test_one_time_validates_but_never_invokes_the_transition():
    """T = 1 returns the filtering row; the callable contract holds."""
    particles = jnp.asarray([[[1.0], [2.0], [3.0]]])
    record = _record(particles, jnp.log(jnp.asarray([[0.2, 0.3, 0.5]])))
    callback: Any = None
    with pytest.raises(ValueError, match="log_transition must be callable"):
        smcx.smoothing_weights(record, callback, None)

    def poison(*_args):
        raise AssertionError("transition must not run at ntime == 1")

    result = smcx.smoothing_weights(record, poison, None)
    np.testing.assert_allclose(
        np.asarray(result),
        np.log(np.asarray([[0.2, 0.3, 0.5]])),
        rtol=1e-6,
    )


def test_degenerate_backward_row_raises_eagerly():
    """A NaN transition output raises the documented error eagerly."""
    particles = jnp.tile(jnp.arange(2.0)[None, :, None], (3, 1, 1))

    def nan_kernel(state, prev_state, params, input_t):
        del params, input_t
        return jnp.nan * state[0] * prev_state[0]

    with pytest.raises(smcx.DegenerateWeightsError):
        smcx.smoothing_weights(
            _record(particles, jnp.full((3, 2), -jnp.log(2.0))),
            nan_kernel,
            None,
        )


def test_jit_propagates_nan_instead_of_raising():
    """Under jit the degenerate result is NaN-filled, never an error."""
    particles = jnp.tile(jnp.arange(2.0)[None, :, None], (3, 1, 1))

    def nan_kernel(state, prev_state, params, input_t):
        del params, input_t
        return jnp.nan * state[0] * prev_state[0]

    record = _record(particles, jnp.full((3, 2), -jnp.log(2.0)))
    result = jax.jit(lambda: smcx.smoothing_weights(record, nan_kernel, None))()
    assert bool(jnp.isnan(result).all())


def test_liu_west_posterior_is_rejected():
    """The parameter-carrying record gets the documented redirect."""
    _, observations = _simulate(jr.key(5), num_timesteps=6)
    lw = smcx.liu_west_filter(
        jr.key(6),
        lambda k, n: jr.normal(k, (n, 1)),
        lambda k, state, params: RHO * state + Q_SD * jr.normal(k, (1,)),
        lambda emission, state, params: (
            -0.5 * ((emission[0] - state[0]) / R_SD) ** 2
        ),
        lambda emission, state, params: (
            -0.5 * ((emission[0] - state[0]) / R_SD) ** 2
        ),
        lambda k, n: jr.normal(k, (n, 1)),
        observations,
        64,
    )
    with pytest.raises(ValueError, match="filtered_params"):
        smcx.smoothing_weights(lw, _log_transition, None)


def test_non_conforming_posterior_is_rejected():
    """A record missing the protocol fields gets the documented error."""
    non_conformer: Any = object()
    with pytest.raises(ValueError, match="ParticleFilterResult"):
        smcx.smoothing_weights(non_conformer, _log_transition, None)
