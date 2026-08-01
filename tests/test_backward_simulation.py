# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Core contracts for particle-filter backward simulation."""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.containers import LiuWestPosterior, ParticleFilterPosterior


def _posterior(particles, log_weights) -> ParticleFilterPosterior:
    log_weights = jnp.asarray(log_weights)
    ntime, num_particles = log_weights.shape
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(0.0),
        filtered_particles=particles,
        filtered_log_weights=log_weights,
        ancestors=jnp.tile(
            jnp.arange(num_particles, dtype=jnp.int32), (ntime, 1)
        ),
        ess=jnp.full((ntime,), num_particles),
        log_evidence_increments=jnp.zeros(ntime),
    )


def _uniform_log_weights(ntime: int, num_particles: int) -> jax.Array:
    return jnp.full((ntime, num_particles), -jnp.log(num_particles))


def test_public_record_is_two_field_sequence_and_exports():
    trajectories = jnp.zeros((2, 3, 1))
    indices = jnp.zeros((2, 3), dtype=jnp.int32)
    result = smcx.ParticleSmootherPosterior(
        smoothed_trajectories=trajectories, backward_indices=indices
    )

    unpacked_trajectories, unpacked_indices = result
    assert result._fields == ("smoothed_trajectories", "backward_indices")
    assert unpacked_trajectories is trajectories
    assert unpacked_indices is indices
    assert callable(smcx.backward_simulation)


def test_unique_parent_draws_use_next_input_and_gather_the_path():
    particles = jnp.broadcast_to(jnp.arange(3.0)[None, :, None], (3, 3, 1))
    log_weights = (
        _uniform_log_weights(3, 3)
        .at[-1]
        .set(jnp.asarray([-jnp.inf, -jnp.inf, 0.0]))
    )
    posterior = _posterior(particles, log_weights)

    def log_transition(state, prev_state, params, input_t):
        del state, params
        return jnp.where(prev_state[0] == input_t[0], 0.0, -jnp.inf)

    result = smcx.backward_simulation(
        jr.key(1),
        posterior,
        log_transition,
        None,
        num_draws=4,
        inputs=jnp.asarray([[99.0], [0.0], [1.0]]),
    )

    np.testing.assert_array_equal(
        result.backward_indices, jnp.asarray([[0, 1, 2]] * 4)
    )
    assert result.backward_indices.dtype == jnp.int32
    gathered = particles[jnp.arange(3)[None, :], result.backward_indices]
    np.testing.assert_array_equal(result.smoothed_trajectories, gathered)


def test_single_time_skips_callback():
    one_time = jnp.asarray([[[1.0], [2.0], [3.0]]])
    posterior = _posterior(one_time, jnp.asarray([[-jnp.inf, 0.0, -jnp.inf]]))

    def must_not_run(*args):
        raise AssertionError("transition callback ran for ntime == 1")

    result = smcx.backward_simulation(
        jr.key(3), posterior, must_not_run, None, num_draws=3
    )

    np.testing.assert_array_equal(result.backward_indices, 1)
    np.testing.assert_array_equal(
        result.smoothed_trajectories, jnp.full((3, 1, 1), 2.0)
    )


def test_non_index_draw_count_reaches_public_validation():
    p = _posterior(jnp.zeros((1, 1, 1)), jnp.zeros((1, 1)))
    count: Any = 1.5
    callback: Any = None
    with pytest.raises(ValueError, match="positive integer"):
        smcx.backward_simulation(jr.key(4), p, callback, None, num_draws=count)


def test_liu_west_parameter_history_is_rejected():
    posterior = _posterior(jnp.zeros((2, 2, 1)), _uniform_log_weights(2, 2))
    liu_west: Any = LiuWestPosterior(
        *posterior,
        filtered_params=jnp.zeros((2, 2, 1)),
    )

    with pytest.raises(ValueError, match="filtered_params"):
        smcx.backward_simulation(
            jr.key(5), liu_west, lambda *_: jnp.asarray(0.0), None, num_draws=1
        )
