# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Core contracts for particle-filter backward simulation."""

import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.containers import LiuWestPosterior, ParticleFilterPosterior


class _MixedParticles(NamedTuple):
    continuous: jax.Array
    category: jax.Array
    flag: jax.Array


def _posterior(particles, log_weights) -> ParticleFilterPosterior:
    log_weights = jnp.asarray(log_weights)
    ntime, num_particles = log_weights.shape
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(0.0),
        filtered_particles=particles,
        filtered_log_weights=log_weights,
        ancestors=jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32),
            (ntime, num_particles),
        ),
        ess=jnp.full((ntime,), num_particles),
        log_evidence_increments=jnp.zeros(ntime),
    )


def _uniform_log_weights(ntime: int, num_particles: int) -> jax.Array:
    return jnp.full((ntime, num_particles), -jnp.log(num_particles))


def test_public_record_is_two_field_sequence_and_exports():
    containers = importlib.import_module("smcx.containers")
    smoothing = importlib.import_module("smcx.smoothing")
    record_type = containers.ParticleSmootherPosterior
    trajectories = jnp.zeros((2, 3, 1))
    indices = jnp.zeros((2, 3), dtype=jnp.int32)
    result = record_type(
        smoothed_trajectories=trajectories,
        backward_indices=indices,
    )

    unpacked_trajectories, unpacked_indices = result
    assert result._fields == ("smoothed_trajectories", "backward_indices")
    assert unpacked_trajectories is trajectories
    assert unpacked_indices is indices
    assert smcx.ParticleSmootherPosterior is record_type
    assert smcx.backward_simulation is smoothing.backward_simulation


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


def test_single_time_skips_callback_and_single_particle_is_exact():
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

    one_particle = jnp.asarray([[[1.0]], [[2.0]], [[4.0]]])
    posterior = _posterior(one_particle, jnp.zeros((3, 1)))
    result = smcx.backward_simulation(
        jr.key(4), posterior, lambda *_: jnp.asarray(0.0), None, num_draws=2
    )

    np.testing.assert_array_equal(result.backward_indices, 0)
    np.testing.assert_array_equal(
        result.smoothed_trajectories,
        jnp.broadcast_to(one_particle[:, 0], (2, 3, 1)),
    )


def test_liu_west_parameter_history_is_rejected():
    posterior = _posterior(jnp.zeros((2, 2, 1)), _uniform_log_weights(2, 2))
    liu_west = LiuWestPosterior(
        *posterior,
        filtered_params=jnp.zeros((2, 2, 1)),
    )

    with pytest.raises(ValueError, match="filtered_params"):
        smcx.backward_simulation(
            jr.key(5), liu_west, lambda *_: jnp.asarray(0.0), None, num_draws=1
        )


def test_traced_degeneracy_invalidates_whole_mixed_dtype_record():
    particles = _MixedParticles(
        continuous=jnp.arange(4.0).reshape(2, 2, 1),
        category=jnp.arange(4, dtype=jnp.int32).reshape(2, 2),
        flag=jnp.ones((2, 2), dtype=bool),
    )
    posterior = _posterior(particles, _uniform_log_weights(2, 2))

    def impossible_transition(state, prev_state, params, input_t):
        del state, prev_state, params, input_t
        return jnp.asarray(-jnp.inf)

    @jax.jit
    def sample(key):
        return smcx.backward_simulation(
            key, posterior, impossible_transition, None, num_draws=3
        )

    result = sample(jr.key(6))

    np.testing.assert_array_equal(result.backward_indices, -1)
    assert np.isnan(result.smoothed_trajectories.continuous).all()
    np.testing.assert_array_equal(result.smoothed_trajectories.category, 0)
    np.testing.assert_array_equal(result.smoothed_trajectories.flag, False)
