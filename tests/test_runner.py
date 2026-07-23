# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public caller-owned particle-filter runner contracts."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import smcx


class BootstrapCarry(NamedTuple):
    """Minimal caller-owned bootstrap state."""

    particles: jax.Array
    log_weights: jax.Array


def test_runner_rejects_an_empty_emission_sequence():
    def initialize(time_index, emission, key):
        raise AssertionError((time_index, emission, key))

    def step(carry, time_index, emission, key):
        raise AssertionError((carry, time_index, emission, key))

    with pytest.raises(
        ValueError, match="emissions must contain at least one row"
    ):
        smcx.run_particle_filter(
            jr.key(0),
            initialize,
            step,
            jnp.empty((0, 1)),
        )


def test_runner_matches_bootstrap_filter_at_a_fixed_key():
    """The runner preserves the established filter key schedule."""
    num_particles = 8
    emissions = jnp.zeros((4, 1))

    def initial_sampler(key, count):
        return jr.normal(key, (count, 1))

    def transition_sampler(key, state):
        return 0.8 * state + 0.1 * jr.normal(key, state.shape)

    def log_observation(emission, state):
        del emission
        return -1.0 + 0.0 * state[0]

    def initialize(time_index, emission, key):
        del time_index
        particles = initial_sampler(key, num_particles)
        log_obs = jax.vmap(log_observation, in_axes=(None, 0))(
            emission, particles
        )
        log_weights, log_sum = smcx.log_normalize(log_obs)
        record = smcx.ParticleFilterRecord(
            particles,
            log_weights,
            jnp.arange(num_particles, dtype=jnp.int32),
            log_sum - jnp.log(jnp.asarray(num_particles)),
        )
        return BootstrapCarry(particles, log_weights), record

    def step(carry, time_index, emission, key):
        del time_index
        resample_key, transition_key = jr.split(key)
        ancestors = smcx.multinomial(
            resample_key,
            smcx.normalize(carry.log_weights),
            num_particles,
        )
        selected = carry.particles[ancestors]
        particle_keys = jr.split(transition_key, num_particles)
        particles = jax.vmap(transition_sampler)(particle_keys, selected)
        log_obs = jax.vmap(log_observation, in_axes=(None, 0))(
            emission, particles
        )
        log_weights, log_sum = smcx.log_normalize(log_obs)
        record = smcx.ParticleFilterRecord(
            particles,
            log_weights,
            ancestors,
            log_sum - jnp.log(jnp.asarray(num_particles)),
        )
        return BootstrapCarry(particles, log_weights), record

    key = jr.key(17)
    expected = smcx.bootstrap_filter(
        key,
        initial_sampler,
        transition_sampler,
        log_observation,
        emissions,
        num_particles,
        resampling_fn=smcx.multinomial,
        resampling_threshold=1.1,
    )
    actual = smcx.run_particle_filter(
        key,
        initialize,
        step,
        emissions,
    )

    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected),
        jax.tree.leaves(actual),
        strict=True,
    ):
        assert jnp.array_equal(expected_leaf, actual_leaf)
