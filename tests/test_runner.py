# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public caller-owned particle-filter runner contracts."""

import math
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


def _transport_callbacks(num_particles, evidence_scale=1.0):
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    reverse = identity[::-1]

    def initialize(time_index, emission, input_t, key):
        noise = 0.01 * jr.normal(key, (num_particles, 1))
        particles = identity[:, None] + input_t + 0.01 * emission + noise
        log_weights, _ = smcx.log_normalize(-0.1 * identity)
        increment = evidence_scale * (
            input_t[0] + time_index + 0.01 * emission[0]
        )
        carry = {
            "algorithm": {
                "particles": particles,
                "log_weights": log_weights,
            },
            "input_checksum": input_t[0],
        }
        record = smcx.ParticleFilterRecord(
            particles,
            log_weights,
            identity,
            increment,
        )
        return carry, record

    def step(carry, time_index, emission, input_t, key):
        particles = carry["algorithm"]["particles"][reverse]
        particles = (
            particles
            + input_t
            + time_index
            + 0.01 * emission
            + 0.01 * jr.normal(key, particles.shape)
        )
        log_weights, _ = smcx.log_normalize(
            -0.05 * jnp.square(particles[:, 0] - emission[0])
        )
        increment = evidence_scale * (
            input_t[0] + time_index + 0.01 * emission[0]
        )
        next_carry = {
            "algorithm": {
                "particles": particles,
                "log_weights": log_weights,
            },
            "input_checksum": carry["input_checksum"] + input_t[0],
        }
        record = smcx.ParticleFilterRecord(
            particles,
            log_weights,
            reverse,
            increment,
        )
        return next_carry, record

    return initialize, step


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


def test_runner_rejects_rank_one_emissions():
    def initialize(time_index, emission, key):
        raise AssertionError((time_index, emission, key))

    def step(carry, time_index, emission, key):
        raise AssertionError((carry, time_index, emission, key))

    with pytest.raises(ValueError, match="emissions must have shape"):
        smcx.run_particle_filter(
            jr.key(0),
            initialize,
            step,
            jnp.zeros(3),
        )


def test_runner_executes_a_custom_kernel_with_time_aligned_inputs():
    num_particles = 16
    emissions = jnp.array([[10.0], [20.0], [30.0]])
    inputs = jnp.array([1.0, 2.0, 3.0])
    initialize, step = _transport_callbacks(num_particles)

    posterior = smcx.run_particle_filter(
        jr.key(5),
        initialize,
        step,
        emissions,
        inputs=inputs,
    )

    expected_increments = jnp.array([1.1, 3.2, 5.3])
    expected_ancestors = jnp.stack([
        jnp.arange(num_particles, dtype=jnp.int32),
        jnp.arange(num_particles - 1, -1, -1, dtype=jnp.int32),
        jnp.arange(num_particles - 1, -1, -1, dtype=jnp.int32),
    ])
    assert isinstance(posterior, smcx.ParticleFilterResult)
    assert posterior.filtered_particles.shape == (3, num_particles, 1)
    assert jnp.array_equal(posterior.ancestors, expected_ancestors)
    assert jnp.allclose(
        posterior.log_evidence_increments,
        expected_increments,
    )
    assert jnp.allclose(
        posterior.ess,
        jax.vmap(smcx.ess)(posterior.filtered_log_weights),
    )
    assert jnp.array_equal(
        smcx.log_ml_increments(posterior),
        posterior.log_evidence_increments,
    )
    assert jnp.array_equal(
        smcx.particle_diversity(posterior),
        jnp.ones(emissions.shape[0]),
    )
    assert "min_ess" in smcx.diagnose(posterior)


@pytest.mark.parametrize("num_timesteps", [1, 3])
def test_runner_final_only_matches_full_custom_history(num_timesteps):
    num_particles = 16
    emissions = jnp.array([[10.0], [20.0], [30.0]])[:num_timesteps]
    inputs = jnp.array([1.0, 2.0, 3.0])[:num_timesteps]
    initialize, step = _transport_callbacks(num_particles)

    full = smcx.run_particle_filter(
        jr.key(8),
        initialize,
        step,
        emissions,
        inputs=inputs,
    )
    final = smcx.run_particle_filter(
        jr.key(8),
        initialize,
        step,
        emissions,
        inputs=inputs,
        store_history=False,
    )

    assert final.filtered_particles.shape == (1, num_particles, 1)
    assert final.filtered_log_weights.shape == (1, num_particles)
    assert final.ancestors.shape == (1, num_particles)
    assert jnp.array_equal(
        full.filtered_particles[-1:], final.filtered_particles
    )
    assert jnp.array_equal(
        full.filtered_log_weights[-1:],
        final.filtered_log_weights,
    )
    assert jnp.array_equal(full.ancestors[-1:], final.ancestors)
    assert jnp.array_equal(full.ess, final.ess)
    assert jnp.array_equal(
        full.log_evidence_increments,
        final.log_evidence_increments,
    )
    assert jnp.array_equal(full.marginal_loglik, final.marginal_loglik)


def test_custom_runner_supports_jit_vmap_and_evidence_gradients():
    num_particles = 8
    emissions = jnp.array([[10.0], [20.0], [30.0]])
    inputs = jnp.array([1.0, 2.0, 3.0])

    def run(key, evidence_scale=1.0):
        initialize, step = _transport_callbacks(
            num_particles,
            evidence_scale,
        )
        return smcx.run_particle_filter(
            key,
            initialize,
            step,
            emissions,
            inputs=inputs,
        )

    eager = run(jr.key(12))
    compiled = jax.jit(run)(jr.key(12))
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        assert jnp.array_equal(eager_leaf, compiled_leaf)

    batched = jax.vmap(run)(jr.split(jr.key(13), 2))
    assert batched.filtered_particles.shape == (2, 3, num_particles, 1)
    assert batched.ess.shape == (2, 3)

    derivative = jax.grad(lambda scale: run(jr.key(14), scale).marginal_loglik)(
        jnp.asarray(1.0)
    )
    expected = jnp.sum(inputs + jnp.arange(3) + 0.01 * emissions[:, 0])
    assert jnp.allclose(derivative, expected)


def test_runner_compensates_cancellation_in_caller_evidence():
    dtype = jnp.asarray(0.0).dtype
    large = 1e16 if dtype == jnp.float64 else 1e8
    emissions = jnp.asarray([large, 1.0, -large])[:, None]
    particles = jnp.zeros((1, 1), dtype=dtype)
    log_weights = jnp.zeros(1, dtype=dtype)
    ancestors = jnp.zeros(1, dtype=jnp.int32)

    def record(increment):
        return smcx.ParticleFilterRecord(
            particles,
            log_weights,
            ancestors,
            increment,
        )

    def initialize(time_index, emission, key):
        del time_index, key
        return particles, record(emission[0])

    def step(carry, time_index, emission, key):
        del time_index, key
        return carry, record(emission[0])

    posterior = smcx.run_particle_filter(
        jr.key(20),
        initialize,
        step,
        emissions,
    )

    # math.fsum is the independent compensated-summation oracle. Casting
    # its result back to the active dtype keeps the assertion f32-honest.
    expected = jnp.asarray(
        math.fsum(map(float, emissions[:, 0])),
        dtype=dtype,
    )
    assert jnp.array_equal(posterior.marginal_loglik, expected)
    assert jnp.array_equal(
        posterior.log_evidence_increments,
        emissions[:, 0],
    )


def test_runner_rejects_rank_two_record_weights():
    particles = jnp.zeros((3, 1))
    record = smcx.ParticleFilterRecord(
        particles,
        jnp.zeros((3, 1)),
        jnp.arange(3, dtype=jnp.int32),
        jnp.asarray(0.0),
    )

    def initialize(time_index, emission, key):
        del time_index, emission, key
        return particles, record

    def step(carry, time_index, emission, key):
        del time_index, emission, key
        return carry, record

    with pytest.raises(ValueError, match="record log_weights must be rank 1"):
        smcx.run_particle_filter(
            jr.key(0),
            initialize,
            step,
            jnp.zeros((1, 1)),
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
