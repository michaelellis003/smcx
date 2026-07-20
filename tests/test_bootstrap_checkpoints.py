# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable bootstrap filtering (ADR-0028)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import smcx

EMISSIONS = jnp.array([[0.2], [-0.4], [0.7], [0.1], [-0.2]])
NUM_PARTICLES = 16


def _initial(key, num_particles):
    return jr.normal(key, (num_particles, 1))


def _transition(key, state):
    return 0.8 * state + 0.3 * jr.normal(key, state.shape)


def _log_observation(emission, state):
    return -0.5 * (emission[0] - state[0]) ** 2


def _assert_tree_equal(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_one_shot_equals_init_then_repeated_step():
    """The same ordered keys give identical direct and one-shot results."""
    key = jr.key(2026)
    step_root, init_key = jr.split(key)
    step_keys = jr.split(step_root, EMISSIONS.shape[0] - 1)
    expected = smcx.bootstrap_filter(
        key,
        _initial,
        _transition,
        _log_observation,
        EMISSIONS,
        NUM_PARTICLES,
    )

    checkpoint, info = smcx.bootstrap_init(
        init_key,
        _initial,
        _log_observation,
        EMISSIONS[0],
        NUM_PARTICLES,
    )
    particles = [checkpoint.state.particles]
    log_weights = [checkpoint.state.log_weights]
    ancestors = [info.ancestors]
    ess = [info.ess]
    increments = [info.log_evidence_increment]
    for step_key, emission in zip(step_keys, EMISSIONS[1:], strict=True):
        checkpoint, info = smcx.bootstrap_step(
            step_key,
            checkpoint,
            _transition,
            _log_observation,
            emission,
        )
        particles.append(checkpoint.state.particles)
        log_weights.append(checkpoint.state.log_weights)
        ancestors.append(info.ancestors)
        ess.append(info.ess)
        increments.append(info.log_evidence_increment)

    actual = smcx.ParticleFilterPosterior(
        marginal_loglik=checkpoint.state.log_marginal_likelihood,
        filtered_particles=jax.tree.map(lambda *xs: jnp.stack(xs), *particles),
        filtered_log_weights=jnp.stack(log_weights),
        ancestors=jnp.stack(ancestors),
        ess=jnp.stack(ess),
        log_evidence_increments=jnp.stack(increments),
    )
    _assert_tree_equal(actual, expected)
