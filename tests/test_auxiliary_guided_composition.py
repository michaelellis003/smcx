# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Compose auxiliary-guided SMC through the public particle-filter runner."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import lax, tree

import smcx
from tests import test_guided as _guided
from tests._kalman import kalman_1d

_EMISSIONS = _guided.Y[:30]
_OPTIMAL_SAMPLE, _LOG_OPTIMAL = _guided._optimal_proposal()


def _prior_proposal(key, state, emission):
    del emission
    return _guided._trans(key, state)


def _log_prior_proposal(emission, new_state, old_state):
    del emission
    return _guided._log_trans(new_state, old_state)


def _predictive_auxiliary(emission, state):
    mean = _guided.A * state[0]
    return -0.5 * (
        jnp.log(2 * jnp.pi * (_guided.Q + _guided.R))
        + (emission[0] - mean) ** 2 / (_guided.Q + _guided.R)
    )


def _flat_auxiliary(emission, state):
    del emission
    return 0.0 * state[0]


def _without_input(callback):
    def wrapped(*args):
        return callback(*args[:-1])

    return wrapped


def _auxiliary_guided_callbacks(
    initial_sampler,
    proposal_sampler,
    log_proposal_fn,
    log_transition_fn,
    log_observation_fn,
    log_auxiliary_fn,
    num_particles,
    *,
    resampling_fn=smcx.systematic,
    resampling_threshold=0.5,
):
    """Build the public-runner recipe documented for issue #99."""
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    log_n = jnp.asarray(math.log(num_particles))

    def evaluate(fn, emission, particles, input_t):
        return jax.vmap(lambda state: fn(emission, state, input_t))(particles)

    def initialize(time_index, emission, *input_and_key):
        del time_index
        input_t, key = (
            input_and_key
            if len(input_and_key) == 2
            else (None, input_and_key[0])
        )
        particles = initial_sampler(key, num_particles, input_t)
        log_g = evaluate(log_observation_fn, emission, particles, input_t)
        log_weights, log_total = smcx.log_normalize(log_g)
        record = smcx.ParticleFilterRecord(
            particles, log_weights, identity, log_total - log_n
        )
        return (particles, log_weights), record

    def step(carry, time_index, emission, *input_and_key):
        del time_index
        input_t, key = (
            input_and_key
            if len(input_and_key) == 2
            else (None, input_and_key[0])
        )
        previous_particles, previous_log_weights = carry
        resample_key, proposal_key = jr.split(key)
        log_auxiliary = evaluate(
            log_auxiliary_fn, emission, previous_particles, input_t
        )
        log_first, first_total = smcx.log_normalize(
            previous_log_weights + log_auxiliary
        )
        do_resample = smcx.ess(log_first) < resampling_threshold * num_particles
        ancestors = lax.cond(
            do_resample,
            lambda: resampling_fn(
                resample_key, smcx.normalize(log_first), num_particles
            ),
            lambda: identity,
        )
        parents = tree.map(lambda leaf: leaf[ancestors], previous_particles)
        keys = jr.split(proposal_key, num_particles)
        particles = jax.vmap(
            lambda key_i, state: proposal_sampler(
                key_i, state, emission, input_t
            )
        )(keys, parents)
        log_f = jax.vmap(lambda new, old: log_transition_fn(new, old, input_t))(
            particles, parents
        )
        log_q = jax.vmap(
            lambda new, old: log_proposal_fn(emission, new, old, input_t)
        )(particles, parents)
        log_g = evaluate(log_observation_fn, emission, particles, input_t)
        log_step = log_g + log_f - log_q
        log_scores = jnp.where(
            do_resample,
            log_step - log_auxiliary[ancestors],
            previous_log_weights + log_step,
        )
        log_weights, second_total = smcx.log_normalize(log_scores)
        increment = jnp.where(
            do_resample,
            first_total + second_total - log_n,
            second_total,
        )
        record = smcx.ParticleFilterRecord(
            particles, log_weights, ancestors, increment
        )
        return (particles, log_weights), record

    return initialize, step


def _assert_close(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(
            actual_leaf, expected_leaf, rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("resampling_threshold", [0.0, 1.1])
@pytest.mark.parametrize("reduction", ["guided", "auxiliary"])
def test_composition_preserves_named_filter_reductions(
    resampling_threshold, reduction
):
    num_particles = 96
    if reduction == "guided":
        proposal, log_q, auxiliary = (
            _OPTIMAL_SAMPLE,
            _LOG_OPTIMAL,
            _flat_auxiliary,
        )
        expected = smcx.guided_filter(
            jr.key(7),
            _guided._init,
            proposal,
            log_q,
            _guided._log_trans,
            _guided._logobs,
            _EMISSIONS[:6],
            num_particles,
            resampling_threshold=resampling_threshold,
        )
    else:
        proposal, log_q, auxiliary = (
            _prior_proposal,
            _log_prior_proposal,
            _predictive_auxiliary,
        )
        expected = smcx.auxiliary_filter(
            jr.key(7),
            _guided._init,
            _guided._trans,
            _guided._logobs,
            auxiliary,
            _EMISSIONS[:6],
            num_particles,
            resampling_threshold=resampling_threshold,
        )
    initialize, step = _auxiliary_guided_callbacks(
        _without_input(_guided._init),
        _without_input(proposal),
        _without_input(log_q),
        _without_input(_guided._log_trans),
        _without_input(_guided._logobs),
        _without_input(auxiliary),
        num_particles,
        resampling_threshold=resampling_threshold,
    )
    actual = smcx.run_particle_filter(
        jr.key(7), initialize, step, _EMISSIONS[:6]
    )
    _assert_close(actual, expected)


def test_optimal_composition_matches_kalman_target():
    exact_logz, exact_means, exact_vars = kalman_1d(
        np.asarray(_EMISSIONS[:, 0]),
        _guided.A,
        _guided.Q,
        _guided.R,
        _guided.M0,
        _guided.P0,
    )
    target = np.array([
        1.0,
        exact_means[-1],
        exact_vars[-1] + exact_means[-1] ** 2,
    ])
    initialize, step = _auxiliary_guided_callbacks(
        _without_input(_guided._init),
        _without_input(_OPTIMAL_SAMPLE),
        _without_input(_LOG_OPTIMAL),
        _without_input(_guided._log_trans),
        _without_input(_guided._logobs),
        _without_input(_predictive_auxiliary),
        512,
        resampling_threshold=1.1,
    )
    rows = []
    for seed in range(12):
        posterior = smcx.run_particle_filter(
            jr.key(seed), initialize, step, _EMISSIONS
        )
        weights = np.exp(posterior.filtered_log_weights[-1])
        particles = np.asarray(posterior.filtered_particles[-1, :, 0])
        rows.append([
            np.exp(float(posterior.marginal_loglik) - exact_logz),
            weights @ particles,
            weights @ particles**2,
        ])
    values = np.asarray(rows)
    # For 12 independent seeds, estimator SE = sample_sd / sqrt(12).
    estimator_se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    # Five estimator SE plus 2e-5 for float32/Metal arithmetic.
    np.testing.assert_array_less(
        np.abs(values.mean(axis=0) - target),
        5 * estimator_se + 2e-5,
    )


def test_structured_input_kernel_obeys_runner_contracts():
    emissions = jnp.array([[0.1], [0.4], [-0.2]])
    inputs = jnp.array([0.3, -0.1, 0.2])
    num_particles = 8

    def initial(key, count, input_t):
        return {
            "x": input_t + 0.1 * jr.normal(key, (count, 1)),
            "tag": jnp.arange(count, dtype=jnp.float32)[:, None],
        }

    def proposal(key, state, emission, input_t):
        mean = 0.6 * state["x"] + 0.2 * input_t + 0.1 * emission
        return {
            "x": mean + 0.2 * jr.normal(key, mean.shape),
            "tag": state["tag"],
        }

    def log_q(emission, new, old, input_t):
        mean = 0.6 * old["x"][0] + 0.2 * input_t[0] + 0.1 * emission[0]
        return -0.5 * (new["x"][0] - mean) ** 2 / 0.04

    def log_f(new, old, input_t):
        mean = 0.6 * old["x"][0] + 0.2 * input_t[0]
        return -0.5 * (new["x"][0] - mean) ** 2 / 0.09

    def log_g(emission, state, input_t):
        mean = state["x"][0] + 0.1 * input_t[0]
        return -0.5 * (emission[0] - mean) ** 2 / 0.5

    def log_m(emission, state, input_t):
        mean = 0.6 * state["x"][0] + 0.3 * input_t[0]
        return -0.5 * (emission[0] - mean) ** 2 / 0.8

    def reverse(key, weights, count):
        del key, weights
        return jnp.arange(count - 1, -1, -1, dtype=jnp.int32)

    initialize, step = _auxiliary_guided_callbacks(
        initial,
        proposal,
        log_q,
        log_f,
        log_g,
        log_m,
        num_particles,
        resampling_fn=reverse,
        resampling_threshold=1.1,
    )

    def run(key, store_history=True):
        return smcx.run_particle_filter(
            key,
            initialize,
            step,
            emissions,
            inputs=inputs,
            store_history=store_history,
        )

    key = jr.key(29)
    with jax.disable_jit():
        eager = run(key)
    compiled = jax.jit(run)(key)
    _assert_close(eager, compiled)
    reverse_indices = jnp.arange(num_particles - 1, -1, -1, dtype=jnp.int32)
    assert jnp.array_equal(
        eager.ancestors[1:], jnp.stack([reverse_indices, reverse_indices])
    )
    final = run(key, store_history=False)
    for full_leaf, final_leaf in zip(
        tree.leaves(eager.filtered_particles),
        tree.leaves(final.filtered_particles),
        strict=True,
    ):
        assert jnp.allclose(full_leaf[-1:], final_leaf, rtol=1e-6)
    assert jnp.allclose(
        eager.filtered_log_weights[-1:],
        final.filtered_log_weights,
        rtol=1e-6,
    )
    assert jnp.allclose(
        jnp.sum(eager.log_evidence_increments),
        eager.marginal_loglik,
        rtol=1e-6,
        atol=1e-6,
    )
