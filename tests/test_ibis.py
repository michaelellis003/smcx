# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Product tests for private IBIS kernels before public assembly."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from smcx._ibis import (
    _advance_ibis_target,
    _ibis_expansion_log_ratio,
    _ibis_prefix_expansion,
    _ibis_prefix_logdensity,
    _IBISPopulation,
)


def _assert_tree_equal(actual, expected) -> None:
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


def test_expansion_log_ratio_retains_low_order_difference():
    proposed_total = jnp.asarray([1e8, -1e8], dtype=jnp.float32)
    proposed_correction = jnp.asarray([1.0, -0.5], dtype=jnp.float32)
    current_total = jnp.asarray([1e8, -1e8], dtype=jnp.float32)
    current_correction = jnp.asarray([0.0, -1.5], dtype=jnp.float32)

    def ratio():
        return _ibis_expansion_log_ratio(
            proposed_total,
            proposed_correction,
            current_total,
            current_correction,
        )

    naive = (proposed_total + proposed_correction) - (
        current_total + current_correction
    )
    np.testing.assert_array_equal(naive, [0.0, 0.0])
    np.testing.assert_array_equal(ratio(), [1.0, 1.0])
    np.testing.assert_array_equal(jax.jit(ratio)(), [1.0, 1.0])


def test_prefix_target_value_and_gradient_exclude_future_factor():
    emissions = jnp.asarray([[0.5], [-1.25], [jnp.nan]], dtype=jnp.float32)
    inputs = jnp.asarray([[0.2], [0.4], [jnp.nan]], dtype=jnp.float32)
    template = jnp.zeros((), dtype=jnp.float32)
    params = jnp.asarray([0.7], dtype=jnp.float32)
    time_index = jnp.asarray(1, dtype=jnp.int32)

    def log_prior(position):
        return -jnp.float32(0.5 * 0.3) * position[0] ** 2

    def log_increment(emission, position, input_t):
        return emission[0] * position[0] - jnp.float32(0.5) * input_t[0] * (
            position[0] ** 2
        )

    def target(position, index):
        return _ibis_prefix_logdensity(
            position,
            index,
            template,
            emissions=emissions,
            inputs=inputs,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )

    expected_value = -0.75 * float(params[0]) - 0.45 * float(params[0]) ** 2
    expected_gradient = -0.75 - 0.9 * float(params[0])
    # The target crosses a sequential f32 scan. Five f32 eps at unit scale
    # admits only its accumulated rounding, not a future-factor contribution.
    atol = float(5 * np.finfo(np.float32).eps)
    for evaluate in (target, jax.jit(target)):
        value, gradient = jax.value_and_grad(evaluate)(params, time_index)
        np.testing.assert_allclose(value, expected_value, rtol=0.0, atol=atol)
        np.testing.assert_allclose(
            gradient,
            [expected_gradient],
            rtol=0.0,
            atol=atol,
        )
        assert bool(jnp.isfinite(value))
        assert bool(jnp.all(jnp.isfinite(gradient)))


def test_prefix_target_passes_none_when_inputs_are_omitted():
    emissions = jnp.asarray([[1.0], [2.0], [4.0]], dtype=jnp.float32)

    def log_prior(_position):
        return jnp.asarray(-0.5, dtype=jnp.float32)

    def log_increment(emission, position, input_t):
        assert input_t is None
        return emission[0] + jnp.zeros_like(position[0])

    def target(position):
        return _ibis_prefix_logdensity(
            position,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.float32),
            emissions=emissions,
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )

    position = jnp.asarray([0.0], dtype=jnp.float32)
    np.testing.assert_array_equal(target(position), 2.5)
    np.testing.assert_array_equal(jax.jit(target)(position), 2.5)


def test_prefix_expansion_retains_cancelling_factor():
    emissions = jnp.asarray([[1e8], [1.0], [-1e8]], dtype=jnp.float32)

    def log_prior(_position):
        return jnp.zeros((), dtype=jnp.float32)

    def log_increment(emission, _position, input_t):
        assert input_t is None
        return emission[0]

    def expansion(position):
        return _ibis_prefix_expansion(
            position,
            jnp.asarray(2, dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.float32),
            emissions=emissions,
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )

    position = jnp.asarray([0.0], dtype=jnp.float32)
    evaluations = (expansion(position), jax.jit(expansion)(position))
    for total, correction in evaluations:
        np.testing.assert_array_equal(total, 0.0)
        np.testing.assert_array_equal(correction, 1.0)
        np.testing.assert_array_equal(total + correction, 1.0)


def test_target_cache_advance_matches_fsum_and_compiled_execution():
    population = _IBISPopulation(
        params=jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32),
        log_target=jnp.asarray([1e8, -1e8, 1.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray([0.5, -0.5, 0.0], dtype=jnp.float32),
    )
    emission = jnp.asarray([0.25], dtype=jnp.float32)
    input_t = jnp.asarray([0.5], dtype=jnp.float32)

    def log_increment(emission_t, params, input_value):
        return emission_t[0] * params[0] + input_value[0]

    def advance(value):
        return _advance_ibis_target(
            value,
            emission,
            input_t,
            log_increment,
        )

    eager = advance(population)
    compiled = jax.jit(advance)(population)
    _assert_tree_equal(compiled, eager)

    next_population, increments = eager
    _assert_tree_equal(next_population.params, population.params)
    np.testing.assert_array_equal(increments, [0.75, 1.0, 1.25])
    expected = np.asarray([
        math.fsum([float(total), float(correction), float(increment)])
        for total, correction, increment in zip(
            population.log_target,
            population.log_target_correction,
            increments,
            strict=True,
        )
    ])
    resolved = np.asarray([
        math.fsum([float(total), float(correction)])
        for total, correction in zip(
            next_population.log_target,
            next_population.log_target_correction,
            strict=True,
        )
    ])
    # Resolve the expansion in the independent host oracle; adding its two
    # f32 components on device can round a small correction away again.
    np.testing.assert_allclose(
        resolved,
        expected,
        rtol=0.0,
        atol=float(np.finfo(np.float32).eps),
    )
