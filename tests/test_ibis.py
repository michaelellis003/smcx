# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Product tests for private IBIS kernels before public assembly."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smcx._ibis import (
    _advance_ibis_target,
    _ibis_expansion_log_ratio,
    _ibis_move_population,
    _ibis_prefix_expansion,
    _ibis_prefix_logdensity,
    _IBISPopulation,
    _run_ibis_custom_mutation_sweep,
    _run_ibis_rwm_sweep,
)


def _assert_tree_equal(actual, expected) -> None:
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


class _GradientMutationState(NamedTuple):
    position: jax.Array


class _GradientMutationInfo(NamedTuple):
    acceptance_rate: jax.Array


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


def test_rwm_sweep_preserves_keyed_multi_sweep_cache_alignment():
    params = jnp.asarray([[0.0], [2.0], [-2.0]], dtype=jnp.float32)
    population = _IBISPopulation(
        params=params,
        log_target=jnp.asarray([0.0, 8.0, -8.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray(
            [0.0, 0.0, -4.0],
            dtype=jnp.float32,
        ),
    )
    emissions = jnp.asarray(
        [[1e8], [1.0], [-1e8], [4.0], [jnp.nan]],
        dtype=jnp.float32,
    )
    proposal_factor = jnp.asarray([[0.6]], dtype=jnp.float32)
    key = jax.random.key(2)
    time_index = jnp.asarray(3, dtype=jnp.int32)
    target_template = jnp.zeros((), dtype=jnp.float32)

    def log_prior(position):
        return -jnp.float32(0.5) * position[0] ** 2

    def log_increment(emission, position, input_t):
        assert input_t is None
        return emission[0] * position[0]

    def sweep(value):
        return _run_ibis_rwm_sweep(
            key,
            value,
            time_index,
            proposal_factor,
            num_steps=3,
            emissions=emissions,
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )

    eager = sweep(population)
    compiled = jax.jit(sweep)(population)
    _assert_tree_equal(compiled, eager)

    next_population, acceptance_rate = eager
    # The committed key fixes all three sweep keys. CPU and Metal lower the
    # nested f32 matrix/scan arithmetic within five eps at scale ten.
    atol = float(5 * np.finfo(np.float32).eps * 10.0)
    np.testing.assert_allclose(
        next_population.params,
        [[0.9171013], [2.3219929], [-0.44291916]],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        next_population.log_target,
        [3.6684053, 9.2879715, -1.7716767],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        next_population.log_target_correction,
        [0.4965639, -0.37383246, -0.5410079],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_array_equal(
        acceptance_rate,
        jnp.asarray(5.0 / 9.0, dtype=population.log_target.dtype),
    )

    fresh_target, fresh_correction = jax.vmap(
        lambda position: _ibis_prefix_expansion(
            position,
            time_index,
            target_template,
            emissions=emissions,
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )
    )(next_population.params)
    _assert_tree_equal(next_population.log_target, fresh_target)
    _assert_tree_equal(next_population.log_target_correction, fresh_correction)
    assert bool(jnp.all(jnp.abs(next_population.log_target_correction) > 0.25))
    assert acceptance_rate.dtype == population.log_target.dtype


def test_rwm_zero_factor_is_exact_identity_with_unit_acceptance():
    params = jnp.asarray([[1.0], [-2.0]], dtype=jnp.float32)
    population = _IBISPopulation(
        params=params,
        log_target=-jnp.float32(0.5) * params[:, 0] ** 2,
        log_target_correction=jnp.zeros(2, dtype=jnp.float32),
    )

    def log_prior(position):
        return -jnp.float32(0.5) * position[0] ** 2

    def log_increment(emission, position, input_t):
        assert input_t is None
        return emission[0] * position[0]

    def sweep(value):
        return _run_ibis_rwm_sweep(
            jax.random.key(91),
            value,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            num_steps=3,
            emissions=jnp.zeros((1, 1), dtype=jnp.float32),
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )

    for next_population, acceptance_rate in (
        sweep(population),
        jax.jit(sweep)(population),
    ):
        _assert_tree_equal(next_population, population)
        np.testing.assert_array_equal(acceptance_rate, 1.0)
        assert acceptance_rate.dtype == population.log_target.dtype


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="mixed f32-parameter/f64-target dtype contract",
)
def test_rwm_keeps_parameter_and_target_arithmetic_dtypes_separate():
    params = jnp.asarray([[1.0], [-2.0]], dtype=jnp.float32)
    population = _IBISPopulation(
        params=params,
        log_target=jnp.asarray([-0.5, -2.0], dtype=jnp.float64),
        log_target_correction=jnp.zeros(2, dtype=jnp.float64),
    )

    def log_prior(position):
        wide_position = position[0].astype(jnp.float64)
        return -jnp.float64(0.5) * wide_position**2

    def log_increment(_emission, _position, input_t):
        assert input_t is None
        return jnp.zeros((), dtype=jnp.float64)

    next_population, acceptance_rate = _run_ibis_rwm_sweep(
        jax.random.key(7),
        population,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        num_steps=1,
        emissions=jnp.zeros((1, 1), dtype=jnp.float32),
        inputs=None,
        log_prior_fn=log_prior,
        log_likelihood_increment_fn=log_increment,
    )

    _assert_tree_equal(next_population, population)
    assert next_population.params.dtype == jnp.float32
    assert next_population.log_target.dtype == jnp.float64
    assert next_population.log_target_correction.dtype == jnp.float64
    assert acceptance_rate.dtype == jnp.float64


def test_custom_mutation_rebuilds_prefix_cache_after_gradient_moves():
    population = _IBISPopulation(
        params=jnp.asarray([[0.5], [-1.5]], dtype=jnp.float32),
        log_target=jnp.asarray([123.0, -456.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray([17.0, -19.0], dtype=jnp.float32),
    )
    emissions = jnp.asarray(
        [[1e8], [1.0], [-1e8], [4.0], [jnp.nan]],
        dtype=jnp.float32,
    )
    time_index = jnp.asarray(3, dtype=jnp.int32)
    target_template = jnp.zeros((), dtype=jnp.float32)

    def log_prior(position):
        return -jnp.float32(0.5) * position[0] ** 2

    def log_increment(emission, position, input_t):
        assert input_t is None
        return emission[0] * position[0]

    def initialize(position, logdensity_fn):
        gradient = jax.grad(logdensity_fn)(position)
        return _GradientMutationState(position + jnp.zeros_like(gradient))

    def mutate(_key, state, logdensity_fn):
        gradient = jax.grad(logdensity_fn)(state.position)
        next_position = state.position + jnp.float32(0.125) * gradient
        return (
            _GradientMutationState(next_position),
            _GradientMutationInfo(jnp.asarray(0.75, dtype=jnp.float32)),
        )

    def sweep(value):
        return _run_ibis_custom_mutation_sweep(
            jax.random.key(17),
            value,
            time_index,
            num_steps=2,
            emissions=emissions,
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
            initialize=initialize,
            mutate=mutate,
        )

    eager = sweep(population)
    compiled = jax.jit(sweep)(population)
    _assert_tree_equal(compiled, eager)

    next_population, acceptance_rate, acceptance_valid = eager
    fresh_target, fresh_correction = jax.vmap(
        lambda position: _ibis_prefix_expansion(
            position,
            time_index,
            target_template,
            emissions=emissions,
            inputs=None,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )
    )(next_population.params)
    _assert_tree_equal(next_population.log_target, fresh_target)
    _assert_tree_equal(next_population.log_target_correction, fresh_correction)
    assert not np.array_equal(next_population.log_target, population.log_target)
    assert not np.array_equal(
        next_population.log_target_correction,
        population.log_target_correction,
    )
    assert bool(jnp.all(jnp.isfinite(next_population.params)))
    np.testing.assert_array_equal(acceptance_rate, 0.75)
    np.testing.assert_array_equal(acceptance_valid, True)


def test_default_move_fits_corrected_weighted_cloud_and_moves_selected_seeds():
    corrected_params = jnp.asarray([[0.0], [2.0], [10.0]], dtype=jnp.float32)
    corrected = _IBISPopulation(
        params=corrected_params,
        log_target=jnp.asarray([0.0, 2.0, 10.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray([0.0, 0.2, 1.0], dtype=jnp.float32),
    )
    selected = _IBISPopulation(
        params=jnp.full((3, 1), 2.0, dtype=jnp.float32),
        log_target=jnp.full(3, 2.0, dtype=jnp.float32),
        log_target_correction=jnp.full(3, 0.2, dtype=jnp.float32),
    )
    weights = jnp.asarray([0.8, 0.15, 0.05], dtype=jnp.float32)
    log_weights = jnp.log(weights)
    key = jax.random.key(31)
    time_index = jnp.asarray(2, dtype=jnp.int32)
    calls = []

    def rwm_kernel(move_key, population, index, proposal_factor):
        calls.append((move_key, population, index, proposal_factor))
        return population, jnp.asarray(0.25, dtype=jnp.float32)

    result = _ibis_move_population(
        key,
        corrected,
        log_weights,
        selected,
        time_index,
        rwm_kernel=rwm_kernel,
        custom_kernel=None,
    )

    assert len(calls) == 1
    move_key, moved_seed, moved_index, proposal_factor = calls[0]
    np.testing.assert_array_equal(
        jax.random.key_data(move_key),
        jax.random.key_data(key),
    )
    _assert_tree_equal(moved_seed, selected)
    _assert_tree_equal(moved_index, time_index)
    host_weights = np.asarray(weights, dtype=np.float64)
    host_params = np.asarray(corrected_params[:, 0], dtype=np.float64)
    mean = host_weights @ host_params
    variance = (2.38**2) * np.sum(host_weights * (host_params - mean) ** 2)
    np.testing.assert_allclose(
        proposal_factor,
        [[math.sqrt(variance)]],
        rtol=0.0,
        atol=float(5 * np.finfo(np.float32).eps * math.sqrt(variance)),
    )
    _assert_tree_equal(result.population, selected)
    np.testing.assert_array_equal(result.acceptance_rate, 0.25)
    np.testing.assert_array_equal(result.acceptance_valid, True)


def test_custom_move_propagates_rebuilt_population_and_validity():
    corrected = _IBISPopulation(
        params=jnp.asarray([[0.0], [1.0]], dtype=jnp.float32),
        log_target=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray([0.0, 0.1], dtype=jnp.float32),
    )
    selected = _IBISPopulation(
        params=jnp.asarray([[1.0], [1.0]], dtype=jnp.float32),
        log_target=jnp.ones(2, dtype=jnp.float32),
        log_target_correction=jnp.full(2, 0.1, dtype=jnp.float32),
    )
    moved = _IBISPopulation(
        params=jnp.asarray([[1.5], [2.5]], dtype=jnp.float32),
        log_target=jnp.asarray([3.0, 5.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray([0.3, 0.5], dtype=jnp.float32),
    )
    key = jax.random.key(41)
    time_index = jnp.asarray(4, dtype=jnp.int32)

    def rwm_kernel(*_args):
        raise AssertionError("default RWM must not run for custom mutation")

    def custom_kernel(move_key, population, index):
        np.testing.assert_array_equal(
            jax.random.key_data(move_key),
            jax.random.key_data(key),
        )
        _assert_tree_equal(population, selected)
        _assert_tree_equal(index, time_index)
        return (
            moved,
            jnp.asarray(0.625, dtype=jnp.float32),
            jnp.asarray(False),
        )

    result = _ibis_move_population(
        key,
        corrected,
        jnp.log(jnp.asarray([0.75, 0.25], dtype=jnp.float32)),
        selected,
        time_index,
        rwm_kernel=rwm_kernel,
        custom_kernel=custom_kernel,
    )

    _assert_tree_equal(result.population, moved)
    np.testing.assert_array_equal(result.acceptance_rate, 0.625)
    np.testing.assert_array_equal(result.acceptance_valid, False)


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
