# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Product tests for private IBIS kernels before public assembly."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
from smcx._ibis import (
    _advance_ibis_target,
    _ibis_expansion_log_ratio,
    _ibis_move_population,
    _ibis_prefix_expansion,
    _ibis_prefix_logdensity,
    _IBISPopulation,
    _run_ibis_custom_mutation_sweep,
    _run_ibis_rwm_sweep,
    _run_ibis_stages,
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
    corrected_params = jnp.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [3.0, 5.0]],
        dtype=jnp.float32,
    )
    corrected = _IBISPopulation(
        params=corrected_params,
        log_target=jnp.asarray([0.0, 2.0, 4.0, 8.0], dtype=jnp.float32),
        log_target_correction=jnp.asarray(
            [0.0, 0.2, 0.4, 0.8],
            dtype=jnp.float32,
        ),
    )
    selected = _IBISPopulation(
        params=jnp.tile(
            jnp.asarray([[2.0, 0.0]], dtype=jnp.float32),
            (4, 1),
        ),
        log_target=jnp.full(4, 2.0, dtype=jnp.float32),
        log_target_correction=jnp.full(4, 0.2, dtype=jnp.float32),
    )
    weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    log_weights = jnp.log(weights)
    key = jax.random.key(31)
    time_index = jnp.asarray(2, dtype=jnp.int32)

    host_weights = np.asarray(weights, dtype=np.float64)
    host_weights /= host_weights.sum()
    host_params = np.asarray(corrected_params, dtype=np.float64)
    centered = host_params - host_weights @ host_params
    weighted_covariance = (centered * host_weights[:, None]).T @ centered
    expected_factor = np.linalg.cholesky(
        (2.38**2 / corrected_params.shape[1]) * weighted_covariance
    )

    def rwm_kernel(move_key, population, index, proposal_factor):
        np.testing.assert_array_equal(
            jax.random.key_data(move_key),
            jax.random.key_data(key),
        )
        _assert_tree_equal(population, selected)
        _assert_tree_equal(index, time_index)
        # Five f32 eps at factor scale admits the host-f64-to-f32 cast and
        # f32 log/exp round trip, while excluding the missing-dimension scale.
        np.testing.assert_allclose(
            proposal_factor,
            expected_factor,
            rtol=0.0,
            atol=float(5 * np.finfo(np.float32).eps * np.max(expected_factor)),
        )
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


def test_ibis_driver_matches_sequential_oracle_and_final_only_storage():
    params = jnp.asarray([[-1.0], [0.5], [2.0]], dtype=jnp.float32)
    emissions = jnp.asarray(
        [[1e8], [1.0], [-1e8], [0.2], [-0.7]],
        dtype=jnp.float32,
    )
    inputs = jnp.asarray(
        [[0.0], [0.0], [0.0], [0.4], [-0.3]],
        dtype=jnp.float32,
    )
    prior_population = _IBISPopulation(
        params=params,
        log_target=jnp.zeros(3, dtype=jnp.float32),
        log_target_correction=jnp.zeros(3, dtype=jnp.float32),
    )
    initial_log_increment = jnp.full(3, 1e8, dtype=jnp.float32)
    stage_root_key = jax.random.key(71)
    expected_resampling_key = jax.random.split(
        jax.random.split(stage_root_key, 5)[3]
    )[0]

    def log_prior(_position):
        return jnp.zeros((), dtype=jnp.float32)

    def log_increment(emission, position, input_t):
        return emission[0] + input_t[0] * position[0]

    def resample_at_fourth(_log_weights, _ess, time_index):
        return time_index == 3

    def fixed_resampler(key, _weights, _num_samples):
        np.testing.assert_array_equal(
            jax.random.key_data(key),
            jax.random.key_data(expected_resampling_key),
        )
        return jnp.asarray([2, 0, 2], dtype=jnp.int32)

    def initialize(position, _target):
        return _GradientMutationState(position)

    def mutate(_key, state, _target):
        return state, _GradientMutationInfo(
            jnp.asarray(0.25, dtype=jnp.float32)
        )

    def run(store_history):
        return _run_ibis_stages(
            stage_root_key,
            prior_population,
            initial_log_increment,
            emissions,
            inputs=inputs,
            num_mcmc_steps=1,
            resampling_threshold=resample_at_fourth,
            resampling_fn=fixed_resampler,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
            mutation_init_fn=initialize,
            mutation_step_fn=mutate,
            store_history=store_history,
        )

    full = run(True)
    final_only = run(False)

    assert full.filtered_params.shape == (5, 3, 1)
    assert full.filtered_log_weights.shape == (5, 3)
    assert final_only.filtered_params.shape == (1, 3, 1)
    assert final_only.filtered_log_weights.shape == (1, 3)
    for field_name in (
        "ess",
        "log_evidence_increments",
        "acceptance_rates",
        "selection_ess",
        "resampled",
    ):
        assert getattr(full, field_name).shape == (5,)
        _assert_tree_equal(
            getattr(final_only, field_name),
            getattr(full, field_name),
        )
    _assert_tree_equal(final_only.state, full.state)
    _assert_tree_equal(final_only.filtered_params[0], full.filtered_params[-1])
    _assert_tree_equal(
        final_only.filtered_log_weights[0],
        full.filtered_log_weights[-1],
    )

    selected_params = params[jnp.asarray([2, 0, 2])]
    expected_params = jnp.stack([
        params,
        params,
        params,
        selected_params,
        selected_params,
    ])
    _assert_tree_equal(full.filtered_params, expected_params)

    uniform = np.full(3, -math.log(3.0), dtype=np.float64)
    final_increment = np.asarray([-1.3, -0.4, -1.3], dtype=np.float64)
    final_raw_weights = uniform + final_increment
    maximum = np.max(final_raw_weights)
    log_normalizer = maximum + np.log(
        np.sum(np.exp(final_raw_weights - maximum))
    )
    final_log_weights = final_raw_weights - log_normalizer
    expected_weights = np.vstack([np.tile(uniform, (4, 1)), final_log_weights])
    # Ten f32 eps covers two rounded log-domain operations per stage across
    # this five-stage oracle; every asserted separation is materially larger.
    f32_atol = float(10 * np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        full.filtered_log_weights,
        expected_weights,
        rtol=0.0,
        atol=f32_atol,
    )

    expected_increments = np.asarray(
        [1e8, 1.0, -1e8, 0.5165765115, -0.9035525151],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        full.log_evidence_increments[:3],
        expected_increments[:3],
    )
    np.testing.assert_allclose(
        full.log_evidence_increments[3:],
        expected_increments[3:],
        rtol=0.0,
        atol=f32_atol,
    )
    np.testing.assert_allclose(
        full.selection_ess,
        [3.0, 3.0, 3.0, 2.45886323, 2.47067465],
        rtol=0.0,
        atol=f32_atol,
    )
    np.testing.assert_allclose(
        full.ess,
        [3.0, 3.0, 3.0, 3.0, 2.47067465],
        rtol=0.0,
        atol=f32_atol,
    )
    np.testing.assert_array_equal(
        full.acceptance_rates,
        [0.0, 0.0, 0.0, 0.25, 0.0],
    )
    np.testing.assert_array_equal(
        full.resampled,
        [False, False, False, True, False],
    )

    marginal = math.fsum([
        float(full.state.log_evidence),
        float(full.state.log_evidence_correction),
    ])
    assert marginal == pytest.approx(0.6130239964, abs=f32_atol)
    naive_increment_sum = float(jnp.sum(full.log_evidence_increments))
    assert naive_increment_sum == pytest.approx(-0.386976, abs=f32_atol)
    assert abs(marginal - naive_increment_sum) > 0.9

    final_population = full.state.population
    fresh_total, fresh_correction = jax.vmap(
        lambda position: _ibis_prefix_expansion(
            position,
            jnp.asarray(4, dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.float32),
            emissions=emissions,
            inputs=inputs,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
        )
    )(final_population.params)
    _assert_tree_equal(final_population.log_target, fresh_total)
    _assert_tree_equal(
        final_population.log_target_correction,
        fresh_correction,
    )
    np.testing.assert_allclose(
        fresh_total,
        [-0.3, -0.6, -0.3],
        rtol=0.0,
        atol=f32_atol,
    )
    np.testing.assert_array_equal(fresh_correction, [1.0, 1.0, 1.0])
    resolved_targets = np.asarray([
        math.fsum([float(total), float(correction)])
        for total, correction in zip(
            fresh_total,
            fresh_correction,
            strict=True,
        )
    ])
    assert resolved_targets == pytest.approx([0.7, 0.4, 0.7], abs=f32_atol)


def test_ibis_driver_stage_keys_are_exact_and_branch_invariant():
    params = jnp.asarray([[0.0], [2.0], [4.0]], dtype=jnp.float32)
    emissions = jnp.ones((3, 1), dtype=jnp.float32)
    prior_population = _IBISPopulation(
        params=params,
        log_target=jnp.zeros(3, dtype=jnp.float32),
        log_target_correction=jnp.zeros(3, dtype=jnp.float32),
    )
    initial_log_increment = jnp.ones(3, dtype=jnp.float32)
    stage_root_key = jax.random.key(83)
    stage_keys = jax.random.split(stage_root_key, 3)
    key_pairs = [jax.random.split(key) for key in stage_keys]
    expected_resampling = {
        tuple(np.asarray(jax.random.key_data(key_pairs[0][0]))): jnp.arange(
            3, dtype=jnp.int32
        ),
        tuple(np.asarray(jax.random.key_data(key_pairs[2][0]))): jnp.arange(
            2, -1, -1, dtype=jnp.int32
        ),
    }

    def log_prior(_position):
        return jnp.zeros((), dtype=jnp.float32)

    def log_increment(emission, position, input_t):
        assert input_t is None
        return emission[0] + jnp.zeros_like(position[0])

    def keyed_resampler(key, _weights, _num_samples):
        key_data = tuple(np.asarray(jax.random.key_data(key)))
        if key_data not in expected_resampling:
            raise AssertionError(f"unexpected resampling key {key_data}")
        return expected_resampling[key_data]

    def initialize(position, _target):
        return _GradientMutationState(position)

    def mutate(key, state, target):
        prefix = target(state.position)
        noise = jax.random.uniform(
            key,
            state.position.shape,
            dtype=state.position.dtype,
        )
        position = jnp.where(
            prefix > 2.5,
            state.position + noise,
            state.position,
        )
        return _GradientMutationState(position), _GradientMutationInfo(
            jnp.asarray(0.5, dtype=jnp.float32)
        )

    def run(resample_at_zero):
        def criterion(_log_weights, _ess, time_index):
            return (time_index == 2) | (
                jnp.asarray(resample_at_zero) & (time_index == 0)
            )

        return _run_ibis_stages(
            stage_root_key,
            prior_population,
            initial_log_increment,
            emissions,
            inputs=None,
            num_mcmc_steps=1,
            resampling_threshold=criterion,
            resampling_fn=keyed_resampler,
            log_prior_fn=log_prior,
            log_likelihood_increment_fn=log_increment,
            mutation_init_fn=initialize,
            mutation_step_fn=mutate,
            store_history=True,
        )

    late_only = run(False)
    early_and_late = run(True)
    np.testing.assert_array_equal(
        late_only.resampled,
        [False, False, True],
    )
    np.testing.assert_array_equal(
        early_and_late.resampled,
        [True, False, True],
    )
    np.testing.assert_array_equal(
        late_only.acceptance_rates,
        [0.0, 0.0, 0.5],
    )
    np.testing.assert_array_equal(
        early_and_late.acceptance_rates,
        [0.5, 0.0, 0.5],
    )

    mutation_key = key_pairs[2][1]
    sweep_key = jax.random.split(mutation_key, 1)[0]
    particle_keys = jax.random.split(sweep_key, 3)
    expected_noise = jax.vmap(
        lambda key: jax.random.uniform(key, (1,), dtype=jnp.float32)
    )(particle_keys)
    expected_final = params[::-1] + expected_noise
    # Five f32 eps at cloud scale covers the separately lowered keyed
    # addition; any wrong mutation key changes the draw materially.
    key_atol = float(
        5 * np.finfo(np.float32).eps * np.max(np.asarray(expected_final))
    )
    np.testing.assert_allclose(
        late_only.filtered_params[-1],
        expected_final,
        rtol=0.0,
        atol=key_atol,
    )
    np.testing.assert_allclose(
        early_and_late.filtered_params[-1],
        expected_final,
        rtol=0.0,
        atol=key_atol,
    )


def test_public_ibis_contract_uses_canonical_rows_and_exact_root_children():
    root_key = jax.random.key(97)
    prior_key, stage_root_key = jax.random.split(root_key)
    stage_keys = jax.random.split(stage_root_key, 2)
    expected_resampling_key = jax.random.split(stage_keys[1])[0]
    params = jnp.asarray([[-1.0], [0.5], [2.0]], dtype=jnp.float32)

    def sample_prior(key, num_particles):
        np.testing.assert_array_equal(
            jax.random.key_data(key),
            jax.random.key_data(prior_key),
        )
        assert num_particles == 3
        return params

    def log_prior(position):
        return -jnp.float32(0.5) * position[0] ** 2

    def log_increment(emission, position, input_t):
        assert emission.shape == (1,)
        assert input_t.shape == (1,)
        return (emission[0] + input_t[0]) * position[0]

    def criterion(_log_weights, _selection_ess, time_index):
        return time_index == 1

    def reverse_resampler(key, _weights, num_samples):
        np.testing.assert_array_equal(
            jax.random.key_data(key),
            jax.random.key_data(expected_resampling_key),
        )
        return jnp.arange(num_samples - 1, -1, -1, dtype=jnp.int32)

    def initialize(position, _target):
        return _GradientMutationState(position)

    def mutate(_key, state, _target):
        return state, _GradientMutationInfo(jnp.asarray(0.5, dtype=jnp.float32))

    posterior = smcx.ibis(
        root_key,
        sample_prior,
        log_prior,
        log_increment,
        jnp.asarray([0.25, -0.5], dtype=jnp.float32),
        3,
        inputs=jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        num_mcmc_steps=1,
        resampling_threshold=criterion,
        resampling_fn=reverse_resampler,
        mutation_init_fn=initialize,
        mutation_step_fn=mutate,
    )

    assert isinstance(posterior, smcx.IBISPosterior)
    assert isinstance(posterior, smcx.ParameterFilterResult)
    assert posterior._fields == (
        "marginal_loglik",
        "filtered_params",
        "filtered_log_weights",
        "ess",
        "log_evidence_increments",
        "acceptance_rates",
        "selection_ess",
        "resampled",
    )
    assert posterior.filtered_params.shape == (2, 3, 1)
    assert posterior.filtered_log_weights.shape == (2, 3)
    _assert_tree_equal(posterior.filtered_params[0], params)
    _assert_tree_equal(posterior.filtered_params[1], params[::-1])
    np.testing.assert_array_equal(posterior.resampled, [False, True])
    np.testing.assert_array_equal(posterior.acceptance_rates, [0.0, 0.5])
    np.testing.assert_array_equal(posterior.ess[1], 3.0)
    assert float(posterior.selection_ess[1]) < 3.0
    np.testing.assert_array_equal(
        posterior.filtered_log_weights[1],
        jnp.full(3, -math.log(3.0), dtype=jnp.float32),
    )


def test_public_ibis_omitted_inputs_and_final_only_storage():
    params = jnp.asarray([[-1.0], [1.0]], dtype=jnp.float32)

    def sample_prior(_key, _num_particles):
        return params

    def log_prior(position):
        return -jnp.float32(0.5) * position[0] ** 2

    def log_increment(emission, position, input_t):
        assert input_t is None
        return emission[0] * position[0]

    def unexpected_resampling(*_args):
        raise AssertionError("zero threshold must skip resampling")

    posterior = smcx.ibis(
        jax.random.key(101),
        sample_prior,
        log_prior,
        log_increment,
        jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32),
        2,
        num_mcmc_steps=1,
        resampling_threshold=0.0,
        resampling_fn=unexpected_resampling,
        store_history=False,
    )

    assert posterior.filtered_params.shape == (1, 2, 1)
    assert posterior.filtered_log_weights.shape == (1, 2)
    for field_name in (
        "ess",
        "log_evidence_increments",
        "acceptance_rates",
        "selection_ess",
        "resampled",
    ):
        assert getattr(posterior, field_name).shape == (3,)
    np.testing.assert_array_equal(posterior.resampled, False)
    np.testing.assert_array_equal(posterior.acceptance_rates, 0.0)
