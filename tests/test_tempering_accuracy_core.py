# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen mathematical and planning contracts for issue #30."""

import math
from dataclasses import replace

import jax.random as jr
import numpy as np
import pytest
from benchmarks.tempering_accuracy.core import (
    ACCURACY_ROOT,
    CHALLENGES,
    ORDER_SEED,
    accuracy_keys,
    build_target,
    centering_summary_count,
    current_cells,
    make_callbacks,
    matched_cells,
    smoke_cells,
    timing_plan,
    waste_free_cells,
)


def _chol_solve(matrix, right):
    lower = np.linalg.cholesky(matrix)
    return np.linalg.solve(lower.T, np.linalg.solve(lower, right))


@pytest.mark.parametrize("geometry", ("G0", "G1"))
@pytest.mark.parametrize("dimension", (4, 32, 128))
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_target_oracle_matches_direct_gaussian_identities(
    geometry, dimension, dtype
):
    target = build_target(geometry, dimension, dtype)
    covariance = target.likelihood_covariance
    identity = np.eye(dimension)
    marginal = identity + covariance

    np.testing.assert_allclose(
        target.posterior_mean,
        _chol_solve(marginal, target.observation),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        target.posterior_covariance,
        identity - _chol_solve(marginal, identity),
        rtol=2e-13,
        atol=2e-13,
    )
    logdet = 2 * np.log(np.diag(np.linalg.cholesky(marginal))).sum()
    expected_log_evidence = -0.5 * (
        dimension * math.log(2 * math.pi)
        + logdet
        + target.observation @ _chol_solve(marginal, target.observation)
    )
    assert target.log_evidence == pytest.approx(expected_log_evidence)


@pytest.mark.parametrize("geometry", ("G0", "G1"))
@pytest.mark.parametrize("dimension", (4, 32, 128))
@pytest.mark.parametrize("dtype", (np.float64, np.float32))
def test_callbacks_include_constants_and_match_dense_target(
    geometry, dimension, dtype
):
    target = build_target(geometry, dimension, dtype)
    callbacks = make_callbacks(target)
    value = np.linspace(-0.4, 0.6, dimension, dtype=dtype)
    residual = target.observation - value.astype(np.float64)
    covariance = target.likelihood_covariance
    logdet = 2 * np.log(np.diag(np.linalg.cholesky(covariance))).sum()
    expected_likelihood = -0.5 * (
        dimension * math.log(2 * math.pi)
        + logdet
        + residual @ _chol_solve(covariance, residual)
    )
    tolerance = 2e-4 if dtype is np.float32 else 2e-12

    assert float(callbacks.log_prior(value)) == pytest.approx(
        -0.5 * (dimension * math.log(2 * math.pi) + value @ value),
        abs=tolerance,
    )
    assert float(callbacks.log_likelihood(value)) == pytest.approx(
        expected_likelihood,
        abs=tolerance,
    )
    sample = callbacks.initial_sampler(jr.key(1), 7)
    assert sample.shape == (7, dimension)
    assert sample.dtype == np.dtype(dtype)


def test_metal_target_rounds_defining_values_before_oracle():
    target = build_target("G1", 32, np.float32)
    rho = float(np.float32(0.9))
    sigma2 = float(np.float32(0.49))
    lags = np.abs(np.subtract.outer(np.arange(32), np.arange(32)))

    np.testing.assert_array_equal(
        target.observation,
        np.linspace(-1, 1, 32, dtype=np.float32).astype(np.float64),
    )
    np.testing.assert_allclose(
        target.likelihood_covariance,
        sigma2 * rho**lags,
        rtol=0,
        atol=0,
    )


def test_accuracy_key_schedule_preserves_prefix_and_is_unique():
    keys = accuracy_keys()
    prefix = jr.split(jr.key(ACCURACY_ROOT), 12)

    assert len(keys) == 32
    np.testing.assert_array_equal(jr.key_data(keys[:12]), jr.key_data(prefix))
    assert len({bytes(jr.key_data(key)) for key in keys}) == 32


def test_registered_cells_and_centering_counts_are_exact():
    current = current_cells()
    matched = matched_cells()
    waste_free = waste_free_cells()

    assert len(current) == 72
    assert len(matched) == len(waste_free) == 12
    assert {(cell.dimension, cell.num_particles) for cell in matched} == set(
        CHALLENGES
    )
    assert {cell.resampler for cell in current} == {"systematic"}
    assert {cell.num_mcmc_steps for cell in current} == {5, 20, 50}
    assert {cell.resampler for cell in matched + waste_free} == {"multinomial"}
    assert {cell.chain_length for cell in waste_free} == {201, 2001}
    assert centering_summary_count(current) == 4_872
    assert centering_summary_count(matched) == 1_356
    assert centering_summary_count(waste_free) == 1_356


def test_smoke_and_timing_plans_are_frozen_and_balanced():
    current = current_cells()
    smoke = smoke_cells()
    plan = timing_plan(current)

    assert ORDER_SEED == 20_260_719
    assert len(smoke) == 4
    assert {
        (cell.geometry, cell.platform, cell.dimension, cell.num_particles)
        for cell in smoke
    } == {
        (geometry, platform, 4, 1_000)
        for geometry in ("G0", "G1")
        for platform in ("cpu", "mps")
    }
    assert len(plan) == 5 * len(current)
    for block in range(5):
        block_cells = [
            replace(cell, block=None) for cell in plan if cell.block == block
        ]
        assert set(block_cells) == set(current)
    assert plan == timing_plan(current)
    assert [cell.block for cell in plan[: len(current)]] == [0] * len(current)
