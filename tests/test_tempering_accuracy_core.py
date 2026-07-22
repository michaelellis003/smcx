# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen mathematical and planning contracts for tempering accuracy."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from benchmarks.tempering_accuracy.core import (
    ACCURACY_ROOT,
    accuracy_keys,
    build_target,
    make_callbacks,
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
    if dtype is np.float64 and not jax.config.read("jax_enable_x64"):
        pytest.skip("CPU-f64 callback contract requires JAX x64")
    target = build_target(geometry, dimension, dtype)
    callbacks = make_callbacks(target)
    assert len(callbacks.device_inputs) == 5
    jax.block_until_ready(callbacks.device_inputs)
    value = np.linspace(-0.4, 0.6, dimension, dtype=dtype)
    residual = target.observation - value.astype(np.float64)
    covariance = target.likelihood_covariance
    logdet = 2 * np.log(np.diag(np.linalg.cholesky(covariance))).sum()
    expected_likelihood = -0.5 * (
        dimension * math.log(2 * math.pi)
        + logdet
        + residual @ _chol_solve(covariance, residual)
    )
    expected_normalizer = -0.5 * (dimension * math.log(2 * math.pi) + logdet)
    tolerance = 2e-4 if dtype is np.float32 else 2e-12

    actual_prior = callbacks.log_prior(value)
    assert isinstance(actual_prior, jax.Array)
    assert float(actual_prior) == pytest.approx(
        -0.5 * (dimension * math.log(2 * math.pi) + value @ value),
        abs=tolerance,
    )
    assert actual_prior.dtype == np.dtype(dtype)
    likelihood_result = callbacks.log_likelihood(value)
    assert isinstance(likelihood_result, jax.Array)
    actual_likelihood = float(likelihood_result)
    assert likelihood_result.dtype == np.dtype(dtype)
    normalizer_result = callbacks.log_likelihood(
        jnp.asarray(target.observation, dtype=dtype)
    )
    assert isinstance(normalizer_result, jax.Array)
    actual_normalizer = float(normalizer_result)
    assert actual_normalizer == pytest.approx(
        expected_normalizer,
        abs=tolerance,
    )
    assert -2 * (actual_likelihood - actual_normalizer) == pytest.approx(
        residual @ _chol_solve(covariance, residual),
        abs=2 * tolerance,
    )
    assert actual_likelihood == pytest.approx(
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
    root = jr.key(ACCURACY_ROOT)
    prefix = jr.split(root, 12)
    extension_root = jr.fold_in(root, 0x54414343)
    extension = np.stack([
        jr.key_data(jr.fold_in(extension_root, i)) for i in range(20)
    ])

    assert len(keys) == 32
    np.testing.assert_array_equal(jr.key_data(keys[:12]), jr.key_data(prefix))
    np.testing.assert_array_equal(jr.key_data(keys[12:]), extension)
    key_bytes = {
        np.asarray(jax.device_get(jr.key_data(key))).tobytes() for key in keys
    }
    assert len(key_bytes) == 32
