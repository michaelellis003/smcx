# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Adaptive covariance factorization regressions."""

import jax.numpy as jnp
import numpy as np
import pytest

from smcx._covariance import _weighted_covariance_factor


def _degenerate_factor(spread):
    particles = jnp.asarray(
        [[0.0, 0.0], [spread, 0.0]],
        dtype=jnp.float32,
    )
    return _weighted_covariance_factor(
        particles,
        jnp.full(2, 0.5, dtype=jnp.float32),
        scale=1.0,
    )


def test_zero_trace_uses_target_dtype_variance_floor():
    factor = _degenerate_factor(0.0)

    assert factor.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(factor)))
    assert np.all(np.diag(np.asarray(factor)) > 0.0)
    factor_variance = np.asarray(factor) @ np.asarray(factor).T
    assert np.all(np.diag(factor_variance) >= np.finfo(np.float32).eps)


def test_near_zero_rank_deficient_factor_remains_positive():
    spread = np.finfo(np.float32).eps / 4
    factor = _degenerate_factor(spread)

    assert factor.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(factor)))
    assert np.all(np.diag(np.asarray(factor)) > 0.0)
    factor_variance = np.asarray(factor) @ np.asarray(factor).T
    expected_null_variance = spread**2 / 8 * 1e-8
    # The Cholesky diagonal is rounded to f32 and squared once. Five eps
    # covers those two operations without admitting the absolute eps floor.
    np.testing.assert_allclose(
        factor_variance[1, 1],
        expected_null_variance,
        rtol=float(5 * np.finfo(np.float32).eps),
        atol=0.0,
    )


@pytest.mark.parametrize("magnitude", [1.0, 1e-3])
def test_factor_leaves_well_conditioned_covariance_unregularized(magnitude):
    particles = magnitude * jnp.asarray(
        [[-1.0, -0.5], [0.0, 1.0], [2.0, -1.5], [0.5, 2.0]],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    factor = _weighted_covariance_factor(
        particles,
        weights,
        scale=0.75,
    )
    assert factor.dtype == jnp.float32

    x = np.asarray(particles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    centered = x - w @ x
    expected = 0.75 * (centered * w[:, None]).T @ centered
    # The f32 factor conversion plus the two-term matrix product contribute
    # at most a few eps; five eps is a conservative f32-honest bound.
    np.testing.assert_allclose(
        np.asarray(factor) @ np.asarray(factor).T,
        expected,
        rtol=float(5 * np.finfo(np.float32).eps),
        atol=0.0,
    )


def test_scale_is_applied_after_covariance_accumulation():
    particles = jnp.asarray(
        [[-628.6092345815442], [328.39444847219517], [-801.0697777975164]],
        dtype=jnp.float64,
    )
    weights = jnp.asarray(
        [0.16770429251437832, 0.4913811098889822, 0.3409145975966395],
        dtype=jnp.float64,
    )
    scale = 413.89206727629573

    factor = _weighted_covariance_factor(
        particles,
        weights,
        scale=scale,
    )

    x = np.asarray(particles)
    w = np.asarray(weights)
    w = w / w.sum()
    centered = x - w @ x
    covariance = (centered * w[:, None]).T @ centered
    expected = np.linalg.cholesky(scale * covariance)
    np.testing.assert_array_equal(np.asarray(factor), expected)
