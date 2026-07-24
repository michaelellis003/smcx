# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Adaptive covariance factorization regressions."""

import jax.numpy as jnp
import numpy as np

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
    factor = _degenerate_factor(np.finfo(np.float32).eps / 4)

    assert factor.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(factor)))
    assert np.all(np.diag(np.asarray(factor)) > 0.0)


def test_factor_leaves_well_conditioned_covariance_unregularized():
    particles = jnp.asarray(
        [[-1.0, -0.5], [0.0, 1.0], [2.0, -1.5], [0.5, 2.0]],
        dtype=jnp.float32,
    )
    weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    factor = _weighted_covariance_factor(
        particles,
        weights,
        scale=0.75,
    )

    x = np.asarray(particles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w /= w.sum()
    centered = x - w @ x
    expected = 0.75 * (centered * w[:, None]).T @ centered
    np.testing.assert_allclose(
        np.asarray(factor) @ np.asarray(factor).T,
        expected,
        rtol=2e-6,
        atol=2e-7,
    )
