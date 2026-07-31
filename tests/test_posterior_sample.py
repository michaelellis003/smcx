# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for joint linear-Gaussian posterior sampling."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import smcx


def test_posterior_sample_matches_fixed_key_scalar_conditional():
    """A two-time oracle binds the gain, variance, keys, and draw axes."""
    means = jnp.zeros((2, 1))
    posterior = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0),
        means,
        jnp.asarray([[[1.0]], [[2.0]]]),
        means,
        jnp.ones((2, 1, 1)),
        jnp.zeros(2),
    )
    key = jr.key(330)
    keys = jr.split(key, 2)
    count = np.int64(3)

    actual = smcx.posterior_sample(
        key,
        posterior,
        jnp.ones((1, 1)),
        num_draws=count,
    )
    terminal = jr.normal(keys[1], (3, 1), dtype=means.dtype)
    earlier = 0.5 * terminal + jnp.sqrt(
        jnp.asarray(0.5, means.dtype)
    ) * jr.normal(keys[0], (3, 1), dtype=means.dtype)
    expected = jnp.stack((earlier, terminal), axis=1)

    # The same backend and key share normal blocks; 16 eps covers the scalar
    # solve and multiply, while a wrong key or conditional differs by O(1).
    tolerance = 16.0 * np.finfo(np.asarray(actual).dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


def test_posterior_sample_one_time_preserves_zero_variance_coordinate():
    """The terminal fallback handles an empty backward pass without jitter."""
    mean = jnp.asarray([[2.0, -1.0]])
    zero = jnp.zeros((1, 2, 2))
    filtered_covariance = jnp.asarray([[[4.0, 0.0], [0.0, 0.0]]])
    posterior = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0),
        mean,
        zero,
        mean,
        filtered_covariance,
        jnp.zeros(1),
    )

    draws = smcx.posterior_sample(
        jr.key(331), posterior, jnp.eye(2), num_draws=4
    )

    assert draws.shape == (4, 1, 2)
    np.testing.assert_array_equal(draws[:, 0, 1], -1.0)
    assert np.all(np.isfinite(draws))
