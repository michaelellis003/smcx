# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for joint linear-Gaussian posterior sampling."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import smcx


def test_posterior_sample_keeps_draw_axis_and_repeats_at_same_key():
    """One draw keeps its axis and an explicit key fixes the result."""
    covariance = jnp.eye(1)
    posterior = smcx.kalman_filter(
        jnp.zeros(1),
        covariance,
        covariance,
        covariance,
        covariance,
        covariance,
        jnp.zeros((2, 1)),
    )
    key = jr.key(330)

    first = smcx.posterior_sample(
        key,
        posterior,
        covariance,
        num_draws=1,
    )
    second = smcx.posterior_sample(
        key,
        posterior,
        covariance,
        num_draws=1,
    )

    assert first.shape == (1, 2, 1)
    np.testing.assert_array_equal(first, second)
