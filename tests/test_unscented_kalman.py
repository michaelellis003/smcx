# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for scaled unscented Kalman filtering."""

import jax.numpy as jnp
import numpy as np

import smcx


def _assert_posterior_close(actual, expected, *, ulps=512):
    """Compare well-conditioned filter fields within an f32/f64 budget."""
    for actual_field, expected_field in zip(actual, expected, strict=True):
        actual_array = np.asarray(actual_field)
        expected_array = np.asarray(expected_field, dtype=actual_array.dtype)
        scale = max(1.0, float(np.max(np.abs(expected_array))))
        atol = ulps * np.finfo(actual_array.dtype).eps * scale
        np.testing.assert_allclose(
            actual_array,
            expected_array,
            rtol=0.0,
            atol=atol,
        )


def test_unscented_kalman_reduces_to_linear_filter():
    """Affine mean callbacks reproduce every exact-filter field."""
    initial_mean = jnp.array([0.2, -0.1])
    initial_covariance = jnp.array([[0.5, 0.03], [0.03, 0.4]])
    transition_matrix = jnp.array([[0.85, 0.1], [-0.05, 0.9]])
    transition_bias = jnp.array([0.02, -0.03])
    transition_covariance = jnp.array([
        [[0.08, 0.01], [0.01, 0.06]],
        [[0.07, 0.00], [0.00, 0.05]],
        [[0.09, -0.01], [-0.01, 0.08]],
    ])
    observation_matrix = jnp.array([[1.0, -0.2]])
    observation_bias = jnp.array([0.04])
    observation_covariance = jnp.array([[[0.3]], [[0.2]], [[0.4]], [[0.25]]])
    emissions = jnp.array([[0.1], [-0.2], [0.3], [0.05]])

    def transition_mean(state):
        return transition_matrix @ state + transition_bias

    def observation_mean(state):
        return observation_matrix @ state + observation_bias

    exact = smcx.kalman_filter(
        initial_mean,
        initial_covariance,
        transition_matrix,
        transition_covariance,
        observation_matrix,
        observation_covariance,
        emissions,
        transition_bias=transition_bias,
        observation_bias=observation_bias,
    )
    unscented = smcx.unscented_kalman_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_covariance,
        observation_mean,
        observation_covariance,
        emissions,
    )

    _assert_posterior_close(unscented, exact)
