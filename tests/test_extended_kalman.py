# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for extended Kalman filtering."""

import jax.numpy as jnp
import numpy as np

import smcx


def test_extended_kalman_reduces_to_linear_filter():
    """Linear mean callbacks reproduce every exact-filter field."""
    initial_mean = jnp.array([0.2, -0.1])
    initial_covariance = jnp.array([[0.5, 0.03], [0.03, 0.4]])
    transition_matrix = jnp.array([[0.85, 0.1], [-0.05, 0.9]])
    transition_bias = jnp.array([0.02, -0.03])
    transition_covariance = jnp.array([[0.08, 0.01], [0.01, 0.06]])
    observation_matrix = jnp.array([[1.0, -0.2]])
    observation_bias = jnp.array([0.04])
    observation_covariance = jnp.array([[0.3]])
    emissions = jnp.array([[0.1], [-0.2], [0.3], [0.05]])

    def transition_mean(state):
        return transition_matrix @ state + transition_bias

    def transition_jacobian(_state):
        return transition_matrix

    def observation_mean(state):
        return observation_matrix @ state + observation_bias

    def observation_jacobian(_state):
        return observation_matrix

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
    extended = smcx.extended_kalman_filter(
        initial_mean,
        initial_covariance,
        transition_mean,
        transition_jacobian,
        transition_covariance,
        observation_mean,
        observation_jacobian,
        observation_covariance,
        emissions,
    )

    for actual, expected in zip(extended, exact, strict=True):
        expected_array = np.asarray(expected)
        scale = max(1.0, float(np.max(np.abs(expected_array))))
        atol = 64 * np.finfo(expected_array.dtype).eps * scale
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)
