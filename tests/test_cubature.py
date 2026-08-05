# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""cubature strategy gates: preset equivalence and exactness (#382)."""

import jax
import jax.numpy as jnp
import numpy as np

import smcx


def _nonlinear_model():
    def transition_mean(state):
        return jnp.array([
            0.9 * state[0] + 0.1 * jnp.sin(state[1]),
            0.8 * state[1],
        ])

    def observation_mean(state):
        return jnp.array([state[0] + 0.05 * state[1] ** 2])

    arrays = dict(
        initial_mean=jnp.zeros(2),
        initial_covariance=jnp.eye(2),
        transition_covariance=0.1 * jnp.eye(2),
        observation_covariance=jnp.array([[0.3]]),
        emissions=jnp.array([[0.2], [-0.1], [0.4]]),
    )
    return transition_mean, observation_mean, arrays


def test_cubature_is_the_documented_unscented_preset():
    """The spherical-radial rule is Unscented(1, 0, 0), by identity."""
    assert smcx.cubature() == smcx.unscented(alpha=1.0, beta=0.0, kappa=0.0)


def test_cubature_filter_matches_the_preset_bitwise():
    """Filtering under cubature() equals the explicit preset exactly."""
    transition_mean, observation_mean, arrays = _nonlinear_model()

    def run(method):
        return smcx.gaussian_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            method=method,
        )

    via_cubature = run(smcx.cubature())
    via_preset = run(smcx.unscented(alpha=1.0, beta=0.0, kappa=0.0))
    for left, right in zip(
        jax.tree.leaves(via_cubature),
        jax.tree.leaves(via_preset),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_cubature_is_exact_on_a_linear_model():
    """All sigma-point rules integrate linear models exactly."""
    transition = jnp.array([[0.9, 0.1], [-0.05, 0.85]])
    observation = jnp.array([[1.0, 0.0], [0.5, 1.0]])
    arrays = dict(
        initial_mean=jnp.array([0.5, -0.25]),
        initial_covariance=jnp.array([[1.0, 0.2], [0.2, 0.8]]),
        transition_covariance=jnp.array([[0.3, 0.05], [0.05, 0.4]]),
        observation_covariance=jnp.array([[0.5, 0.1], [0.1, 0.6]]),
        emissions=jnp.array([[0.3, -0.1], [0.6, 0.2], [-0.4, 0.9]]),
    )
    via_cubature = smcx.gaussian_filter(
        arrays["initial_mean"],
        arrays["initial_covariance"],
        lambda state: transition @ state,
        arrays["transition_covariance"],
        lambda state: observation @ state,
        arrays["observation_covariance"],
        arrays["emissions"],
        method=smcx.cubature(),
    )
    exact = smcx.kalman_filter(
        arrays["initial_mean"],
        arrays["initial_covariance"],
        transition,
        arrays["transition_covariance"],
        observation,
        arrays["observation_covariance"],
        arrays["emissions"],
    )
    eps = float(jnp.finfo(via_cubature.filtered_means.dtype).eps)
    np.testing.assert_allclose(
        np.asarray(via_cubature.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=1e3 * eps,
        atol=1e2 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(via_cubature.marginal_loglik),
        np.asarray(exact.marginal_loglik),
        rtol=1e3 * eps,
    )


def test_cubature_smoother_matches_the_preset_bitwise():
    """The smoother path composes with the preset unchanged."""
    transition_mean, observation_mean, arrays = _nonlinear_model()

    def run(method):
        filtered = smcx.gaussian_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            transition_mean,
            arrays["transition_covariance"],
            observation_mean,
            arrays["observation_covariance"],
            arrays["emissions"],
            method=method,
        )
        return smcx.gaussian_smoother(filtered, transition_mean, method=method)

    via_cubature = run(smcx.cubature())
    via_preset = run(smcx.unscented(alpha=1.0, beta=0.0, kappa=0.0))
    for left, right in zip(
        jax.tree.leaves(via_cubature),
        jax.tree.leaves(via_preset),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
