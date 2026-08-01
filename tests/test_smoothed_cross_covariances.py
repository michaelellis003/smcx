# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for linear-Gaussian smoothed adjacent-state covariances."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx


def _scalar_posterior(
    coefficient: jax.Array,
) -> smcx.GaussianSmootherPosterior:
    """Build a consistent two-time scalar smoother record."""
    dtype = coefficient.dtype
    filtered_variance = jnp.asarray(1.0, dtype=dtype)
    process_variance = jnp.asarray(0.7, dtype=dtype)
    terminal_variance = jnp.asarray(0.6, dtype=dtype)
    predicted_variance = filtered_variance * coefficient**2 + process_variance
    gain = filtered_variance * coefficient / predicted_variance
    initial_smoothed_variance = filtered_variance + gain**2 * (
        terminal_variance - predicted_variance
    )
    return smcx.GaussianSmootherPosterior(
        jnp.asarray(0.0, dtype=dtype),
        jnp.zeros((2, 1), dtype=dtype),
        jnp.stack((filtered_variance, predicted_variance)).reshape(2, 1, 1),
        jnp.zeros((2, 1), dtype=dtype),
        jnp.stack((filtered_variance, terminal_variance)).reshape(2, 1, 1),
        jnp.zeros(2, dtype=dtype),
        jnp.zeros((2, 1), dtype=dtype),
        jnp.stack((initial_smoothed_variance, terminal_variance)).reshape(
            2, 1, 1
        ),
    )


def _scalar_cross_covariance(coefficient: jax.Array) -> jax.Array:
    posterior = _scalar_posterior(coefficient)
    return smcx.smoothed_cross_covariances(
        posterior,
        coefficient.reshape(1, 1),
    )[0, 0, 0]


def test_cross_covariances_transform_and_gradient_match_scalar_oracle() -> None:
    """Batching, compilation, and autodiff preserve the analytic result."""
    coefficients = jnp.asarray([0.6, 0.8, 1.1], dtype=jnp.float32)
    expected = 0.6 * coefficients / (coefficients**2 + 0.7)
    expected_gradient = (
        0.6 * (0.7 - coefficients**2) / (coefficients**2 + 0.7) ** 2
    )
    direct = jnp.stack([
        _scalar_cross_covariance(coefficient) for coefficient in coefficients
    ])
    vectorized = jax.vmap(_scalar_cross_covariance)(coefficients)
    compiled = jax.jit(jax.vmap(_scalar_cross_covariance))(coefficients)
    gradient = jax.vmap(jax.grad(_scalar_cross_covariance))

    # The scalar expressions have unit-scale conditioning. Thirty-two eps
    # covers one Cholesky, two triangular solves, and the autodiff graph.
    tolerance = 32 * float(np.finfo(np.float32).eps)
    for actual in (direct, vectorized, compiled):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=tolerance,
        )
    for actual in (gradient(coefficients), jax.jit(gradient)(coefficients)):
        np.testing.assert_allclose(
            actual,
            expected_gradient,
            rtol=0.0,
            atol=tolerance,
        )


def test_cross_covariances_accept_singular_unfactored_matrices() -> None:
    """Only each positive-time predicted covariance needs a factorization."""
    dtype = jnp.float32
    transition = jnp.asarray([[0.5, 0.25], [1.0, 0.5]], dtype=dtype)
    initial = jnp.diag(jnp.asarray([1.0, 0.0], dtype=dtype))
    terminal = jnp.diag(jnp.asarray([0.25, 0.0], dtype=dtype))
    predicted = jnp.asarray([[0.75, 0.5], [0.5, 1.25]], dtype=dtype)
    posterior = smcx.GaussianSmootherPosterior(
        jnp.asarray(0.0, dtype=dtype),
        jnp.zeros((2, 2), dtype=dtype),
        jnp.stack((initial, predicted)),
        jnp.zeros((2, 2), dtype=dtype),
        jnp.stack((initial, terminal)),
        jnp.zeros(2, dtype=dtype),
        jnp.zeros((2, 2), dtype=dtype),
        jnp.stack((
            jnp.diag(jnp.asarray([23.0 / 121.0, 0.0], dtype=dtype)),
            terminal,
        )),
    )
    expected = jnp.asarray([[1.0 / 22.0, 0.0], [0.0, 0.0]], dtype=dtype)

    actual = smcx.smoothed_cross_covariances(posterior, transition)

    np.testing.assert_allclose(
        actual[0],
        expected,
        rtol=0.0,
        atol=8 * float(np.finfo(np.float32).eps),
    )


@pytest.mark.parametrize(
    "transition",
    [jnp.zeros((2, 2)), jnp.empty((0, 2, 2))],
    ids=["static", "timed-empty"],
)
def test_cross_covariances_one_time_returns_exact_empty(
    transition: jax.Array,
) -> None:
    """A one-time record accepts singular fields and factors nothing."""
    zero_covariance = jnp.zeros((1, 2, 2))
    posterior = smcx.GaussianSmootherPosterior(
        jnp.asarray(0.0),
        jnp.zeros((1, 2)),
        zero_covariance,
        jnp.zeros((1, 2)),
        zero_covariance,
        jnp.zeros(1),
        jnp.zeros((1, 2)),
        zero_covariance,
    )

    eager = smcx.smoothed_cross_covariances(posterior, transition)
    compiled = jax.jit(smcx.smoothed_cross_covariances)(posterior, transition)

    assert eager.shape == (0, 2, 2)
    np.testing.assert_array_equal(compiled, eager)


@pytest.mark.parametrize(
    ("posterior_update", "transition", "message"),
    [
        (
            {"smoothed_means": jnp.zeros((2, 2))},
            jnp.ones((1, 1)),
            "smoothed_means must have shape",
        ),
        (
            {"smoothed_means": jnp.zeros((2, 1), dtype=jnp.int32)},
            jnp.ones((1, 1)),
            "smoothed_means must have a floating dtype",
        ),
        (
            {"smoothed_covariances": jnp.zeros((2, 1, 2))},
            jnp.ones((1, 1)),
            "smoothed_covariances must have shape",
        ),
        (
            {"smoothed_covariances": jnp.zeros((2, 1, 1), dtype=jnp.int32)},
            jnp.ones((1, 1)),
            "smoothed_covariances must have a floating dtype",
        ),
        (
            {"smoothed_covariances": -jnp.ones((2, 1, 1))},
            jnp.ones((1, 1)),
            "smoothed_covariances must be positive semidefinite",
        ),
        (
            {"predicted_covariances": jnp.zeros((2, 1, 1))},
            jnp.ones((1, 1)),
            r"predicted_covariances\[1:\] must be positive definite",
        ),
        (
            {},
            jnp.ones((1, 1), dtype=jnp.int32),
            "transition_matrix must have a floating dtype",
        ),
        (
            {},
            jnp.ones((2, 1, 1)),
            "transition_matrix must have shape",
        ),
    ],
)
def test_cross_covariances_reject_invalid_boundaries(
    posterior_update: dict[str, jax.Array],
    transition: jax.Array,
    message: str,
) -> None:
    """The standalone boundary validates the full retained record."""
    posterior = _scalar_posterior(jnp.asarray(0.8))._replace(**posterior_update)

    with pytest.raises(ValueError, match=message):
        smcx.smoothed_cross_covariances(posterior, transition)
