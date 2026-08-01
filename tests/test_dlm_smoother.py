# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Conjugate DLM retrospective-analysis tests (W&H 1997, ch. 4)."""

from fractions import Fraction
from inspect import unwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="frozen CPU/x64 arithmetic contract",
)
def test_dlm_smoother_matches_timed_evolution_fraction_oracle():
    """The scale-free backward recursion matches exact rational arithmetic."""
    observations = jnp.asarray([1.0, -0.5, 0.75])
    evolution = jnp.asarray([[[0.25]], [[float(Fraction(2, 3))]]])

    filtered = smcx.dlm_filter(
        jnp.asarray([0.5]),
        jnp.asarray([[3.0]]),
        jnp.eye(1),
        jnp.asarray([1.0]),
        observations,
        scale_free_transition_covariance=evolution,
        prior_shape=5.0,
        prior_scale=2.0,
    )
    posterior = smcx.dlm_smoother(
        filtered,
        jnp.eye(1),
        scale_free_transition_covariance=evolution,
    )

    # Exact substitution in the scalar recurrences gives these Fractions.
    # Sixty-four eps covers their conversion and a few CPU/x64 operations.
    expected_means = [Fraction(95, 208), Fraction(33, 104), Fraction(51, 104)]
    expected_covariances = [Fraction(21, 52), Fraction(5, 13), Fraction(7, 13)]
    eps = np.finfo(np.asarray(posterior.smoothed_means).dtype).eps
    np.testing.assert_allclose(
        np.asarray(posterior.smoothed_means[:, 0]),
        [float(value) for value in expected_means],
        rtol=0.0,
        atol=64 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(posterior.smoothed_scale_free_covariances[:, 0, 0]),
        [float(value) for value in expected_covariances],
        rtol=0.0,
        atol=64 * eps,
    )
    np.testing.assert_allclose(posterior.scale_shapes[-1], float(Fraction(8)))
    np.testing.assert_allclose(
        posterior.scale_estimates[-1], float(Fraction(145, 104)), rtol=1e-13
    )
    assert type(posterior)._fields == (
        *type(filtered)._fields,
        "smoothed_means",
        "smoothed_scale_free_covariances",
    )
    for retained, original in zip(posterior[:6], filtered, strict=True):
        np.testing.assert_array_equal(retained, original)
    np.testing.assert_array_equal(
        posterior.smoothed_means[-1], posterior.filtered_means[-1]
    )
    np.testing.assert_array_equal(
        posterior.smoothed_scale_free_covariances[-1],
        posterior.filtered_scale_free_covariances[-1],
    )


def test_dlm_smoother_canonicalizes_released_f32_roundoff_under_jit():
    """A released producer crosses only the obsolete symmetry boundary."""
    dtype = jnp.float32
    initial_covariance = jnp.asarray(
        [
            [10.720845, -0.54652244, 3.1010354],
            [-0.54652244, 0.9465956, -0.46825573],
            [3.1010354, -0.46825573, 1.4951488],
        ],
        dtype=dtype,
    )
    transition = jnp.asarray(
        [
            [-0.84977686, -0.20510903, -0.20211807],
            [-0.15791872, 0.11738589, 0.00980542],
            [0.5523522, 0.53633, -0.14769909],
        ],
        dtype=dtype,
    )
    evolution = 0.1 * jnp.eye(3, dtype=dtype)
    filtered = smcx.dlm_filter(
        jnp.zeros(3, dtype=dtype),
        initial_covariance,
        transition,
        jnp.asarray([0.63770854, -0.22345555, -1.3328147], dtype=dtype),
        jnp.asarray(
            [0.25107875, 1.1324171, 0.37735105, 1.4852368, 2.473733],
            dtype=dtype,
        ),
        scale_free_transition_covariance=evolution,
    )
    raw = np.asarray(filtered.filtered_scale_free_covariances).copy()
    transpose = raw.swapaxes(-1, -2)
    eps = np.finfo(raw.dtype).eps
    assert not np.array_equal(raw, transpose)

    def run(record, matrix, covariance):
        return smcx.dlm_smoother(
            record, matrix, scale_free_transition_covariance=covariance
        )

    eager = run(filtered, transition, evolution)
    compiled = jax.jit(run)(filtered, transition, evolution)
    np.testing.assert_array_equal(
        eager.filtered_scale_free_covariances, 0.5 * (raw + transpose)
    )
    np.testing.assert_array_equal(raw, filtered.filtered_scale_free_covariances)
    np.testing.assert_array_equal(
        compiled.filtered_scale_free_covariances,
        compiled.filtered_scale_free_covariances.swapaxes(-1, -2),
    )
    np.testing.assert_array_equal(
        eager.smoothed_scale_free_covariances[-1],
        eager.filtered_scale_free_covariances[-1],
    )
    for eager_field, compiled_field in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(
            eager_field, compiled_field, rtol=32 * eps, atol=32 * eps
        )
    with pytest.raises(ValueError, match="scalar"):
        jax.jit(
            lambda value: unwrap(smcx.dlm_smoother)(
                filtered, transition, discount=value
            )
        )(jnp.asarray([0.9, 0.8, 0.7], dtype=dtype))
