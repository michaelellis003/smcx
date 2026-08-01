# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Conjugate DLM retrospective-analysis tests (W&H 1997, ch. 4)."""

from fractions import Fraction

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
    m0 = Fraction(1, 2)
    c0 = Fraction(3)
    n0 = Fraction(5)
    s0 = Fraction(2)
    observations = [Fraction(1), Fraction(-1, 2), Fraction(3, 4)]
    evolution = [Fraction(1, 4), Fraction(2, 3)]

    filtered = smcx.dlm_filter(
        jnp.asarray([float(m0)]),
        jnp.asarray([[float(c0)]]),
        jnp.eye(1),
        jnp.asarray([1.0]),
        jnp.asarray([float(value) for value in observations]),
        scale_free_transition_covariance=jnp.asarray([
            [[float(value)]] for value in evolution
        ]),
        prior_shape=float(n0),
        prior_scale=float(s0),
    )
    posterior = smcx.dlm_smoother(
        filtered,
        jnp.eye(1),
        scale_free_transition_covariance=jnp.asarray([
            [[float(value)]] for value in evolution
        ]),
    )

    # Exact scalar forward recursion gives filtered
    # (m, C-tilde) = (7/8, 3/4), (3/16, 1/2), (51/104, 7/13).
    # Substitution in B_t = C_t / (C_t + W_t) gives the exact
    # retrospective moments below. Sixty-four eps covers only the
    # float conversion and a few scalar operations in CPU/x64.
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
        "filtered_means",
        "filtered_scale_free_covariances",
        "scale_shapes",
        "scale_estimates",
        "marginal_loglik",
        "log_evidence_increments",
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
