# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for smcx.weights — validated against hand-computed values."""

import jax.numpy as jnp

from smcx.weights import log_normalize, normalize


class TestLogNormalize:
    """Tests for log_normalize."""

    def test_uniform_weights(self):
        """Uniform log-weights [0, 0, 0] -> log(1/3) each."""
        lw = jnp.array([0.0, 0.0, 0.0])
        log_norm, log_z = log_normalize(lw)
        expected = jnp.full(3, jnp.log(1.0 / 3.0))
        assert jnp.allclose(log_norm, expected, atol=1e-7)
        # logsumexp([0, 0, 0]) = log(3)
        assert jnp.allclose(log_z, jnp.log(3.0), atol=1e-7)

    def test_degenerate_weights(self):
        """One particle has all weight, rest are -inf."""
        lw = jnp.array([0.0, -jnp.inf, -jnp.inf])
        log_norm, log_z = log_normalize(lw)
        assert jnp.allclose(log_norm[0], 0.0, atol=1e-7)
        assert log_norm[1] == -jnp.inf
        assert log_norm[2] == -jnp.inf
        # logsumexp([0, -inf, -inf]) = 0
        assert jnp.allclose(log_z, 0.0, atol=1e-7)

    def test_normalized_sum_to_zero(self):
        """Normalized log-weights should logsumexp to 0."""
        lw = jnp.array([1.0, 2.0, 3.0])
        log_norm, _ = log_normalize(lw)
        log_total = jnp.logaddexp.reduce(log_norm)
        assert jnp.allclose(log_total, 0.0, atol=1e-7)

    def test_numerical_stability_extreme(self):
        """Large magnitude log-weights should not overflow/underflow."""
        lw = jnp.array([1000.0, 1000.0, 999.0])
        log_norm, log_ev = log_normalize(lw)
        assert jnp.all(jnp.isfinite(log_norm))
        assert jnp.isfinite(log_ev)

    def test_large_negative_weights(self):
        """Very negative log-weights should still normalize correctly."""
        lw = jnp.array([-1000.0, -1000.0, -1001.0])
        log_norm, log_ev = log_normalize(lw)
        assert jnp.all(jnp.isfinite(log_norm))
        assert jnp.isfinite(log_ev)


class TestNormalize:
    """Tests for normalize (exp + normalize)."""

    def test_sums_to_one(self):
        """Normalized weights should sum to 1."""
        lw = jnp.array([1.0, 2.0, 3.0, 4.0])
        w = normalize(lw)
        assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-7)

    def test_uniform(self):
        """Equal log-weights should give equal normalized weights."""
        lw = jnp.zeros(5)
        w = normalize(lw)
        assert jnp.allclose(w, 0.2, atol=1e-7)

    def test_all_positive(self):
        """All normalized weights should be non-negative."""
        lw = jnp.array([-10.0, 0.0, 10.0])
        w = normalize(lw)
        assert jnp.all(w >= 0.0)
