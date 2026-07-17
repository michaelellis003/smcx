# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Guided filter tests (ADR-0008 item 2, ported from the MLX suite)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import smcx

A, Q, R = 0.9, 0.25, 1.0
M0, P0 = 0.0, 1.0
T = 60
N = 1_500


def _data(seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(M0, np.sqrt(P0))
    for t in range(1, T):
        x[t] = A * x[t - 1] + rng.normal(0.0, np.sqrt(Q))
    y = x + rng.normal(0.0, np.sqrt(R), T)
    return jnp.asarray(y)[:, None]


Y = _data()


def _init(key, n):
    return M0 + jnp.sqrt(P0) * jr.normal(key, (n, 1))


def _trans(key, z):
    return A * z + jnp.sqrt(Q) * jr.normal(key, z.shape)


def _logobs(y, z):
    return -0.5 * (jnp.log(2 * jnp.pi * R) + (y[0] - z[0]) ** 2 / R)


def _log_trans(z_new, z_old):
    return -0.5 * (jnp.log(2 * jnp.pi * Q) + (z_new[0] - A * z_old[0]) ** 2 / Q)


def _optimal_proposal():
    # Locally optimal q(z_t | z_{t-1}, y_t) for the LGSSM
    # (Doucet, Godsill & Andrieu 2000).
    s_star = 1.0 / (1.0 / Q + 1.0 / R)
    sd = jnp.sqrt(s_star)

    def sample(key, z, y):
        m = s_star * (A * z / Q + y / R)
        return m + sd * jr.normal(key, z.shape)

    def log_q(y, z_new, z_old):
        m = s_star * (A * z_old[0] / Q + y[0] / R)
        return -0.5 * (
            jnp.log(2 * jnp.pi * s_star) + (z_new[0] - m) ** 2 / s_star
        )

    return sample, log_q


class TestGuidedReducesToBootstrap:
    """q = f must reproduce the bootstrap filter."""

    def test_prior_proposal_matches_bootstrap(self):
        # With q = f the correction f/q cancels mathematically, and the
        # per-particle key streams coincide, so the two filters agree
        # to floating-point tolerance at the same key.
        def prop(key, z, y):
            return _trans(key, z)

        def log_q(y, z_new, z_old):
            return _log_trans(z_new, z_old)

        key = jr.key(7)
        guided = smcx.guided_filter(
            key, _init, prop, log_q, _log_trans, _logobs, Y, N
        )
        boot = smcx.bootstrap_filter(key, _init, _trans, _logobs, Y, N)
        assert jnp.allclose(
            guided.marginal_loglik, boot.marginal_loglik, rtol=1e-6
        )
        assert jnp.allclose(
            guided.filtered_particles, boot.filtered_particles, rtol=1e-6
        )


class TestOptimalProposalVarianceReduction:
    """The locally optimal proposal reduces log-ML variance."""

    def test_optimal_proposal_lowers_log_ml_variance(self):
        sample, log_q = _optimal_proposal()
        keys = [jr.key(s) for s in range(12)]
        guided_lls = jnp.stack([
            smcx.guided_filter(
                k, _init, sample, log_q, _log_trans, _logobs, Y, N
            ).marginal_loglik
            for k in keys
        ])
        boot_lls = jnp.stack([
            smcx.bootstrap_filter(
                k, _init, _trans, _logobs, Y, N
            ).marginal_loglik
            for k in keys
        ])
        assert jnp.var(guided_lls) < jnp.var(boot_lls)


class TestGuidedPosterior:
    """Shapes, finiteness, and jit-compatibility."""

    def test_shapes_and_finiteness(self):
        sample, log_q = _optimal_proposal()
        post = smcx.guided_filter(
            jr.key(1), _init, sample, log_q, _log_trans, _logobs, Y, N
        )
        assert post.filtered_particles.shape == (T, N, 1)
        assert post.filtered_log_weights.shape == (T, N)
        assert post.ancestors.shape == (T, N)
        assert post.ess.shape == (T,)
        assert jnp.isfinite(post.marginal_loglik)
        # Normalized log weights at every step.
        assert jnp.allclose(
            jax.nn.logsumexp(post.filtered_log_weights, axis=1),
            0.0,
            atol=1e-5,
        )

    def test_jit_compiles(self):
        sample, log_q = _optimal_proposal()

        @jax.jit
        def run(key):
            return smcx.guided_filter(
                key, _init, sample, log_q, _log_trans, _logobs, Y, N
            ).marginal_loglik

        assert jnp.isfinite(run(jr.key(3)))


class TestDegenerateRaises:
    """Collapsed weights raise host-side in eager execution."""

    def test_impossible_observation_raises(self):
        def impossible(y, z):
            return jnp.array(-jnp.inf)

        with pytest.raises(smcx.DegenerateWeightsError):
            smcx.bootstrap_filter(jr.key(2), _init, _trans, impossible, Y, 200)
