# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`smcx.guided_filter` against an exact LGSSM oracle."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

A, Q, R = 0.9, 0.25, 1.0
M0, P0 = 0.0, 1.0
T = 60
N = 1_500


def _data(seed=0):
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


def _misspecified_proposal():
    """A valid, deliberately non-optimal Gaussian proposal."""
    proposal_var = 0.6

    def sample(key, z, y):
        mean = A * z + 0.25 * (y - A * z)
        return mean + jnp.sqrt(proposal_var) * jr.normal(key, z.shape)

    def log_q(y, z_new, z_old):
        mean = A * z_old[0] + 0.25 * (y[0] - A * z_old[0])
        return -0.5 * (
            jnp.log(2 * jnp.pi * proposal_var)
            + (z_new[0] - mean) ** 2 / proposal_var
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

    def test_input_aware_prior_proposal_matches_bootstrap(self):
        inputs = jnp.array([1.0, 2.0, 3.0])
        emissions = jnp.array([[2.0], [5.0], [9.0]])

        def initial_sampler(key, n, input_t):
            del key
            return jnp.full((n, 1), input_t[0])

        def transition_sampler(key, state, input_t):
            del key
            return state + input_t

        def proposal_sampler(key, state, emission, input_t):
            del key, emission
            return state + input_t

        def log_proposal_fn(emission, new_state, old_state, input_t):
            del emission, new_state, old_state
            return 0.0 * input_t[0]

        def log_transition_fn(new_state, old_state, input_t):
            del new_state, old_state
            return 0.0 * input_t[0]

        def log_observation_fn(emission, state, input_t):
            error = emission[0] - state[0] - input_t[0]
            return -0.5 * error**2

        key = jr.key(7)
        guided = smcx.guided_filter(
            key,
            initial_sampler,
            proposal_sampler,
            log_proposal_fn,
            log_transition_fn,
            log_observation_fn,
            emissions,
            4,
            resampling_threshold=0.0,
            inputs=inputs,
        )
        bootstrap = smcx.bootstrap_filter(
            key,
            initial_sampler,
            transition_sampler,
            log_observation_fn,
            emissions,
            4,
            resampling_threshold=0.0,
            inputs=inputs,
        )

        assert jnp.array_equal(
            guided.filtered_particles, bootstrap.filtered_particles
        )
        assert guided.marginal_loglik == bootstrap.marginal_loglik


class TestGuidedProposalReferences:
    """General proposals preserve the exact filtering target."""

    def test_general_proposals_match_exact_target(self):
        optimal = _optimal_proposal()

        def prior_sample(key, z, y):
            del y
            return _trans(key, z)

        def prior_log_q(y, z_new, z_old):
            del y
            return _log_trans(z_new, z_old)

        proposals = {
            "prior": (prior_sample, prior_log_q),
            "optimal": optimal,
            "misspecified": _misspecified_proposal(),
        }
        exact_logz, exact_means, exact_vars = kalman_1d(
            np.asarray(Y[:, 0]), A, Q, R, M0, P0
        )
        exact_targets = np.array([
            1.0,
            exact_means[-1],
            exact_vars[-1] + exact_means[-1] ** 2,
        ])
        for sample, log_q in proposals.values():
            rows = []
            # R=12 independent committed seeds; SE(mean) = sd / sqrt(R).
            for seed in range(12):
                post = smcx.guided_filter(
                    jr.key(seed),
                    _init,
                    sample,
                    log_q,
                    _log_trans,
                    _logobs,
                    Y,
                    N,
                )
                logz = float(post.marginal_loglik)
                weights = np.exp(
                    np.asarray(post.filtered_log_weights[-1], np.float64)
                )
                particles = np.asarray(
                    post.filtered_particles[-1, :, 0], np.float64
                )
                rows.append([
                    np.exp(logz - exact_logz),
                    weights @ particles,
                    weights @ (particles**2),
                ])
            values = np.asarray(rows)
            estimator_se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            # 2e-5 is the explicit f32/Metal arithmetic budget.
            np.testing.assert_array_less(
                np.abs(values.mean(axis=0) - exact_targets),
                5 * estimator_se + 2e-5,
            )


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
