# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.bootstrap_filter.

Cross-validates against the Dynamax Kalman filter (exact solution
for linear Gaussian SSMs).
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from smcjax.bootstrap import bootstrap_filter
from tests.conftest import _mvn_logpdf, _mvn_sample

# ---------------------------------------------------------------------------
# Helpers to define the LGSSM for smcjax
# ---------------------------------------------------------------------------


def _make_smcjax_fns(lgssm_params):
    """Build (initial_sampler, transition_sampler, log_obs_fn) closures."""
    m0 = lgssm_params['initial_mean']
    P0 = lgssm_params['initial_cov']
    F = lgssm_params['dynamics_weights']
    Q = lgssm_params['dynamics_cov']
    H = lgssm_params['emissions_weights']
    R = lgssm_params['emissions_cov']

    def initial_sampler(key, n):
        return _mvn_sample(key, m0, P0, shape=(n,))

    def transition_sampler(key, state):
        mean = (F @ state[:, None]).squeeze(-1)
        return _mvn_sample(key, mean, Q)

    def log_observation_fn(emission, state):
        mean = (H @ state[:, None]).squeeze(-1)
        return _mvn_logpdf(emission, mean, R)

    return initial_sampler, transition_sampler, log_observation_fn


# ---------------------------------------------------------------------------
# Test: bootstrap filter vs. Kalman filter (exact)
# ---------------------------------------------------------------------------


class TestBootstrapVsKalman:
    """Bootstrap PF on a linear Gaussian SSM should approximate the Kalman filter."""

    def test_log_marginal_likelihood(self, lgssm_params, lgssm_data):
        """PF log-ML should be close to the Kalman exact log-ML."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)

        # Exact Kalman
        kalman_post = lgssm_filter(params, emissions)
        exact_ll = float(kalman_post.marginal_loglik)

        # Bootstrap PF with many particles
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf_post = bootstrap_filter(
            key=jr.PRNGKey(123),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=10_000,
        )
        pf_ll = float(pf_post.marginal_loglik)

        assert pf_ll == pytest.approx(exact_ll, rel=0.05), (
            f'PF log-ML {pf_ll:.2f} vs Kalman {exact_ll:.2f}'
        )

    def test_filtered_means(self, lgssm_params, lgssm_data):
        """PF weighted means should track the Kalman filtered means."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)

        kalman_post = lgssm_filter(params, emissions)
        kalman_means = kalman_post.filtered_means  # (T, 1)

        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf_post = bootstrap_filter(
            key=jr.PRNGKey(456),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=10_000,
        )

        # Compute weighted mean of particles at each time step
        from smcjax.weights import normalize

        weights = jnp.array(
            [normalize(pf_post.filtered_log_weights[t]) for t in range(50)]
        )  # (T, N)
        pf_means = jnp.sum(
            weights[:, :, None] * pf_post.filtered_particles, axis=1
        )  # (T, 1)

        assert jnp.allclose(pf_means, kalman_means, atol=0.15), (
            f'Max error: {jnp.max(jnp.abs(pf_means - kalman_means)):.4f}'
        )


# ---------------------------------------------------------------------------
# Test: convergence with increasing particles
# ---------------------------------------------------------------------------


class TestBootstrapConvergence:
    """PF estimates should improve with more particles."""

    def test_log_ml_converges(self, lgssm_params, lgssm_data):
        """Log-ML variance decreases with more particles."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)
        exact_ll = float(lgssm_filter(params, emissions).marginal_loglik)

        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        errors = []
        for n in [100, 1_000, 10_000]:
            pf = bootstrap_filter(
                key=jr.PRNGKey(999),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=n,
            )
            errors.append(abs(float(pf.marginal_loglik) - exact_ll))

        # Error should generally decrease with more particles
        assert errors[-1] < errors[0], f'Error did not decrease: {errors}'


class TestBootstrapESSTrace:
    """ESS trace should be reasonable."""

    def test_ess_bounded(self, lgssm_params, lgssm_data):
        """ESS should be between 1 and N at every time step."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        n = 1_000
        pf = bootstrap_filter(
            key=jr.PRNGKey(111),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=n,
        )
        assert jnp.all(pf.ess >= 0.9)  # ESS >= ~1
        assert jnp.all(pf.ess <= n + 0.1)


class TestBootstrapLogEvidenceIncrements:
    """log_evidence_increments field should be consistent."""

    def test_log_evidence_increments_shape(self, lgssm_params, lgssm_data):
        """Shape should be (ntime,)."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf = bootstrap_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        assert pf.log_evidence_increments.shape == (emissions.shape[0],)

    def test_log_evidence_increments_sum_to_marginal(
        self, lgssm_params, lgssm_data
    ):
        """Increments should sum to marginal_loglik."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf = bootstrap_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        total = float(jnp.sum(pf.log_evidence_increments))
        assert total == pytest.approx(float(pf.marginal_loglik), abs=1e-6)

    def test_log_evidence_increments_finite(self, lgssm_params, lgssm_data):
        """All increments should be finite."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcjax_fns(lgssm_params)
        pf = bootstrap_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        assert jnp.all(jnp.isfinite(pf.log_evidence_increments))
