# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.auxiliary_filter.

Cross-validates against:
1. Dynamax Kalman filter (exact solution for linear Gaussian SSMs)
2. smcjax bootstrap filter (APF with flat auxiliary = bootstrap)
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from smcjax.auxiliary import auxiliary_filter
from smcjax.bootstrap import bootstrap_filter
from tests.conftest import _mvn_logpdf, _mvn_sample

# ---------------------------------------------------------------------------
# Helpers to define the LGSSM for smcjax APF
# ---------------------------------------------------------------------------


def _make_smcjax_fns(lgssm_params):
    """Build (initial, transition, log_obs, log_aux) closures."""
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

    def log_auxiliary_fn(emission, state):
        """Look-ahead: p(y_{t+1} | mu_{t+1}) where mu = F @ x_t."""
        predicted_mean = (H @ F @ state[:, None]).squeeze(-1)
        return _mvn_logpdf(emission, predicted_mean, R)

    return (
        initial_sampler,
        transition_sampler,
        log_observation_fn,
        log_auxiliary_fn,
    )


# ---------------------------------------------------------------------------
# Test: APF vs. Kalman filter (exact)
# ---------------------------------------------------------------------------


class TestAuxiliaryVsKalman:
    """APF on a linear Gaussian SSM should approximate Kalman."""

    def test_auxiliary_log_ml_matches_kalman(self, lgssm_params, lgssm_data):
        """APF log-ML should be close to Kalman exact log-ML."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)

        # Exact Kalman
        kalman_post = lgssm_filter(params, emissions)
        exact_ll = float(kalman_post.marginal_loglik)

        # APF with many particles
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcjax_fns(lgssm_params)
        pf_post = auxiliary_filter(
            key=jr.PRNGKey(123),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            emissions=emissions,
            num_particles=10_000,
        )
        pf_ll = float(pf_post.marginal_loglik)

        assert pf_ll == pytest.approx(exact_ll, rel=0.05), (
            f'APF log-ML {pf_ll:.2f} vs Kalman {exact_ll:.2f}'
        )


# ---------------------------------------------------------------------------
# Test: flat auxiliary fn → same as bootstrap
# ---------------------------------------------------------------------------


class TestAuxiliaryFlatMatchesBootstrap:
    """APF with log_auxiliary_fn=0 should match bootstrap."""

    def test_auxiliary_flat_auxiliary_matches_bootstrap(
        self, lgssm_params, lgssm_data
    ):
        """With flat auxiliary, APF log-ML ≈ bootstrap log-ML."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, _ = _make_smcjax_fns(lgssm_params)

        def flat_auxiliary_fn(emission, state):
            return jnp.float64(0.0)

        # Run both with same key and particle count
        key = jr.PRNGKey(42)
        n = 5_000

        bpf_post = bootstrap_filter(
            key=key,
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=n,
        )

        apf_post = auxiliary_filter(
            key=key,
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=flat_auxiliary_fn,
            emissions=emissions,
            num_particles=n,
        )

        bpf_ll = float(bpf_post.marginal_loglik)
        apf_ll = float(apf_post.marginal_loglik)

        # Both are Monte Carlo estimates; with N=5000 and T=50,
        # std of log-ML ≈ O(1), so atol=3 is ~3 sigma.
        assert apf_ll == pytest.approx(bpf_ll, abs=3.0), (
            f'APF {apf_ll:.2f} vs bootstrap {bpf_ll:.2f}'
        )


# ---------------------------------------------------------------------------
# Test: ESS trace
# ---------------------------------------------------------------------------


class TestAuxiliaryESSTrace:
    """ESS trace should be reasonable."""

    def test_auxiliary_ess_bounded(self, lgssm_params, lgssm_data):
        """ESS should be between 1 and N at every time step."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcjax_fns(lgssm_params)
        n = 1_000
        pf = auxiliary_filter(
            key=jr.PRNGKey(111),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            emissions=emissions,
            num_particles=n,
        )
        assert jnp.all(pf.ess >= 0.9)  # ESS >= ~1
        assert jnp.all(pf.ess <= n + 0.1)


class TestAuxiliaryLogEvidenceIncrements:
    """log_evidence_increments field should be consistent."""

    def test_log_evidence_increments_shape(self, lgssm_params, lgssm_data):
        """Shape should be (ntime,)."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcjax_fns(lgssm_params)
        pf = auxiliary_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        assert pf.log_evidence_increments.shape == (emissions.shape[0],)

    def test_log_evidence_increments_sum_to_marginal(
        self, lgssm_params, lgssm_data
    ):
        """Increments should sum to marginal_loglik."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcjax_fns(lgssm_params)
        pf = auxiliary_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        total = float(jnp.sum(pf.log_evidence_increments))
        assert total == pytest.approx(float(pf.marginal_loglik), abs=1e-6)

    def test_log_evidence_increments_finite(self, lgssm_params, lgssm_data):
        """All increments should be finite."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcjax_fns(lgssm_params)
        pf = auxiliary_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        assert jnp.all(jnp.isfinite(pf.log_evidence_increments))
