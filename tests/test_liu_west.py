# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.liu_west_filter.

Validates parameter recovery on a linear Gaussian SSM with unknown
observation noise variance, degeneracy to APF with fixed parameters,
and the effect of shrinkage on posterior spread.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from smcjax.liu_west import liu_west_filter
from tests.conftest import _mvn_logpdf, _mvn_sample

# ---------------------------------------------------------------------------
# Test model: 1-D LGSSM with unknown observation noise variance
#
#   z_0  ~ N(0, 1)
#   z_t  = 0.9 * z_{t-1} + eps,  eps ~ N(0, 0.25)
#   y_t  = z_t + eta,            eta ~ N(0, sigma_y^2)
#
# Parameter to estimate: sigma_y^2 (true value = 1.0)
# ---------------------------------------------------------------------------


def _make_liu_west_fns():
    """Build closures for Liu-West filter on LGSSM with unknown obs noise."""
    m0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    F = jnp.array([[0.9]])
    Q = jnp.array([[0.25]])
    H = jnp.array([[1.0]])

    def initial_sampler(key, n):
        return _mvn_sample(key, m0, P0, shape=(n,))

    def transition_sampler(key, state, params):
        mean = (F @ state[:, None]).squeeze(-1)
        return _mvn_sample(key, mean, Q)

    def log_observation_fn(emission, state, params):
        sigma_y_sq = jnp.exp(params[0])
        R = jnp.array([[sigma_y_sq]], dtype=jnp.float64)
        mean = (H @ state[:, None]).squeeze(-1)
        return _mvn_logpdf(emission, mean, R)

    def log_auxiliary_fn(emission, state, params):
        sigma_y_sq = jnp.exp(params[0])
        R = jnp.array([[sigma_y_sq]], dtype=jnp.float64)
        predicted_mean = (H @ F @ state[:, None]).squeeze(-1)
        return _mvn_logpdf(emission, predicted_mean, R)

    def param_initial_sampler(key, n):
        # Prior on log(sigma_y^2) ~ N(0, 0.5^2)
        return jnp.float64(0.5) * jr.normal(key, (n, 1))

    return (
        initial_sampler,
        transition_sampler,
        log_observation_fn,
        log_auxiliary_fn,
        param_initial_sampler,
    )


class TestLiuWestRecoversParams:
    """Liu-West filter should recover known parameters."""

    def test_liu_west_recovers_known_params(self, lgssm_params, lgssm_data):
        """Posterior param mean should be near true value."""
        _, emissions = lgssm_data
        (
            init_fn,
            trans_fn,
            obs_fn,
            aux_fn,
            param_init_fn,
        ) = _make_liu_west_fns()

        post = liu_west_filter(
            key=jr.PRNGKey(42),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            param_initial_sampler=param_init_fn,
            emissions=emissions,
            num_particles=5_000,
            shrinkage=0.95,
        )

        # True log(sigma_y^2) = log(1.0) = 0.0
        # Get final time step parameter posterior mean
        from smcjax.weights import normalize

        final_weights = normalize(post.filtered_log_weights[-1])
        final_params = post.filtered_params[-1]  # (N, 1)
        posterior_mean = float(
            jnp.sum(final_weights[:, None] * final_params, axis=0)[0]
        )

        # log(sigma_y^2) should be near 0 (= log(1.0))
        assert posterior_mean == pytest.approx(0.0, abs=0.5), (
            f'Posterior mean log(sigma_y^2) = {posterior_mean:.3f}, expected ~0.0'
        )


class TestLiuWestFixedParamsMatchesAPF:
    """With delta prior (fixed params), Liu-West should match APF."""

    def test_liu_west_fixed_params_matches_apf(self, lgssm_params, lgssm_data):
        """Log-ML with delta prior on true params ≈ APF log-ML."""
        from smcjax.auxiliary import auxiliary_filter

        _, emissions = lgssm_data

        m0 = lgssm_params['initial_mean']
        P0 = lgssm_params['initial_cov']
        F = lgssm_params['dynamics_weights']
        Q = lgssm_params['dynamics_cov']
        H = lgssm_params['emissions_weights']
        R = lgssm_params['emissions_cov']

        # APF closures (no params)
        def apf_init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def apf_trans(key, state):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def apf_obs(emission, state):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def apf_aux(emission, state):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        # Liu-West closures (params ignored since fixed)
        def lw_init(key, n):
            return apf_init(key, n)

        def lw_trans(key, state, params):
            return apf_trans(key, state)

        def lw_obs(emission, state, params):
            return apf_obs(emission, state)

        def lw_aux(emission, state, params):
            return apf_aux(emission, state)

        def lw_param_init(key, n):
            # Delta prior: all particles get same param (log(1.0) = 0.0)
            return jnp.zeros((n, 1))

        key = jr.PRNGKey(99)
        n = 5_000

        apf_post = auxiliary_filter(
            key=key,
            initial_sampler=apf_init,
            transition_sampler=apf_trans,
            log_observation_fn=apf_obs,
            log_auxiliary_fn=apf_aux,
            emissions=emissions,
            num_particles=n,
        )

        lw_post = liu_west_filter(
            key=key,
            initial_sampler=lw_init,
            transition_sampler=lw_trans,
            log_observation_fn=lw_obs,
            log_auxiliary_fn=lw_aux,
            param_initial_sampler=lw_param_init,
            emissions=emissions,
            num_particles=n,
            shrinkage=0.99,  # minimal smoothing
        )

        apf_ll = float(apf_post.marginal_loglik)
        lw_ll = float(lw_post.marginal_loglik)

        assert lw_ll == pytest.approx(apf_ll, abs=5.0), (
            f'Liu-West {lw_ll:.2f} vs APF {apf_ll:.2f}'
        )


class TestLiuWestShrinkage:
    """Shrinkage parameter should affect posterior spread."""

    def test_liu_west_shrinkage_affects_spread(self, lgssm_params, lgssm_data):
        """Lower shrinkage → wider parameter posterior."""
        _, emissions = lgssm_data
        (
            init_fn,
            trans_fn,
            obs_fn,
            aux_fn,
            param_init_fn,
        ) = _make_liu_west_fns()

        spreads = []
        for a in [0.80, 0.99]:
            post = liu_west_filter(
                key=jr.PRNGKey(7),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                log_auxiliary_fn=aux_fn,
                param_initial_sampler=param_init_fn,
                emissions=emissions,
                num_particles=2_000,
                shrinkage=a,
            )
            # Compute weighted variance of final param posterior
            from smcjax.weights import normalize

            w = normalize(post.filtered_log_weights[-1])
            p = post.filtered_params[-1, :, 0]
            mean = jnp.sum(w * p)
            var = jnp.sum(w * (p - mean) ** 2)
            spreads.append(float(var))

        # Lower shrinkage (0.80) should give wider spread
        assert spreads[0] > spreads[1], (
            f'Spread with a=0.80: {spreads[0]:.4f}, a=0.99: {spreads[1]:.4f}'
        )


class TestLiuWestJIT:
    """Liu-West filter should be JIT-compilable."""

    def test_liu_west_jit_compiles(self):
        m0 = jnp.array([0.0])
        P0 = jnp.array([[1.0]])
        F = jnp.array([[0.9]])
        Q = jnp.array([[0.25]])
        H = jnp.array([[1.0]])

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            R = jnp.array([[jnp.exp(params[0])]])
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            R = jnp.array([[jnp.exp(params[0])]])
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        emissions = jnp.ones((10, 1))

        @jax.jit
        def run(key):
            return liu_west_filter(
                key=key,
                initial_sampler=init,
                transition_sampler=trans,
                log_observation_fn=obs,
                log_auxiliary_fn=aux,
                param_initial_sampler=param_init,
                emissions=emissions,
                num_particles=50,
                shrinkage=0.95,
            )

        result = run(jr.PRNGKey(0))
        assert result.filtered_particles.shape == (10, 50, 1)
        assert result.filtered_params.shape == (10, 50, 1)
        assert jnp.isfinite(result.marginal_loglik)


class TestLiuWestLogEvidenceIncrements:
    """log_evidence_increments field should be consistent."""

    def test_log_evidence_increments_shape(self, lgssm_params, lgssm_data):
        """Shape should be (ntime,)."""
        _, emissions = lgssm_data
        (
            init_fn,
            trans_fn,
            obs_fn,
            aux_fn,
            param_init_fn,
        ) = _make_liu_west_fns()

        post = liu_west_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            param_initial_sampler=param_init_fn,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )
        assert post.log_evidence_increments.shape == (emissions.shape[0],)

    def test_log_evidence_increments_sum_to_marginal(
        self, lgssm_params, lgssm_data
    ):
        """Increments should sum to marginal_loglik."""
        _, emissions = lgssm_data
        (
            init_fn,
            trans_fn,
            obs_fn,
            aux_fn,
            param_init_fn,
        ) = _make_liu_west_fns()

        post = liu_west_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            param_initial_sampler=param_init_fn,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )
        total = float(jnp.sum(post.log_evidence_increments))
        assert total == pytest.approx(float(post.marginal_loglik), abs=1e-6)

    def test_log_evidence_increments_finite(self, lgssm_params, lgssm_data):
        """All increments should be finite."""
        _, emissions = lgssm_data
        (
            init_fn,
            trans_fn,
            obs_fn,
            aux_fn,
            param_init_fn,
        ) = _make_liu_west_fns()

        post = liu_west_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            log_auxiliary_fn=aux_fn,
            param_initial_sampler=param_init_fn,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )
        assert jnp.all(jnp.isfinite(post.log_evidence_increments))
