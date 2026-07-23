# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for Liu-West filtering against a conjugate Gaussian reference.

Algorithm: Liu and West (2001),
https://doi.org/10.1007/978-1-4757-3437-9_10.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from smcx.liu_west import liu_west_filter
from tests.conftest import _mvn_logpdf, _mvn_sample

CONJUGATE_OBSERVATIONS = np.array([
    0.8734662860069428,
    0.9899723320744257,
    0.37135331956539325,
    1.2533790653278962,
    1.9729231490940866,
    -1.7452417576798576,
    0.9513622280736571,
    -0.2966459285882617,
    0.21853507678488962,
    2.4276737220149176,
    -1.7605528589701298,
    0.7853769754843833,
    2.233763064237592,
    0.9343393602611758,
    1.188773811596476,
    0.9904719964866113,
    0.10964724312826299,
    0.990649780519261,
    0.8309278983072693,
    0.7997507118602487,
    0.6251269352868399,
    0.8784983383012326,
    0.769817280537482,
    0.7284232125271259,
    0.9645402294462921,
    1.7320430772147475,
    1.2461958903087298,
    0.6305917672806888,
    0.46055383962628593,
    1.4727239413660862,
])
CONJUGATE_EXACT_MEAN = 0.7809514783860596
CONJUGATE_EXACT_VARIANCE = 0.01887601887601888
CONJUGATE_EXACT_LOGZ = -43.459162306696996


def _normal_logpdf_1d(value, mean, variance):
    return -0.5 * (
        jnp.log(2.0 * jnp.pi * variance) + (value - mean) ** 2 / variance
    )


def _make_conjugate_fns():
    """Build the independent-state conjugate Liu-West validation model."""
    prior_var = 4.0
    state_var = 0.35
    obs_var = 0.20
    marginal_var = state_var + obs_var

    def initial_sampler(key, n):
        return math.sqrt(state_var) * jr.normal(key, (n, 1))

    def param_initial_sampler(key, n):
        return math.sqrt(prior_var) * jr.normal(key, (n, 1))

    def transition_sampler(key, state, params):
        del state
        return params + math.sqrt(state_var) * jr.normal(key, params.shape)

    def log_observation_fn(emission, state, params):
        del params
        return _normal_logpdf_1d(emission[0], state[0], obs_var)

    def log_auxiliary_fn(emission, state, params):
        del state
        return _normal_logpdf_1d(emission[0], params[0], marginal_var)

    return (
        initial_sampler,
        transition_sampler,
        log_observation_fn,
        log_auxiliary_fn,
        param_initial_sampler,
    )


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


class TestLiuWestConjugateReference:
    """Liu-West output is characterized against an exact posterior."""

    def test_evidence_and_parameter_moments_pass_five_se_gate(self):
        """Twelve committed runs match exact evidence and two moments."""
        init_fn, trans_fn, obs_fn, aux_fn, param_init_fn = _make_conjugate_fns()
        emissions = jnp.asarray(CONJUGATE_OBSERVATIONS)[:, None]
        rows = []
        # Each row is an independent full filter. Thus the estimator SE of
        # the across-run mean is sample_sd / sqrt(R), with R=12.
        for seed in range(12):
            post = liu_west_filter(
                key=jr.key(seed),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                log_auxiliary_fn=aux_fn,
                param_initial_sampler=param_init_fn,
                emissions=emissions,
                num_particles=5_000,
                shrinkage=0.95,
            )
            weights = np.exp(
                np.asarray(post.filtered_log_weights[-1], np.float64)
            )
            params = np.asarray(post.filtered_params[-1, :, 0], np.float64)
            rows.append([
                math.exp(float(post.marginal_loglik) - CONJUGATE_EXACT_LOGZ),
                weights @ params,
                weights @ (params**2),
            ])

        values = np.asarray(rows)
        expected = np.array([
            1.0,
            CONJUGATE_EXACT_MEAN,
            CONJUGATE_EXACT_VARIANCE + CONJUGATE_EXACT_MEAN**2,
        ])
        estimator_se = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        # 2e-5 is the explicit f32/Metal arithmetic budget.
        np.testing.assert_array_less(
            np.abs(values.mean(axis=0) - expected),
            5 * estimator_se + 2e-5,
        )


class TestLiuWestFixedParamsMatchesAPF:
    """With delta prior (fixed params), Liu-West should match APF."""

    def test_liu_west_fixed_params_matches_apf(self, lgssm_params, lgssm_data):
        """Log-ML with delta prior on true params ≈ APF log-ML."""
        from smcx.auxiliary import auxiliary_filter

        _, emissions = lgssm_data

        m0 = lgssm_params["initial_mean"]
        P0 = lgssm_params["initial_cov"]
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

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
            f"Liu-West {lw_ll:.2f} vs APF {apf_ll:.2f}"
        )


class TestLiuWestShrinkage:
    """Shrinkage parameter should affect posterior spread."""

    def test_liu_west_shrinkage_affects_spread(self, lgssm_params, lgssm_data):
        """Lower shrinkage → wider parameter posterior on average."""
        from smcx.weights import normalize

        _, emissions = lgssm_data
        (
            init_fn,
            trans_fn,
            obs_fn,
            aux_fn,
            param_init_fn,
        ) = _make_liu_west_fns()

        def _spread(seed, a):
            post = liu_west_filter(
                key=jr.PRNGKey(seed),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                log_auxiliary_fn=aux_fn,
                param_initial_sampler=param_init_fn,
                emissions=emissions,
                num_particles=2_000,
                shrinkage=a,
            )
            w = normalize(post.filtered_log_weights[-1])
            p = post.filtered_params[-1, :, 0]
            mean = jnp.sum(w * p)
            return float(jnp.sum(w * (p - mean) ** 2))

        # Average across several seeds: a single particle filter run is
        # too noisy to compare two shrinkage settings reliably.
        seeds = list(range(8))
        spread_low = sum(_spread(s, 0.80) for s in seeds) / len(seeds)
        spread_high = sum(_spread(s, 0.99) for s in seeds) / len(seeds)

        assert spread_low > spread_high, (
            f"Mean spread (a=0.80): {spread_low:.4f}, "
            f"(a=0.99): {spread_high:.4f}"
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
        # float32 (Metal) carries ~7 significant digits; float64 gets
        # the sharp absolute bound.
        f64 = jnp.asarray(post.marginal_loglik).dtype == jnp.float64
        if f64:
            assert total == pytest.approx(float(post.marginal_loglik), abs=1e-6)
        else:
            assert total == pytest.approx(float(post.marginal_loglik), rel=1e-5)

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


class TestLiuWestInputs:
    """Inputs follow parameters in every Liu-West callback."""

    def test_inputs_condition_initial_state_and_follow_params(self):
        inputs = jnp.array([1.0, 2.0, 3.0])
        emissions = jnp.array([[2.0], [5.0], [9.0]])

        def initial_sampler(key, n, input_t):
            del key
            return jnp.full((n, 1), input_t[0])

        def param_initial_sampler(key, n):
            del key
            return jnp.zeros((n, 2))

        def transition_sampler(key, state, params, input_t):
            del key
            return state + input_t + 0.0 * params[1]

        def log_observation_fn(emission, state, params, input_t):
            error = emission[0] - state[0] - input_t[0]
            return -0.5 * error**2 + 0.0 * params[1]

        def log_auxiliary_fn(emission, state, params, input_t):
            del emission, state
            return 0.0 * params[1] + 0.0 * input_t[0]

        post = liu_west_filter(
            key=jr.key(0),
            initial_sampler=initial_sampler,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation_fn,
            log_auxiliary_fn=log_auxiliary_fn,
            param_initial_sampler=param_initial_sampler,
            emissions=emissions,
            num_particles=4,
            resampling_threshold=0.0,
            inputs=inputs,
        )

        expected = jnp.broadcast_to(jnp.array([1.0, 3.0, 6.0])[:, None], (3, 4))
        assert jnp.array_equal(post.filtered_particles[:, :, 0], expected)
        assert post.marginal_loglik == pytest.approx(0.0)
