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

import smcx.liu_west as lw
from smcx.exceptions import DegenerateWeightsError
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


def _run_covariance_case(
    parameter_cloud: jax.Array,
) -> lw.LiuWestPosterior:
    """Run one Liu-West propagation for a supplied parameter cloud."""
    dtype = parameter_cloud.dtype
    num_particles = parameter_cloud.shape[0]

    def initial_sampler(key, n):
        del key
        return jnp.zeros((n, 1), dtype=dtype)

    def param_initial_sampler(key, n):
        del key
        assert n == num_particles
        return parameter_cloud

    def transition_sampler(key, state, params):
        del key, params
        return state

    def log_density(emission, state, params):
        del emission, state, params
        return jnp.asarray(0.0, dtype=dtype)

    return liu_west_filter(
        key=jr.key(129),
        initial_sampler=initial_sampler,
        transition_sampler=transition_sampler,
        log_observation_fn=log_density,
        log_auxiliary_fn=log_density,
        param_initial_sampler=param_initial_sampler,
        emissions=jnp.zeros((2, 1), dtype=dtype),
        num_particles=num_particles,
        shrinkage=0.95,
        resampling_threshold=0.0,
    )


class TestLiuWestCovarianceKernel:
    """Parameter perturbations preserve represented covariance support."""

    def test_zero_weight_leading_outlier_does_not_set_moment_anchor(self):
        params = jnp.array([[1e10], [0.0], [1.0]], dtype=jnp.float32)
        weights = jnp.array([0.0, 0.5, 0.5], dtype=jnp.float32)
        shrinkage = jnp.asarray(0.95, dtype=jnp.float32)
        kernel_variance = 1.0 - shrinkage**2

        shrunk, factor = lw._parameter_kernel(
            params,
            weights,
            shrinkage,
            kernel_variance,
        )

        # Five eps cover the centered f32 reductions and eigendecomposition.
        tolerance = float(5 * np.finfo(np.float32).eps)
        np.testing.assert_allclose(
            shrunk[1:],
            np.array([[0.025], [0.975]], dtype=np.float32),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            factor @ factor.T,
            np.array([[0.25 * (1.0 - 0.95**2)]], dtype=np.float32),
            rtol=tolerance,
            atol=tolerance,
        )

    def test_zero_weight_extreme_is_masked_before_moment_arithmetic(self):
        extreme = jnp.asarray(3e38, dtype=jnp.float32)
        params = jnp.array(
            [[extreme], [-extreme], [-extreme]],
            dtype=jnp.float32,
        )
        weights = jnp.array([0.0, 0.5, 0.5], dtype=jnp.float32)
        shrinkage = jnp.asarray(0.95, dtype=jnp.float32)

        shrunk, factor = lw._parameter_kernel(
            params,
            weights,
            shrinkage,
            1.0 - shrinkage**2,
        )

        assert jnp.all(jnp.isfinite(shrunk))
        np.testing.assert_array_equal(shrunk, -extreme)
        np.testing.assert_array_equal(factor, jnp.zeros((1, 1)))

    def test_float32_zero_spread_does_not_drift(self):
        cloud = jnp.zeros((17, 2), dtype=jnp.float32)

        actual = _run_covariance_case(cloud).filtered_params[-1]

        assert actual.dtype == cloud.dtype
        np.testing.assert_array_equal(actual, cloud)

    def test_translated_constant_does_not_create_spread(self):
        cloud = jnp.full((1_000, 2), 1e10, dtype=jnp.float32)

        actual = _run_covariance_case(cloud).filtered_params[-1]

        assert actual.dtype == cloud.dtype
        np.testing.assert_array_equal(actual, cloud)

    def test_rank_deficient_cloud_stays_in_its_support(self):
        first = jnp.array([9_990.0, 10_000.0, 10_010.0], jnp.float32)
        cloud = jnp.stack((first, 0.5 * first), axis=1)

        actual = _run_covariance_case(cloud).filtered_params[-1]

        assert jnp.all(jnp.isfinite(actual))
        # Sixteen eps cover eigensolver and matrix-product rounding.
        tolerance = float(16 * np.finfo(np.float32).eps * np.max(np.abs(cloud)))
        np.testing.assert_allclose(
            actual[:, 1],
            0.5 * actual[:, 0],
            rtol=0.0,
            atol=tolerance,
        )

    def test_near_singular_cloud_remains_finite(self):
        first = jnp.array(
            [9_990.0, 10_000.0, 10_010.0, 10_020.0],
            jnp.float32,
        )
        second = 0.5 * first
        second = second.at[-1].add(jnp.float32(0.01))
        cloud = jnp.stack((first, second), axis=1)

        actual = _run_covariance_case(cloud).filtered_params[-1]

        assert actual.dtype == cloud.dtype
        assert jnp.all(jnp.isfinite(actual))

    def test_ordinary_spread_preserves_kernel_variance(self):
        pattern = jnp.array(
            [[-1.0, -2.0], [-1.0, 2.0], [1.0, -2.0], [1.0, 2.0]],
            jnp.float32,
        )
        cloud = jnp.tile(pattern, (5_000, 1))

        actual = _run_covariance_case(cloud).filtered_params[-1]
        noise = np.asarray(actual - jnp.float32(0.95) * cloud)
        second_moment = np.mean(noise**2, axis=0)
        expected = (1.0 - 0.95**2) * np.array([1.0, 4.0])
        # For Gaussian noise, SE(sample second moment) =
        # sqrt(2 / N) * variance. Five SE is the stochastic-test gate.
        estimator_se = np.sqrt(2.0 / cloud.shape[0]) * expected
        np.testing.assert_array_less(
            np.abs(second_moment - expected),
            5 * estimator_se,
        )


def test_uncompiled_step_matches_compiled_scan():
    _, transition, observation, auxiliary, _ = _make_conjugate_fns()
    num_particles, shrinkage = 16, jnp.asarray(0.95)
    carry = lw._LiuWestStepCarry(
        jnp.zeros((num_particles, 1)),
        jnp.linspace(-1.0, 1.0, num_particles)[:, None],
        jnp.full(num_particles, -math.log(num_particles)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.arange(num_particles, dtype=jnp.int32),
        jnp.asarray(False),
    )
    signature = lw._validate_dense_initial_cloud(carry.particles, 16, name="x")

    def advance(current):
        return lw._liu_west_step(
            current,
            lw._LiuWestStepInput(jnp.asarray([0.5]), None, jnp.int32(1)),
            jr.key(23),
            transition_sampler=transition,
            log_observation_fn=observation,
            log_auxiliary_fn=auxiliary,
            resampling_fn=lw.systematic,
            resampling_threshold=1.1,
            log_num_particles=jnp.asarray(math.log(num_particles)),
            shrinkage=shrinkage,
            kernel_variance=1.0 - shrinkage**2,
            state_signature=signature,
        )

    with jax.disable_jit():
        eager = advance(carry)
    compiled = jax.jit(advance)(carry)
    # Fixed-key tolerance is five float32 eps for CPU/Metal rounding.
    tolerance = float(5 * np.finfo(np.float32).eps)
    for actual, expected in zip(
        jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True
    ):
        np.testing.assert_allclose(
            actual, expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("value", [-jnp.inf, jnp.inf, jnp.nan])
def test_nonfinite_first_stage_raises(value):
    """A discarded invalid look-ahead normalizer still fails the run."""
    initial, transition, observation, _, param_initial = _make_conjugate_fns()

    def invalid_auxiliary(emission, state, params):
        del emission, state, params
        return value

    with pytest.raises(DegenerateWeightsError):
        liu_west_filter(
            jr.key(24),
            initial,
            transition,
            observation,
            invalid_auxiliary,
            param_initial,
            jnp.asarray(CONJUGATE_OBSERVATIONS[:2])[:, None],
            16,
            resampling_threshold=0.0,
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


def test_invalid_ancestor_precedes_induced_degeneracy():
    def initial(key, count):
        del key
        return jnp.arange(count, dtype=jnp.float32)[:, None]

    def transition(key, state, params):
        del key, params
        return state

    def observation(emission, state, params):
        del emission, params
        return jnp.where(state[0] == 3, -jnp.inf, 0.0)

    def invalid_resampler(key, weights, count):
        del key, weights
        return jnp.full(count, count, dtype=jnp.int32)

    with pytest.raises(ValueError, match=r"entries.*\[0, 4\)"):
        liu_west_filter(
            jr.key(159),
            initial,
            transition,
            observation,
            lambda emission, state, params: jnp.asarray(0.0),
            lambda key, count: jnp.zeros((count, 1)),
            jnp.zeros((2, 1)),
            4,
            resampling_fn=invalid_resampler,
            resampling_threshold=1.1,
        )


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
