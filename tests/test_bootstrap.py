# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`smcx.bootstrap_filter` against independent references.

The permanent gates use a dependency-free float64 Kalman recurrence and the
frozen Dynamax fixture in :mod:`tests._lgssm_reference`. A one-time isolated
campaign (2026-07-18; input-aware scalar LGSSM, T=30, N=4096, 128 seeds)
also compared evidence and filtering moments out of process. Mean
``Z_hat / Z_exact`` (Monte Carlo SE) was ``0.993628 (0.006205)`` for smcx,
``0.995324 (0.006113)`` for particles, and ``1.003881 (0.005597)`` for TFP;
all exact and cross-implementation moment discrepancies were below five SE.
The frozen campaign case SHA-256 was
``d59064d711ba96f3d61da207c79b8f1b4526eade25620dd40ec10f6ed47d2689``;
smcx/particles used seeds 20261000--20261127 and TFP used
20261200--20261327. Its complete case preimage, exact target, and runner
settings are retained in :mod:`tests.test_reference_data`.

Pinned external authorities (no code copied; neither is imported here):

* particles 0.4, commit f71e94a21a11c73b58e2d694775b1b1d379b8854,
  MIT source and license:
  https://github.com/nchopin/particles/blob/f71e94a21a11c73b58e2d694775b1b1d379b8854/particles/state_space_models.py#L299-L351
  https://github.com/nchopin/particles/blob/f71e94a21a11c73b58e2d694775b1b1d379b8854/LICENSE
* TensorFlow Probability 0.25.0, commit
  9709569d9c1159dc54154044f679edc4a15bd26b, Apache-2.0 source and license:
  https://github.com/tensorflow/probability/blob/9709569d9c1159dc54154044f679edc4a15bd26b/tensorflow_probability/python/experimental/mcmc/particle_filter.py#L441-L638
  https://github.com/tensorflow/probability/blob/9709569d9c1159dc54154044f679edc4a15bd26b/LICENSE
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from smcx.bootstrap import bootstrap_filter
from tests._kalman import kalman_1d
from tests._lgssm_reference import EXACT_LOG_LIKELIHOOD, REFERENCE_TIMES
from tests._lgssm_reference import FILTERED_MEANS as EXACT_FILTERED_MEANS
from tests._lgssm_reference import FILTERED_VARIANCES as EXACT_FILTERED_VARS
from tests.conftest import _mvn_logpdf, _mvn_sample

# ---------------------------------------------------------------------------
# Helpers to define the LGSSM for smcx
# ---------------------------------------------------------------------------


def _make_smcx_fns(lgssm_params):
    """Build (initial_sampler, transition_sampler, log_obs_fn) closures."""
    m0 = lgssm_params["initial_mean"]
    P0 = lgssm_params["initial_cov"]
    F = lgssm_params["dynamics_weights"]
    Q = lgssm_params["dynamics_cov"]
    H = lgssm_params["emissions_weights"]
    R = lgssm_params["emissions_cov"]

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
    """Bootstrap PF on a linear Gaussian SSM matches the Kalman filter."""

    def test_evidence_and_filtering_moments(self, lgssm_params, lgssm_data):
        """Evidence and selected moments pass committed five-SE gates."""
        _, emissions = lgssm_data

        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
        ratios, means, second_moments = [], [], []
        # Each row is one independent estimator. Therefore the SE of its
        # across-run mean is sample_sd / sqrt(R), with R=20 fixed seeds.
        for seed in range(20):
            post = bootstrap_filter(
                key=jr.key(seed),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=2_048,
            )
            ratios.append(
                math.exp(float(post.marginal_loglik) - EXACT_LOG_LIKELIHOOD)
            )
            weights = np.exp(
                np.asarray(
                    post.filtered_log_weights[REFERENCE_TIMES],
                    dtype=np.float64,
                )
            )
            particles = np.asarray(
                post.filtered_particles[REFERENCE_TIMES, :, 0],
                dtype=np.float64,
            )
            means.append(np.sum(weights * particles, axis=1))
            second_moments.append(np.sum(weights * particles**2, axis=1))

        def assert_five_se(observed, expected):
            values = np.asarray(observed, dtype=np.float64)
            estimator_se = values.std(axis=0, ddof=1) / math.sqrt(
                values.shape[0]
            )
            # 2e-5 is the explicit f32/Metal arithmetic budget.
            np.testing.assert_array_less(
                np.abs(values.mean(axis=0) - expected),
                5 * estimator_se + 2e-5,
            )

        assert_five_se(ratios, 1.0)
        assert_five_se(means, EXACT_FILTERED_MEANS)
        assert_five_se(
            second_moments,
            EXACT_FILTERED_VARS + EXACT_FILTERED_MEANS**2,
        )


# ---------------------------------------------------------------------------
# Test: convergence with increasing particles
# ---------------------------------------------------------------------------


class TestBootstrapConvergence:
    """PF estimates should improve with more particles."""

    def test_log_ml_variance_decreases_at_alpha_0_05(
        self, lgssm_params, lgssm_data
    ):
        """More particles reduce log-evidence variance at fixed alpha."""
        _, emissions = lgssm_data

        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
        variances = []
        # Use disjoint committed seeds so the two samples are independent.
        # A single seeded absolute error need not decrease monotonically.
        campaigns = (
            (256, range(3_000, 3_016)),
            (4_096, range(4_000, 4_016)),
        )
        for num_particles, seeds in campaigns:
            ratios = []
            for seed in seeds:
                pf = bootstrap_filter(
                    key=jr.key(seed),
                    initial_sampler=init_fn,
                    transition_sampler=trans_fn,
                    log_observation_fn=obs_fn,
                    emissions=emissions,
                    num_particles=num_particles,
                )
                ratios.append(
                    math.exp(float(pf.marginal_loglik) - EXACT_LOG_LIKELIHOOD)
                )
            variances.append(np.var(ratios, ddof=1))

        # Under equal normal variances, s_low^2 / s_high^2 ~ F(15, 15).
        # The one-sided alpha=.05 critical value is 2.4034470714953375.
        variance_ratio = variances[0] / variances[1]
        assert variance_ratio > 2.4034470714953375, variances


class TestBootstrapESSTrace:
    """ESS trace should be reasonable."""

    def test_ess_bounded(self, lgssm_params, lgssm_data):
        """ESS should be between 1 and N at every time step."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
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
        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
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
        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
        pf = bootstrap_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        total = float(jnp.sum(pf.log_evidence_increments))
        f64 = jnp.asarray(pf.marginal_loglik).dtype == jnp.float64
        if f64:
            assert total == pytest.approx(float(pf.marginal_loglik), abs=1e-6)
        else:
            # A 50-term f32 reduction accumulates several ulps on Metal.
            assert total == pytest.approx(float(pf.marginal_loglik), rel=1e-5)

    def test_log_evidence_increments_finite(self, lgssm_params, lgssm_data):
        """All increments should be finite."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
        pf = bootstrap_filter(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            log_observation_fn=obs_fn,
            emissions=emissions,
            num_particles=1_000,
        )
        assert jnp.all(jnp.isfinite(pf.log_evidence_increments))


class TestBootstrapInputs:
    """Per-step exogenous inputs reach every model callback."""

    def test_inputs_reach_initial_transition_and_observation_callbacks(self):
        inputs = jnp.array([1.0, 2.0, 3.0])
        emissions = jnp.array([[2.0], [5.0], [9.0]])

        def initial_sampler(key, n, input_t):
            del key
            return jnp.full((n, 1), input_t[0])

        def transition_sampler(key, state, input_t):
            del key
            return state + input_t

        def log_observation_fn(emission, state, input_t):
            error = emission[0] - state[0] - input_t[0]
            return -0.5 * error**2

        post = bootstrap_filter(
            key=jr.key(0),
            initial_sampler=initial_sampler,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation_fn,
            emissions=emissions,
            num_particles=4,
            inputs=inputs,
        )

        expected = jnp.array([1.0, 3.0, 6.0])
        expected_cloud = jnp.broadcast_to(expected[:, None], (3, 4))
        assert jnp.array_equal(post.filtered_particles[:, :, 0], expected_cloud)
        assert post.marginal_loglik == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("inputs", "message"),
        [
            (jnp.zeros((3, 1, 1)), "inputs must have shape"),
            (jnp.zeros((2, 1)), "inputs must have leading dimension"),
        ],
    )
    def test_inputs_reject_malformed_shapes_at_public_entry(
        self, inputs, message
    ):
        def initial_sampler(key, n, input_t):
            del key, input_t
            return jnp.zeros((n, 1))

        def transition_sampler(key, state, input_t):
            del key, input_t
            return state

        def log_observation_fn(emission, state, input_t):
            del emission, state, input_t
            return jnp.array(0.0)

        with pytest.raises(ValueError, match=message):
            bootstrap_filter(
                key=jr.key(0),
                initial_sampler=initial_sampler,
                transition_sampler=transition_sampler,
                log_observation_fn=log_observation_fn,
                emissions=jnp.zeros((3, 1)),
                num_particles=4,
                inputs=inputs,
            )

    def test_controlled_lgssm_matches_kalman_oracle(self):
        a, b = 0.9, 0.7
        q, r = 0.25, 1.0
        m0, p0 = 0.0, 1.0
        num_timesteps = 40
        rng = np.random.default_rng(9)
        inputs = rng.normal(size=num_timesteps)
        states = np.empty(num_timesteps)
        states[0] = rng.normal(m0, math.sqrt(p0))
        for t in range(1, num_timesteps):
            states[t] = (
                a * states[t - 1]
                + b * inputs[t]
                + rng.normal(0.0, math.sqrt(q))
            )
        observations = states + rng.normal(
            0.0, math.sqrt(r), size=num_timesteps
        )
        exact_loglik, _, _ = kalman_1d(
            observations, a, q, r, m0, p0, b=b, u=inputs
        )

        def initial_sampler(key, n, input_t):
            del input_t
            return m0 + math.sqrt(p0) * jr.normal(key, (n, 1))

        def transition_sampler(key, state, input_t):
            return (
                a * state
                + b * input_t
                + math.sqrt(q) * jr.normal(key, state.shape)
            )

        def log_observation_fn(emission, state, input_t):
            del input_t
            return -0.5 * (
                math.log(2.0 * math.pi * r) + (emission[0] - state[0]) ** 2 / r
            )

        emissions_arr = jnp.asarray(observations)[:, None]
        inputs_arr = jnp.asarray(inputs)
        estimates = np.asarray([
            bootstrap_filter(
                jr.key(seed),
                initial_sampler,
                transition_sampler,
                log_observation_fn,
                emissions_arr,
                5_000,
                inputs=inputs_arr,
            ).marginal_loglik
            for seed in range(8)
        ])

        # For R independent estimates, SE(mean(log Z-hat)) = s/sqrt(R).
        # The lower bound includes the lognormal approximation to Jensen
        # bias, -s^2/2; five SE gives an MC-error-honest tolerance.
        sd = estimates.std(ddof=1)
        se = sd / math.sqrt(estimates.size)
        error = estimates.mean() - exact_loglik
        assert -(5.0 * se + 0.5 * sd**2) <= error <= 5.0 * se

    def test_inputs_remain_dynamic_under_jit(self):
        def initial_sampler(key, n, input_t):
            del key
            return jnp.full((n, 1), input_t[0])

        def transition_sampler(key, state, input_t):
            del key
            return state + input_t

        def log_observation_fn(emission, state, input_t):
            return 0.0 * (emission[0] + state[0] + input_t[0])

        @jax.jit
        def run(input_values):
            post = bootstrap_filter(
                jr.key(0),
                initial_sampler,
                transition_sampler,
                log_observation_fn,
                jnp.zeros((4, 1)),
                4,
                inputs=input_values,
                store_history=False,
            )
            return post.filtered_particles[0, 0, 0]

        zero_result = run(jnp.zeros((4, 1)))
        one_result = run(jnp.ones((4, 1)))
        assert jnp.array_equal(zero_result, jnp.array(0.0))
        assert jnp.array_equal(one_result, jnp.array(4.0))

    def test_ignored_inputs_preserve_key_stream_and_numerics(self):
        emissions = jnp.linspace(-0.5, 0.5, 5)[:, None]

        def initial_sampler(key, n):
            return jr.normal(key, (n, 1))

        def transition_sampler(key, state):
            return 0.8 * state + 0.3 * jr.normal(key, state.shape)

        def log_observation_fn(emission, state):
            return -0.5 * (emission[0] - state[0]) ** 2

        def initial_sampler_u(key, n, input_t):
            del input_t
            return initial_sampler(key, n)

        def transition_sampler_u(key, state, input_t):
            del input_t
            return transition_sampler(key, state)

        def log_observation_fn_u(emission, state, input_t):
            del input_t
            return log_observation_fn(emission, state)

        legacy = bootstrap_filter(
            jr.key(11),
            initial_sampler,
            transition_sampler,
            log_observation_fn,
            emissions,
            32,
        )
        input_aware = bootstrap_filter(
            jr.key(11),
            initial_sampler_u,
            transition_sampler_u,
            log_observation_fn_u,
            emissions,
            32,
            inputs=jnp.zeros((5, 2)),
        )

        for legacy_field, input_field in zip(
            jax.tree_util.tree_leaves(legacy),
            jax.tree_util.tree_leaves(input_aware),
            strict=True,
        ):
            assert jnp.array_equal(legacy_field, input_field)
