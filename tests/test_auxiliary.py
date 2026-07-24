# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`smcx.auxiliary_filter` against an exact LGSSM oracle."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from smcx._numerics import _neumaier_add
from smcx.auxiliary import auxiliary_filter
from smcx.bootstrap import bootstrap_filter
from tests._lgssm_reference import EXACT_LOG_LIKELIHOOD, REFERENCE_TIMES
from tests._lgssm_reference import FILTERED_MEANS as EXACT_FILTERED_MEANS
from tests._lgssm_reference import FILTERED_VARIANCES as EXACT_FILTERED_VARS
from tests.conftest import _mvn_logpdf, _mvn_sample

# ---------------------------------------------------------------------------
# Helpers to define the LGSSM for smcx APF
# ---------------------------------------------------------------------------


def _make_smcx_fns(lgssm_params):
    """Build (initial, transition, log_obs, log_aux) closures."""
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

    def test_evidence_and_filtering_moments(self, lgssm_params, lgssm_data):
        """Evidence and selected moments pass committed five-SE gates."""
        _, emissions = lgssm_data

        init_fn, trans_fn, obs_fn, aux_fn = _make_smcx_fns(lgssm_params)
        ratios, means, second_moments = [], [], []
        # Each row is independent, so SE(mean) = sample_sd / sqrt(R), R=20.
        for seed in range(20):
            post = auxiliary_filter(
                key=jr.key(seed),
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                log_auxiliary_fn=aux_fn,
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
# Test: flat auxiliary fn → same as bootstrap
# ---------------------------------------------------------------------------


class TestAuxiliaryFlatMatchesBootstrap:
    """APF with log_auxiliary_fn=0 should match bootstrap."""

    def test_auxiliary_flat_auxiliary_matches_bootstrap(
        self, lgssm_params, lgssm_data
    ):
        """With flat auxiliary, APF log-ML ≈ bootstrap log-ML."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, _ = _make_smcx_fns(lgssm_params)

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
            f"APF {apf_ll:.2f} vs bootstrap {bpf_ll:.2f}"
        )


def test_uncompiled_step_matches_public_and_compiled_scan(
    lgssm_params, lgssm_data
):
    """The pure APF step preserves the fixed-key public scan contract."""
    from smcx._utils import _validate_particle_cloud
    from smcx.auxiliary import (
        _auxiliary_step,
        _AuxiliaryStepCarry,
        _AuxiliaryStepInput,
    )
    from smcx.containers import ParticleState
    from smcx.resampling import systematic

    _, emissions = lgssm_data
    emissions = emissions[:2]
    initial, transition, observation, auxiliary = _make_smcx_fns(lgssm_params)
    key, num_particles = jr.key(19), 16
    expected = auxiliary_filter(
        key,
        initial,
        transition,
        observation,
        auxiliary,
        emissions,
        num_particles,
        resampling_threshold=1.1,
    )
    state = ParticleState(
        expected.filtered_particles[0],
        expected.filtered_log_weights[0],
        expected.log_evidence_increments[0],
    )
    carry = _AuxiliaryStepCarry(
        state,
        jnp.zeros_like(state.log_marginal_likelihood),
        expected.ancestors[0],
    )
    step_input = _AuxiliaryStepInput(
        emissions[1], None, jnp.asarray(1, dtype=jnp.int32)
    )
    signature = _validate_particle_cloud(
        state.particles, num_particles, name="initial_sampler output"
    )

    def advance(current, current_input, current_key):
        return _auxiliary_step(
            current,
            current_input,
            current_key,
            transition_sampler=transition,
            log_observation_fn=observation,
            log_auxiliary_fn=auxiliary,
            resampling_fn=systematic,
            resampling_threshold=1.1,
            log_num_particles=jnp.asarray(math.log(num_particles)),
            state_signature=signature,
        )

    step_key = jr.split(jr.split(key)[0], 1)[0]
    with jax.disable_jit():
        eager = advance(carry, step_input, step_key)
    compiled = jax.jit(advance)(carry, step_input, step_key)
    record = tuple(field[1] for field in expected[1:])
    expected_total, expected_correction = _neumaier_add(
        expected.log_evidence_increments[0],
        jnp.zeros_like(expected.log_evidence_increments[0]),
        expected.log_evidence_increments[1],
    )
    expected_leaves = (
        *record[:2],
        expected_total,
        expected_correction,
        record[2],
        *record,
    )
    # Fixed keys remove MC error; five f32 eps covers compiler rounding.
    tolerance = float(5 * np.finfo(np.float32).eps)
    for actual in (eager, compiled):
        for value, target in zip(
            jax.tree.leaves(actual), expected_leaves, strict=True
        ):
            np.testing.assert_allclose(
                value, target, rtol=tolerance, atol=tolerance
            )


# ---------------------------------------------------------------------------
# Test: ESS trace
# ---------------------------------------------------------------------------


class TestAuxiliaryESSTrace:
    """ESS trace should be reasonable."""

    def test_auxiliary_ess_bounded(self, lgssm_params, lgssm_data):
        """ESS should be between 1 and N at every time step."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcx_fns(lgssm_params)
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
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcx_fns(lgssm_params)
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
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcx_fns(lgssm_params)
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
        # float32 (Metal) carries ~7 significant digits; float64 gets
        # the sharp absolute bound.
        f64 = jnp.asarray(pf.marginal_loglik).dtype == jnp.float64
        if f64:
            assert total == pytest.approx(float(pf.marginal_loglik), abs=1e-6)
        else:
            assert total == pytest.approx(float(pf.marginal_loglik), rel=1e-5)

    def test_log_evidence_increments_finite(self, lgssm_params, lgssm_data):
        """All increments should be finite."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn, aux_fn = _make_smcx_fns(lgssm_params)
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


class TestAuxiliaryInputs:
    """Input-aware APF preserves its bootstrap reduction."""

    def test_flat_auxiliary_matches_input_aware_bootstrap(self):
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

        def flat_auxiliary_fn(emission, state, input_t):
            del emission, state
            return 0.0 * input_t[0]

        bootstrap = bootstrap_filter(
            jr.key(0),
            initial_sampler,
            transition_sampler,
            log_observation_fn,
            emissions,
            4,
            resampling_threshold=0.0,
            inputs=inputs,
        )
        auxiliary = auxiliary_filter(
            jr.key(0),
            initial_sampler,
            transition_sampler,
            log_observation_fn,
            flat_auxiliary_fn,
            emissions,
            4,
            resampling_threshold=0.0,
            inputs=inputs,
        )

        assert jnp.array_equal(
            auxiliary.filtered_particles, bootstrap.filtered_particles
        )
        assert jnp.array_equal(
            auxiliary.filtered_log_weights, bootstrap.filtered_log_weights
        )
        assert auxiliary.marginal_loglik == bootstrap.marginal_loglik
