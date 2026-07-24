# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tempered SMC tests against a conjugate Gaussian target.

Conjugate ground truth: prior N(0, s0^2 I_d), likelihood
N(y_obs; x, sl^2 I_d) => log Z = sum_i log N(y_i; 0, s0^2 + sl^2)
exactly, and the posterior is Gaussian with known moments.

Algorithm: Del Moral, Doucet, and Jasra (2006),
https://doi.org/10.1111/j.1467-9868.2006.00553.x.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

D = 3
S0, SL = 2.0, 0.5
Y_OBS = np.array([1.0, -0.7, 0.4])

POST_VAR = 1.0 / (1.0 / S0**2 + 1.0 / SL**2)
POST_MEAN = POST_VAR * Y_OBS / SL**2
LOGZ_TRUE = float(
    np.sum(
        -0.5
        * (np.log(2 * np.pi * (S0**2 + SL**2)) + Y_OBS**2 / (S0**2 + SL**2))
    )
)


def _model():
    y = jnp.asarray(Y_OBS)

    def init(key, n):
        return S0 * jr.normal(key, (n, D))

    def log_prior(x):
        return -0.5 * jnp.sum(math.log(2 * math.pi * S0**2) + (x / S0) ** 2)

    def log_lik(x):
        return -0.5 * jnp.sum(
            math.log(2 * math.pi * SL**2) + ((y - x) / SL) ** 2
        )

    return init, log_prior, log_lik


def _run(seed, n=4000, **kw):
    init, log_prior, log_lik = _model()
    return smcx.temper(jr.key(seed), init, log_prior, log_lik, n, **kw)


def _small_tempering_model():
    observation = jnp.array([0.25], dtype=jnp.float64)

    def init(_key, n):
        return jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)[:, None]

    def log_prior(x):
        return -0.5 * jnp.sum(x**2)

    def log_lik(x):
        return -0.5 * jnp.sum((observation - x) ** 2 / 0.7)

    return init, log_prior, log_lik


class _MutationState(NamedTuple):
    position: jax.Array
    logdensity: jax.Array
    step_index: jax.Array


class _MutationInfo(NamedTuple):
    acceptance_rate: jax.Array
    is_accepted: jax.Array


def _mutation_init(position, tempered_logdensity_fn):
    return _MutationState(
        position,
        tempered_logdensity_fn(position),
        jnp.zeros((), dtype=jnp.int32),
    )


def _mutation_step(key, state, tempered_logdensity_fn):
    proposal_key, accept_key = jr.split(key)
    scale = 0.15 + 0.05 * state.step_index.astype(state.position.dtype)
    proposal = state.position + scale * jr.normal(
        proposal_key, state.position.shape
    )
    proposal_logdensity = tempered_logdensity_fn(proposal)
    log_ratio = proposal_logdensity - state.logdensity
    acceptance_rate = jnp.exp(jnp.minimum(0.0, log_ratio))
    is_accepted = jr.uniform(accept_key) < acceptance_rate
    next_state = _MutationState(
        jnp.where(is_accepted, proposal, state.position),
        jnp.where(is_accepted, proposal_logdensity, state.logdensity),
        state.step_index + 1,
    )
    return next_state, _MutationInfo(acceptance_rate, is_accepted)


def _bad_mutation_init(position, tempered_logdensity_fn):
    state = _mutation_init(position, tempered_logdensity_fn)
    return state._replace(position=position[None])


def _bad_mutation_step(key, state, tempered_logdensity_fn):
    next_state, info = _mutation_step(key, state, tempered_logdensity_fn)
    return next_state, info._replace(acceptance_rate=jnp.ones(2))


class TestMutationCallback:
    """Caller-owned mutation state composes with tempering and JIT."""

    def test_stateful_kernel_matches_eager_and_compiled_execution(self):
        init, log_prior, log_lik = _small_tempering_model()

        def run():
            return smcx.temper(
                jr.key(41),
                init,
                log_prior,
                log_lik,
                8,
                num_mcmc_steps=3,
                target_ess=0.6,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=_mutation_step,
            )

        with jax.disable_jit():
            eager = run()
        compiled = run()

        for eager_value, compiled_value in zip(eager, compiled, strict=True):
            np.testing.assert_allclose(
                eager_value, compiled_value, rtol=2e-6, atol=2e-6
            )
        assert np.all(np.asarray(compiled.acceptance_rates) >= 0.0)
        assert np.all(np.asarray(compiled.acceptance_rates) <= 1.0)

    @pytest.mark.parametrize(
        "callbacks",
        [
            {"mutation_init_fn": _mutation_init},
            {"mutation_step_fn": _mutation_step},
        ],
    )
    def test_mutation_callbacks_must_be_supplied_together(self, callbacks):
        init, log_prior, log_lik = _small_tempering_model()
        with pytest.raises(ValueError, match="must be supplied together"):
            smcx.temper(
                jr.key(42),
                init,
                log_prior,
                log_lik,
                5,
                **callbacks,
            )

    @pytest.mark.parametrize(
        ("initialize", "step", "message"),
        [
            (_bad_mutation_init, _mutation_step, "position must have shape"),
            (_mutation_init, _bad_mutation_step, "must be a scalar float"),
        ],
    )
    def test_malformed_mutation_contract_raises(
        self, initialize, step, message
    ):
        init, log_prior, log_lik = _small_tempering_model()
        with pytest.raises(ValueError, match=message):
            smcx.temper(
                jr.key(43),
                init,
                log_prior,
                log_lik,
                5,
                mutation_init_fn=initialize,
                mutation_step_fn=step,
            )

    def test_requires_at_least_one_mutation_step(self):
        init, log_prior, log_lik = _small_tempering_model()
        with pytest.raises(ValueError, match="num_mcmc_steps must be >= 1"):
            smcx.temper(
                jr.key(44),
                init,
                log_prior,
                log_lik,
                5,
                num_mcmc_steps=0,
            )


class TestEvidence:
    """MC-calibrated gate against the exact conjugate evidence."""

    def test_evidence_and_posterior_moments_r12(self):
        rows = []
        # Each row comes from an independent complete SMC run. Therefore
        # SE(mean) = the across-run sample SD / sqrt(R), with R=12.
        for seed in range(12):
            post = _run(seed)
            particles = np.asarray(post.particles, dtype=np.float64)
            rows.append([
                np.exp(float(post.marginal_loglik) - LOGZ_TRUE),
                *particles.mean(axis=0).tolist(),
                *(particles**2).mean(axis=0).tolist(),
            ])
        values = np.asarray(rows)
        expected = np.concatenate((
            np.ones(1),
            POST_MEAN,
            POST_VAR + POST_MEAN**2,
        ))
        estimator_se = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
        # 2e-5 is the explicit f32/Metal arithmetic budget.
        np.testing.assert_array_less(
            np.abs(values.mean(axis=0) - expected),
            5 * estimator_se + 2e-5,
        )


class TestSchedule:
    """Adaptive ESS-bisection schedule properties."""

    def test_temperatures_increase_and_end_at_one(self):
        post = _run(1)
        temps = np.array(post.temperatures, dtype=np.float64)
        assert np.all(np.diff(temps) > 0)
        assert temps[-1] == pytest.approx(1.0, abs=1e-6)
        assert temps[0] > 0.0

    def test_intermediate_ess_hits_target(self):
        n = 4000
        post = _run(2, n=n, target_ess=0.5)
        e = np.array(post.ess)
        if len(e) > 1:
            assert np.allclose(e[:-1], 0.5 * n, rtol=0.05)
        assert e[-1] >= 0.45 * n

    def test_flat_likelihood_single_jump(self):
        # sl huge => likelihood nearly flat => one stage to phi = 1.
        y = jnp.asarray(Y_OBS)
        init, log_prior, _ = _model()

        def log_lik_flat(x):
            return -0.5 * jnp.sum(
                math.log(2 * math.pi * 1e6) + ((y - x) ** 2) / 1e6
            )

        post = smcx.temper(jr.key(3), init, log_prior, log_lik_flat, 1000)
        assert post.temperatures.shape == (1,)
        assert float(post.temperatures[0]) == pytest.approx(1.0)


class TestMechanics:
    """Acceptance, determinism, degeneracy, container."""

    def test_acceptance_rates_sane(self):
        post = _run(4)
        acc = np.array(post.acceptance_rates)
        assert np.all(acc > 0.05) and np.all(acc < 0.95)

    def test_deterministic_per_key(self):
        a = _run(5, n=1000)
        b = _run(5, n=1000)
        # Bit-identical determinism per key is the contract, so exact
        # comparison is intended (not a tolerance bug).
        assert np.array_equal(
            np.array(a.marginal_loglik), np.array(b.marginal_loglik)
        )
        assert np.array_equal(np.array(a.particles), np.array(b.particles))

    def test_distinct_hash_equal_likelihood_uses_second_behavior(self):
        init, log_prior, _ = _small_tempering_model()

        def shifted_log_likelihood(center, value):
            return -0.5 * jnp.sum((value - center) ** 2 / 0.2)

        class HashEqualLikelihood:
            def __init__(self, center):
                self.center = center

            def __hash__(self):
                return 1

            def __eq__(self, other):
                return isinstance(other, HashEqualLikelihood)

            def __call__(self, value):
                return shifted_log_likelihood(self.center, value)

        class FreshLikelihood:
            def __init__(self, center):
                self.center = center

            def __call__(self, value):
                return shifted_log_likelihood(self.center, value)

        smcx.temper(
            jr.key(30),
            init,
            log_prior,
            HashEqualLikelihood(-1.0),
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )
        actual = smcx.temper(
            jr.key(31),
            init,
            log_prior,
            HashEqualLikelihood(1.0),
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )
        expected = smcx.temper(
            jr.key(31),
            init,
            log_prior,
            FreshLikelihood(1.0),
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )

        for expected_value, actual_value in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual_value),
                np.asarray(expected_value),
            )

    @pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="frozen CPU/f64 arithmetic contract",
    )
    def test_rwm_sweep_preserves_frozen_fixed_key_output(self):
        init, log_prior, log_lik = _small_tempering_model()
        posterior = smcx.temper(
            jr.key(314159),
            init,
            log_prior,
            log_lik,
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )

        # Linux/x64 and macOS/arm64 CPU lowerings differed by at most
        # 6.7e-16 in this frozen f64 fixture.  The 1e-15 absolute budget is
        # less than five binary64 eps at unit scale: it admits only backend
        # rounding while still rejecting meaningful numerical drift.
        frozen_atol = 1e-15
        np.testing.assert_allclose(
            np.asarray(posterior.particles),
            np.array([
                [1.5109879397100636],
                [0.8820825513186982],
                [0.0],
                [0.3108199100404425],
                [-0.4867093863813025],
            ]),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.log_weights),
            np.full(5, -1.6094379124341003),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.marginal_loglik),
            np.asarray(-0.33449690533561793),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.temperatures),
            np.array([1.0]),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.ess),
            np.array([4.5218752201463674]),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.acceptance_rates),
            np.array([0.4000000134110451]),
            rtol=0.0,
            atol=frozen_atol,
        )

    def test_degenerate_likelihood_raises(self):
        init, log_prior, _ = _model()

        def impossible(x):
            return jnp.array(-jnp.inf)

        with pytest.raises(smcx.DegenerateWeightsError):
            smcx.temper(jr.key(6), init, log_prior, impossible, 500)

    def test_container_shapes(self):
        post = _run(7, n=1000)
        k = post.temperatures.shape[0]
        assert post.particles.shape == (1000, D)
        assert post.log_weights.shape == (1000,)
        assert post.ess.shape == (k,)
        assert post.acceptance_rates.shape == (k,)
        assert np.allclose(
            np.array(post.log_weights), -math.log(1000), atol=1e-5
        )
