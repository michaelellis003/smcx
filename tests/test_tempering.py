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
import smcx.types as smcx_types
from smcx.types import (
    PRNGKeyT,
    StaticLogDensity,
    TemperingMutationInfo,
    TemperingMutationState,
)

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


@pytest.mark.parametrize(
    ("generic_name", "tempering_name"),
    [
        ("StaticMutationState", "TemperingMutationState"),
        ("StaticMutationInfo", "TemperingMutationInfo"),
        ("StaticMutationInitFn", "TemperingMutationInitFn"),
        ("StaticMutationStepFn", "TemperingMutationStepFn"),
    ],
)
def test_static_mutation_protocols_alias_tempering_names(
    generic_name, tempering_name
):
    assert getattr(smcx_types, generic_name) is getattr(
        smcx_types, tempering_name
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


def _small_tempering_model(dtype=jnp.float64):
    observation = jnp.array([0.25], dtype=dtype)

    def init(_key, n):
        return jnp.linspace(-1.0, 1.0, n, dtype=dtype)[:, None]

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


def _upper_target_run(scale, *, num_particles=2, max_stages):
    def init(_key, count):
        return jnp.linspace(0.0, 1.0, count, dtype=jnp.float32)[:, None]

    def log_prior(x):
        return jnp.asarray(0.0, dtype=x.dtype)

    def log_likelihood(x):
        return jnp.asarray(scale, dtype=x.dtype) * x[0]

    def mutation_step(_key, state, _tempered_logdensity_fn):
        info = _MutationInfo(
            jnp.asarray(1.0, dtype=state.position.dtype),
            jnp.asarray(True),
        )
        return state, info

    def identity_resampling(_key, _weights, num_samples):
        return jnp.arange(num_samples, dtype=jnp.int32)

    return smcx.temper(
        jr.key(31),
        init,
        log_prior,
        log_likelihood,
        num_particles,
        num_mcmc_steps=1,
        target_ess=1.0 - float(np.finfo(np.float32).eps),
        resampling_fn=identity_resampling,
        mutation_init_fn=_mutation_init,
        mutation_step_fn=mutation_step,
        max_stages=max_stages,
    )


def _bad_mutation_init(position, tempered_logdensity_fn):
    state = _mutation_init(position, tempered_logdensity_fn)
    return state._replace(position=position[None])


def _bad_mutation_step(key, state, tempered_logdensity_fn):
    next_state, info = _mutation_step(key, state, tempered_logdensity_fn)
    return next_state, info._replace(acceptance_rate=jnp.ones(2))


def _nonconvertible_mutation_step(key, state, tempered_logdensity_fn):
    next_state, info = _mutation_step(key, state, tempered_logdensity_fn)
    return next_state, info._replace(acceptance_rate=object())


def _fixed_acceptance_step(rate):
    def step(_key, state, _tempered_logdensity_fn):
        info = _MutationInfo(
            jnp.asarray(rate, dtype=state.position.dtype),
            jnp.asarray(False),
        )
        return state, info

    return step


def _cancelling_invalid_acceptance_step(
    _key,
    state,
    _tempered_logdensity_fn,
):
    rate = jnp.where(state.step_index == 0, -0.1, 1.1)
    return state._replace(step_index=state.step_index + 1), _MutationInfo(
        rate,
        jnp.asarray(False),
    )


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
            (
                _mutation_init,
                _nonconvertible_mutation_step,
                "must be a scalar float",
            ),
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

    @pytest.mark.parametrize("rate", [-0.1, 1.1, np.nan, np.inf])
    def test_acceptance_rate_must_be_a_probability(self, rate):
        init, log_prior, log_lik = _small_tempering_model()

        with pytest.raises(
            ValueError,
            match="acceptance_rate must be finite and in \\[0, 1\\]",
        ):
            smcx.temper(
                jr.key(47),
                init,
                log_prior,
                log_lik,
                5,
                num_mcmc_steps=2,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=_fixed_acceptance_step(rate),
            )

    def test_each_acceptance_rate_is_checked_before_reduction(self):
        init, log_prior, log_lik = _small_tempering_model()

        with pytest.raises(ValueError, match="must be finite and in"):
            smcx.temper(
                jr.key(48),
                init,
                log_prior,
                log_lik,
                5,
                num_mcmc_steps=2,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=_cancelling_invalid_acceptance_step,
            )

    def test_negative_subnormal_acceptance_rate_is_rejected_across_jit(self):
        """A traced float32 value below zero never flushes into the domain."""
        negative_subnormal = jnp.asarray(
            -jnp.finfo(jnp.float32).smallest_subnormal,
            dtype=jnp.float32,
        )

        def init(_key: PRNGKeyT, count: int) -> jax.Array:
            return jnp.full((count, 1), negative_subnormal, dtype=jnp.float32)

        def log_density(_position: jax.Array) -> jax.Array:
            return jnp.asarray(0.0, dtype=jnp.float32)

        def mutation_step(
            _key: PRNGKeyT,
            state: TemperingMutationState,
            _tempered_logdensity_fn: StaticLogDensity,
        ) -> tuple[TemperingMutationState, TemperingMutationInfo]:
            return state, _MutationInfo(
                state.position[0],
                jnp.asarray(False),
            )

        def run() -> None:
            smcx.temper(
                jr.key(168),
                init,
                log_density,
                log_density,
                4,
                num_mcmc_steps=1,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=mutation_step,
                max_stages=1,
            )

        message = "acceptance_rate must be finite and in \\[0, 1\\]"
        with jax.disable_jit(), pytest.raises(ValueError, match=message):
            run()
        with pytest.raises(ValueError, match=message):
            run()

    def test_invalid_acceptance_rate_is_rejected_at_a_later_stage(self):
        init, log_prior, log_lik = _small_tempering_model()

        def step(_key, state, tempered_logdensity_fn):
            probe = jnp.zeros_like(state.position)
            phi = (tempered_logdensity_fn(probe) - log_prior(probe)) / log_lik(
                probe
            )
            rate = jnp.where(phi < 0.9, 0.5, 1.1)
            return state, _MutationInfo(rate, jnp.asarray(False))

        def run(max_stages):
            return smcx.temper(
                jr.key(50),
                init,
                log_prior,
                log_lik,
                5,
                num_mcmc_steps=1,
                target_ess=0.95,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=step,
                max_stages=max_stages,
            )

        with pytest.raises(RuntimeError, match="within 1 stages"):
            run(1)
        with pytest.raises(ValueError, match="must be finite and in"):
            run(2)

    def test_valid_acceptance_rates_do_not_change_fixed_key_inference(self):
        init, log_prior, log_lik = _small_tempering_model()
        rates = (0.0, -0.0, 0.4, 1.0)
        posteriors = [
            smcx.temper(
                jr.key(49),
                init,
                log_prior,
                log_lik,
                5,
                num_mcmc_steps=2,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=_fixed_acceptance_step(rate),
            )
            for rate in rates
        ]

        baseline = posteriors[0]
        for rate, posterior in zip(rates, posteriors, strict=True):
            acceptance_rates = np.asarray(posterior.acceptance_rates)
            np.testing.assert_array_equal(
                acceptance_rates,
                np.full(
                    acceptance_rates.shape,
                    rate,
                    dtype=acceptance_rates.dtype,
                ),
            )
            for field_name in type(baseline)._fields:
                if field_name == "acceptance_rates":
                    continue
                np.testing.assert_array_equal(
                    getattr(posterior, field_name),
                    getattr(baseline, field_name),
                )

    @pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="JAX sub-byte float support is backend-specific",
    )
    @pytest.mark.parametrize(
        ("dtype", "invalid_rate"),
        [
            (jnp.float4_e2m1fn, -0.5),
            (jnp.float8_e4m3fn, np.nan),
        ],
    )
    def test_low_precision_acceptance_rate_preserves_domain_and_mean(
        self,
        dtype,
        invalid_rate,
    ):
        init, log_prior, log_lik = _small_tempering_model()

        def run(rate_value):
            rate = jnp.asarray(rate_value, dtype=dtype)

            def step(
                _key: PRNGKeyT,
                state: TemperingMutationState,
                _tempered_logdensity_fn: StaticLogDensity,
            ) -> tuple[TemperingMutationState, TemperingMutationInfo]:
                return state, _MutationInfo(rate, jnp.asarray(False))

            return smcx.temper(
                jr.key(168),
                init,
                log_prior,
                log_lik,
                100,
                num_mcmc_steps=1,
                mutation_init_fn=_mutation_init,
                mutation_step_fn=step,
            )

        posterior = run(0.5)

        assert posterior.acceptance_rates.dtype == dtype
        np.testing.assert_array_equal(
            np.asarray(posterior.acceptance_rates),
            np.full(
                posterior.acceptance_rates.shape,
                0.5,
                dtype=np.asarray(jnp.asarray(0.5, dtype=dtype)).dtype,
            ),
        )
        with pytest.raises(ValueError, match="must be finite and in"):
            run(invalid_rate)

    def test_requires_at_least_one_mutation_step(self):
        init, log_prior, log_lik = _small_tempering_model()
        with pytest.raises(
            ValueError, match="num_mcmc_steps must be a positive integer"
        ):
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

    @pytest.mark.parametrize("num_particles", [63, 8192])
    def test_float32_upper_target_finishes_for_uniform_cloud_sizes(
        self,
        num_particles,
    ):
        posterior = _upper_target_run(
            0.0,
            num_particles=num_particles,
            max_stages=1,
        )

        np.testing.assert_array_equal(
            np.asarray(posterior.temperatures),
            [1.0],
        )

    def test_float32_upper_target_finishes_in_four_short_stages(self):
        posterior = _upper_target_run(0.002, max_stages=4)
        temperatures = np.asarray(posterior.temperatures, dtype=np.float64)

        assert temperatures.shape == (4,)
        assert np.all(np.diff(np.concatenate(([0.0], temperatures))) > 0.0)
        np.testing.assert_array_equal(temperatures[-1], 1.0)

    def test_float32_upper_target_honors_raised_stage_budget(self):
        with pytest.raises(
            RuntimeError,
            match="within 4 stages; reached phi=",
        ):
            _upper_target_run(0.02, max_stages=4)

        posterior = _upper_target_run(0.02, max_stages=64)
        temperatures = np.asarray(posterior.temperatures, dtype=np.float64)

        assert temperatures.shape[0] > 4
        assert np.all(np.diff(np.concatenate(([0.0], temperatures))) > 0.0)
        np.testing.assert_array_equal(temperatures[-1], 1.0)

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

    def test_large_constant_likelihood_offset_does_not_add_stages(self):
        """A representable constant must not change the ESS schedule."""

        def init(_key, n):
            return jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32)[:, None]

        def log_prior(x):
            return -0.5 * jnp.sum(x**2)

        def log_likelihood(_x):
            return jnp.asarray(2**24, dtype=jnp.float32)

        post = smcx.temper(
            jr.key(30),
            init,
            log_prior,
            log_likelihood,
            3,
            num_mcmc_steps=1,
            target_ess=0.9,
        )

        np.testing.assert_array_equal(np.asarray(post.temperatures), [1.0])
        np.testing.assert_allclose(np.asarray(post.ess), [3.0], rtol=2e-6)


class TestMechanics:
    """Acceptance, determinism, degeneracy, container."""

    @pytest.mark.parametrize(
        "spread",
        [0.0, np.finfo(np.float32).eps / 4],
    )
    @pytest.mark.parametrize("num_particles", [1, 2])
    def test_adaptive_mutation_handles_zero_and_near_zero_spread(
        self,
        spread,
        num_particles,
    ):
        def init(_key, count):
            values = spread * jnp.arange(count, dtype=jnp.float32)
            return values[:, None]

        def log_density(position):
            return -0.5 * jnp.sum(position**2)

        posterior = smcx.temper(
            jr.key(47),
            init,
            log_density,
            log_density,
            num_particles,
            num_mcmc_steps=1,
        )

        assert posterior.particles.dtype == jnp.float32
        assert np.all(np.isfinite(np.asarray(posterior.particles)))

    @pytest.mark.parametrize(
        ("drifting_name", "message"),
        [
            ("prior", "log_prior_fn output must have shape"),
            ("likelihood", "log_likelihood_fn output must have shape"),
        ],
    )
    def test_rejects_density_contract_drift_during_mutation(
        self, drifting_name, message
    ) -> None:
        outputs = iter((jnp.asarray(0.0), jnp.zeros(2)))

        def stable_density(_position):
            return jnp.asarray(0.0)

        def drifting_density(_position):
            return next(outputs)

        log_prior = (
            drifting_density if drifting_name == "prior" else stable_density
        )
        log_likelihood = (
            drifting_density
            if drifting_name == "likelihood"
            else stable_density
        )

        with pytest.raises(ValueError, match=message):
            smcx.temper(
                jr.key(46),
                lambda _key, count: jnp.arange(count)[:, None].astype(
                    jnp.float32
                ),
                log_prior,
                log_likelihood,
                2,
                num_mcmc_steps=1,
            )

    def test_exact_zero_uniform_does_not_accept_bad_float32_move(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def init(_key, _num_particles):
            return jnp.array([[0.0], [1.0]], dtype=jnp.float32)

        def log_prior(x):
            return -1e6 * jnp.sum(x**2)

        def log_likelihood(_x):
            return jnp.array(0.0, dtype=jnp.float32)

        def zero_uniform(_key, shape=(), dtype=None):
            return jnp.zeros(shape, dtype=dtype)

        def unit_normal(_key, shape, dtype=None):
            return jnp.ones(shape, dtype=dtype)

        monkeypatch.setattr(jr, "uniform", zero_uniform)
        monkeypatch.setattr(jr, "normal", unit_normal)
        with jax.enable_x64(False):
            posterior = smcx.temper(
                jr.key(45),
                init,
                log_prior,
                log_likelihood,
                2,
            )

        np.testing.assert_array_equal(
            posterior.particles,
            np.array([[0.0], [1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            posterior.acceptance_rates,
            np.array([0.0], dtype=np.float32),
        )

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
        jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
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

    @pytest.mark.skipif(
        jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
        reason="frozen CPU/f64 arithmetic contract",
    )
    def test_rwm_sweep_preserves_frozen_two_stage_output(self):
        init, log_prior, log_lik = _small_tempering_model()
        posterior = smcx.temper(
            jr.key(314159),
            init,
            log_prior,
            log_lik,
            5,
            num_mcmc_steps=2,
            target_ess=0.95,
        )

        # This uses the same five-binary64-eps backend-rounding budget as the
        # established one-stage frozen fixture immediately above.
        frozen_atol = 1e-15
        np.testing.assert_allclose(
            np.asarray(posterior.particles),
            np.array([
                [0.5566520237390912],
                [0.42931421610876214],
                [0.0],
                [0.3006901431829462],
                [-0.04198666790556421],
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
            np.asarray(-0.3510029236122849),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.temperatures),
            np.array([0.6627762304582339, 1.0]),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.ess),
            np.array([4.75, 4.873686009537485]),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.acceptance_rates),
            np.array([0.4000000134110451, 0.30000000447034836]),
            rtol=0.0,
            atol=frozen_atol,
        )
        np.testing.assert_allclose(
            np.asarray(posterior.log_evidence_increments),
            np.array([-0.23537493635122342, -0.1156279872610615]),
            rtol=0.0,
            atol=frozen_atol,
        )

    @pytest.mark.skipif(
        jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
        reason="frozen mixed x64/f32 arithmetic contract",
    )
    def test_f32_schedule_preserves_uniform_weight_representation(self):
        init, log_prior, log_lik = _small_tempering_model(jnp.float32)
        calls = 0

        def schedule(phi, log_weights, _log_likelihoods):
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0.5
            if float(log_weights[0]) > -1.60943793:
                return 1.0
            return min(1.0, phi + 0.25)

        posterior = smcx.temper(
            jr.key(314159),
            init,
            log_prior,
            log_lik,
            5,
            num_mcmc_steps=2,
            schedule_fn=schedule,
        )

        assert posterior.log_weights.dtype == jnp.float64
        assert calls == 2
        np.testing.assert_array_equal(posterior.temperatures, [0.5, 1.0])

    @pytest.mark.parametrize("value", [-jnp.inf, jnp.inf, jnp.nan])
    def test_nonfinite_reweight_normalizer_raises(self, value):
        init, log_prior, _ = _model()

        def impossible(x):
            del x
            return value

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


def test_rejects_out_of_range_custom_resampler():
    def invalid_resampler(key, weights, count):
        del key, weights
        return jnp.full(count, -1, dtype=jnp.int32)

    with pytest.raises(ValueError, match=r"entries must be in \[0, 4\)"):
        _run(159, n=4, resampling_fn=invalid_resampler)


class TestScheduleCallback:
    """Caller-owned tempering schedules replace the ESS bisection."""

    def test_fixed_linear_schedule_is_honored_exactly(self):
        init, log_prior, log_lik = _small_tempering_model()

        def quarter_steps(phi, normalized_log_weights, log_likelihoods):
            del normalized_log_weights, log_likelihoods
            return min(1.0, phi + 0.25)

        posterior = smcx.temper(
            jr.key(17),
            init,
            log_prior,
            log_lik,
            8,
            num_mcmc_steps=1,
            schedule_fn=quarter_steps,
        )

        np.testing.assert_allclose(
            np.asarray(posterior.temperatures, dtype=np.float64),
            [0.25, 0.5, 0.75, 1.0],
            rtol=0.0,
            atol=0.0,
        )

    @pytest.mark.parametrize("bad", [0.0, 1.5, float("nan")])
    def test_invalid_schedule_return_raises(self, bad):
        init, log_prior, log_lik = _small_tempering_model()

        def stuck(phi, normalized_log_weights, log_likelihoods):
            del normalized_log_weights, log_likelihoods
            return bad

        with pytest.raises(ValueError, match="schedule_fn must return"):
            smcx.temper(
                jr.key(19),
                init,
                log_prior,
                log_lik,
                8,
                num_mcmc_steps=1,
                schedule_fn=stuck,
            )


def test_stage_increments_sum_to_the_marginal():
    """Per-stage evidence increments compose the compensated total."""
    posterior = _run(3, n=256)

    np.testing.assert_allclose(
        float(jnp.sum(posterior.log_evidence_increments)),
        float(posterior.marginal_loglik),
        rtol=1e-10,
        atol=1e-10,
    )
    assert posterior.log_evidence_increments.shape == (
        posterior.temperatures.shape[0],
    )
