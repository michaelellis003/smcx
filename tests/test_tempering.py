# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tempered SMC tests against exact and independent implementations.

Conjugate ground truth: prior N(0, s0^2 I_d), likelihood
N(y_obs; x, sl^2 I_d) => log Z = sum_i log N(y_i; 0, s0^2 + sl^2)
exactly, and the posterior is Gaussian with known moments.

One-time isolated validation on this exact target (2026-07-18; N=4000,
12 fixed seeds) produced mean log evidence (Monte Carlo SE) ``-5.108088
(.009799)`` for smcx, ``-5.121923 (.007684)`` for particles, and
``-5.122697 (.005115)`` for BlackJAX, versus ``-5.12131172107733`` exact.
All posterior mean and variance coordinates passed five-SE exact and
cross-implementation gates. No outside package is imported by these tests.

Pinned authorities (no code copied):

* particles 0.4, f71e94a21a11c73b58e2d694775b1b1d379b8854, MIT:
  https://github.com/nchopin/particles/blob/f71e94a21a11c73b58e2d694775b1b1d379b8854/particles/smc_samplers.py#L800-L958
  https://github.com/nchopin/particles/blob/f71e94a21a11c73b58e2d694775b1b1d379b8854/LICENSE
* BlackJAX 1.6.2, a9ef478c69d730a2caa13ca4b2d735c580e0feec,
  Apache-2.0:
  https://github.com/blackjax-devs/blackjax/blob/a9ef478c69d730a2caa13ca4b2d735c580e0feec/blackjax/smc/adaptive_tempered.py
  https://github.com/blackjax-devs/blackjax/blob/a9ef478c69d730a2caa13ca4b2d735c580e0feec/LICENSE

Algorithm: Del Moral, Doucet, and Jasra (2006),
https://doi.org/10.1111/j.1467-9868.2006.00553.x
"""

import importlib
import math

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


def _small_cache_model():
    observation = jnp.array([0.25], dtype=jnp.float64)

    def init(_key, n):
        return jnp.linspace(-1.0, 1.0, n, dtype=jnp.float64)[:, None]

    def log_prior(x):
        return -0.5 * jnp.sum(x**2)

    def log_lik(x):
        return -0.5 * jnp.sum((observation - x) ** 2 / 0.7)

    return init, log_prior, log_lik


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

    def test_rwm_factory_is_reused_for_same_callbacks(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        tempering = importlib.import_module("smcx.tempering")
        tempering._cached_rwm_sweep.cache_clear()
        original_factory = tempering._build_rwm_sweep
        builds = 0

        def recording_factory(*args, **kwargs):
            nonlocal builds
            builds += 1
            return original_factory(*args, **kwargs)

        monkeypatch.setattr(
            tempering,
            "_build_rwm_sweep",
            recording_factory,
        )
        init, log_prior, log_lik = _small_cache_model()
        try:
            for seed in (10, 11):
                smcx.temper(
                    jr.key(seed),
                    init,
                    log_prior,
                    log_lik,
                    5,
                    num_mcmc_steps=2,
                    target_ess=0.6,
                )
            cache_info = tempering._cached_rwm_sweep.cache_info()
        finally:
            tempering._cached_rwm_sweep.cache_clear()

        assert builds == 1
        assert cache_info.hits == 1
        assert cache_info.misses == 1

    @pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="frozen CPU/x64 arithmetic contract",
    )
    def test_rwm_factory_preserves_frozen_fixed_key_output(self):
        init, log_prior, log_lik = _small_cache_model()
        posterior = smcx.temper(
            jr.key(314159),
            init,
            log_prior,
            log_lik,
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )

        np.testing.assert_array_equal(
            np.asarray(posterior.particles),
            np.array([
                [1.5109879397100636],
                [0.8820825513186982],
                [0.0],
                [0.3108199100404425],
                [-0.4867093863813025],
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_weights),
            np.full(5, -1.6094379124341003),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik),
            np.asarray(-0.33449690533561793),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.temperatures),
            np.array([1.0]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess),
            np.array([4.5218752201463674]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.acceptance_rates),
            np.array([0.4000000134110451]),
        )

    def test_unhashable_callbacks_use_uncached_rwm_factory(self):
        class UnhashableCallback:
            __hash__ = None

            def __init__(self, callback):
                self.callback = callback

            def __call__(self, value):
                return self.callback(value)

        init, log_prior, log_lik = _small_cache_model()
        expected = smcx.temper(
            jr.key(2718),
            init,
            log_prior,
            log_lik,
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )
        actual = smcx.temper(
            jr.key(2718),
            init,
            UnhashableCallback(log_prior),
            UnhashableCallback(log_lik),
            5,
            num_mcmc_steps=2,
            target_ess=0.6,
        )

        for expected_value, actual_value in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual_value),
                np.asarray(expected_value),
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
