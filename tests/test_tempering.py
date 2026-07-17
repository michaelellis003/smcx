# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tempered SMC tests (ported from the MLX suite; ADR-0008 item 6).

Conjugate ground truth: prior N(0, s0^2 I_d), likelihood
N(y_obs; x, sl^2 I_d) => log Z = sum_i log N(y_i; 0, s0^2 + sl^2)
exactly, and the posterior is Gaussian with known moments.
"""

import math

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


class TestEvidence:
    """MC-calibrated gate against the exact conjugate evidence."""

    def test_logz_gate_r20(self):
        r_keys = 20
        vals = np.array([float(_run(s).marginal_loglik) for s in range(r_keys)])
        sd = vals.std(ddof=1)
        err = vals.mean() - LOGZ_TRUE
        upper = 3 * sd / math.sqrt(r_keys)
        lower = -(upper + 0.5 * sd**2)
        assert lower <= err <= upper, (err, sd, LOGZ_TRUE)

    def test_posterior_moments(self):
        post = _run(0, n=8000)
        x = np.array(post.particles, dtype=np.float64)
        # Equal-weight draws after final resample+moves; MCMC
        # autocorrelation inflates the SE of the mean beyond
        # sd/sqrt(n) — generous bounds.
        assert np.allclose(x.mean(axis=0), POST_MEAN, atol=0.06)
        assert np.allclose(x.var(axis=0), POST_VAR, rtol=0.15)


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
