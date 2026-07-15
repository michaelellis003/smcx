# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Tempered SMC tests (spec: feat-8-tempering; ADR-0008 item 6).

Conjugate ground truth: prior N(0, s0^2 I_d), likelihood
N(y_obs; x, sl^2 I_d) => log Z = sum_i log N(y_i; 0, s0^2 + sl^2)
exactly, and the posterior is Gaussian with known moments.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx

D = 3
S0, SL = 2.0, 0.5
Y_OBS = np.array([1.0, -0.7, 0.4])

# Analytic posterior and evidence (conjugate Gaussian).
POST_VAR = 1.0 / (1.0 / S0**2 + 1.0 / SL**2)
POST_MEAN = POST_VAR * Y_OBS / SL**2
LOGZ_TRUE = float(
    np.sum(
        -0.5
        * (np.log(2 * np.pi * (S0**2 + SL**2)) + Y_OBS**2 / (S0**2 + SL**2))
    )
)


def _model():
    y = mx.array(Y_OBS.astype(np.float32))

    def init(key, n):
        return S0 * mx.random.normal((n, D), key=key)

    def log_prior(x):
        return -0.5 * mx.sum(math.log(2 * math.pi * S0**2) + (x / S0) ** 2)

    def log_lik(x):
        return -0.5 * mx.sum(
            math.log(2 * math.pi * SL**2) + ((y - x) / SL) ** 2
        )

    return init, log_prior, log_lik


def _run(seed, n=4000, **kw):
    init, log_prior, log_lik = _model()
    return smcx.temper(mx.random.key(seed), init, log_prior, log_lik, n, **kw)


class TestEvidence:
    """MC-calibrated gate against the exact conjugate evidence."""

    def test_logz_gate_r20(self):
        r_keys = 20
        vals = np.array([_run(s).marginal_loglik.item() for s in range(r_keys)])
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
        # sd/sqrt(n) — use a 5x-inflated 5*SE bound:
        # 5 * 5 * sqrt(POST_VAR/n) ~ 0.038.
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
        # Every stage but the last is solved to ESS = target*N by
        # bisection (tolerance loose: f32 ESS + bisection width);
        # the final stage jumps to 1.0 with ESS >= target*N.
        if len(e) > 1:
            assert np.allclose(e[:-1], 0.5 * n, rtol=0.05)
        assert e[-1] >= 0.45 * n

    def test_flat_likelihood_single_jump(self):
        # sl huge => likelihood nearly flat => ESS at phi=1 already
        # above target => exactly one stage.
        y = mx.array(Y_OBS.astype(np.float32))
        init, log_prior, _ = _model()

        def log_lik_flat(x):
            return -0.5 * mx.sum(
                math.log(2 * math.pi * 1e6) + ((y - x) ** 2) / 1e6
            )

        post = smcx.temper(
            mx.random.key(3), init, log_prior, log_lik_flat, 1000
        )
        assert post.temperatures.shape == (1,)
        assert post.temperatures[0].item() == pytest.approx(1.0)


class TestMechanics:
    """Acceptance, determinism, degeneracy, container."""

    def test_acceptance_rates_sane(self):
        post = _run(4)
        acc = np.array(post.acceptance_rates)
        assert np.all(acc > 0.05) and np.all(acc < 0.95)

    def test_deterministic_per_key(self):
        a = _run(5, n=1000)
        b = _run(5, n=1000)
        assert a.marginal_loglik.item() == b.marginal_loglik.item()
        assert np.array_equal(np.array(a.particles), np.array(b.particles))

    def test_degenerate_likelihood_raises(self):
        init, log_prior, _ = _model()

        def impossible(x):
            return mx.array(-mx.inf)

        with pytest.raises(smcx.DegenerateWeightsError):
            smcx.temper(mx.random.key(6), init, log_prior, impossible, 500)

    def test_container_shapes(self):
        post = _run(7, n=1000)
        k = post.temperatures.shape[0]
        assert post.particles.shape == (1000, D)
        assert post.log_weights.shape == (1000,)
        assert post.ess.shape == (k,)
        assert post.acceptance_rates.shape == (k,)
        # equal weights after the final resample + moves
        assert np.allclose(
            np.array(post.log_weights), -math.log(1000), atol=1e-5
        )
