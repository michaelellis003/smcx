# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Guided filter tests (spec: feat-7-guided; ADR-0008 item 2)."""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

A, Q, R = 0.9, 0.5, 0.3
M0, P0 = 0.0, 1.0
T = 50


def _base():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def init(key, n):
        return M0 + sp * mx.random.normal((n, 1), key=key)

    def trans_sample(key, s):
        return A * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    def log_trans(new, old):
        return -0.5 * (
            math.log(2 * math.pi * Q) + (new[0] - A * old[0]) ** 2 / Q
        )

    return init, trans_sample, logobs, log_trans


def _optimal_proposal():
    # Locally optimal q(x_t | x_{t-1}, y_t) for the LGSSM:
    # s* = (1/Q + 1/R)^-1, m* = s*(A x/Q + y/R) (Doucet et al. 2000).
    s_star = 1.0 / (1.0 / Q + 1.0 / R)
    sd_star = math.sqrt(s_star)

    def prop_sample(key, s, y):
        m = s_star * (A * s / Q + y / R)
        return m + sd_star * mx.random.normal(s.shape, key=key)

    def log_prop(y, new, old):
        m = s_star * (A * old[0] / Q + y[0] / R)
        return -0.5 * (
            math.log(2 * math.pi * s_star) + (new[0] - m) ** 2 / s_star
        )

    return prop_sample, log_prop


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(M0, math.sqrt(P0))
    for t in range(1, T):
        x[t] = A * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return x + rng.normal(0, math.sqrt(R), T)


Y = _data()
LOGZ_TRUE, _, _ = kalman_1d(Y, A, Q, R, M0, P0)
Y_MX = mx.array(Y.astype(np.float32))[:, None]


class TestPriorProposalReduction:
    """q = f reduces to bootstrap (mathematically; f32-tight here)."""

    def test_prior_proposal_matches_bootstrap(self):
        init, trans_sample, logobs, log_trans = _base()
        sq = math.sqrt(Q)

        def prop_sample(key, s, y):
            # identical draw and key consumption; y ignored
            return A * s + sq * mx.random.normal(s.shape, key=key)

        def log_prop(y, new, old):
            return log_trans(new, old)

        a = smcx.guided_filter(
            mx.random.key(7),
            init,
            prop_sample,
            log_prop,
            log_trans,
            logobs,
            Y_MX,
            1000,
        )
        b = smcx.bootstrap_filter(
            mx.random.key(7), init, trans_sample, logobs, Y_MX, 1000
        )
        # f and q produce identical values, but (obs + f) - q rounds
        # at the intermediate add, so the reduction is mathematical,
        # not bitwise: ulp-level weight perturbations can also flip
        # resampling picks at strata edges. Compare at f32-honest
        # tolerances on the statistics, not indices.
        assert a.marginal_loglik.item() == pytest.approx(
            b.marginal_loglik.item(), abs=2e-3
        )
        assert np.allclose(
            np.array(a.ess), np.array(b.ess), rtol=5e-3, atol=0.5
        )


class TestKalmanGate:
    """Gate + the variance-reduction property of the optimal proposal."""

    def _logzs(self, r_keys, n, use_guided):
        init, trans_sample, logobs, log_trans = _base()
        prop_sample, log_prop = _optimal_proposal()
        out = []
        for s in range(r_keys):
            if use_guided:
                post = smcx.guided_filter(
                    mx.random.key(s),
                    init,
                    prop_sample,
                    log_prop,
                    log_trans,
                    logobs,
                    Y_MX,
                    n,
                )
            else:
                post = smcx.bootstrap_filter(
                    mx.random.key(s), init, trans_sample, logobs, Y_MX, n
                )
            out.append(post.marginal_loglik.item())
        return np.array(out)

    def test_log_ml_gate_r20_optimal_proposal(self):
        vals = self._logzs(20, 10_000, use_guided=True)
        sd = vals.std(ddof=1)
        err = vals.mean() - LOGZ_TRUE
        upper = 3 * sd / math.sqrt(20)
        assert -(upper + 0.5 * sd**2) <= err <= upper, (err, sd)

    def test_optimal_proposal_reduces_logz_variance(self):
        # The locally optimal proposal has zero conditional variance
        # of the incremental weight given ancestors (Doucet, Godsill
        # & Andrieu 2000) — for this LGSSM the SD gap vs bootstrap is
        # large (several-fold), far beyond R=20 estimation noise.
        sd_guided = self._logzs(20, 2000, use_guided=True).std(ddof=1)
        sd_boot = self._logzs(20, 2000, use_guided=False).std(ddof=1)
        assert sd_guided < sd_boot


class TestValidationAndOptions:
    """Arity, inputs, store_history."""

    def test_arity_mismatch_raises_named_error(self):
        init, _, logobs, log_trans = _base()
        prop_sample, log_prop = _optimal_proposal()
        with pytest.raises(TypeError, match="proposal_sampler"):
            smcx.guided_filter(
                mx.random.key(0),
                init,
                prop_sample,  # 3-arg while inputs supplied
                log_prop,
                log_trans,
                logobs,
                Y_MX,
                100,
                inputs=mx.zeros((T,)),
            )

    def test_store_history_final_only(self):
        init, _, logobs, log_trans = _base()
        prop_sample, log_prop = _optimal_proposal()
        post = smcx.guided_filter(
            mx.random.key(1),
            init,
            prop_sample,
            log_prop,
            log_trans,
            logobs,
            Y_MX,
            500,
            store_history=False,
        )
        assert post.filtered_particles.shape == (1, 500, 1)
        assert isinstance(post, smcx.ParticleFilterResult)
