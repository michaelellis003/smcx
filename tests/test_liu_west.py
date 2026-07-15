# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Liu-West filter tests (spec: feat-10-liu-west).

The Liu-West filter is a labeled-approximate method (non-vanishing
bias, discount sensitivity — Kantas et al. 2015); tests target its
contract, not exactness: parameter concentration around truth,
point-mass reduction to the APF, the variance-matched discount
kernel, and the container/degeneracy conventions.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

A_TRUE, Q, R = 0.9, 0.5, 0.3
M0, P0 = 0.0, 1.0
T = 60


def _model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def init(key, n):
        return M0 + sp * mx.random.normal((n, 1), key=key)

    def trans(key, s, params):
        return params[0] * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s, params):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    def logaux(y, s, params):
        v = Q + R
        return -0.5 * (
            math.log(2 * math.pi * v) + (y[0] - params[0] * s[0]) ** 2 / v
        )

    def param_init(key, n):
        # U(0.5, 1.3) prior over the AR coefficient.
        return 0.5 + 0.8 * mx.random.uniform(shape=(n, 1), key=key)

    def param_point_mass(key, n):
        return mx.full((n, 1), A_TRUE)

    return init, trans, logobs, logaux, param_init, param_point_mass


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(M0, math.sqrt(P0))
    for t in range(1, T):
        x[t] = A_TRUE * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return x + rng.normal(0, math.sqrt(R), T)


Y = _data()
LOGZ_TRUE, _, _ = kalman_1d(Y, A_TRUE, Q, R, M0, P0)
Y_MX = mx.array(Y.astype(np.float32))[:, None]

INIT, TRANS, LOGOBS, LOGAUX, PARAM_INIT, PARAM_POINT = _model()


def _run(seed, n=4000, param_init=PARAM_INIT, **kw):
    return smcx.liu_west_filter(
        mx.random.key(seed),
        INIT,
        TRANS,
        LOGOBS,
        LOGAUX,
        param_init,
        Y_MX,
        n,
        **kw,
    )


class TestParameterLearning:
    """The filter's reason to exist."""

    def test_param_posterior_concentrates_near_truth(self):
        post = _run(0, n=8000)
        means = np.array(smcx.param_weighted_mean(post))[:, 0]
        # Final-step posterior mean near the true AR coefficient.
        # Liu-West carries method bias; +-0.08 is the honest band
        # for T=60 observations (posterior sd itself ~0.03-0.05).
        assert abs(means[-1] - A_TRUE) < 0.08
        # And it should have LEARNED: closer than the prior mean.
        assert abs(means[-1] - A_TRUE) < abs(0.9 - means[0]) + 0.05

    def test_param_spread_shrinks_over_time(self):
        post = _run(1, n=8000)
        qs = np.array(
            smcx.param_weighted_quantile(post, mx.array([0.05, 0.95]))
        )
        width = qs[:, 1, 0] - qs[:, 0, 0]
        assert width[-1] < 0.5 * width[0]

    def test_shrinkage_preserves_marginal_spread(self):
        # The discount kernel is variance-MATCHED by construction:
        # shrink by a, jitter with h^2 = 1 - a^2 times the weighted
        # covariance, so the marginal parameter spread is invariant
        # to a (Liu & West 2001, the point of the discount). Assert
        # that: final spreads at a=0.85 and a=0.995 agree within an
        # MC band. A broken kernel fails loudly — shrinkage without
        # jitter collapses the ratio toward 0; jitter without
        # shrinkage inflates it. Measured per-seed ratio sd is ~0.12
        # (both backends), so the 3-seed band [0.7, 1.4] sits >4 SE
        # out. A single-seed strict ordering here is a coin flip —
        # it flipped between local M-series and CI's paravirtual
        # Metal device.
        def mean_width(shrinkage: float) -> float:
            widths = []
            for seed in range(3):
                post = _run(seed, n=4000, shrinkage=shrinkage)
                qs = np.array(
                    smcx.param_weighted_quantile(post, mx.array([0.05, 0.95]))
                )
                widths.append((qs[:, 1, 0] - qs[:, 0, 0])[-10:].mean())
            return float(np.mean(widths))

        ratio = mean_width(0.85) / mean_width(0.995)
        assert 0.7 < ratio < 1.4


class TestPointMassReduction:
    """Point-mass params: Liu-West ~ APF at known parameters."""

    def test_logz_matches_apf_statistically(self):
        # Same algorithm structure, different RNG consumption: the
        # comparison is tier-2 statistical (design §9b), both vs the
        # same fixture: |mean diff| <= 3*sqrt(sd_a^2/R + sd_b^2/R).
        r_keys = 10
        lw_vals = np.array([
            _run(s, n=4000, param_init=PARAM_POINT).marginal_loglik.item()
            for s in range(r_keys)
        ])

        def logaux2(y, s):
            v = Q + R
            return -0.5 * (
                math.log(2 * math.pi * v) + (y[0] - A_TRUE * s[0]) ** 2 / v
            )

        def trans2(key, s):
            return A_TRUE * s + math.sqrt(Q) * mx.random.normal(
                s.shape, key=key
            )

        def logobs2(y, s):
            return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

        apf_vals = np.array([
            smcx.auxiliary_filter(
                mx.random.key(s), INIT, trans2, logobs2, logaux2, Y_MX, 4000
            ).marginal_loglik.item()
            for s in range(r_keys)
        ])
        diff = lw_vals.mean() - apf_vals.mean()
        bound = 3 * math.sqrt(
            lw_vals.std(ddof=1) ** 2 / r_keys
            + apf_vals.std(ddof=1) ** 2 / r_keys
        )
        assert abs(diff) <= bound, (diff, bound)

    def test_logz_gate_vs_kalman_at_true_params(self):
        r_keys = 10
        vals = np.array([
            _run(s, n=4000, param_init=PARAM_POINT).marginal_loglik.item()
            for s in range(r_keys)
        ])
        sd = vals.std(ddof=1)
        err = vals.mean() - LOGZ_TRUE
        upper = 3 * sd / math.sqrt(r_keys)
        assert -(upper + 0.5 * sd**2) <= err <= upper, (err, sd)


class TestStructure:
    """Container, degeneracy, store_history."""

    def test_container_shapes_and_invariants(self):
        post = _run(3, n=500)
        assert isinstance(post, smcx.ParticleFilterResult)
        assert post.filtered_params.shape == (T, 500, 1)
        assert post.filtered_particles.shape == (T, 500, 1)
        total = np.array(post.log_evidence_increments, dtype=np.float64).sum()
        assert post.marginal_loglik.item() == pytest.approx(total, abs=5e-4)
        e = np.array(post.ess)
        assert np.all(e >= 1 - 1e-4) and np.all(e <= 500 * (1 + 1e-4))

    def test_deterministic_per_key(self):
        a = _run(4, n=500)
        b = _run(4, n=500)
        assert a.marginal_loglik.item() == b.marginal_loglik.item()
        assert np.array_equal(
            np.array(a.filtered_params), np.array(b.filtered_params)
        )

    def test_degenerate_raises(self):
        def impossible(y, s, params):
            return mx.array(-mx.inf)

        with pytest.raises(smcx.DegenerateWeightsError):
            smcx.liu_west_filter(
                mx.random.key(5),
                INIT,
                TRANS,
                impossible,
                LOGAUX,
                PARAM_INIT,
                Y_MX,
                200,
            )

    def test_store_history_final_only(self):
        post = _run(6, n=500, store_history=False)
        assert post.filtered_particles.shape == (1, 500, 1)
        assert post.filtered_params.shape == (1, 500, 1)
        assert post.ess.shape == (T,)
