# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics tests (spec: feat-9-diagnostics).

Property patterns follow smcjax's test contract; hand-built
containers exercise exact cases (any ParticleFilterResult is
accepted structurally).
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from smcx.containers import LiuWestPosterior, ParticleFilterPosterior
from tests._kalman import kalman_1d

A, Q, R = 0.9, 0.5, 0.3
M0, P0 = 0.0, 1.0
T = 50


def _model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def init(key, n):
        return M0 + sp * mx.random.normal((n, 1), key=key)

    def trans(key, s):
        return A * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    def emit(key, s):
        return s + math.sqrt(R) * mx.random.normal(s.shape, key=key)

    return init, trans, logobs, emit


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(M0, math.sqrt(P0))
    for t in range(1, T):
        x[t] = A * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return x + rng.normal(0, math.sqrt(R), T)


Y = _data()
_, KMEANS, KVARS = kalman_1d(Y, A, Q, R, M0, P0)
Y_MX = mx.array(Y.astype(np.float32))[:, None]

INIT, TRANS, LOGOBS, EMIT = _model()
POST = smcx.bootstrap_filter(
    mx.random.key(0), INIT, TRANS, LOGOBS, Y_MX, 10_000
)


def _make_posterior(log_w, particles, ancestors=None):
    """Hand-built single-step posterior for exact-case tests."""
    t, n = log_w.shape
    if ancestors is None:
        ancestors = mx.broadcast_to(mx.arange(n, dtype=mx.int32), (t, n))
    return ParticleFilterPosterior(
        marginal_loglik=mx.array(0.0),
        filtered_particles=particles,
        filtered_log_weights=log_w,
        ancestors=ancestors,
        ess=mx.stack([smcx.ess(log_w[i]) for i in range(t)]),
        log_evidence_increments=mx.zeros((t,)),
    )


class TestWeightedSummaries:
    """Against the Kalman oracle and exact hand cases."""

    def test_mean_tracks_kalman(self):
        means = np.array(smcx.weighted_mean(POST))[:, 0]
        assert np.allclose(means, KMEANS, atol=0.15)

    def test_variance_tracks_kalman(self):
        var = np.array(smcx.weighted_variance(POST))[:, 0]
        # Var estimates have larger MC error than means; 30% rel
        # covers ESS ~ 2000 at f32.
        assert np.allclose(var, KVARS, rtol=0.3, atol=0.02)

    def test_quantiles_monotone_and_contain_truth(self):
        qs = mx.array([0.05, 0.5, 0.95])
        quant = np.array(smcx.weighted_quantile(POST, qs))
        assert np.all(np.diff(quant, axis=1) >= -1e-6)
        lo, hi = quant[:, 0, 0], quant[:, 2, 0]
        coverage = np.mean((lo <= KMEANS) & (hi >= KMEANS))
        assert coverage > 0.9

    def test_quantiles_ignore_zero_weight_particles(self):
        # Two live particles at 1.0/3.0; a huge dead outlier at 100.
        lw = mx.array([[0.0, 0.0, -mx.inf]])
        vals = mx.array([[[1.0], [3.0], [100.0]]])
        post = _make_posterior(lw, vals)
        med = smcx.weighted_quantile(post, mx.array([0.5]))[0, 0, 0]
        assert 1.0 <= med.item() <= 3.0

    def test_param_summaries(self):
        lw = mx.zeros((2, 4))
        params = mx.array([
            [[1.0], [2.0], [3.0], [4.0]],
            [[2.0], [2.0], [2.0], [2.0]],
        ])
        post = LiuWestPosterior(
            marginal_loglik=mx.array(0.0),
            filtered_particles=params,
            filtered_log_weights=lw,
            ancestors=mx.zeros((2, 4), dtype=mx.int32),
            ess=mx.full((2,), 4.0),
            log_evidence_increments=mx.zeros((2,)),
            filtered_params=params,
        )
        m = np.array(smcx.param_weighted_mean(post))
        assert m[0, 0] == pytest.approx(2.5, rel=1e-5)
        assert m[1, 0] == pytest.approx(2.0, rel=1e-5)
        qs = np.array(smcx.param_weighted_quantile(post, mx.array([0.5])))
        assert 2.0 <= qs[0, 0, 0] <= 3.0


class TestFaithfulness:
    """Diversity, Pareto-k, tail-ESS."""

    def test_diversity_identity_and_collapsed(self):
        lw = mx.zeros((2, 8))
        vals = mx.zeros((2, 8, 1))
        idpost = _make_posterior(lw, vals)
        assert np.allclose(np.array(smcx.particle_diversity(idpost)), 1.0)
        collapsed = _make_posterior(
            lw, vals, ancestors=mx.zeros((2, 8), dtype=mx.int32)
        )
        assert np.allclose(
            np.array(smcx.particle_diversity(collapsed)), 1.0 / 8
        )

    def test_pareto_k_ordering(self):
        n = 2000
        rng = np.random.default_rng(1)
        gauss = rng.normal(0, 0.5, n).astype(np.float32)
        heavy = (-np.log(rng.uniform(size=n))).astype(np.float32)  # k ~ 1
        uniform = np.zeros(n, dtype=np.float32)
        post = _make_posterior(
            mx.array(np.stack([uniform, gauss, heavy])),
            mx.zeros((3, n, 1)),
        )
        k = np.array(smcx.pareto_k_diagnostic(post))
        assert k[0] < 0.1  # uniform: prior mean ~ 0.05
        assert k[1] < 0.7  # well-behaved
        assert k[2] > 0.7  # pareto(1) importance weights
        assert k[0] < k[1] < k[2]

    def test_tail_ess_uniform_is_q_fraction(self):
        n = 4000
        lw = mx.zeros((1, n))
        vals = mx.random.normal((1, n, 1), key=mx.random.key(2))
        post = _make_posterior(lw, vals)
        te = smcx.tail_ess(post, q=0.05).item()
        # Uniform weights: each tail holds ~q*N effective particles.
        assert te == pytest.approx(0.05 * n, rel=0.15)

    def test_tail_ess_bounded_by_ess(self):
        te = np.array(smcx.tail_ess(POST))
        e = np.array(POST.ess)
        assert np.all(te <= e * (1 + 1e-4))
        assert np.all(te >= 0)


class TestModelComparison:
    """Scores, Bayes factors, replication."""

    def test_cumulative_log_score_ends_at_marginal(self):
        s = np.array(smcx.cumulative_log_score(POST), dtype=np.float64)
        assert s[-1] == pytest.approx(POST.marginal_loglik.item(), abs=5e-4)

    def test_log_bayes_factor(self):
        assert smcx.log_bayes_factor(-10.0, -12.5).item() == pytest.approx(2.5)

    def test_replicated_log_ml_deterministic(self):
        def run(key):
            return smcx.bootstrap_filter(
                key, INIT, TRANS, LOGOBS, Y_MX, 200
            ).marginal_loglik

        vals = smcx.replicated_log_ml(mx.random.key(3), run, 4)
        assert vals.shape == (4,)
        keys = mx.random.split(mx.random.key(3), 4)
        manual = [run(keys[i]).item() for i in range(4)]
        assert np.allclose(np.array(vals), manual)


class TestPredictive:
    """Posterior predictive draws and CRPS."""

    def test_predictive_shapes_and_center(self):
        pred = smcx.posterior_predictive_sample(
            mx.random.key(4), POST, TRANS, EMIT, num_samples=500
        )
        assert pred.shape == (T, 500, 1)
        centers = np.array(mx.mean(pred, axis=1))[:, 0]
        # E[y_{t+1} | y_{1:t}] = A * filtered_mean_t; MC + filter
        # error at 500 draws: sd ~ sqrt(A^2 P + Q + R) ~ 1 => 5*SE
        # ~ 0.22, plus filter-mean error.
        assert np.allclose(centers, A * KMEANS, atol=0.3)

    def test_crps_exact_two_point(self):
        # predictions {0, 1}, obs 0.5: E|Y-y| = 0.5, E|Y-Y'| = 0.5
        # => CRPS = 0.25 exactly.
        val = smcx.crps(mx.array([0.0, 1.0]), 0.5).item()
        assert val == pytest.approx(0.25, rel=1e-6)

    def test_crps_zero_for_perfect_and_nonnegative(self):
        assert smcx.crps(mx.full((8,), 2.0), 2.0).item() == pytest.approx(
            0.0, abs=1e-7
        )
        preds = mx.random.normal((256,), key=mx.random.key(5))
        assert smcx.crps(preds, 0.3).item() >= 0.0


class TestDiagnose:
    """Summary dict and warnings."""

    def test_clean_posterior_no_warnings(self):
        # Uniform weights, identity ancestors: provably clean.
        n = 1000
        post = _make_posterior(
            mx.zeros((4, n)),
            mx.random.normal((4, n, 1), key=mx.random.key(9)),
        )
        report = smcx.diagnose(post)
        assert report["warnings"] == []

    def test_real_run_report_is_self_consistent(self):
        # This fixture legitimately grazes two flags (min ESS ~ 890
        # of 10k just under the 10% default; one step with Pareto-k
        # ~ 0.87): assert the warnings agree with the numbers rather
        # than asserting health the data doesn't have.
        n = 10_000
        report = smcx.diagnose(POST)
        assert report["min_ess"] > 500  # no deep collapse
        assert (report["min_ess"] < 0.1 * n) == any(
            "ESS" in w for w in report["warnings"]
        )
        assert (report["min_diversity"] < 0.1) == any(
            "diversity" in w for w in report["warnings"]
        )
        thresh = min(1.0 - 1.0 / math.log10(n), 0.7)
        assert (report["max_pareto_k"] > thresh) == any(
            "Pareto" in w for w in report["warnings"]
        )

    def test_collapsed_weights_warn(self):
        n = 500
        lw = np.full((3, n), -100.0, dtype=np.float32)
        lw[:, 0] = 0.0  # one particle carries everything
        post = _make_posterior(
            mx.array(lw),
            mx.zeros((3, n, 1)),
            ancestors=mx.zeros((3, n), dtype=mx.int32),
        )
        report = smcx.diagnose(post)
        assert len(report["warnings"]) >= 2
        assert report["min_ess"] == pytest.approx(1.0, rel=1e-3)

    def test_adaptive_pareto_threshold(self):
        # N=500: min(1 - 1/log10(500), 0.7) = min(0.629, 0.7) = 0.629.
        n = 500
        rng = np.random.default_rng(6)
        lw = rng.normal(0, 2.0, (1, n)).astype(np.float32)
        post = _make_posterior(mx.array(lw), mx.zeros((1, n, 1)))
        k = float(np.array(smcx.pareto_k_diagnostic(post))[0])
        report = smcx.diagnose(post)
        thresh = min(1.0 - 1.0 / math.log10(n), 0.7)
        has_warning = any("Pareto" in w for w in report["warnings"])
        assert has_warning == (k > thresh)
