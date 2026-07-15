# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Distributions tests (spec: feat-5-distributions; ADR-0012).

Density tests compare against f64 references built from math/numpy
(never another library's runtime output; deterministic, tier-1).
Sampler tests are tier-2 moment tests: tolerance = 5x the derived SE
of the estimator, derivation in a comment at each site.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from smcx import distributions as dist

K0 = mx.random.key(0)


class TestLgamma:
    """Lanczos g=7/n=9 (Godfrey/GSL set; mlx-audit validated)."""

    @pytest.mark.parametrize(
        "x", [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.7, 10.0, 100.0, 1e4]
    )
    def test_matches_math_lgamma(self, x):
        got = dist.lgamma(mx.array(x)).item()
        # abs floor for the exact zeros at x=1, 2 (rel is undefined
        # against 0; audit-measured error there ~1e-7).
        assert got == pytest.approx(math.lgamma(x), rel=3e-6, abs=5e-7)

    @pytest.mark.parametrize("x", [-0.5, -1.5, -2.7])
    def test_reflection_negative_x(self, x):
        got = dist.lgamma(mx.array(x)).item()
        assert got == pytest.approx(math.lgamma(x), rel=1e-4)


class TestLogpdfs:
    """f64 closed-form references, f32-honest tolerances (rel 1e-5)."""

    def test_normal(self):
        x, loc, scale = 1.3, 0.5, 2.0
        ref = (
            -0.5 * math.log(2 * math.pi * scale**2)
            - 0.5 * ((x - loc) / scale) ** 2
        )
        got = dist.normal_logpdf(
            mx.array(x), mx.array(loc), mx.array(scale)
        ).item()
        assert got == pytest.approx(ref, rel=1e-5)

    def test_exponential(self):
        # rate parameterization: log(rate) - rate*x
        got = dist.exponential_logpdf(mx.array(2.0), mx.array(1.5)).item()
        assert got == pytest.approx(math.log(1.5) - 1.5 * 2.0, rel=1e-5)

    def test_laplace(self):
        x, loc, b = 0.7, 0.0, 2.0
        ref = -math.log(2 * b) - abs(x - loc) / b
        got = dist.laplace_logpdf(
            mx.array(x), mx.array(loc), mx.array(b)
        ).item()
        assert got == pytest.approx(ref, rel=1e-5)

    def test_uniform_inside_and_outside_support(self):
        inside = dist.uniform_logpdf(
            mx.array(0.5), mx.array(0.0), mx.array(2.0)
        ).item()
        assert inside == pytest.approx(-math.log(2.0), rel=1e-5)
        outside = dist.uniform_logpdf(
            mx.array(2.5), mx.array(0.0), mx.array(2.0)
        ).item()
        assert outside == float("-inf")

    def test_bernoulli_log1p_form_at_extreme_p(self):
        # p = 1e-7: naive log(1-p) in f32 loses all precision;
        # log1p keeps it. logpmf(0, p) = log1p(-p) ~ -1e-7.
        got = dist.bernoulli_logpmf(mx.array(0.0), mx.array(1e-7)).item()
        assert got == pytest.approx(-1e-7, rel=1e-3)

    def test_categorical(self):
        logits = mx.array([0.1, 1.2, -0.3])
        ref = 1.2 - math.log(sum(math.exp(v) for v in [0.1, 1.2, -0.3]))
        got = dist.categorical_logpmf(mx.array(1), logits).item()
        assert got == pytest.approx(ref, rel=1e-5)

    def test_gamma_shape_rate(self):
        x, a, b = 2.5, 3.0, 1.5
        ref = a * math.log(b) - math.lgamma(a) + (a - 1) * math.log(x) - b * x
        got = dist.gamma_logpdf(mx.array(x), mx.array(a), mx.array(b)).item()
        assert got == pytest.approx(ref, rel=1e-5)

    def test_studentt(self):
        x, df, loc, scale = 1.7, 4.0, 0.5, 2.0
        z = (x - loc) / scale
        ref = (
            math.lgamma((df + 1) / 2)
            - math.lgamma(df / 2)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
            - (df + 1) / 2 * math.log1p(z * z / df)
        )
        got = dist.studentt_logpdf(
            mx.array(x), mx.array(df), mx.array(loc), mx.array(scale)
        ).item()
        assert got == pytest.approx(ref, rel=1e-5)

    def test_mvnormal_matches_numpy_f64(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(3, 3))
        cov = a @ a.T + 3 * np.eye(3)
        mean = np.array([1.0, -2.0, 0.5])
        x = rng.normal(size=3)
        d = 3
        ref = -0.5 * (
            d * math.log(2 * math.pi)
            + np.linalg.slogdet(cov)[1]
            + (x - mean) @ np.linalg.solve(cov, x - mean)
        )
        f = dist.chol_factor(cov)
        got = dist.mvnormal_logpdf(
            mx.array(x.astype(np.float32)),
            mx.array(mean.astype(np.float32)),
            f,
        ).item()
        assert got == pytest.approx(float(ref), rel=1e-4)


class TestCholFactor:
    """The silent-Cholesky guard (numerical-methods.md finding)."""

    def test_spd_reconstructs(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(5, 5))
        cov = a @ a.T + 5 * np.eye(5)
        f = dist.chol_factor(cov)
        l_np = np.array(f.scale_tril, dtype=np.float64)
        assert np.allclose(l_np @ l_np.T, cov, rtol=1e-4, atol=1e-4)

    def test_rank_deficient_never_silent_garbage(self):
        # Rank-2 covariance in d=5 — the post-resampling-collapse
        # case where MLX f32 cholesky returns finite garbage.
        rng = np.random.default_rng(2)
        v = rng.normal(size=(5, 2))
        cov = v @ v.T  # rank 2, PSD
        f = dist.chol_factor(cov)
        l_np = np.array(f.scale_tril, dtype=np.float64)
        recon = l_np @ l_np.T
        # Must reconstruct cov up to the jitter scale, never 23x off.
        assert np.linalg.norm(recon - cov) / np.linalg.norm(cov) < 1e-2

    def test_inverse_consistent(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(4, 4))
        cov = a @ a.T + 4 * np.eye(4)
        f = dist.chol_factor(cov)
        li = np.array(f.inv_scale_tril, dtype=np.float64)
        l_np = np.array(f.scale_tril, dtype=np.float64)
        assert np.allclose(li @ l_np, np.eye(4), atol=1e-4)


class TestSamplers:
    """Tier-2 moment tests, 5*SE tolerances."""

    def test_exponential_mean(self):
        n, rate = 100_000, 2.0
        x = dist.exponential_sample(K0, rate, (n,))
        # SE(mean) = (1/rate)/sqrt(n) = 0.00158; 5*SE = 0.0079
        assert mx.mean(x).item() == pytest.approx(1 / rate, abs=0.008)
        assert mx.min(x).item() >= 0.0

    def test_categorical_frequencies(self):
        n = 100_000
        p = np.array([0.1, 0.6, 0.3])
        idx = dist.categorical_sample(
            K0, mx.array(np.log(p).astype(np.float32)), n
        )
        freq = np.bincount(np.array(idx), minlength=3) / n
        # SE(freq_i) = sqrt(p(1-p)/n) <= 0.00155; 5*SE
        assert np.all(np.abs(freq - p) < 5 * np.sqrt(p * (1 - p) / n))

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 10.0])
    def test_gamma_moments(self, alpha):
        n, rate = 100_000, 1.0
        x = dist.gamma_sample(mx.random.key(int(alpha * 10)), alpha, rate, (n,))
        arr = np.array(x, dtype=np.float64)
        assert np.all(np.isfinite(arr)) and arr.min() > 0
        # mean = a/b, var = a/b^2; SE(mean) = sqrt(a)/sqrt(n);
        # SE(var) ~ var*sqrt(2/n + kurtosis-term) — use 6*SE slack
        # for the variance at small alpha (heavy right tail).
        assert arr.mean() == pytest.approx(alpha, abs=5 * math.sqrt(alpha / n))
        assert arr.var() == pytest.approx(
            alpha, rel=6 * math.sqrt((2 + 6 / alpha) / n)
        )

    def test_mvnormal_sample_covariance(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(3, 3))
        cov = a @ a.T + 3 * np.eye(3)
        f = dist.chol_factor(cov)
        n = 200_000
        x = dist.mvnormal_sample(K0, mx.zeros((3,)), f, (n,))
        emp = np.cov(np.array(x, dtype=np.float64).T)
        # SE of a covariance entry ~ sqrt((c_ii*c_jj + c_ij^2)/n);
        # bound with 5*SE of the largest diagonal (~9): ~0.10
        assert np.allclose(emp, cov, atol=0.15)

    def test_deterministic_per_key(self):
        a = dist.gamma_sample(mx.random.key(9), 2.0, 1.0, (64,))
        b = dist.gamma_sample(mx.random.key(9), 2.0, 1.0, (64,))
        assert np.array_equal(np.array(a), np.array(b))

    def test_gamma_sample_compiles(self):
        fn = mx.compile(lambda k: dist.gamma_sample(k, 2.0, 1.0, (256,)))
        got = fn(mx.random.key(5))
        want = dist.gamma_sample(mx.random.key(5), 2.0, 1.0, (256,))
        assert np.allclose(np.array(got), np.array(want))
