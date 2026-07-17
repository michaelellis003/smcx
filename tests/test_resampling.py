# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resampling (spec: feat-2-resampling; ADR-0004/0009).

Statistical tolerances are MC-error-honest per AGENTS.md: unbiasedness
uses 5x the derived standard error of the mean count. For multinomial,
Var(count_i) = M*w_i*(1-w_i), so the SE of the K-key mean is
sqrt(M*w_i*(1-w_i)/K); stratified/systematic/residual counts have
variance at most the multinomial's (Douc & Cappe 2005), so the same
bound is conservative for all schemes.
"""

import mlx.core as mx
import numpy as np
import pytest

import smcx
from smcx import resampling

SCHEMES = [
    smcx.systematic,
    smcx.stratified,
    smcx.multinomial,
    smcx.residual,
]
IDS = ["systematic", "stratified", "multinomial", "residual"]

WEIGHTS_SMALL = np.array([0.05, 0.35, 0.10, 0.30, 0.20], dtype=np.float32)


def _counts(idx: mx.array, n: int) -> np.ndarray:
    return np.bincount(np.array(idx), minlength=n)


class TestContract:
    """ADR-0004: (key, weights, num_samples) -> int32 ancestors."""

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_shape_dtype_bounds(self, fn):
        w = mx.array(WEIGHTS_SMALL)
        idx = fn(mx.random.key(0), w, 12)
        assert idx.shape == (12,)
        assert idx.dtype == mx.int32
        arr = np.array(idx)
        assert arr.min() >= 0 and arr.max() < 5

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_deterministic_per_key(self, fn):
        w = mx.array(WEIGHTS_SMALL)
        a = fn(mx.random.key(7), w, 32)
        b = fn(mx.random.key(7), w, 32)
        assert np.array_equal(np.array(a), np.array(b))

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_one_hot_weights_select_only_that_index(self, fn):
        w = mx.array([0.0, 0.0, 1.0, 0.0], dtype=mx.float32)
        idx = fn(mx.random.key(1), w, 16)
        assert np.all(np.array(idx) == 2)

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_zero_weight_particles_never_selected(self, fn):
        w = mx.array([0.5, 0.0, 0.25, 0.0, 0.25], dtype=mx.float32)
        idx = fn(mx.random.key(2), w, 64)
        got = set(np.array(idx).tolist())
        assert 1 not in got and 3 not in got

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_bounds_at_1e6_adversarial_tail(self, fn):
        # f32 CDF ends at ~0.9999995: raw bisect can hit N (mlx-audit
        # hazard 2). A few dominant weights + a huge near-zero tail.
        n = 1_000_000
        w = mx.full((n,), 1e-8)
        w = mx.where(mx.arange(n) < 3, mx.array(0.33), w)
        w = w / mx.sum(w)
        idx = fn(mx.random.key(3), w, n)
        arr = np.array(idx)
        assert arr.min() >= 0 and arr.max() < n


class TestMonotonicity:
    """Design §5: monotone ancestors (gather locality invariant)."""

    @pytest.mark.parametrize(
        "fn", SCHEMES[:3], ids=IDS[:3]
    )  # residual: two monotone segments, documented
    def test_ancestors_nondecreasing(self, fn):
        n = 100_000
        w = mx.random.uniform(shape=(n,), key=mx.random.key(4))
        w = w / mx.sum(w)
        idx = np.array(fn(mx.random.key(5), w, n))
        assert np.all(np.diff(idx) >= 0)


class TestExactCases:
    """Deterministic identities."""

    def test_systematic_uniform_weights_is_arange(self):
        # cdf_i = (i+1)/N, u0 in (0,1): every particle gets exactly
        # one offspring, so ancestors == arange(N).
        n = 64
        w = mx.full((n,), 1.0 / n)
        idx = smcx.systematic(mx.random.key(6), w, n)
        assert np.array_equal(np.array(idx), np.arange(n))

    def test_residual_deterministic_part_guaranteed(self):
        # floor(M*w) copies of each particle are guaranteed.
        w = mx.array(WEIGHTS_SMALL)
        m = 20
        floor_counts = np.floor(m * WEIGHTS_SMALL).astype(int)
        for key in range(5):
            c = _counts(smcx.residual(mx.random.key(key), w, m), 5)
            assert np.all(c >= floor_counts)
            assert c.sum() == m


class TestUnbiasedness:
    """E[counts] = M*w within 5*SE (derivation in module docstring)."""

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_expected_counts(self, fn):
        w64 = np.array([0.02, 0.08, 0.4, 0.3, 0.15, 0.05])
        w = mx.array(w64.astype(np.float32))
        m, k = 60, 600
        total = np.zeros(6)
        for s in range(k):
            total += _counts(fn(mx.random.key(s), w, m), 6)
        mean = total / k
        se = np.sqrt(m * w64 * (1 - w64) / k)
        assert np.all(np.abs(mean - m * w64) <= 5 * se + 1e-9)


class TestVarianceOrdering:
    """Theorem-backed count-variance orderings only.

    stratified <= multinomial and residual <= multinomial (Douc &
    Cappe 2005). Systematic is deliberately NOT ordered against
    stratified (counter-example exists; see design §9).
    """

    def _count_var(self, fn, w64, m, k):
        w = mx.array(w64.astype(np.float32))
        counts = np.stack([
            _counts(fn(mx.random.key(s), w, m), len(w64)) for s in range(k)
        ])
        return counts.var(axis=0).sum()

    def test_orderings(self):
        w64 = np.array([0.02, 0.08, 0.4, 0.3, 0.15, 0.05])
        m, k = 60, 800
        v_mult = self._count_var(smcx.multinomial, w64, m, k)
        v_strat = self._count_var(smcx.stratified, w64, m, k)
        v_resid = self._count_var(smcx.residual, w64, m, k)
        # Gaps on this fixture are large (multinomial var ~ m*w*(1-w)
        # ~ 3.4 total; stratified/residual < 1); K=800 keeps 5*SE of
        # the variance estimate well under the gap.
        assert v_strat < v_mult
        assert v_resid < v_mult


class TestKernelPaths:
    """ADR-0009: Metal kernel and take-chain fallback agree exactly."""

    def test_metal_and_fallback_identical(self):
        if mx.default_device() != mx.Device(mx.gpu):
            pytest.skip("GPU-only comparison")
        n = 50_000
        cdf = mx.cumsum(mx.full((n,), 1.0 / n))
        cdf = cdf / cdf[-1]
        u = mx.sort(mx.random.uniform(shape=(n,), key=mx.random.key(8)))
        a = resampling._searchsorted_metal(cdf, u)
        b = resampling._searchsorted_take(cdf, u)
        assert np.array_equal(np.array(a), np.array(b))

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_fallback_path_on_cpu_device(self, fn):
        prev = mx.default_device()
        mx.set_default_device(mx.Device(mx.cpu))
        try:
            w = mx.array(WEIGHTS_SMALL)
            idx = fn(mx.random.key(9), w, 16)
            arr = np.array(idx)
            assert arr.min() >= 0 and arr.max() < 5
        finally:
            mx.set_default_device(prev)


class TestCompile:
    """Every scheme compiles with static shapes."""

    @pytest.mark.parametrize("fn", SCHEMES, ids=IDS)
    def test_compiled_equals_uncompiled(self, fn):
        w = mx.array(WEIGHTS_SMALL)
        key = mx.random.key(10)
        got = mx.compile(lambda k, ww: fn(k, ww, 24))(key, w)
        want = fn(key, w, 24)
        assert np.array_equal(np.array(got), np.array(want))


class TestSortedUniforms:
    """Devroye exponential-spacings helper (numerical-methods.md §2)."""

    def test_sorted_and_in_unit_interval(self):
        u = resampling._sorted_uniforms(mx.random.key(11), 100_000)
        arr = np.array(u)
        assert arr.min() >= 0.0 and arr.max() < 1.0
        # f32 blocked cumsum: sub-slot local non-monotonicity is
        # possible and harmless for search; sortedness must hold to
        # within one representable step here at 1e5.
        assert np.all(np.diff(arr) >= -2e-7)

    def test_distribution_matches_uniform_order_stats(self):
        # Mean of sorted uniforms ~ j/(m+1); spot-check the median.
        u = resampling._sorted_uniforms(mx.random.key(12), 200_001)
        med = np.array(u)[100_000]
        # SE of the sample median of U(0,1) at m=2e5 is ~1/(2*sqrt(m))
        # ~ 1.1e-3; 5*SE bound.
        assert abs(med - 0.5) < 5.6e-3


class TestSystematicBisectSemantics:
    """ADR-0017: systematic follows exact right-bisect semantics.

    The ancestor of grid point q_j = (u0 + j)/m is the count of cdf
    entries <= q_j (clipped to N-1) — checked against a NumPy oracle
    run on the same f32 CDF and the same u0, so equality is exact.
    """

    @pytest.mark.parametrize("seed", [0, 7])
    @pytest.mark.parametrize("peaked", [False, True])
    def test_matches_numpy_right_bisect_oracle(self, seed, peaked):
        n = 5_000
        key = mx.random.key(seed)
        w_key = mx.random.key(seed + 100)
        raw = mx.random.uniform(shape=(n,), key=w_key) + 1e-3
        if peaked:
            raw = raw * raw * raw * raw  # concentrate mass
        w = raw / mx.sum(raw)
        ancestors = np.array(resampling.systematic(key, w, n))

        u0 = np.float32(mx.random.uniform(key=key).item())
        cdf = np.array(resampling._normalized_cdf(w))
        # Match the kernel's f32 op order exactly: (u0 + j) then / n,
        # then the sub-1 endpoint clamp (numerics review).
        q = (u0 + np.arange(n, dtype=np.float32)) / np.float32(n)
        q = np.minimum(q, np.float32(1.0) - np.float32(2**-24))
        expected = np.clip(np.searchsorted(cdf, q, side="right"), 0, n - 1)
        assert np.array_equal(ancestors, expected)


class TestGridEndpointGuard:
    """The query grid must stay strictly below 1.0 (numerics review).

    ``(u0 + (m-1))/m`` rounds to exactly 1.0 in f32 with probability
    ~ulp(m)/2 per call (near-certain at m >= 2^23); an unclamped
    right-bisect then returns n and the clip pins the ancestor to
    particle n-1 regardless of its weight — selecting a
    zero-probability ancestor when trailing weights are zero. The
    counting formulation's counts>0 fill-forward could never do this.
    """

    def test_never_selects_zero_weight_trailing_particle(self):
        m = 2**24
        w = mx.array([1.0] + [0.0] * 7)
        hit_edge = False
        for seed in range(24):
            key = mx.random.key(seed)
            u0 = np.float32(mx.random.uniform(key=key).item())
            q_last = (u0 + np.float32(m - 1)) / np.float32(m)
            if q_last == np.float32(1.0):
                hit_edge = True
                ancestors = resampling.systematic(key, w, m)
                assert int(mx.max(ancestors).item()) == 0
                break
        assert hit_edge, "no edge key found in 24 seeds (P ~ 2^-24)"
