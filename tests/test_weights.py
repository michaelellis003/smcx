# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for log-space weight utilities (spec: feat-1-weights).

Tolerances are f32-honest: mx defaults to float32, which carries ~7
decimal digits; deterministic reductions here are exact to a few ulp
(mlx reductions are blocked/pairwise, see docs/research/mlx-audit.md),
so 1e-5 relative / 1e-6 absolute bounds are comfortable, not tight.
"""

import math

import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import smcx


def _np_logsumexp(x: np.ndarray) -> float:
    """f64 reference logsumexp (max-shifted)."""
    m = np.max(x)
    if np.isneginf(m):
        return float("-inf")
    return float(m + np.log(np.sum(np.exp(x - m))))


class TestLogNormalize:
    """log_normalize: normalizer + shift invariance + degeneracy."""

    def test_normalized_logsumexp_is_zero(self):
        lw = mx.array([-1.0, -2.0, -3.0, -0.5])
        log_norm, _ = smcx.log_normalize(lw)
        assert mx.logsumexp(log_norm).item() == pytest.approx(0.0, abs=1e-6)

    def test_normalizer_matches_f64_reference(self):
        rng = np.random.default_rng(0)
        lw64 = rng.normal(-5.0, 3.0, size=1000)
        _, log_z = smcx.log_normalize(mx.array(lw64.astype(np.float32)))
        assert log_z.item() == pytest.approx(_np_logsumexp(lw64), rel=1e-5)

    def test_shift_invariance_of_normalized_part(self):
        lw = mx.array([-1.0, -2.0, -3.0])
        a, _ = smcx.log_normalize(lw)
        b, _ = smcx.log_normalize(lw + 123.5)
        assert np.allclose(np.array(a), np.array(b), atol=1e-5)

    def test_extreme_spread_no_overflow(self):
        # exp(88.8) overflows f32; the max shift must happen internally.
        lw = mx.array([1000.0, 999.0, 0.0])
        log_norm, log_z = smcx.log_normalize(lw)
        assert log_z.item() == pytest.approx(
            1000.0 + math.log(1.0 + math.e**-1.0), rel=1e-6
        )
        # ulp(1000) in f32 is 6.1e-5: subtracting a ~1000-magnitude
        # normalizer leaves absolute error of that order in each
        # entry, so 0 ± 1e-6 is unattainable at this spread — 2e-4
        # (~3 ulp) is the honest bound.
        assert mx.logsumexp(log_norm).item() == pytest.approx(0.0, abs=2e-4)

    def test_all_degenerate_signals_neg_inf_normalizer(self):
        # Design §6: pure functions do NOT raise; the loop shell
        # raises DegenerateWeightsError on this signal. MLX's LSE
        # returns -inf (not NaN) on all -inf input.
        lw = mx.full((4,), -mx.inf)
        _, log_z = smcx.log_normalize(lw)
        assert math.isinf(log_z.item()) and log_z.item() < 0


class TestNormalize:
    """normalize: probability-space weights sum to one."""

    @pytest.mark.parametrize("n", [10, 1_000, 100_000])
    def test_sums_to_one(self, n):
        key = mx.random.key(n)
        lw = mx.random.normal((n,), key=key) * 5.0
        w = smcx.normalize(lw)
        # Pairwise-summed f32: error ~1e-7 per doubling; 1e-5 is ample.
        assert mx.sum(w).item() == pytest.approx(1.0, abs=1e-5)

    def test_matches_exp_of_log_normalize(self):
        lw = mx.array([-1.0, -2.0, 0.5])
        w = smcx.normalize(lw)
        log_norm, _ = smcx.log_normalize(lw)
        assert np.allclose(np.array(w), np.array(mx.exp(log_norm)))


class TestEss:
    """ess/log_ess via the 2*LSE(l) - LSE(2l) identity."""

    def test_uniform_weights_give_n(self):
        n = 1024
        lw = mx.zeros((n,))
        assert smcx.ess(lw).item() == pytest.approx(n, rel=1e-5)

    def test_uniform_is_shift_invariant(self):
        n = 512
        assert smcx.ess(mx.full((n,), -7.3)).item() == pytest.approx(
            n, rel=1e-5
        )

    def test_one_hot_gives_one(self):
        lw = mx.array([0.0, -mx.inf, -mx.inf, -mx.inf])
        assert smcx.ess(lw).item() == pytest.approx(1.0, rel=1e-5)

    def test_two_equal_particles_give_two(self):
        lw = mx.array([3.0, 3.0, -mx.inf])
        assert smcx.ess(lw).item() == pytest.approx(2.0, rel=1e-5)

    def test_matches_direct_f64_reference(self):
        rng = np.random.default_rng(1)
        lw64 = rng.normal(0.0, 4.0, size=10_000)
        w = np.exp(lw64 - _np_logsumexp(lw64))
        expected = 1.0 / np.sum(w**2)
        got = smcx.ess(mx.array(lw64.astype(np.float32))).item()
        # Identity form error <= 1.6e-6 at N=1e6 (numerical-methods.md);
        # rel=1e-4 covers the f32 input quantization at spread ~4 nats.
        assert got == pytest.approx(expected, rel=1e-4)

    def test_log_ess_is_log_of_ess(self):
        lw = mx.array([-1.0, -2.0, -3.0, -4.0])
        assert smcx.log_ess(lw).item() == pytest.approx(
            math.log(smcx.ess(lw).item()), rel=1e-5
        )

    def test_all_degenerate_gives_nan(self):
        # The isnan(ess) degeneracy signal from design §6.
        lw = mx.full((4,), -mx.inf)
        assert math.isnan(smcx.ess(lw).item())

    @settings(deadline=None, max_examples=50)
    @given(
        seed=st.integers(0, 2**32 - 1),
        n=st.integers(2, 4096),
        scale=st.floats(0.0, 30.0),
    )
    def test_bounds_property(self, seed, n, scale):
        rng = np.random.default_rng(seed)
        lw = mx.array(rng.normal(0.0, scale, size=n).astype(np.float32))
        e = smcx.ess(lw).item()
        # 1 <= ESS <= N up to f32 slack on the identity.
        assert 1.0 - 1e-4 <= e <= n * (1.0 + 1e-4)


class TestCompileVmap:
    """mx.compile equivalence and mx.vmap batching."""

    def test_compiled_equals_uncompiled(self):
        lw = mx.random.normal((1000,), key=mx.random.key(2))

        def pipeline(x):
            log_norm, log_z = smcx.log_normalize(x)
            return log_norm, log_z, smcx.ess(x), smcx.normalize(x)

        got = mx.compile(pipeline)(lw)
        want = pipeline(lw)
        for g, w in zip(got, want, strict=True):
            assert np.allclose(np.array(g), np.array(w), atol=1e-6)

    def test_vmap_over_batch_matches_per_row(self):
        batch = mx.random.normal((8, 256), key=mx.random.key(3))
        got = mx.vmap(smcx.ess)(batch)
        want = mx.stack([smcx.ess(batch[i]) for i in range(8)])
        assert np.allclose(np.array(got), np.array(want), rtol=1e-5)


class TestRuntimeTypechecking:
    """ADR-0007: the beartype hook enforces annotations in tests."""

    def test_wrong_ndim_rejected(self):
        # A (B, N) matrix must be rejected by " num_particles".
        with pytest.raises(Exception, match=r"type[- ]?check|Float"):
            smcx.log_normalize(mx.zeros((3, 4)))

    def test_wrong_dtype_rejected(self):
        with pytest.raises(Exception, match=r"type[- ]?check|Float"):
            smcx.log_normalize(mx.zeros((4,), dtype=mx.int32))
