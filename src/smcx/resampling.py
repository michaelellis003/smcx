# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Native resampling schemes (ADR-0004 contract, ADR-0009 kernels).

Every scheme has the BlackJAX-compatible signature
``(key, weights, num_samples) -> int32 ancestors`` with
probability-space weights, so smcjax call sites port unchanged.
Ancestor indices from ``systematic``, ``stratified``, and
``multinomial`` are nondecreasing — a design invariant (sorted
gathers run ~4.9x faster than random on M-series; design §5).
``residual`` returns its deterministic block first, then iid
residual draws.

Kernels:

- ``systematic``/``stratified``/``multinomial`` locate their query
  grids / sorted uniforms in the CDF by right binary search: a fused
  ``mx.fast.metal_kernel`` on the GPU (no vmap/vjp — ADR-0009/0017;
  under vmap use the take-chain explicitly), with a pure-MLX
  take-chain fallback elsewhere. Query grids are clamped strictly
  below 1 (``_BELOW_ONE``).
- ``multinomial`` draws already-sorted uniforms in O(N) via
  exponential spacings (Devroye 1986, Ch. V.3.1) using
  ``-log1p(-u)`` — ``mx.random.uniform`` can return exactly 0.

Guards (mlx-audit hazards 1-2): CDFs are normalized by their final
element and all indices are clipped to ``[0, N-1]`` — a raw f32
bisect provably reaches N at N=1e6.
"""

import math

import mlx.core as mx
from jaxtyping import Float, Int32

from smcx.types import KeyT

# Avoids 0/0 on all-zero CDFs (outputs are masked wherever this can
# engage; f32 min normal is ~1.18e-38, see ADR-0003 FTZ note).
_TINY = 1e-30

# Largest f32 below 1. Query grids are clamped here: (u0 + (m-1))/m
# rounds to exactly 1.0 with probability ~ulp(m)/2 per call
# (near-certain at m >= 2^23), and an unclamped right-bisect of 1.0
# returns n — the clip then selects particle n-1 regardless of its
# weight (numerics review, 2026-07-16). The counting formulation's
# counts>0 fill-forward guarded this implicitly.
_BELOW_ONE = 1.0 - 2.0**-24

_METAL_SEARCHSORTED_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)u_shape[0]) return;
    T x = u[i];
    uint lo = 0;
    uint hi = (uint)cdf_shape[0];
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (cdf[mid] <= x) { lo = mid + 1; } else { hi = mid; }
    }
    uint n = (uint)cdf_shape[0];
    out[i] = (int)min(lo, n - 1);
"""

_metal_kernel_cache: list = []


def _normalized_cdf(
    weights: Float[mx.array, " num_particles"],
) -> Float[mx.array, " num_particles"]:
    """Cumulative distribution normalized so the final entry is 1."""
    cdf = mx.cumsum(weights)
    return cdf / mx.maximum(cdf[-1], _TINY)


def _searchsorted_metal(cdf: mx.array, u: mx.array) -> mx.array:
    """Fused right-bisect on GPU (one dispatch; ADR-0009)."""
    if not _metal_kernel_cache:
        _metal_kernel_cache.append(
            mx.fast.metal_kernel(
                name="smcx_searchsorted",
                input_names=["cdf", "u"],
                output_names=["out"],
                source=_METAL_SEARCHSORTED_SRC,
                ensure_row_contiguous=True,
            )
        )
    (out,) = _metal_kernel_cache[0](
        inputs=[cdf, u],
        template=[("T", cdf.dtype)],
        grid=(u.shape[0], 1, 1),
        # 1024 (the Apple-GPU max) measured 1.75x over 256 at N=1e6
        # fresh-process (perf-analysis.md 2026-07-16 late section).
        threadgroup=(min(1024, max(u.shape[0], 1)), 1, 1),
        output_shapes=[u.shape],
        output_dtypes=[mx.int32],
    )
    return out


def _searchsorted_take(cdf: mx.array, u: mx.array) -> mx.array:
    """Portable right-bisect: ceil(log2 N)+1 rounds of mx.take."""
    n = cdf.shape[0]
    lo = mx.zeros(u.shape, dtype=mx.int32)
    hi = mx.full(u.shape, n, dtype=mx.int32)
    for _ in range(math.ceil(math.log2(max(n, 2))) + 1):
        mid = (lo + hi) // 2
        go_right = mx.take(cdf, mx.clip(mid, 0, n - 1)) <= u
        lo = mx.where(go_right, mid + 1, lo)
        hi = mx.where(go_right, hi, mid)
    return mx.clip(lo, 0, n - 1)


def _searchsorted(cdf: mx.array, u: mx.array) -> mx.array:
    """Dispatch: Metal kernel on GPU, take-chain elsewhere."""
    if mx.default_device() == mx.Device(mx.gpu):
        return _searchsorted_metal(cdf, u)
    return _searchsorted_take(cdf, u)


def _sorted_uniforms(key: KeyT, num_samples: int) -> mx.array:
    """Sorted U(0,1) order statistics in O(N), no sort.

    Devroye (1986, Ch. V.3.1): normalized running sums of iid Exp(1)
    spacings. ``-log1p(-u)`` never sees log(0) (uniform can return
    exactly 0). f32 blocked cumsum can produce sub-slot local
    non-monotonicity (<= ~2e-7) — harmless for binary search.
    """
    e = -mx.log1p(-mx.random.uniform(shape=(num_samples + 1,), key=key))
    s = mx.cumsum(e)
    return s[:-1] / mx.maximum(s[-1], _TINY)


def _fill_forward_ancestors(
    starts: mx.array, counts: mx.array, num_samples: int
) -> mx.array:
    """Offspring counts -> monotone ancestors via scatter-max/cummax.

    Scatters each positive-count particle's index at its output start
    position into a -1-filled array, then forward-fills with cummax:
    among particles sharing a start (zero-count collisions), the one
    with offspring carries the largest index, so scatter-max resolves
    ties correctly (Murray-Lee-Jacob 2016 formulation).
    """
    n = counts.shape[0]
    out = mx.full((num_samples,), -1, dtype=mx.int32)
    vals = mx.where(
        counts > 0, mx.arange(n, dtype=mx.int32), mx.array(-1, dtype=mx.int32)
    )
    out = out.at[mx.clip(starts, 0, num_samples - 1)].maximum(vals)
    return mx.cummax(out)


def systematic(
    key: KeyT,
    weights: Float[mx.array, " num_particles"],
    num_samples: int,
) -> Int32[mx.array, " num_samples"]:
    """Systematic resampling via the fused right-bisect (ADR-0017).

    One shared uniform places an evenly spaced grid on the CDF; each
    grid point's ancestor is its exact right-bisect count. Offspring
    deviate by less than one from ``num_samples * weights``
    (Kitagawa 1996). Note
    systematic resampling does not dominate stratified in variance
    for all test functions (Douc & Cappe 2005).

    Args:
        key: PRNG key.
        weights: Normalized probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.
    """
    m = num_samples
    u0 = mx.random.uniform(key=key)
    # ADR-0017 (supersedes the ADR-0009 counting choice here): the
    # offset grid bisected by the shared right-bisect kernel measured
    # 1.7-2.6x over the ceil/cummax/scatter counting chain at
    # N=1e4..1e6 compiled (tg=1024), and the sorted queries keep the
    # per-thread search warp-coherent. Ancestors are exact
    # #{cdf_i <= q_j} counts (clipped), nondecreasing by construction
    # (sorted queries on a sorted CDF), preserving the
    # monotone-gather invariant. Under vmap use the take-chain
    # fallback explicitly (the fused kernel has no vmap; ADR-0009).
    q = mx.minimum((u0 + mx.arange(m)) / m, _BELOW_ONE)
    return _searchsorted(_normalized_cdf(weights), q)


def stratified(
    key: KeyT,
    weights: Float[mx.array, " num_particles"],
    num_samples: int,
) -> Int32[mx.array, " num_samples"]:
    """Stratified resampling: one uniform per stratum.

    Count variance is dominated by multinomial's for every test
    function (Douc & Cappe 2005).

    Args:
        key: PRNG key.
        weights: Normalized probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.
    """
    m = num_samples
    v = mx.random.uniform(shape=(m,), key=key)
    u = mx.minimum((mx.arange(m) + v) / m, _BELOW_ONE)
    return _searchsorted(_normalized_cdf(weights), u)


def multinomial(
    key: KeyT,
    weights: Float[mx.array, " num_particles"],
    num_samples: int,
) -> Int32[mx.array, " num_samples"]:
    """Multinomial (iid) resampling via sorted uniforms.

    Never uses ``mx.random.categorical(num_samples=...)``, whose
    O(N*M) memory is unusable at resampling scale (mlx-audit hazard
    1). Sorted queries also keep the ancestor gather monotone.

    Args:
        key: PRNG key.
        weights: Normalized probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.
    """
    u = mx.minimum(_sorted_uniforms(key, num_samples), _BELOW_ONE)
    return _searchsorted(_normalized_cdf(weights), u)


def residual(
    key: KeyT,
    weights: Float[mx.array, " num_particles"],
    num_samples: int,
) -> Int32[mx.array, " num_samples"]:
    """Residual resampling: guaranteed floor counts + iid remainder.

    Each particle receives ``floor(num_samples * w_i)`` offspring
    deterministically; the remaining draws are iid from the residual
    distribution (Liu & Chen 1998). Count variance is dominated by
    multinomial's. Output is the deterministic block (nondecreasing)
    followed by the residual draws — not globally monotone.

    Args:
        key: PRNG key.
        weights: Normalized probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        int32 ancestor indices.
    """
    m = num_samples
    w = weights / mx.maximum(mx.sum(weights), _TINY)
    scaled = m * w
    floor_counts = mx.floor(scaled).astype(mx.int32)
    cum = mx.cumsum(floor_counts)
    num_deterministic = cum[-1]
    det = _fill_forward_ancestors(cum - floor_counts, floor_counts, m)
    # Fixed-shape residual block: draw m iid uniforms and mask; only
    # positions >= num_deterministic are consumed, so the surplus
    # draws are discarded, keeping shapes static under mx.compile.
    resid = mx.maximum(scaled - floor_counts, 0.0)
    cdf_r = mx.cumsum(resid)
    cdf_r = cdf_r / mx.maximum(cdf_r[-1], _TINY)
    u = mx.random.uniform(shape=(m,), key=mx.random.split(key)[1])
    res_idx = _searchsorted(cdf_r, u)
    return mx.where(mx.arange(m) < num_deterministic, det, res_idx)
