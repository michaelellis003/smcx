# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 4: mx.fast.metal_kernel — fused binary-search searchsorted vs 21-round mx.take version."""

import math
import time

import mlx.core as mx
import numpy as np

mx.random.seed(0)

SRC = """
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

kernel = mx.fast.metal_kernel(
    name="ssorted",
    input_names=["cdf", "u"],
    output_names=["out"],
    source=SRC,
    ensure_row_contiguous=True,
)


def metal_searchsorted(cdf, u):
    (out,) = kernel(
        inputs=[cdf, u],
        template=[("T", mx.float32)],
        grid=(u.shape[0], 1, 1),
        threadgroup=(min(256, u.shape[0]), 1, 1),
        output_shapes=[u.shape],
        output_dtypes=[mx.int32],
    )
    return out


def take_searchsorted(cdf, u, n):
    lo = mx.zeros(u.shape, dtype=mx.int32)
    hi = mx.full(u.shape, n, dtype=mx.int32)
    for _ in range(int(math.ceil(math.log2(n))) + 1):
        mid = (lo + hi) // 2
        v = mx.take(cdf, mx.clip(mid, 0, n - 1))
        gr = v <= u
        lo = mx.where(gr, mid + 1, lo)
        hi = mx.where(gr, hi, mid)
    return mx.clip(lo, 0, n - 1)


def bench(fn, *args, iters=50, warmup=5):
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


print("=== correctness check ===")
for N in (10_000, 1_000_000):
    w = mx.random.uniform(shape=(N,))
    w = w / mx.sum(w)
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    u = mx.sort(mx.random.uniform(shape=(N,)))
    mx.eval(cdf, u)
    a = metal_searchsorted(cdf, u)
    b = take_searchsorted(cdf, u, N)
    ref = np.searchsorted(np.array(cdf), np.array(u), side="right")
    ref = np.clip(ref, 0, N - 1)
    mx.eval(a, b)
    am = np.mean(np.array(a) == ref)
    bm = np.mean(np.array(b) == ref)
    print(
        f"N={N}: metal matches np {am * 100:.4f}%  take-version matches np {bm * 100:.4f}%  metal==take {np.mean(np.array(a) == np.array(b)) * 100:.4f}%"
    )

print()
print("=== searchsorted alone ===")
for N in (10_000, 100_000, 1_000_000):
    w = mx.random.uniform(shape=(N,))
    w = w / mx.sum(w)
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    u0 = mx.random.uniform(shape=())
    u = (u0 + mx.arange(N)) / N  # systematic points
    mx.eval(cdf, u)
    t_take = bench(mx.compile(lambda c, q: take_searchsorted(c, q, N)), cdf, u)
    t_metal = bench(metal_searchsorted, cdf, u)
    print(
        f"N={N:>8}: take-21-rounds(compiled) {t_take * 1e3:8.3f} ms   metal_kernel {t_metal * 1e3:8.3f} ms   speedup {t_take / t_metal:5.2f}x"
    )

print()
print("=== full resample pipeline (normalize+cumsum+search+gather) ===")
for N in (10_000, 100_000, 1_000_000):
    logw = mx.random.normal((N,))
    parts = mx.random.normal((N,))
    key = mx.random.key(1)
    mx.eval(logw, parts, key)

    def pipeline_take(logw, parts, key):
        w = mx.softmax(logw)
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        u = (mx.random.uniform(key=key) + mx.arange(N)) / N
        idx = take_searchsorted(cdf, u, N)
        return mx.take(parts, idx)

    def pipeline_metal(logw, parts, key):
        w = mx.softmax(logw)
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        u = (mx.random.uniform(key=key) + mx.arange(N)) / N
        idx = metal_searchsorted(cdf, u)
        return mx.take(parts, idx)

    t_take = bench(mx.compile(pipeline_take), logw, parts, key)
    t_metal = bench(
        pipeline_metal, logw, parts, key
    )  # metal kernel not compilable? try
    try:
        t_metal_c = bench(mx.compile(pipeline_metal), logw, parts, key)
    except Exception as e:
        t_metal_c = float("nan")
        print("  compile(pipeline_metal) failed:", type(e).__name__, e)
    print(
        f"N={N:>8}: pipeline take {t_take * 1e3:8.3f} ms   pipeline metal {t_metal * 1e3:8.3f} ms   "
        f"pipeline metal compiled {t_metal_c * 1e3:8.3f} ms   speedup {t_take / t_metal:5.2f}x"
    )
