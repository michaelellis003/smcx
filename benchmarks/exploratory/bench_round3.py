# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Round 3: bounded async pipelining, real CPU/GPU overlap, fair conditional-resample, capture API."""

import math
import time
from collections import deque

import mlx.core as mx

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
        threadgroup=(256, 1, 1),
        output_shapes=[u.shape],
        output_dtypes=[mx.int32],
    )
    return out


def make_step(N):
    def step(key, p, logw, y):
        w = mx.softmax(logw)
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        k1, k2 = mx.random.split(key)
        u = (mx.random.uniform(key=k1) + mx.arange(N)) / N
        idx = metal_searchsorted(cdf, u)
        p2 = mx.take(p, idx)
        p2 = 0.9 * p2 + mx.random.normal((N,), key=k2) * 0.5
        lw = -0.5 * (y - p2) ** 2
        return p2, lw, mx.logsumexp(lw) - math.log(N)

    return step


print("=== R6. bounded async pipelining: async_eval + eval of step t-k ===")
for N in (10_000, 1_000_000):
    step = mx.compile(make_step(N))
    T = 100
    keys = mx.random.split(mx.random.key(42), T)
    ys = mx.random.normal((T,))
    for lag in (0, 2, 4, 8, None):  # None = pure async, 0 = eval every step
        p = mx.random.normal((N,))
        logw = mx.zeros((N,))
        mx.eval(keys, ys, p, logw)
        mx.eval(step(keys[0], p, logw, ys[0]))
        mx.synchronize()
        mx.reset_peak_memory()
        q = deque()
        t0 = time.perf_counter()
        incs = []
        for t in range(T):
            p, logw, inc = step(keys[t], p, logw, ys[t])
            incs.append(inc)
            if lag is None:
                mx.async_eval(p, logw)
            elif lag == 0:
                mx.eval(p, logw)
            else:
                mx.async_eval(p, logw)
                q.append((p, logw))
                if len(q) > lag:
                    old = q.popleft()
                    mx.eval(*old)
        mx.eval(p, logw, incs)
        mx.synchronize()
        dt = time.perf_counter() - t0
        lbl = {None: "async-inf", 0: "eval/step"}.get(lag, f"lag-{lag}")
        print(
            f"N={N:>8} {lbl:>9}: {dt / T * 1e6:7.0f} us/step  peak {mx.get_peak_memory() / 1e6:8.1f} MB"
        )
    print()

print("=== R7. genuine CPU/GPU overlap: ~10ms GPU chain vs ~10ms CPU job ===")
CPU = mx.new_stream(mx.cpu)
N = 2 * 10**6
p0 = mx.random.normal((N,))
key = mx.random.key(0)
mx.eval(p0, key)


def gpu_heavy(p, key):
    y = p
    for _ in range(200):
        y = mx.abs(y * 1.0001 + 0.001)
    return y


g = mx.compile(gpu_heavy)
mx.eval(g(p0, key))
mx.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    mx.eval(g(p0, key))
mx.synchronize()
t_gpu = (time.perf_counter() - t0) / 10
cpu_src = mx.random.normal((300_000,))
mx.eval(cpu_src)
with mx.stream(CPU):
    z = mx.sort(cpu_src)
    mx.eval(z)
t0 = time.perf_counter()
for _ in range(10):
    with mx.stream(CPU):
        z = mx.sort(cpu_src)
    mx.eval(z)
t_cpu = (time.perf_counter() - t0) / 10
print(
    f"GPU chain alone: {t_gpu * 1e3:.2f} ms   CPU sort alone: {t_cpu * 1e3:.2f} ms"
)
mx.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    y = g(p0, key)
    mx.async_eval(y)
    with mx.stream(CPU):
        z = mx.sort(cpu_src)
    mx.eval(z)
    mx.eval(y)
mx.synchronize()
t_par = (time.perf_counter() - t0) / 10
print(
    f"overlapped: {t_par * 1e3:.2f} ms  (serial sum would be {(t_gpu + t_cpu) * 1e3:.2f} ms; overlap efficiency "
    f"{(t_gpu + t_cpu - t_par) / min(t_gpu, t_cpu) * 100:.0f}%)"
)

print()
print(
    "=== R8. fair conditional resample: fully-compiled branchless vs compiled-two-phase + .item() ==="
)
for N in (10_000, 1_000_000):
    T = 100
    keys = mx.random.split(mx.random.key(7), T)
    ys = mx.random.normal((T,))
    mx.eval(keys, ys)

    # branchless single compiled step incl. where(select)
    def make_bl(N):
        def stepbl(key, p, logw, y):
            wn = mx.softmax(logw)
            ess = 1.0 / mx.sum(wn**2)
            do = ess < 0.5 * N
            cdf = mx.cumsum(wn)
            cdf = cdf / cdf[-1]
            k1, k2 = mx.random.split(key)
            u = (mx.random.uniform(key=k1) + mx.arange(N)) / N
            idx = mx.where(do, metal_searchsorted(cdf, u), mx.arange(N))
            p2 = mx.take(p, idx)
            logw_r = mx.where(do, mx.zeros((N,)), logw)
            p2 = 0.9 * p2 + mx.random.normal((N,), key=k2) * 0.5
            lw = logw_r + (-0.5 * (y - p2) ** 2)
            return p2, lw

        return stepbl

    sbl = mx.compile(make_bl(N))
    p = mx.random.normal((N,))
    logw = mx.zeros((N,))
    mx.eval(p, logw)
    mx.eval(sbl(keys[0], p, logw, ys[0]))
    mx.synchronize()
    t0 = time.perf_counter()
    for t in range(T):
        p, logw = sbl(keys[t], p, logw, ys[t])
        mx.async_eval(p, logw)
    mx.eval(p, logw)
    mx.synchronize()
    print(
        f"N={N:>8} branchless compiled + async:      {(time.perf_counter() - t0) / T * 1e6:7.0f} us/step"
    )

    # two-phase: compiled ess fn, python branch, compiled resample+mutate OR mutate-only
    def make_parts(N):
        def essf(logw):
            wn = mx.softmax(logw)
            return 1.0 / mx.sum(wn**2)

        def res_mut(key, p, logw, y):
            wn = mx.softmax(logw)
            cdf = mx.cumsum(wn)
            cdf = cdf / cdf[-1]
            k1, k2 = mx.random.split(key)
            u = (mx.random.uniform(key=k1) + mx.arange(N)) / N
            p2 = mx.take(p, metal_searchsorted(cdf, u))
            p2 = 0.9 * p2 + mx.random.normal((N,), key=k2) * 0.5
            lw = -0.5 * (y - p2) ** 2
            return p2, lw

        def mut(key, p, logw, y):
            k1, k2 = mx.random.split(key)
            p2 = 0.9 * p + mx.random.normal((N,), key=k2) * 0.5
            lw = logw + (-0.5 * (y - p2) ** 2)
            return p2, lw

        return mx.compile(essf), mx.compile(res_mut), mx.compile(mut)

    essf, res_mut, mut = make_parts(N)
    p = mx.random.normal((N,))
    logw = mx.zeros((N,))
    mx.eval(p, logw)
    mx.eval(res_mut(keys[0], p, logw, ys[0]))
    mx.eval(mut(keys[0], p, logw, ys[0]))
    mx.eval(essf(logw))
    mx.synchronize()
    t0 = time.perf_counter()
    for t in range(T):
        e = essf(logw)
        if e.item() < 0.5 * N:
            p, logw = res_mut(keys[t], p, logw, ys[t])
        else:
            p, logw = mut(keys[t], p, logw, ys[t])
        mx.async_eval(p, logw)
    mx.eval(p, logw)
    mx.synchronize()
    print(
        f"N={N:>8} two-phase compiled + .item branch: {(time.perf_counter() - t0) / T * 1e6:7.0f} us/step"
    )

print()
print("=== R9. capture API presence ===")
print(
    "start_capture:",
    hasattr(mx.metal, "start_capture"),
    " stop_capture:",
    hasattr(mx.metal, "stop_capture"),
)
print(
    "mx.metal.device_info:",
    mx.metal.device_info() if hasattr(mx.metal, "device_info") else "n/a",
)
import mlx.core as _mx

print(
    "set_wired_limit:",
    hasattr(_mx, "set_wired_limit"),
    " set_memory_limit:",
    hasattr(_mx, "set_memory_limit"),
    " set_cache_limit:",
    hasattr(_mx, "set_cache_limit"),
    " clear_cache:",
    hasattr(_mx, "clear_cache"),
)
