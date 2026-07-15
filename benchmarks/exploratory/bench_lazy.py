# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 2: lazy eval / pipelining — eval cadence k in {1,5,25,T}, async_eval, .item() syncs, graph growth."""

import math
import time

import mlx.core as mx

mx.random.seed(0)


def make_step(N):
    LOG2 = int(math.ceil(math.log2(N))) + 1

    def step(key, p, logw, y):
        w = mx.softmax(logw)
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        k1, k2 = mx.random.split(key)
        u = (mx.random.uniform(key=k1) + mx.arange(N)) / N
        lo = mx.zeros((N,), dtype=mx.int32)
        hi = mx.full((N,), N, dtype=mx.int32)
        for _ in range(LOG2):
            mid = (lo + hi) // 2
            v = mx.take(cdf, mx.clip(mid, 0, N - 1))
            gr = v <= u
            lo = mx.where(gr, mid + 1, lo)
            hi = mx.where(gr, hi, mid)
        idx = mx.clip(lo, 0, N - 1)
        p2 = mx.take(p, idx)
        p2 = 0.9 * p2 + mx.random.normal((N,), key=k2) * 0.5
        lw = -0.5 * (y - p2) ** 2
        ess = 1.0 / mx.sum(mx.softmax(lw) ** 2)
        return p2, lw, mx.logsumexp(lw) - math.log(N), ess

    return step


def run(N, T, cadence, use_item=False):
    """cadence: int k (eval every k steps) or 'async' or 'async2' (lag-2)."""
    step = mx.compile(make_step(N))
    keys = mx.random.split(mx.random.key(42), T)
    ys = mx.random.normal((T,))
    p = mx.random.normal((N,))
    logw = mx.zeros((N,))
    mx.eval(keys, ys, p, logw)
    # warmup trace
    mx.eval(step(keys[0], p, logw, ys[0]))
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    incs = []
    prev = None
    for t in range(T):
        p, logw, inc, ess = step(keys[t], p, logw, ys[t])
        incs.append(inc)
        if use_item:
            _ = ess.item()  # forces full sync each step
        elif cadence == "async":
            mx.async_eval(p, logw, inc)
        elif cadence == "async2":
            mx.async_eval(p, logw, inc)
            if prev is not None:
                mx.eval(
                    prev[0]
                )  # ensure step t-1 done; ess of t-1 available cheaply
            prev = (ess,)
        elif (t + 1) % cadence == 0:
            mx.eval(p, logw)
    mx.eval(p, logw, incs)
    mx.synchronize()
    dt = time.perf_counter() - t0
    return dt, mx.get_peak_memory() / 1e6


print("=== 2a. eval cadence sweep (T=100) ===")
for N in (10_000, 100_000, 1_000_000):
    T = 100
    for cad in (1, 5, 25, 100, "async", "async2"):
        dt, peak = run(N, T, cad)
        label = f"k={cad}" if isinstance(cad, int) else cad
        print(
            f"N={N:>8} T={T} cadence={label:>7}: {dt * 1e3:8.1f} ms total  {dt / T * 1e6:8.0f} us/step  peak {peak:8.1f} MB"
        )
    dt, peak = run(N, T, 1, use_item=True)
    print(
        f"N={N:>8} T={T} cadence=item/step: {dt * 1e3:8.1f} ms total  {dt / T * 1e6:8.0f} us/step  peak {peak:8.1f} MB"
    )
    print()

print("=== 2b. T=500 at N=1e4 (small-N dispatch-bound regime) ===")
for cad in (1, 5, 25, "async"):
    dt, peak = run(10_000, 500, cad)
    label = f"k={cad}" if isinstance(cad, int) else cad
    print(
        f"N=10000 T=500 cadence={label:>6}: {dt * 1e3:8.1f} ms total  {dt / 500 * 1e6:8.0f} us/step  peak {peak:.1f} MB"
    )

print()
print("=== 2c. bare .item() sync cost ===")
x = mx.random.normal((1000,))
mx.eval(x)
mx.synchronize()
s = mx.sum(x)
mx.eval(s)
t0 = time.perf_counter()
for _ in range(1000):
    _ = s.item()  # already evaluated: pure conversion
print(
    f".item() on already-evaluated scalar: {(time.perf_counter() - t0):.3f} us/call".replace(
        "us", "us? actually total ms:"
    )
)
t0 = time.perf_counter()
for _ in range(1000):
    _ = s.item()
print(
    f".item() evaluated scalar: {(time.perf_counter() - t0) * 1e3:.1f} us/call"
)
# unevaluated + gpu busy
t0 = time.perf_counter()
for _ in range(200):
    y = mx.sum(mx.abs(x * 1.0001))
    _ = y.item()
print(
    f".item() forcing tiny graph eval:     {(time.perf_counter() - t0) / 200 * 1e6:.1f} us/call"
)
