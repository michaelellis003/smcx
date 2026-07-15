# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 1: mx.compile mechanics — fusion, dispatch overhead, shapeless, cache, Python graph-build cost."""

import math
import time

import mlx.core as mx

mx.random.seed(0)


def bench(fn, *args, iters=50, warmup=5):
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


print("=== 1a. elementwise chain: fusion + per-op dispatch overhead ===")
for N in (1024, 100_000, 1_000_000):
    x = mx.random.normal((N,))
    mx.eval(x)
    K = 64

    def chain_ew(x):
        y = x
        for _ in range(K):
            y = mx.abs(y * 1.0001 + 0.001)
        return y

    t_lazy = bench(chain_ew, x)
    t_comp = bench(mx.compile(chain_ew), x)
    print(
        f"N={N:>8} K={K}: uncompiled {t_lazy * 1e3:8.3f} ms  compiled {t_comp * 1e3:8.3f} ms  "
        f"-> per-op overhead approx {(t_lazy - t_comp) / K * 1e6:7.2f} us/op  speedup {t_lazy / t_comp:5.1f}x"
    )

print()
print("=== 1b. gather (take) chains: does compile help when nothing fuses? ===")
for N in (10_000, 100_000, 1_000_000):
    x = mx.random.normal((N,))
    idx = mx.random.randint(0, N, (N,))
    mx.eval(x, idx)
    K = 21  # like the binary-search round count

    def chain_take(x, idx):
        y = x
        for _ in range(K):
            y = mx.take(y, idx) * 1.0001
        return y

    t_lazy = bench(chain_take, x, idx)
    t_comp = bench(mx.compile(chain_take), x, idx)
    print(
        f"N={N:>8} K={K} take+mul rounds: uncompiled {t_lazy * 1e3:8.3f} ms  compiled {t_comp * 1e3:8.3f} ms  speedup {t_lazy / t_comp:4.2f}x"
    )

print()
print("=== 1c. reduction breaking fusion ===")
N = 1_000_000
x = mx.random.normal((N,))
mx.eval(x)


def ew_only(x):
    y = x
    for _ in range(16):
        y = mx.abs(y * 1.0001 + 0.001)
    return y


def ew_with_reductions(x):
    y = x
    for _ in range(4):
        for _ in range(4):
            y = mx.abs(y * 1.0001 + 0.001)
        y = y - mx.logsumexp(y) / N  # scalar reduction in the middle
    return y


print(
    f"16 ew ops compiled:              {bench(mx.compile(ew_only), x) * 1e3:.3f} ms"
)
print(
    f"16 ew ops + 4 logsumexp compiled:{bench(mx.compile(ew_with_reductions), x) * 1e3:.3f} ms"
)
print(
    f"16 ew ops + 4 logsumexp lazy:    {bench(ew_with_reductions, x) * 1e3:.3f} ms"
)

print()
print("=== 1d. compile call overhead + cache across invocations ===")
f = mx.compile(lambda a: a * 2.0 + 1.0)
a = mx.random.normal((16,))
mx.eval(a)
mx.eval(f(a))
mx.synchronize()
t0 = time.perf_counter()
M = 2000
for _ in range(M):
    out = f(a)
    mx.eval(out)
mx.synchronize()
print(
    f"tiny compiled fn, eval each call: {(time.perf_counter() - t0) / M * 1e6:.1f} us/call"
)
t0 = time.perf_counter()
outs = []
for _ in range(M):
    outs.append(f(a))
mx.eval(outs)
mx.synchronize()
print(
    f"tiny compiled fn, eval at end:    {(time.perf_counter() - t0) / M * 1e6:.1f} us/call"
)

# retrace cost on shape change
g = mx.compile(lambda a: mx.abs(a * 1.0001 + 0.001))
for shape in ((1000,), (2000,), (1000,), (2000,)):
    b = mx.random.normal(shape)
    mx.eval(b)
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(g(b))
    mx.synchronize()
    print(
        f"compiled call shape={shape}: {(time.perf_counter() - t0) * 1e6:.0f} us (first call at a shape = trace)"
    )

print()
print("=== 1e. shapeless=True ===")
h = mx.compile(lambda a: mx.abs(a * 1.0001 + 0.001), shapeless=True)
for shape in ((1000,), (2000,), (4000,), (2000,)):
    b = mx.random.normal(shape)
    mx.eval(b)
    mx.synchronize()
    t0 = time.perf_counter()
    mx.eval(h(b))
    mx.synchronize()
    print(
        f"shapeless compiled call shape={shape}: {(time.perf_counter() - t0) * 1e6:.0f} us"
    )
# shapeless risk demo: shape-dependent constant baked in
try:

    def bad(a):
        return a / a.shape[0]  # python int from shape -> baked constant

    hb = mx.compile(bad, shapeless=True)
    r1 = hb(mx.ones((10,)))
    r2 = hb(mx.ones((20,)))
    mx.eval(r1, r2)
    print(
        f"shapeless shape-const bake: f(ones(10))[0]={r1[0].item():.4f} f(ones(20))[0]={r2[0].item():.4f} "
        f"(correct would be 0.1 and 0.05)"
    )
except Exception as e:
    print("shapeless bake test raised:", e)

print()
print("=== 1f. Python graph construction cost per step (no eval) ===")
# representative SMC step graph, uncompiled: time pure graph build (lazy, nothing runs)
N = 100_000
LOG2 = int(math.ceil(math.log2(N))) + 1


def step_graph(key, p, logw, y):
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
    p = mx.take(p, idx)
    p = 0.9 * p + mx.random.normal((N,), key=k2) * 0.5
    lw = -0.5 * (y - p) ** 2
    return p, lw, mx.logsumexp(lw) - math.log(N)


key = mx.random.key(0)
p = mx.random.normal((N,))
logw = mx.zeros((N,))
y = mx.array(1.0)
mx.eval(key, p, logw, y)
t0 = time.perf_counter()
R = 200
for _ in range(R):
    out = step_graph(key, p, logw, y)
t_build = (time.perf_counter() - t0) / R
print(f"uncompiled step graph build (no eval): {t_build * 1e6:.0f} us/step")
del out

cstep = mx.compile(step_graph)
mx.eval(cstep(key, p, logw, y))
t0 = time.perf_counter()
for _ in range(R):
    out2 = cstep(key, p, logw, y)
t_call = (time.perf_counter() - t0) / R
print(f"compiled step call (no eval):          {t_call * 1e6:.0f} us/step")
print(
    f"-> at T=500 that is {t_build * 500 * 1e3:.1f} ms uncompiled vs {t_call * 500 * 1e3:.1f} ms compiled of pure Python"
)
mx.eval(out2)
