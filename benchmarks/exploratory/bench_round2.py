# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Round 2: roofline bandwidth, full SMC step end-to-end (take vs metal kernel x cadence),
threadgroup sweep, shapeless safety on real step, fixed streams overlap tests.
"""

import math
import time

import mlx.core as mx

mx.random.seed(0)

# ---------- bandwidth roofline ----------
print("=== R0. memory bandwidth roofline (M3 Pro nominal 150 GB/s) ===")
n = 10**8
a = mx.random.normal((n,))
b = mx.random.normal((n,))
mx.eval(a, b)
mx.synchronize()


def add(a, b):
    return a + b


for _ in range(3):
    mx.eval(add(a, b))
mx.synchronize()
t0 = time.perf_counter()
iters = 10
for _ in range(iters):
    mx.eval(add(a, b))
mx.synchronize()
dt = (time.perf_counter() - t0) / iters
print(
    f"a+b at 1e8 f32: {dt * 1e3:.2f} ms -> {(3 * 4 * n) / dt / 1e9:.1f} GB/s effective (12 B/elem)"
)
t0 = time.perf_counter()
for _ in range(iters):
    mx.eval(a * 1.0001)
mx.synchronize()
dt = (time.perf_counter() - t0) / iters
print(
    f"a*c at 1e8 f32: {dt * 1e3:.2f} ms -> {(2 * 4 * n) / dt / 1e9:.1f} GB/s effective (8 B/elem)"
)
del a, b

# random gather bandwidth
n = 10**7
src = mx.random.normal((n,))
idx = mx.random.randint(0, n, (n,))
mx.eval(src, idx)
mx.synchronize()
for _ in range(3):
    mx.eval(mx.take(src, idx))
mx.synchronize()
t0 = time.perf_counter()
for _ in range(iters):
    mx.eval(mx.take(src, idx))
mx.synchronize()
dt = (time.perf_counter() - t0) / iters
print(
    f"random take 1e7: {dt * 1e3:.2f} ms -> {(3 * 4 * n) / dt / 1e9:.1f} GB/s effective; sorted-idx gather:"
)
idx_sorted = mx.sort(idx)
mx.eval(idx_sorted)
t0 = time.perf_counter()
for _ in range(iters):
    mx.eval(mx.take(src, idx_sorted))
mx.synchronize()
dt2 = (time.perf_counter() - t0) / iters
print(
    f"sorted take 1e7: {dt2 * 1e3:.2f} ms -> {(3 * 4 * n) / dt2 / 1e9:.1f} GB/s effective (locality gain {dt / dt2:.2f}x)"
)
del src, idx, idx_sorted

# ---------- metal searchsorted kernel ----------
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


def metal_searchsorted(cdf, u, tg=256):
    (out,) = kernel(
        inputs=[cdf, u],
        template=[("T", mx.float32)],
        grid=(u.shape[0], 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[u.shape],
        output_dtypes=[mx.int32],
    )
    return out


print()
print("=== R1. threadgroup-size sweep, metal searchsorted N=1e6 ===")
N = 10**6
w = mx.random.uniform(shape=(N,))
w = w / mx.sum(w)
cdf = mx.cumsum(w)
cdf = cdf / cdf[-1]
u = (mx.random.uniform(shape=()) + mx.arange(N)) / N
mx.eval(cdf, u)
for tg in (32, 64, 128, 256, 512, 1024):
    for _ in range(3):
        mx.eval(metal_searchsorted(cdf, u, tg))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        mx.eval(metal_searchsorted(cdf, u, tg))
    mx.synchronize()
    print(f"tg={tg:>5}: {(time.perf_counter() - t0) / 30 * 1e3:.3f} ms")


# ---------- full SMC step ----------
def make_step(N, use_metal):
    LOG2 = int(math.ceil(math.log2(N))) + 1

    def take_ss(cdf, u):
        lo = mx.zeros((N,), dtype=mx.int32)
        hi = mx.full((N,), N, dtype=mx.int32)
        for _ in range(LOG2):
            mid = (lo + hi) // 2
            v = mx.take(cdf, mx.clip(mid, 0, N - 1))
            gr = v <= u
            lo = mx.where(gr, mid + 1, lo)
            hi = mx.where(gr, hi, mid)
        return mx.clip(lo, 0, N - 1)

    def step(key, p, logw, y):
        w = mx.softmax(logw)
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        k1, k2 = mx.random.split(key)
        u = (mx.random.uniform(key=k1) + mx.arange(N)) / N
        idx = metal_searchsorted(cdf, u) if use_metal else take_ss(cdf, u)
        p2 = mx.take(p, idx)
        p2 = 0.9 * p2 + mx.random.normal((N,), key=k2) * 0.5
        lw = -0.5 * (y - p2) ** 2
        return p2, lw, mx.logsumexp(lw) - math.log(N)

    return step


def run(N, T, cadence, use_metal):
    step = mx.compile(make_step(N, use_metal))
    keys = mx.random.split(mx.random.key(42), T)
    ys = mx.random.normal((T,))
    p = mx.random.normal((N,))
    logw = mx.zeros((N,))
    mx.eval(keys, ys, p, logw)
    mx.eval(step(keys[0], p, logw, ys[0]))
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    incs = []
    for t in range(T):
        p, logw, inc = step(keys[t], p, logw, ys[t])
        incs.append(inc)
        if cadence == "async":
            mx.async_eval(p, logw)
        elif (t + 1) % cadence == 0:
            mx.eval(p, logw)
    mx.eval(p, logw, incs)
    mx.synchronize()
    return time.perf_counter() - t0, mx.get_peak_memory() / 1e6


print()
print(
    "=== R2. full bootstrap-like step: take vs metal kernel x cadence (T=100) ==="
)
for N in (10_000, 100_000, 1_000_000):
    for use_metal in (False, True):
        for cad in (1, "async"):
            dt, peak = run(N, 100, cad, use_metal)
            lbl = "metal" if use_metal else "take "
            print(
                f"N={N:>8} {lbl} cadence={cad!s:>5}: {dt / 100 * 1e6:8.0f} us/step  peak {peak:7.1f} MB"
            )
    print()

print("=== R3. shapeless=True on the real step: is it safe? ===")
stepfn = make_step(100, False)  # closure bakes N=100 python constants anyway
try:
    c = mx.compile(stepfn, shapeless=True)
    k = mx.random.key(0)
    out = c(k, mx.random.normal((100,)), mx.zeros((100,)), mx.array(0.5))
    mx.eval(out)
    print(
        "shapeless compile of step traced OK at N=100 (but N is baked into arange/log/rounds -> unusable across N; confirmed conceptually)"
    )
except Exception as e:
    print("shapeless step failed:", type(e).__name__, e)

print()
print("=== R4. streams: overlap + small-f64-diag cost (fixed) ===")
CPU = mx.new_stream(mx.cpu)
N = 10**6
T = 50
keys = mx.random.split(mx.random.key(0), T)
p0 = mx.random.normal((N,))
mx.eval(keys, p0)


def gpu_work(p, key):
    y = p
    for _ in range(30):
        y = mx.abs(y * 1.0001 + 0.001)
    return y + mx.random.normal((N,), key=key) * 0.01


f = mx.compile(gpu_work)
mx.eval(f(p0, keys[0]))
mx.synchronize()


def run_mode(mode):
    p = p0
    mx.synchronize()
    t0 = time.perf_counter()
    ds = []
    for t in range(T):
        p = f(p, keys[t])
        if mode == "small_cpu_f64":
            with mx.stream(CPU):
                sub = p[::1000].astype(mx.float64)  # 1000 elems
                d = mx.logsumexp(sub)
            ds.append(d)
        elif mode == "big_cpu_sort":
            with mx.stream(CPU):
                d = mx.sum(mx.sort(p))
            ds.append(d)
        elif mode == "big_cpu_sort_lagged":
            if t % 10 == 0:
                with mx.stream(CPU):
                    d = mx.sum(mx.sort(p))
                ds.append(d)
        mx.async_eval(p)
    mx.eval(p, ds)
    mx.synchronize()
    return time.perf_counter() - t0


for mode in ("none", "small_cpu_f64", "big_cpu_sort", "big_cpu_sort_lagged"):
    run_mode(mode)
    ts = [run_mode(mode) for _ in range(3)]
    print(
        f"mode={mode:>20}: min {min(ts) * 1e3:8.1f} ms ({min(ts) / T * 1e6:6.0f} us/step)"
    )

# pure overlap check: GPU async then CPU work vs serial
print()
big_gpu = lambda: mx.eval(f(p0, keys[0]))
mx.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    y = f(p0, keys[0])
    mx.async_eval(y)
    with mx.stream(CPU):
        z = mx.sum(mx.sort(p0))
    mx.eval(z)
    mx.eval(y)
mx.synchronize()
t_overlap = (time.perf_counter() - t0) / 20
t0 = time.perf_counter()
for _ in range(20):
    y = f(p0, keys[0])
    mx.eval(y)
    with mx.stream(CPU):
        z = mx.sum(mx.sort(p0))
    mx.eval(z)
mx.synchronize()
t_serial = (time.perf_counter() - t0) / 20
print(
    f"GPU step + CPU sort: overlapped {t_overlap * 1e3:.2f} ms vs serialized {t_serial * 1e3:.2f} ms"
)

print()
print(
    "=== R5. conditional resampling: python .item() branch vs branchless where ==="
)


def make_cond_step(N, use_metal, branch):
    LOG2 = int(math.ceil(math.log2(N))) + 1

    def resample(cdf, u):
        if use_metal:
            return metal_searchsorted(cdf, u)
        lo = mx.zeros((N,), dtype=mx.int32)
        hi = mx.full((N,), N, dtype=mx.int32)
        for _ in range(LOG2):
            mid = (lo + hi) // 2
            v = mx.take(cdf, mx.clip(mid, 0, N - 1))
            gr = v <= u
            lo = mx.where(gr, mid + 1, lo)
            hi = mx.where(gr, hi, mid)
        return mx.clip(lo, 0, N - 1)

    def mutate_reweight(key, p, logw_r, y):
        p2 = 0.9 * p + mx.random.normal((N,), key=key) * 0.5
        lw = logw_r + (-0.5 * (y - p2) ** 2)
        return p2, lw

    return resample, mutate_reweight


for N in (10_000, 1_000_000):
    resample, mutre = make_cond_step(N, True, True)
    c_mutre = mx.compile(mutre)
    T = 100
    keys = mx.random.split(mx.random.key(7), T)
    ys = mx.random.normal((T,))
    mx.eval(keys, ys)
    # A: python branch on ess.item(), resample ~always triggered (threshold 1.0)
    p = mx.random.normal((N,))
    logw = mx.zeros((N,))
    mx.eval(p, logw)
    mx.synchronize()
    t0 = time.perf_counter()
    for t in range(T):
        wn = mx.softmax(logw)
        ess = 1.0 / mx.sum(wn**2)
        if ess.item() < 0.5 * N:  # sync!
            cdf = mx.cumsum(wn)
            cdf = cdf / cdf[-1]
            u = (mx.random.uniform(key=keys[t]) + mx.arange(N)) / N
            p = mx.take(p, resample(cdf, u))
            logw = mx.zeros((N,))
        p, logw = c_mutre(keys[t], p, logw, ys[t])
        mx.async_eval(p, logw)
    mx.eval(p, logw)
    mx.synchronize()
    print(
        f"N={N:>8} python-branch (.item ess/step): {(time.perf_counter() - t0) / T * 1e6:8.0f} us/step"
    )
