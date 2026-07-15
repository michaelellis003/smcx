# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 7: pitfalls — RNG throughput, vmap vs batched, contiguity copies, sync traps."""

import time

import mlx.core as mx
import numpy as np

mx.random.seed(0)


def bench(fn, *args, iters=30, warmup=5):
    for _ in range(warmup):
        mx.eval(fn(*args))
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


print("=== 7a. RNG throughput (keyed) ===")
key = mx.random.key(0)
for n in (10**6, 10**7):
    t_n = bench(lambda k: mx.random.normal((n,), key=k), key)
    t_u = bench(lambda k: mx.random.uniform(shape=(n,), key=k), key)
    print(
        f"n={n:>9}: normal {t_n * 1e3:7.3f} ms ({n / t_n / 1e9:5.2f} Gsamp/s)   uniform {t_u * 1e3:7.3f} ms ({n / t_u / 1e9:5.2f} Gsamp/s)"
    )
# CPU comparison at 1e7
with mx.stream(mx.cpu):
    t_cpu = bench(lambda k: mx.random.normal((10**7,), key=k), key)
print(
    f"normal 1e7 on CPU stream: {t_cpu * 1e3:.3f} ms ({10**7 / t_cpu / 1e9:.2f} Gsamp/s)"
)

print()
print("=== 7b. vmap per-particle transition vs batched arithmetic ===")
N = 100_000
xs = mx.random.normal((N,))
keys = mx.random.split(mx.random.key(1), N)
mx.eval(xs, keys)


def transition_one(key, x):
    return 0.9 * x + 0.5 * mx.random.normal(key=key)


vm = mx.vmap(transition_one, in_axes=(0, 0))


def batched(key, x):
    return 0.9 * x + 0.5 * mx.random.normal((N,), key=key)


k1 = mx.random.key(2)
t_vm = bench(vm, keys, xs)
t_vm_c = bench(mx.compile(vm), keys, xs)
t_b = bench(batched, k1, xs)
t_b_c = bench(mx.compile(batched), k1, xs)
print(
    f"vmap(per-particle key): {t_vm * 1e3:8.3f} ms  compiled {t_vm_c * 1e3:8.3f} ms"
)
print(
    f"batched (one key):      {t_b * 1e3:8.3f} ms  compiled {t_b_c * 1e3:8.3f} ms"
)
print(
    f"vmap trace/build overhead per call (uncompiled): includes retracing? ratio {t_vm / t_b:.1f}x"
)

print()
print("=== 7c. contiguity: transpose/slice copies ===")
A = mx.random.normal((4000, 4000))
mx.eval(A)
t_c = bench(lambda a: mx.sum(a * 1.0001, axis=1), A)
At = A.T
mx.eval(At)
t_t = bench(lambda a: mx.sum(a * 1.0001, axis=1), At)
print(
    f"row-reduce contiguous: {t_c * 1e3:.3f} ms   on transposed view: {t_t * 1e3:.3f} ms  ratio {t_t / t_c:.2f}x"
)
t_reshape_c = bench(lambda a: mx.reshape(a, (-1,)), A)
t_reshape_t = bench(lambda a: mx.reshape(a, (-1,)), At)
print(
    f"reshape contiguous: {t_reshape_c * 1e3:.3f} ms   reshape transposed (copies): {t_reshape_t * 1e3:.3f} ms"
)
# strided slice
B = A[:, ::2]
mx.eval(B)
t_s = bench(lambda a: mx.sum(a * 1.0001), B)
t_full = bench(lambda a: mx.sum(a * 1.0001), A[:, :2000])
print(
    f"sum over strided slice (4000x2000): {t_s * 1e3:.3f} ms   over contiguous-ish slice: {t_full * 1e3:.3f} ms"
)

print()
print("=== 7d. implicit sync traps ===")
x = mx.random.normal((10**6,))
mx.eval(x)
# scalar python conditional forces eval
f = mx.compile(lambda a: mx.abs(a * 1.0001 + 0.001))
mx.eval(f(x))
mx.synchronize()
t0 = time.perf_counter()
T = 100
y = x
for _ in range(T):
    y = f(y)
    if float(mx.max(y)) > 1e9:  # python conditional -> sync every step
        break
mx.synchronize()
print(
    f"loop with python-float conditional/step: {(time.perf_counter() - t0) / T * 1e6:.0f} us/step"
)
t0 = time.perf_counter()
y = x
for _ in range(T):
    y = f(y)
mx.eval(y)
mx.synchronize()
print(
    f"loop without conditional (eval at end):  {(time.perf_counter() - t0) / T * 1e6:.0f} us/step"
)
# numpy conversion
t0 = time.perf_counter()
for _ in range(20):
    _ = np.array(y)
print(
    f"np.array(1e6 f32, evaluated): {(time.perf_counter() - t0) / 20 * 1e6:.0f} us"
)

print()
print("=== 7e. where-based branchless resample-skip vs python branch ===")
N = 10**6
logw = mx.random.normal((N,))
parts = mx.random.normal((N,))
key = mx.random.key(3)
mx.eval(logw, parts, key)


def full_step_branchless(logw, parts, key, do_resample):
    # always compute resample, select with where (compiled graph fixed)
    w = mx.softmax(logw)
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    u = (mx.random.uniform(key=key) + mx.arange(N)) / N
    lo = mx.zeros((N,), dtype=mx.int32)
    hi = mx.full((N,), N, dtype=mx.int32)
    for _ in range(21):
        mid = (lo + hi) // 2
        v = mx.take(cdf, mx.clip(mid, 0, N - 1))
        gr = v <= u
        lo = mx.where(gr, mid + 1, lo)
        hi = mx.where(gr, hi, mid)
    idx = mx.where(do_resample, mx.clip(lo, 0, N - 1), mx.arange(N))
    return mx.take(parts, idx)


c = mx.compile(full_step_branchless)
t_always = bench(c, logw, parts, key, mx.array(True))
print(
    f"branchless step (resample always computed): {t_always * 1e3:.3f} ms — cost paid even when skipped"
)
