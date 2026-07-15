# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""B follow-ups: monotonicity of exp-spacing uniforms; multinomial via
expspacing+sortmerge; stratified via sortmerge; N=1e7 scaling.
"""

import math
import time

import mlx.core as mx
import numpy as np


def timeit(fn, *args, reps=20, warmup=4):
    for _ in range(warmup):
        mx.eval(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3


def searchsorted(cdf, u, n):
    lo = mx.zeros(u.shape, dtype=mx.int32)
    hi = mx.full(u.shape, n, dtype=mx.int32)
    for _ in range(int(math.ceil(math.log2(n))) + 1):
        mid = (lo + hi) // 2
        v = mx.take(cdf, mx.clip(mid, 0, n - 1))
        go_right = v <= u
        lo = mx.where(go_right, mid + 1, lo)
        hi = mx.where(go_right, hi, mid)
    return mx.clip(lo, 0, n - 1)


def sorted_uniforms_expspacing(key, n):
    u = mx.random.uniform(shape=(n + 1,), key=key)
    e = -mx.log1p(-u)
    s = mx.cumsum(e)
    return s[:n] / s[n]


def sortmerge_anc(cdf, u, n):
    merged = mx.concatenate([cdf, u])
    order = mx.argsort(merged)
    inv = mx.zeros((2 * n,), dtype=mx.int32)
    inv = inv.at[order].add(mx.arange(2 * n, dtype=mx.int32))
    anc = inv[n:] - mx.arange(n, dtype=mx.int32)
    return mx.clip(anc, 0, n - 1)


def systematic_counting(w, u0, n):
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    s = mx.clip(mx.ceil(n * cdf - u0).astype(mx.int32), 0, n)
    s = mx.concatenate([s[:-1], mx.array([n], dtype=mx.int32)])
    starts = mx.concatenate([mx.array([0], dtype=mx.int32), s[:-1]])
    vals = mx.where(s > starts, mx.arange(n, dtype=mx.int32), -1)
    arr = mx.full((n,), -1, dtype=mx.int32)
    arr = arr.at[mx.clip(starts, 0, n - 1)].maximum(vals)
    return mx.cummax(arr)


def systematic_bsearch(w, u0, n):
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    u = (mx.arange(n, dtype=mx.float32) + u0) / n
    return searchsorted(cdf, u, n)


print("=" * 70)
print("B3. monotonicity violations in exp-spacing sorted uniforms (f32 cumsum)")
print("=" * 70)
for n in [100_000, 1_000_000]:
    viol_tot, worst = 0, 0.0
    for k in range(5):
        us = np.array(sorted_uniforms_expspacing(mx.random.key(k), n))
        d = np.diff(us)
        viol = int((d < 0).sum())
        viol_tot += viol
        if viol:
            worst = max(worst, float(-d[d < 0].min()))
    print(
        f"  N={n:>9,}: total violations over 5 keys = {viol_tot}, worst backstep = {worst:.2e} "
        f"(one slot = {1 / n:.1e})"
    )

print()
print("=" * 70)
print(
    "B4. multinomial: expspacing+sortmerge vs expspacing+bsearch vs sort+bsearch"
)
print("=" * 70)
rng = np.random.default_rng(2)
for n in [10_000, 100_000, 1_000_000]:
    w64 = rng.exponential(1.0, n)
    w = mx.array((w64 / w64.sum()).astype(np.float32))
    key = mx.random.key(3)

    def multi_sm(w, k, n=n):
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        us = sorted_uniforms_expspacing(k, n)
        return sortmerge_anc(cdf, us, n)

    def multi_bs(w, k, n=n):
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        us = sorted_uniforms_expspacing(k, n)
        return searchsorted(cdf, us, n)

    def strat_sm(w, k, n=n):
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        u = (
            mx.arange(n, dtype=mx.float32)
            + mx.random.uniform(shape=(n,), key=k)
        ) / n
        return sortmerge_anc(cdf, u, n)

    def strat_bs(w, k, n=n):
        cdf = mx.cumsum(w)
        cdf = cdf / cdf[-1]
        u = (
            mx.arange(n, dtype=mx.float32)
            + mx.random.uniform(shape=(n,), key=k)
        ) / n
        return searchsorted(cdf, u, n)

    f1 = mx.compile(multi_sm)
    f2 = mx.compile(multi_bs)
    f3 = mx.compile(strat_sm)
    f4 = mx.compile(strat_bs)
    # correctness: multi_sm vs multi_bs on same key
    a1 = np.array(multi_sm(w, key))
    a2 = np.array(multi_bs(w, key))
    print(
        f"  N={n:>9,}: multi sm-vs-bs diffs={int((a1 != a2).sum())} | "
        f"t(multi-sm)={timeit(f1, w, key):6.3f}  t(multi-bs)={timeit(f2, w, key):6.3f}  "
        f"t(strat-sm)={timeit(f3, w, key):6.3f}  t(strat-bs)={timeit(f4, w, key):6.3f}"
    )

print()
print("=" * 70)
print("B5. N=1e7: counting vs bsearch (systematic), peak memory")
print("=" * 70)
n = 10_000_000
w64 = rng.exponential(1.0, n)
w = mx.array((w64 / w64.sum()).astype(np.float32))
u0 = mx.array(0.5, dtype=mx.float32)
f_ct = mx.compile(lambda w, u0, n=n: systematic_counting(w, u0, n))
f_bs = mx.compile(lambda w, u0, n=n: systematic_bsearch(w, u0, n))
mx.reset_peak_memory()
t_ct = timeit(f_ct, w, u0, reps=10)
m_ct = mx.get_peak_memory() / 1e6
mx.reset_peak_memory()
t_bs = timeit(f_bs, w, u0, reps=10)
m_bs = mx.get_peak_memory() / 1e6
a1 = np.array(f_ct(w, u0))
a2 = np.array(f_bs(w, u0))
print(
    f"  N=1e7: counting={t_ct:.2f} ms ({m_ct:.0f} MB peak)   bsearch={t_bs:.2f} ms ({m_bs:.0f} MB peak)   diffs={int((a1 != a2).sum())}"
)
