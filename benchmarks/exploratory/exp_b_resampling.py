# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 2: resampling kernels — binary search vs scatter/cummax counting vs
sort-merge; sorted uniforms via exponential spacings; timings on GPU.
"""

import math
import time

import mlx.core as mx
import numpy as np


def timeit(fn, *args, reps=30, warmup=5):
    for _ in range(warmup):
        mx.eval(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3  # ms


# ---------------- kernels ----------------
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


def systematic_bsearch(w, u0, n):
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    u = (mx.arange(n, dtype=mx.float32) + u0) / n
    return searchsorted(cdf, u, n)


def systematic_counting(w, u0, n):
    """O(N) no-search systematic: offspring counts via ceil, ancestors via
    scatter-max + cummax (Murray-Lee-Jacob offspring->ancestor style).
    """
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    s = mx.ceil(n * cdf - u0).astype(mx.int32)
    s = mx.clip(s, 0, n)
    # force last to n (guards f32 shortfall)
    s = mx.concatenate([s[:-1], mx.array([n], dtype=mx.int32)])
    starts = mx.concatenate([mx.array([0], dtype=mx.int32), s[:-1]])
    has_off = s > starts
    idx = mx.arange(n, dtype=mx.int32)
    vals = mx.where(has_off, idx, -1)
    arr = mx.full((n,), -1, dtype=mx.int32)
    arr = arr.at[mx.clip(starts, 0, n - 1)].maximum(vals)
    return mx.cummax(arr)


def systematic_sortmerge(w, u0, n):
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    u = (mx.arange(n, dtype=mx.float32) + u0) / n
    merged = mx.concatenate([cdf, u])
    order = mx.argsort(merged)
    # inverse permutation via scatter-add into zeros
    inv = mx.zeros((2 * n,), dtype=mx.int32)
    inv = inv.at[order].add(mx.arange(2 * n, dtype=mx.int32))
    # query j sits at merged position inv[n+j]; #cdf entries before it = inv[n+j]-j
    anc = inv[n:] - mx.arange(n, dtype=mx.int32)
    return mx.clip(anc, 0, n - 1)


def sorted_uniforms_expspacing(key, n):
    """Devroye: U_(i) = S_i / S_{n+1}, S = cumsum of iid Exp(1). O(N), no sort."""
    u = mx.random.uniform(shape=(n + 1,), key=key)
    e = -mx.log1p(-u)
    s = mx.cumsum(e)
    return s[:n] / s[n]


def multinomial_expspacing(w, key, n):
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    us = sorted_uniforms_expspacing(key, n)
    return searchsorted(cdf, us, n)


def multinomial_sort(w, key, n):
    cdf = mx.cumsum(w)
    cdf = cdf / cdf[-1]
    us = mx.sort(mx.random.uniform(shape=(n,), key=key))
    return searchsorted(cdf, us, n)


# ---------------- correctness ----------------
print("=" * 70)
print("B1. correctness: counting & sortmerge vs bsearch vs numpy f64")
print("=" * 70)
rng = np.random.default_rng(1)
for n in [10_000, 1_000_000]:
    w64 = rng.exponential(1.0, n)
    w64[:5] *= 1e4  # a few dominant
    w = mx.array((w64 / w64.sum()).astype(np.float32))
    u0v = 0.37
    u0 = mx.array(u0v, dtype=mx.float32)
    a_bs = np.array(systematic_bsearch(w, u0, n))
    a_ct = np.array(systematic_counting(w, u0, n))
    a_sm = np.array(systematic_sortmerge(w, u0, n))
    # f64 reference
    cdf64 = np.cumsum(w64 / w64.sum())
    cdf64 /= cdf64[-1]
    q64 = (np.arange(n) + u0v) / n
    a_ref = np.searchsorted(cdf64, q64, side="right")
    for name, a in [("bsearch", a_bs), ("counting", a_ct), ("sortmerge", a_sm)]:
        mism = int((a != a_ref).sum())
        maxd = int(np.abs(a.astype(np.int64) - a_ref).max()) if mism else 0
        print(
            f"  N={n:>8}: {name:9s} vs f64-ref: {mism} mismatches (max |di|={maxd}); "
            f"vs bsearch: {int((a != a_bs).sum())} diffs"
        )

# unbiasedness of counting scheme, small N many keys
n = 1000
w64 = rng.exponential(1.0, n)
w64[:3] *= 300
wp = w64 / w64.sum()
w = mx.array(wp.astype(np.float32))
counts = np.zeros(n)
K = 500
for k in range(K):
    u0 = mx.random.uniform(key=mx.random.key(k))
    a = np.array(systematic_counting(w, u0, n))
    counts += np.bincount(a, minlength=n)
exp_counts = n * wp
z = counts / K - exp_counts
print(
    f"  counting unbiasedness (N=1000, 500 keys): max |E[c]-Nw| = {np.abs(z).max():.4f} "
    f"(systematic per-key deviation <1 by construction)"
)

# sorted uniforms distribution sanity
n = 1_000_000
us = np.array(sorted_uniforms_expspacing(mx.random.key(7), n))
print(
    f"  exp-spacing sorted uniforms: monotone={bool(np.all(np.diff(us) >= 0))}, "
    f"KS-vs-U(0,1) sup|F-x|={np.abs(us - (np.arange(1, n + 1) / (n + 1))).max():.2e} "
    f"(expect O(1/sqrt N)={1 / np.sqrt(n):.1e})"
)

# ---------------- timing ----------------
print()
print("=" * 70)
print("B2. GPU timings (median ms, compiled)")
print("=" * 70)
for n in [10_000, 100_000, 1_000_000]:
    w64 = rng.exponential(1.0, n)
    w = mx.array((w64 / w64.sum()).astype(np.float32))
    u0 = mx.array(0.5, dtype=mx.float32)
    key = mx.random.key(3)

    f_bs = mx.compile(lambda w, u0, n=n: systematic_bsearch(w, u0, n))
    f_ct = mx.compile(lambda w, u0, n=n: systematic_counting(w, u0, n))
    f_sm = mx.compile(lambda w, u0, n=n: systematic_sortmerge(w, u0, n))
    f_me = mx.compile(lambda w, k, n=n: multinomial_expspacing(w, k, n))
    f_ms = mx.compile(lambda w, k, n=n: multinomial_sort(w, k, n))
    f_sort = mx.compile(
        lambda k, n=n: mx.sort(mx.random.uniform(shape=(n,), key=k))
    )
    f_su = mx.compile(lambda k, n=n: sorted_uniforms_expspacing(k, n))

    t_bs = timeit(f_bs, w, u0)
    t_ct = timeit(f_ct, w, u0)
    t_sm = timeit(f_sm, w, u0)
    t_me = timeit(f_me, w, key)
    t_ms = timeit(f_ms, w, key)
    t_sort = timeit(f_sort, key)
    t_su = timeit(f_su, key)
    print(
        f"  N={n:>9,}: syst-bsearch={t_bs:7.3f}  syst-counting={t_ct:7.3f}  "
        f"syst-sortmerge={t_sm:7.3f}"
    )
    print(
        f"              multi-expspacing={t_me:7.3f}  multi-sort={t_ms:7.3f}  "
        f"(sort-only={t_sort:.3f}, expspacing-only={t_su:.3f})"
    )
