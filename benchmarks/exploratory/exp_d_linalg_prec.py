# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Areas 5+6+misc: truncated-normal bias check, weighted covariance,
Cholesky jitter, Liu-West lerp, denormals, f16 gathers, Kahan-vs-Neumaier
classic failure, degenerate softmax.
"""

import math
import time

import mlx.core as mx
import numpy as np

print("=" * 70)
print("D0. truncated_normal bias vs exact (f64 erf on CPU stream)")
print("=" * 70)


def phi(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


for lo, hi in [(0.0, 1.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)]:
    exact = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
    t = mx.random.truncated_normal(
        lo, hi, shape=(1 << 22,), key=mx.random.key(5)
    )
    m = float(mx.mean(t).item())
    se = float(mx.array(np.array(t)).std().item()) / math.sqrt(1 << 22)
    print(
        f"  TN[{lo},{hi}]: sample mean={m:.5f}  exact={exact:.5f}  diff={m - exact:+.5f}  (MC se={se:.5f})"
    )

print()
print("=" * 70)
print("D1. weighted covariance: two-pass vs single-pass vs shifted, f32")
print("     N=1e5, d=50, std=1, growing mean offset")
print("=" * 70)
rng = np.random.default_rng(4)
N, d = 100_000, 50
A = rng.standard_normal((d, d)) * 0.3
S_true_dir = A  # correlation structure
Z = rng.standard_normal((N, d))
lw = -rng.exponential(2.0, N)
lw -= lw.max()
w64 = np.exp(lw)
w64 /= w64.sum()
for off in [0.0, 10.0, 100.0, 1000.0, 10000.0]:
    X64 = Z @ A.T + off
    mu64 = w64 @ X64
    Xc = X64 - mu64
    C64 = (
        Xc * w64[:, None]
    ).T @ Xc  # f64 reference (no Bessel; population form)
    X = mx.array(X64.astype(np.float32))
    w = mx.array(w64.astype(np.float32))
    # two-pass
    mu = w @ X
    Xc32 = X - mu
    C2 = (Xc32 * w[:, None]).T @ Xc32
    # single-pass E[xx^T] - mu mu^T
    Exx = (X * w[:, None]).T @ X
    C1 = Exx - mx.outer(mu, mu)
    # shifted single-pass (shift by first particle)
    sh = X[0]
    Xs = X - sh
    mus = w @ Xs
    C3 = (Xs * w[:, None]).T @ Xs - mx.outer(mus, mus)
    fro = np.linalg.norm(C64)
    e2 = np.linalg.norm(np.array(C2, dtype=np.float64) - C64) / fro
    e1 = np.linalg.norm(np.array(C1, dtype=np.float64) - C64) / fro
    e3 = np.linalg.norm(np.array(C3, dtype=np.float64) - C64) / fro
    print(
        f"  offset={off:8.0f}: two-pass={e2:.2e}  single-pass={e1:.2e}  shifted-1pass={e3:.2e}"
    )

print()
print("=" * 70)
print("D2. f32 Cholesky jitter threshold on near-singular particle covariance")
print("=" * 70)
d = 50
# eigenvalues spanning 1 .. lam_min
for lam_min in [1e-4, 1e-6, 1e-7, 1e-8]:
    lams = np.geomspace(1.0, lam_min, d)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    S64 = (Q * lams) @ Q.T
    S32 = S64.astype(np.float32)
    tr_over_d = float(np.trace(S32)) / d
    ok_at = None
    for jit_rel in [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
        S = mx.array(S32) + jit_rel * tr_over_d * mx.eye(d)
        L = mx.linalg.cholesky(S, stream=mx.cpu)
        ln = np.array(L)
        if np.all(np.isfinite(ln)):
            ok_at = jit_rel
            break
    print(
        f"  eig span 1..{lam_min:.0e} (cond {1 / lam_min:.0e}): first success at jitter = "
        f"{ok_at if ok_at is not None else '>1e-4'} * tr/d"
    )

print()
print("=" * 70)
print("D3. Liu-West shrinkage: a*x+(1-a)*m  vs  m+a*(x-m), f32")
print("=" * 70)
a = np.float32(0.995)
for mval, spread in [(1000.0, 0.01), (1e5, 1.0), (1.0, 1e-4)]:
    x64 = mval + rng.standard_normal(100_000) * spread
    m64 = float(np.mean(x64))
    ref = a.astype(np.float64) * x64 + (1 - a.astype(np.float64)) * m64
    x32 = x64.astype(np.float32)
    m32 = np.float32(m64)
    f1 = a * x32 + (np.float32(1) - a) * m32
    f2 = m32 + a * (x32 - m32)
    # error relative to the *spread* (what matters: jitter geometry)
    e1 = np.abs(f1.astype(np.float64) - ref).max() / spread
    e2 = np.abs(f2.astype(np.float64) - ref).max() / spread
    print(
        f"  mean={mval:8.0f} spread={spread:6.0e}: lerp-form err/spread={e1:.2e}   shifted-form err/spread={e2:.2e}"
    )

print()
print("=" * 70)
print("F1. Metal denormal (subnormal) behavior in f32")
print("=" * 70)
tiny = np.float32(1e-44)  # subnormal (min normal = 1.18e-38)
x = mx.array(tiny)
print(f"  input 1e-44 stored as: {x.item():.6e} (numpy f32 keeps {tiny:.6e})")
print(f"  GPU x*1      = {(x * 1.0).item():.6e}")
print(f"  GPU x+x      = {(x + x).item():.6e}")
print(
    f"  GPU exp(-95) = {mx.exp(mx.array(-95.0)).item():.6e}  (true 5.5e-42, subnormal)"
)
print(
    f"  GPU exp(-90) = {mx.exp(mx.array(-90.0)).item():.6e}  (true 8.2e-40, subnormal)"
)
print(f"  GPU log(exp(-95)) = {mx.log(mx.exp(mx.array(-95.0))).item():.4f}")
print(f"  CPU exp(-95) = {mx.exp(mx.array(-95.0), stream=mx.cpu).item():.6e}")
print(
    f"  GPU 1e-38 * 1e-2 = {(mx.array(np.float32(1e-38)) * 0.01).item():.6e} (subnormal product)"
)

print()
print("=" * 70)
print("F2. gather bandwidth: f32 vs f16/bf16 payload, N=1e6, varying d")
print("=" * 70)


def timeit(fn, *args, reps=30, warmup=5):
    for _ in range(warmup):
        mx.eval(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn(*args))
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3


N = 1_000_000
idx = mx.array(rng.integers(0, N, N).astype(np.int32))
for d in [4, 16, 64]:
    X32 = mx.array(rng.standard_normal((N, d)).astype(np.float32))
    X16 = X32.astype(mx.float16)
    Xb16 = X32.astype(mx.bfloat16)
    g32 = mx.compile(lambda X, i: mx.take(X, i, axis=0))
    g16u = mx.compile(
        lambda X, i: mx.take(X, i, axis=0).astype(mx.float32)
    )  # gather f16, upcast
    t32 = timeit(g32, X32, idx)
    t16 = timeit(g32, X16, idx)
    t16u = timeit(g16u, X16, idx)
    tb16 = timeit(g32, Xb16, idx)
    print(
        f"  d={d:3d}: gather f32={t32:.3f} ms   f16={t16:.3f}   f16+upcast={t16u:.3f}   bf16={tb16:.3f}"
    )

print()
print("=" * 70)
print("G1. Kahan vs Neumaier classic failure case (f32)")
print("=" * 70)
seq = np.array([1.0, 3e7, 1.0, -3e7], dtype=np.float32)  # true sum = 2


def kahan(xs):
    s = np.float32(0)
    c = np.float32(0)
    for x in xs:
        y = np.float32(x - c)
        t = np.float32(s + y)
        c = np.float32(np.float32(t - s) - y)
        s = t
    return float(s)


def neumaier(xs):
    s = np.float32(xs[0])
    c = np.float32(0)
    for x in xs[1:]:
        t = np.float32(s + x)
        c = np.float32(
            c
            + (
                np.float32(np.float32(s - t) + x)
                if abs(s) >= abs(x)
                else np.float32(np.float32(x - t) + s)
            )
        )
        s = t
    return float(np.float32(s + c))


print(
    f"  seq=[1, 3e7, 1, -3e7]: true=2  kahan={kahan(seq)}  neumaier={neumaier(seq)}  naive={float(np.sum(seq)):.1f}"
)

print()
print("=" * 70)
print("G2. degenerate weights: softmax/logsumexp all -inf")
print("=" * 70)
allinf = mx.full((4,), -np.inf)
print(f"  logsumexp(all -inf) = {mx.logsumexp(allinf).item()}")
print(f"  softmax(all -inf)   = {np.array(mx.softmax(allinf))}")
print(
    f"  ESS identity on all -inf = {mx.exp(-(mx.logsumexp(2 * (allinf - mx.logsumexp(allinf))))).item()}"
)
