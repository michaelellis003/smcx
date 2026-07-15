# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 3: normal/gamma/other samplers in f32."""

import math

import mlx.core as mx
import numpy as np

print("=" * 70)
print("C1. mx.random.normal tail reach + implementation probe")
print("=" * 70)
# empirical max |z| over 8 x 2^24 draws (n=1.34e8, expected continuous max ~ 5.67)
mx_max = 0.0
for s in range(8):
    z = mx.random.normal(shape=(1 << 24,), key=mx.random.key(s))
    mx_max = max(mx_max, float(mx.max(mx.abs(z)).item()))
n_tot = 8 * (1 << 24)
# expected max for continuous N(0,1): Phi^-1(1 - 1/(2n)) approx
from math import sqrt

exp_max = sqrt(2) * float(
    np.array(
        mx.erfinv(mx.array(1 - 1.0 / n_tot, dtype=mx.float64), stream=mx.cpu)
    )
)
print(
    f"  max |z| over {n_tot:.2e} draws: {mx_max:.4f}  (continuous-theory expected ~{exp_max:.2f})"
)
# what's the largest value sqrt(2)*erfinv can produce in f32?
one_m = np.float32(1.0) - np.float32(2**-24)
z_cap = float(
    (math.sqrt(2) * mx.erfinv(mx.array(one_m, dtype=mx.float32))).item()
)
print(
    f"  sqrt(2)*erfinv(1-2^-24) in f32 = {z_cap:.4f}  <- cap if normal = erfinv(uniform)"
)
print(f"  erfinv(1.0) f32 = {mx.erfinv(mx.array(1.0)).item()}")
# Box-Muller cap given observed u_min ~ 1e-9..2^-32:
print(
    f"  Box-Muller cap sqrt(-2 log 2^-32) = {math.sqrt(2 * 32 * math.log(2)):.3f}"
)

print()
print("=" * 70)
print("C2. erfinv f32 accuracy vs f64 (CPU stream) incl. tail")
print("=" * 70)
xs = np.concatenate([
    np.linspace(-0.999, 0.999, 4001),
    1 - np.geomspace(1e-7, 1e-3, 200),
    -(1 - np.geomspace(1e-7, 1e-3, 200)),
])
r32 = np.array(mx.erfinv(mx.array(xs.astype(np.float32))))
r64 = np.array(mx.erfinv(mx.array(xs, dtype=mx.float64), stream=mx.cpu))
rel = np.abs(r32 - r64) / np.maximum(np.abs(r64), 1e-30)
print(f"  central |x|<=0.999: max rel err = {rel[:4001].max():.2e}")
print(f"  tail 1-x in [1e-7,1e-3]: max rel err = {rel[4001:].max():.2e}")
# z-accuracy at the deepest reachable tail
for eps in [1e-3, 1e-5, 1e-7, 2**-24]:
    x32 = np.float32(1 - eps)
    z32 = float(mx.erfinv(mx.array(x32)).item())
    z64 = float(
        np.array(
            mx.erfinv(
                mx.array(np.float64(x32), dtype=mx.float64), stream=mx.cpu
            )
        )
    )
    print(
        f"  erfinv(1-{eps:.1e}): f32={z32:.6f} f64={z64:.6f} relerr={abs(z32 - z64) / z64:.2e}"
    )

print()
print("=" * 70)
print("C3. mx.random.gumbel / exponential-from-uniform safety")
print("=" * 70)
ninf = 0
nnan = 0
gmax = -1e9
gmin = 1e9
for s in range(4):
    g = mx.random.gumbel(shape=(1 << 24,), key=mx.random.key(s))
    ninf += int(mx.sum(mx.isinf(g)).item())
    nnan += int(mx.sum(mx.isnan(g)).item())
    gmax = max(gmax, float(mx.max(g).item()))
    gmin = min(gmin, float(mx.min(g).item()))
print(
    f"  gumbel over {4 * (1 << 24):.1e} draws: inf={ninf} nan={nnan} range=[{gmin:.3f},{gmax:.3f}]"
)
u = mx.random.uniform(shape=(1 << 24,), key=mx.random.key(0))
e1 = -mx.log(u)  # unsafe if u==0
e2 = -mx.log1p(-u)  # safe: u <= 1-2^-24
print(
    f"  -log(u): inf count={int(mx.sum(mx.isinf(e1)).item())}  max={mx.max(e1).item():.3f}"
)
print(
    f"  -log1p(-u): inf count={int(mx.sum(mx.isinf(e2)).item())}  max={mx.max(e2).item():.3f} (cap=-log(2^-24)={24 * math.log(2):.2f})"
)

print()
print("=" * 70)
print("C4. truncated_normal in far tail")
print("=" * 70)
for lo, hi in [(0.0, 1.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0)]:
    t = mx.random.truncated_normal(
        lo, hi, shape=(1 << 20,), key=mx.random.key(1)
    )
    tn = np.array(t)
    bad = int(((tn < lo) | (tn > hi)).sum()) + int(np.isnan(tn).sum())
    print(
        f"  [{lo},{hi}]: mean={tn.mean():.4f} min={tn.min():.4f} max={tn.max():.4f} out-of-range/nan={bad}"
    )

print()
print("=" * 70)
print("C5. Marsaglia-Tsang gamma, fixed-round masked rejection (pure MLX)")
print("=" * 70)


def gamma_mt(key, alpha, n, rounds):
    """Alpha >= 1 Marsaglia-Tsang; fixed number of masked rounds; returns
    (sample, unresolved_mask_count, per-round acceptance est).
    """
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    x = mx.zeros((n,))
    done = mx.zeros((n,), dtype=mx.bool_)
    acc_first = None
    keys = mx.random.split(mx.random.key(key), rounds * 2)
    for r in range(rounds):
        z = mx.random.normal(shape=(n,), key=keys[2 * r])
        u = mx.random.uniform(shape=(n,), key=keys[2 * r + 1])
        v = (1.0 + c * z) ** 3
        ok_v = v > 0
        # log acceptance: log u < 0.5 z^2 ... use the standard log test
        logu = mx.log1p(-u)  # log(1-u) ~ log(u') same distribution, avoids u=0
        accept = ok_v & (
            logu < (0.5 * z * z + d - d * v + d * mx.log(mx.maximum(v, 1e-30)))
        )
        newly = accept & ~done
        x = mx.where(newly, d * v, x)
        if r == 0:
            acc_first = float(mx.mean(accept.astype(mx.float32)).item())
        done = done | accept
    return x, int(mx.sum(~done).item()), acc_first


n = 1_000_000
for alpha in [1.0, 2.0, 4.0, 10.0, 100.0]:
    x, unresolved, acc = gamma_mt(42, alpha, n, rounds=6)
    xn = np.array(x)
    xn = xn[xn > 0]
    m, v = xn.mean(), xn.var()
    print(
        f"  alpha={alpha:6.1f}: round-1 acceptance={acc:.4f}  unresolved after 6 rounds={unresolved}"
        f"  mean={m:.4f} (exp {alpha})  var={v:.4f} (exp {alpha})"
    )
p_fail = 1 - 0.9517  # worst-case M-T acceptance lower bound ~0.95
for N in [1e6, 1e7]:
    k = math.ceil((math.log(1e-9) - math.log(N)) / math.log(p_fail))
    print(
        f"  rounds for P(any unresolved among N={N:.0e}) < 1e-9 at accept=0.95: {k}"
    )

print()
print("=" * 70)
print("C6. Wilson-Hilferty gamma approx accuracy (quantiles vs f64 MC truth)")
print("=" * 70)
rng = np.random.default_rng(9)
qs = [0.001, 0.01, 0.5, 0.99, 0.999]
for alpha in [2.0, 10.0, 100.0]:
    ref = rng.gamma(alpha, 1.0, 10_000_000)
    z = rng.standard_normal(10_000_000)
    wh = (
        alpha
        * np.maximum(1 - 1 / (9 * alpha) + z / (3 * np.sqrt(alpha)), 0.0) ** 3
    )
    qr = np.quantile(ref, qs)
    qw = np.quantile(wh, qs)
    rel = np.abs(qw - qr) / qr
    print(
        f"  alpha={alpha:6.1f}: quantile rel errs "
        + " ".join(f"q{q}={e:.1%}" for q, e in zip(qs, rel))
    )
