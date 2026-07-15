# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 1 (summation/LSE) + Area 4 (ESS identities) experiments."""

import mlx.core as mx
import numpy as np

print("=" * 70)
print("A1. Is mx.logsumexp max-shifted? Extreme-spread probes (f32, GPU)")
print("=" * 70)
probes = {
    "[-1e38, -1e38]": [-1e38, -1e38],
    "[1e38-ish large: 88, 88]": [88.0, 88.0],  # exp overflows f32 at ~88.72
    "[500, 500]": [500.0, 500.0],  # naive exp -> inf
    "[500, 0]": [500.0, 0.0],
    "[-500, 0]": [-500.0, 0.0],
    "[-1000,-1000]": [-1000.0, -1000.0],
    "[-inf, 0]": [-np.inf, 0.0],
    "[-inf, -inf]": [-np.inf, -np.inf],
}
for name, v in probes.items():
    r = mx.logsumexp(mx.array(v, dtype=mx.float32)).item()
    # f64 reference
    a = np.array(v, dtype=np.float64)
    m = np.max(a)
    ref = m + np.log(np.sum(np.exp(a - m))) if np.isfinite(m) else -np.inf
    print(f"  LSE({name}) = {r!r}   ref={ref!r}")

print()
print(
    "logcumsumexp extreme:",
    mx.logcumsumexp(mx.array([500.0, 0.0, -500.0])).tolist(),
)

print()
print("=" * 70)
print("A2. mx.logsumexp accuracy at N=1e6, f32, wide spread")
print("=" * 70)
rng = np.random.default_rng(0)
for spread in [10, 50, 200]:
    lw64 = -rng.exponential(spread, size=1_000_000)
    lw64[0] = 0.0
    ref = np.logaddexp.reduce(lw64)
    r32 = mx.logsumexp(mx.array(lw64.astype(np.float32))).item()
    # naive f32: log(sum(exp)))
    naive = float(np.log(np.sum(np.exp(lw64.astype(np.float32)))))
    print(
        f"  spread={spread:4d}: mlx LSE err={r32 - ref:+.3e}  naive-f32 err={naive - ref:+.3e}  ref={ref:.6f}"
    )

print()
print("=" * 70)
print("A3. log-Z carry: naive vs Kahan vs Neumaier vs pairwise, f32, T=1e4")
print("=" * 70)


def run_sums(inc64):
    inc32 = inc64.astype(np.float32)
    ref = float(np.sum(inc64))
    # naive f32
    s = np.float32(0.0)
    for x in inc32:
        s = np.float32(s + x)
    naive = float(s)
    # Kahan
    s = np.float32(0.0)
    c = np.float32(0.0)
    for x in inc32:
        y = np.float32(x - c)
        t = np.float32(s + y)
        c = np.float32(np.float32(t - s) - y)
        s = t
    kahan = float(s)
    # Neumaier (Kahan-Babuska)
    s = np.float32(inc32[0])
    c = np.float32(0.0)
    for x in inc32[1:]:
        t = np.float32(s + x)
        if abs(s) >= abs(x):
            c = np.float32(c + np.float32(np.float32(s - t) + x))
        else:
            c = np.float32(c + np.float32(np.float32(x - t) + s))
        s = t
    neumaier = float(np.float32(s + c))
    # pairwise (numpy sum on f32 is pairwise)
    pw = float(np.sum(inc32))
    return ref, naive, kahan, neumaier, pw


cases = {}
# typical SMC: increments all same sign, similar magnitude (per-step loglik)
cases["typical: inc ~ N(-1.2, 0.3), T=1e4"] = rng.normal(-1.2, 0.3, 10_000)
# large |logZ|: per-step -50 (e.g. high-dim obs), T=1e4
cases["big: inc ~ N(-50, 5), T=1e4"] = rng.normal(-50, 5, 10_000)
# adversarial for Kahan: one giant increment mid-stream (Kahan fails when |addend| >> |sum|... actually when next |x| > |s|)
adv = rng.normal(-1.0, 0.3, 10_000)
adv[5000] = (
    1.0e7  # giant positive increment (contrived: e.g. one enormous evidence jump)
)
cases["adversarial: giant +1e7 spike at t=5000"] = adv
# sign-alternating large
alt = np.tile([1e4, -1e4 + 0.1], 5000).astype(np.float64)
cases["alternating +-1e4 with +0.1 net per pair"] = alt

for name, inc in cases.items():
    ref, naive, kahan, neumaier, pw = run_sums(inc)
    print(f"  {name}")
    print(f"    ref(f64)={ref:+.6f}")
    print(
        f"    naive err={naive - ref:+.3e}  kahan err={kahan - ref:+.3e}  "
        f"neumaier err={neumaier - ref:+.3e}  pairwise err={pw - ref:+.3e}"
    )

print()
print("=" * 70)
print(
    "A4. ESS: 2*LSE(l)-LSE(2l) identity vs direct sum of squared normalized weights"
)
print("=" * 70)


def ess_ref(lw64):
    lw = lw64 - np.logaddexp.reduce(lw64)
    return float(np.exp(-np.logaddexp.reduce(2 * lw)))


N = 1_000_000
scenarios = {}
lw = -rng.exponential(1.0, N)
scenarios["mild (spread ~1)"] = lw
lw = -rng.exponential(30.0, N)
scenarios["heavy (spread ~30)"] = lw
lw = np.full(N, -60.0)
lw[0] = 0.0
scenarios["degenerate: 1 particle carries all"] = lw
lw = np.full(N, -20.0)
lw[:10] = 0.0
scenarios["10 dominant + 1e6 at -20"] = lw
lw = np.zeros(N)
scenarios["uniform (ESS=N)"] = lw
lw = -rng.exponential(5.0, N)
lw[::2] -= 40
scenarios["bimodal spread"] = lw

for name, lw64 in scenarios.items():
    ref = ess_ref(lw64)
    l32 = mx.array(lw64.astype(np.float32))
    # identity form, f32 mlx
    ident = float(mx.exp(-(mx.logsumexp(2 * (l32 - mx.logsumexp(l32))))).item())
    # direct: normalize in prob space then sum of squares
    w = mx.softmax(l32)
    direct = float((1.0 / mx.sum(w * w)).item())
    print(
        f"  {name:38s} ref={ref:.6g}  identity relerr={abs(ident - ref) / ref:.2e}  direct relerr={abs(direct - ref) / ref:.2e}"
    )

print()
print("=" * 70)
print("A5. CESS in log domain (Zhou-Johansen-Aston) f32 check")
print("=" * 70)
# CESS = N * (sum W_prev * v)^2 / sum W_prev * v^2 ; v = incremental weights exp(l)
# log CESS = log N + 2*LSE(logW + l) - LSE(logW + 2l)
N = 1_000_000
logW64 = -rng.exponential(3.0, N)
logW64 -= np.logaddexp.reduce(logW64)
l64 = -rng.exponential(25.0, N)
num = np.logaddexp.reduce(logW64 + l64)
den = np.logaddexp.reduce(logW64 + 2 * l64)
ref = float(np.exp(np.log(N) + 2 * num - den))
lW = mx.array(logW64.astype(np.float32))
l = mx.array(l64.astype(np.float32))
cess32 = float(
    mx.exp(
        np.log(N) + 2 * mx.logsumexp(lW + l) - mx.logsumexp(lW + 2 * l)
    ).item()
)
# prob-space direct
W = mx.exp(lW)
v = mx.exp(l)
direct = float((N * mx.sum(W * v) ** 2 / mx.sum(W * v * v)).item())
print(
    f"  ref={ref:.6g}  log-identity relerr={abs(cess32 - ref) / ref:.2e}  prob-direct relerr={abs(direct - ref) / ref:.2e}"
)
print("  (prob-direct underflows when l very negative; log form does not)")

print()
print("=" * 70)
print("A6. sorting before summation - does it matter after max-shift LSE?")
print("=" * 70)
lw64 = -rng.exponential(50.0, 1_000_000)
lw64[0] = 0.0
ref = np.logaddexp.reduce(lw64)
l32 = lw64.astype(np.float32)
unsorted_ = float(mx.logsumexp(mx.array(l32)).item())
sorted_inc = float(mx.logsumexp(mx.array(np.sort(l32))).item())
print(
    f"  LSE unsorted err={unsorted_ - ref:+.3e}   LSE ascending-sorted err={sorted_inc - ref:+.3e}"
)
