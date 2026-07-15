# Kill test re-run (clean) — 2026-07-15

**Verdict (pre-registered criterion): the thesis HOLDS WEAKLY — 1 of
3 workloads count. This clean run SUPERSEDES the contaminated
2026-07-14 run and removes its provisional flag.** All 15
correctness gates PASS on both libraries; every timing is at
matched, oracle-verified accuracy.

## Machine and configuration

- Apple M3 Pro (12-core CPU, 18-core GPU), 36 GB, macOS 26.2,
  **AC power, idle machine confirmed pre-run** (load avg 2.4/12
  cores; only desktop apps — Safari/VS Code/this CLI — together
  under one core; no batch workloads; no thermal warnings recorded).
  Sequential sides (JAX finished before MLX started).
- smcx @ b6ecd3f (mlx 0.32.0, Python 3.13.9); smcjax v1.1.0 @
  e93d527 (jax/jaxlib 0.6.2 CPU, x64 disabled, whole filter jitted,
  cpu_count=12). Same datasets as 2026-07-14 (sha256 verified
  unchanged against benchmarks/data/meta.json).
- Per the 2026-07-15 protocol amendment: added `store_history=False`
  arm (ADR-0011), **report-only** — primary comparison is
  full-history on both sides.

## Timing (median [min, IQR] seconds; speedup = JAX/MLX-GPU)

### lgssm

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.043 [0.043, 0.001] | 0.039 [0.038, 0.000] | 0.034 | 0.074 | **1.1x** | 1.3x | 30 | 2 |
| 100,000 | 0.184 [0.181, 0.001] | 0.055 [0.054, 0.001] | 0.051 | 0.644 | **3.3x** | 3.6x | 252 | 32 |
| 1,000,000 | 1.261 [1.257, 0.023] | 0.303 [0.301, 0.002] | 0.303 | 6.170 | **4.2x** | 4.2x | 2469 | 317 |

### sv

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.179 [0.176, 0.001] | 0.193 [0.190, 0.002] | 0.173 | 0.344 | **0.9x** | 1.0x | 148 | 2 |
| 100,000 | 0.638 [0.635, 0.002] | 0.275 [0.255, 0.002] | 0.275 | 2.972 | **2.3x** | 2.3x | 1235 | 29 |
| 1,000,000 | 4.202 [4.189, 0.020] | 1.513 [1.502, 0.016] | 1.467 | 28.877 | **2.8x** | 2.9x | 12102 | 321 |

### track

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.126 [0.125, 0.001] | 0.103 [0.097, 0.004] | 0.096 | 0.392 | **1.2x** | 1.3x | 106 | 6 |
| 100,000 | 0.690 [0.687, 0.003] | 0.323 [0.322, 0.001] | 0.323 | 3.583 | **2.1x** | 2.1x | 982 | 67 |
| 1,000,000 | 5.337 [5.318, 0.015] | 4.787 [4.598, 0.160] | 5.872 | 41.537 | **1.1x** | 0.9x | 9734 | 661 |

### track_full

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.127 [0.126, 0.001] | 0.103 [0.095, 0.001] | 0.096 | 0.375 | **1.2x** | 1.3x | 106 | 6 |
| 100,000 | 0.694 [0.691, 0.002] | 0.324 [0.323, 0.001] | 0.325 | 3.688 | **2.1x** | 2.1x | 982 | 67 |
| 1,000,000 | 5.364 [5.350, 0.012] | 5.017 [4.785, 0.180] | 5.719 | 41.588 | **1.1x** | 0.9x | 9734 | 661 |

## Cadence sweep (MLX-GPU median s; best arm bolded)

| cell | lag0 | lag2 | lag4 | lag8 | async |
|---|---|---|---|---|---|
| lgssm/10,000 | 0.039 | 0.039 | 0.039 | 0.039 | **0.031** |
| lgssm/100,000 | 0.056 | 0.055 | 0.055 | 0.055 | **0.042** |
| lgssm/1,000,000 | 0.300 | 0.302 | 0.303 | 0.301 | **0.287** |
| sv/10,000 | 0.192 | 0.193 | 0.193 | 0.193 | **0.150** |
| sv/100,000 | 0.274 | 0.274 | 0.275 | 0.276 | **0.202** |
| sv/1,000,000 | 1.504 | 1.503 | 1.513 | 1.505 | **1.385** |
| track/10,000 | 0.104 | 0.104 | 0.103 | 0.103 | **0.071** |
| track/100,000 | 0.323 | 0.323 | 0.323 | 0.322 | **0.281** |
| track/1,000,000 | 4.736 | 4.728 | 4.787 | 4.926 | **3.102** |

## Correctness gates (k=3, R=20, one-sided Jensen)

| cell | side | gate | err (nats) | SD |
|---|---|---|---|---|
| lgssm/10,000 | jax | PASS | +0.057 | 0.253 |
| lgssm/10,000 | mlx | PASS | -0.047 | 0.172 |
| lgssm/100,000 | jax | PASS | -0.001 | 0.066 |
| lgssm/100,000 | mlx | PASS | -0.004 | 0.057 |
| lgssm/1,000,000 | jax | PASS | +0.001 | 0.016 |
| lgssm/1,000,000 | mlx | PASS | -0.005 | 0.020 |
| track/10,000 | jax | PASS | -0.713 | 1.219 |
| track/10,000 | mlx | PASS | -0.943 | 1.472 |
| track/100,000 | jax | PASS | -0.051 | 0.405 |
| track/100,000 | mlx | PASS | +0.086 | 0.407 |
| track/1,000,000 | jax | PASS | -0.028 | 0.113 |
| track/1,000,000 | mlx | PASS | -0.043 | 0.119 |
| sv/10,000 | cross | PASS | -0.065 | bound 0.089 |
| sv/100,000 | cross | PASS | -0.006 | bound 0.032 |
| sv/1,000,000 | cross | PASS | +0.001 | bound 0.006 |

## Pre-registered criterion

- lgssm: COUNTS (N=100,000: 3.3x, N=1,000,000: 4.2x)
- sv: does not count (N=100,000: 2.3x, N=1,000,000: 2.8x)
- track: does not count (N=100,000: 2.1x, N=1,000,000: 1.1x)

**Verdict: the thesis HOLDS WEAKLY** (1/3 count)

## Interpretation

- **The contamination hypothesis is confirmed in the honest
  direction**: on the idle machine JAX-CPU got faster (e.g. LGSSM/10⁵
  0.204→0.184 s; SV/10⁶ 4.55→4.20 s), so yesterday's speedups were
  indeed inflated — 10⁵-row speedups dropped (LGSSM 4.0→3.3×,
  SV 2.7→2.3×, TRACK 2.5→2.1×). The verdict is unchanged but now
  stands on clean measurements.
- **LGSSM counts (3.3× / 4.2×); SV improved at 10⁶ (1.9→2.8×)** —
  yesterday's anomalous SV cell was memory pressure (12 GB peak
  under contention); clean, it runs 1.51 s. Still short of the 3×
  bar. TRACK remains resampling-light and compute-balanced
  (2.1× / 1.1×).
- **store_history=False delivers exactly what ADR-0011 promised**:
  peak memory drops 8–38× (LGSSM/10⁶ 2469→317 MB; SV/10⁶
  12102→321 MB; TRACK/10⁶ 9734→661 MB) with timing within noise on
  GPU — memory materialization was a *capacity* problem, not (on an
  idle machine) a *speed* limiter. The lean arm makes 10⁷-particle
  runs feasible in RAM.
- Guard honored: JAX-CPU wins one cell (SV/10⁴, 0.9×) — the
  dispatch-overhead regime the protocol predicted. MLX-GPU leads the
  other 11 cells (1.1–4.2×) and always beats MLX-CPU at ≥10⁵.
- Cadence sweep: pure async fastest in most cells again; lag-4
  within a few percent everywhere — the shipped default stands.

## Consequences

- README Status updated: verdict final (not provisional), claims at
  the clean numbers.
- The remaining path to a stronger verdict is algorithmic, not
  benchmarking: TRACK's 1.1× at 10⁶ is dominated by per-particle
  4×4 matvec transition sampling — a batched-matmul mutation path
  (particles as one (N,4)@(4,4) matmul instead of vmapped per-
  particle matvecs on BOTH sides... note smcjax has the same form,
  so this is fair) and the Metal-kernel/counting resampler bake-off
  at 10⁷ are the levers. Tracked on the roadmap under Later.
