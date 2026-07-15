# Kill test — optimized implementation, 2026-07-15 (current verdict)

**Verdict (pre-registered criterion, unchanged since pre-code): the
thesis HOLDS — 3 of 3 workloads count.** LGSSM 3.4×/6.2×, SV
3.7×/7.8×, TRACK 4.2×/5.6× at 10⁵/10⁶ particles, all ≥3× at both N;
all 15 correctness gates PASS on both libraries. Supersedes the
morning's holds-weakly result, which measured the pre-optimization
implementation.

## What changed since the morning run (full disclosure)

The *criterion, gates, datasets (hashes unchanged), and JAX baseline*
are identical. Two smcx implementation improvements landed in
between, both motivated by the profiling in
docs/research/perf-analysis.md:

1. **Value-branch conditional resampling** (internal): the filter now
   skips the resample pipeline on skip steps, exactly as smcjax's
   `lax.cond` always did — bit-identical results (tested). This is
   fairness-restoring, not an edge.
2. **Batched TRACK closures** (`batched=True`, ADR-0013): the TRACK
   cells' MLX side uses batched callbacks so the 4×4 transition runs
   as one GEMM. This levels a *compiler* asymmetry — XLA fuses
   vmapped matvecs automatically, MLX does not — and is disclosed
   here per the ADR. LGSSM/SV still use the per-particle convention
   matching smcjax's form (their vmap is free).
3. **Fresh process per cell** (protocol amendment 2026-07-15): the
   in-process sweep was shown to accumulate state (one cell inflated
   1.8×).

## Machine and configuration

- Apple M3 Pro, 36 GB, macOS 26.2, AC power, idle (load ≤2.4/12
  before both runs; desktop apps only). Sequential sides; fresh
  process per cell.
- smcx @ 30c3898 (mlx 0.32.0, Python 3.13.9); smcjax v1.1.0 @
  e93d527 (jax 0.6.2 CPU, x64 off, one jitted program). R=20 per
  primary cell; k=3 one-sided-Jensen gates; datasets per
  benchmarks/data/meta.json.

## Timing (median [min, IQR] seconds; speedup = JAX/MLX-GPU)

### lgssm

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.044 [0.043, 0.001] | 0.039 [0.038, 0.000] | 0.035 | 0.072 | **1.1x** | 1.3x | 30 | 2 |
| 100,000 | 0.184 [0.182, 0.002] | 0.054 [0.051, 0.001] | 0.053 | 0.504 | **3.4x** | 3.5x | 251 | 10 |
| 1,000,000 | 1.257 [1.253, 0.005] | 0.202 [0.200, 0.001] | 0.196 | 4.918 | **6.2x** | 6.4x | 2457 | 96 |

### sv

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.181 [0.176, 0.001] | 0.191 [0.188, 0.002] | 0.173 | 0.340 | **0.9x** | 1.0x | 148 | 3 |
| 100,000 | 0.638 [0.633, 0.003] | 0.170 [0.162, 0.003] | 0.155 | 1.709 | **3.7x** | 4.1x | 1230 | 10 |
| 1,000,000 | 4.191 [4.124, 0.007] | 0.536 [0.529, 0.004] | 0.527 | 16.715 | **7.8x** | 8.0x | 12054 | 96 |

### track

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.127 [0.126, 0.001] | 0.082 [0.080, 0.001] | 0.073 | 0.168 | **1.6x** | 1.7x | 106 | 5 |
| 100,000 | 0.689 [0.686, 0.004] | 0.165 [0.161, 0.002] | 0.163 | 1.304 | **4.2x** | 4.2x | 980 | 20 |
| 1,000,000 | 5.332 [5.314, 0.017] | 0.947 [0.939, 0.006] | 0.909 | 13.385 | **5.6x** | 5.9x | 9714 | 196 |

### track_full

| N | JAX-CPU | MLX-GPU (lag4) | GPU nohist | MLX-CPU | speedup | nohist speedup | GPU peak MB | nohist MB |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 0.127 [0.126, 0.001] | 0.082 [0.080, 0.001] | 0.074 | 0.164 | **1.5x** | 1.7x | 106 | 5 |
| 100,000 | 0.693 [0.687, 0.002] | 0.164 [0.161, 0.002] | 0.162 | 1.332 | **4.2x** | 4.3x | 980 | 20 |
| 1,000,000 | 5.361 [5.353, 0.009] | 0.948 [0.940, 0.008] | 0.924 | 13.481 | **5.7x** | 5.8x | 9714 | 196 |

## Cadence sweep (MLX-GPU median s; best arm bolded)

| cell | lag0 | lag2 | lag4 | lag8 | async |
|---|---|---|---|---|---|
| lgssm/10,000 | 0.039 | 0.039 | 0.039 | 0.038 | **0.030** |
| lgssm/100,000 | 0.053 | 0.053 | 0.054 | 0.054 | **0.052** |
| lgssm/1,000,000 | 0.201 | 0.202 | 0.202 | 0.201 | **0.197** |
| sv/10,000 | 0.191 | 0.191 | 0.191 | 0.190 | **0.150** |
| sv/100,000 | 0.170 | 0.170 | **0.170** | 0.170 | 0.171 |
| sv/1,000,000 | 0.538 | 0.539 | **0.536** | 0.540 | 0.539 |
| track/10,000 | 0.082 | 0.082 | 0.082 | 0.082 | **0.060** |
| track/100,000 | 0.164 | 0.165 | 0.165 | 0.164 | **0.159** |
| track/1,000,000 | 0.944 | 0.944 | 0.947 | 0.942 | **0.924** |

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
| track/10,000 | mlx | PASS | -0.753 | 1.287 |
| track/100,000 | jax | PASS | -0.051 | 0.405 |
| track/100,000 | mlx | PASS | -0.104 | 0.328 |
| track/1,000,000 | jax | PASS | -0.028 | 0.113 |
| track/1,000,000 | mlx | PASS | -0.029 | 0.106 |
| sv/10,000 | cross | PASS | -0.065 | bound 0.089 |
| sv/100,000 | cross | PASS | -0.006 | bound 0.032 |
| sv/1,000,000 | cross | PASS | +0.001 | bound 0.006 |

## Pre-registered criterion

- lgssm: COUNTS (N=100,000: 3.4x, N=1,000,000: 6.2x)
- sv: COUNTS (N=100,000: 3.7x, N=1,000,000: 7.8x)
- track: COUNTS (N=100,000: 4.2x, N=1,000,000: 5.6x)

**Verdict: the thesis HOLDS** (3/3 count)

## Interpretation

- The two optimizations did what the profiling projected (SV/10⁶
  1.51→0.54 s vs projected 0.53; TRACK/10⁶ 0.95 s vs projected
  0.90). LGSSM/10⁵ landed at 3.4× (projection 4.6× assumed the
  morning's slightly slower JAX row).
- `store_history=False` now also *helps* memory dramatically at zero
  speed cost everywhere (e.g. SV/10⁶ 12.1 GB → 96 MB — a 125×
  reduction with the resample-skip landing first).
- The N=10⁴ row remains the protocol-predicted dispatch-bound regime
  (0.9–1.7×): typical small-N filtering has no GPU story, exactly as
  the thesis always said.
- What the verdict now supports claiming: on M-series unified
  memory, oracle-verified SMC runs **3–8× faster than a strong
  12-core JAX-CPU baseline at 10⁵–10⁶ particles across
  resampling-bound, compute-bound, and multivariate workloads**.
