# Representation and history profile — 2026-07-19

Status: **complete, correct, and performance-eligible**.

## Metadata

| Field | Value |
|---|---|
| Hardware | Apple M3 Pro, 12 CPU cores, 36 GiB unified memory |
| OS | macOS 26.2 (25C56) |
| Power / thermal | AC power; no warning before, after, or after extraction for every cell |
| Python | 3.13.9 |
| JAX / jaxlib | 0.10.2 / 0.10.2 |
| jax-mps | 0.10.10, safe dispatch |
| NumPy | 2.5.1 |
| smcx | 1.3.0 |
| Source | `651666b414ce3c16fa7398d4e190c433715bfb5a`, clean, source SHA-256 `b8aebca1ae24b8cca5d279b2eb62b65cb8d89c08be8faa73e5b27c16b72f9167` |
| Timing design | Five isolated process blocks; one warm-up and seven fenced repeats per block |
| Primary estimate | Median of five per-process steady medians |
| Order / inference / validation seeds | `20260719` / `20260719` / `20260720` |

The campaign executed 100 timing workers and 10 independent validation
workers: 800 timed calls and 184 validation replicates. All cells completed,
passed their registered gates, and remained power/thermal eligible.

## Tracking model: dense array versus two-leaf PyTree

The model is a four-coordinate linear-Gaussian tracker at `N=10,000` and
`T=200`. The PyTree stores position and velocity in two semantic leaves; the
dense arm stores one four-vector. Both covariance regimes were crossed with
history off/on.

| Covariance | State | History | CPU, ms | MPS, ms | CPU executable peak, MiB | MPS device peak, MiB |
|---|---|---:|---:|---:|---:|---:|
| correlated | dense | off | 133.095 | 412.586 | 0.23 | 8.32 |
| correlated | dense | on | 140.914 | 417.701 | 45.78 | 137.45 |
| correlated | PyTree | off | 140.782 | 414.159 | 0.23 | 8.63 |
| correlated | PyTree | on | 147.505 | 423.973 | 45.78 | 137.45 |
| diagonal | dense | off | 133.070 | 417.986 | 0.23 | 8.22 |
| diagonal | dense | on | 141.103 | 426.731 | 45.78 | 137.45 |
| diagonal | PyTree | off | 142.056 | 420.047 | 0.23 | 8.63 |
| diagonal | PyTree | on | 147.917 | 431.699 | 45.78 | 137.45 |

Matched PyTree/dense steady ratios were `1.047`--`1.068` on CPU and
`1.004`--`1.015` on MPS. The PyTree is therefore a flexibility and semantic
structure feature, not a speed optimization for this model. The measured CPU
cost includes the workload's leaf-flattening callback adapter.

Matched history-on/off ratios were `1.041`--`1.060` on CPU and
`1.012`--`1.028` on MPS. Retaining the full tracking result requires
45.78 MiB; the CPU executable peak equals that output-size floor. The MPS
allocator peak is 137.45 MiB, about three times the retained result, but no
longer exhibits the operand-wide update blow-up seen under jax-mps 0.10.9.

CPU/MPS timing ratios are withheld for tracking because backend rounding led
to different adaptive resampling counts: 163/164 on CPU versus 161/165 on MPS,
depending on covariance regime. Representation and history ratios remain
valid because each within-backend pair had identical per-block work counters.

## Liu--West state and parameter history

The forced-resampling P1 arm used `N=10,000`, `T=100`, parameter dimension
one, shrinkage `0.95`, and threshold `1.1`. Every run therefore performed
exactly 99 resampling decisions.

| Backend | History | Steady, ms | Process RSS, MiB | Executable/device peak, MiB |
|---|---:|---:|---:|---:|
| CPU | off | 72.421 | 322.36 | 0.15 |
| CPU | on | 72.790 | 366.41 | 15.26 |
| MPS | off | 246.529 | 201.62 | 7.82 |
| MPS | on | 254.034 | 203.72 | 45.94 |

History cost `1.005x` on CPU and `1.030x` on MPS. MPS cost `3.404x` CPU
without history and `3.490x` with history under exactly matched work.

The independent twelve-replicate oracle gates passed on both backends. The
mean evidence ratios were `1.011` (CPU) and `1.006` (MPS); parameter-mean
errors were `-0.001052` and `0.000303`, below tolerances `0.006295` and
`0.006152`, respectively. Parameter raw-second-moment gates also passed.

## Backend finding

jax-mps 0.10.9 lowered scan history updates through operand-wide operations;
the same 45.78 MiB tracking output previously reached approximately 1.16 GiB
for the dense state and 1.52 GiB for the two-leaf state. Version 0.10.10 ships
native MLX slice updates and reduces the current peak to 137.45 MiB while
making history runtime nearly neutral. This supports the dependency-floor fix
in ADR-0026 and does not support an smcx-specific history rewrite.

Authoritative upstream sources:

- jax-mps 0.10.10 release:
  <https://github.com/tillahoffmann/jax-mps/releases/tag/v0.10.10>
- Native `slice_update` lowering, PR #219:
  <https://github.com/tillahoffmann/jax-mps/pull/219>
- Native dynamic-slice lowering, PR #220:
  <https://github.com/tillahoffmann/jax-mps/pull/220>
- Shared clamped slice starts, PR #222:
  <https://github.com/tillahoffmann/jax-mps/pull/222>
- JAX PyTree definition and registration model:
  <https://docs.jax.dev/en/latest/pytrees.html>

Reproduction command:

```bash
uv run python -m benchmarks.profiling.run \
  --profile representation --platforms cpu mps \
  --output-dir /tmp/smcx-profiling-representation-20260719-651666b-01010
```
