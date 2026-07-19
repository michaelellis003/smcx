# Structured latent-state PyTree benchmark — 2026-07-19

## Result

For this fixed-size, no-history bootstrap-filter workload, small state trees
were inexpensive. Relative to one dense `D=16` array, the median steady-time
deltas were **+1.1% / +1.6%** on CPU and **-0.3% / +0.7%** on safe MPS for
2 / 4 leaves. The negative MPS result is smaller than the measured dispersion
and is not evidence of a speedup. The 16-leaf stress case made the cost visible:
**+6.6% on CPU and +3.7% on MPS**.

Cold `lower + compile` API latency rose **10.1% on CPU and 8.9% on MPS** at
16 leaves. Median process peak RSS rose **8.0% on CPU and 1.4% on MPS**.
The CPU executable-memory estimate changed by only 0.017%, and MPS allocator
peak changed by 0.34%. Within this workload, the evidence supports using a
small number of semantically meaningful state leaves while avoiding gratuitous
fragmentation into many scalar leaves.

Every dense/structured layout produced the exact same output checksum within
each backend, and there were no failed cells. This dated Markdown report is
the committed result; raw per-run benchmark data remains a reproducible local
artifact rather than repository history.

## Steady runtime

The primary estimate is the median of five fresh-process medians; each process
median contains seven fenced executions after one first execution/warm-up.
`Min` and `IQR` below are computed over all 35 individual executions. `Block
IQR` is the IQR of the five process medians.

| Backend | State layout | Primary median (ms) | Delta vs dense | Min (ms) | IQR (ms) | Block IQR (ms) |
|---|---:|---:|---:|---:|---:|---:|
| CPU | dense `(16,)` | 26.199 | — | 25.805 | 0.337 | 0.230 |
| CPU | 2 × `(8,)` | 26.493 | +1.1% | 26.084 | 0.259 | 0.089 |
| CPU | 4 × `(4,)` | 26.630 | +1.6% | 25.195 | 0.335 | 0.510 |
| CPU | 16 × `(1,)` | 27.929 | +6.6% | 27.163 | 0.278 | 0.207 |
| safe MPS | dense `(16,)` | 167.951 | — | 164.604 | 3.645 | 1.954 |
| safe MPS | 2 × `(8,)` | 167.407 | -0.3% | 160.012 | 5.063 | 1.463 |
| safe MPS | 4 × `(4,)` | 169.132 | +0.7% | 162.929 | 3.370 | 1.736 |
| safe MPS | 16 × `(1,)` | 174.240 | +3.7% | 166.086 | 5.801 | 3.154 |

This is a within-backend representation comparison, not a CPU-versus-MPS
performance comparison. The workload is too narrow to support the latter.

## Compile lifecycle

Each fresh process first compiled and executed a trivial backend-startup burn.
The measured workload then timed `jax.jit(...).lower(...)` and
`lowered.compile()` separately with the persistent compilation cache disabled.
The table reports medians across five fresh processes. Total compile `min/IQR`
also comes from those five values. The independently aggregated lowering and
backend-compile medians need not sum exactly to the total median.

MPS performs substantial work on first execution that is not represented by
its short `compile()` call, so first fenced execution is reported separately.

| Backend | State layout | Lower (ms) | Backend compile (ms) | Total compile median [min, IQR] (ms) | Delta | First execution (ms) | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU | dense | 46.288 | 314.894 | 363.432 [350.155, 17.877] | — | 29.544 | — |
| CPU | 2-leaf | 48.516 | 313.172 | 359.734 [347.181, 29.341] | -1.0% | 30.035 | +1.7% |
| CPU | 4-leaf | 47.763 | 331.870 | 379.633 [364.592, 18.891] | +4.5% | 30.015 | +1.6% |
| CPU | 16-leaf | 51.134 | 348.474 | 400.237 [392.888, 32.132] | +10.1% | 31.548 | +6.8% |
| safe MPS | dense | 49.684 | 18.808 | 68.390 [66.490, 3.265] | — | 221.841 | — |
| safe MPS | 2-leaf | 48.370 | 19.529 | 67.902 [67.025, 2.079] | -0.7% | 220.316 | -0.7% |
| safe MPS | 4-leaf | 49.210 | 19.918 | 69.127 [68.466, 3.795] | +1.1% | 222.142 | +0.1% |
| safe MPS | 16-leaf | 53.093 | 21.401 | 74.494 [73.694, 4.266] | +8.9% | 227.130 | +2.4% |

The small negative deltas are within run-to-run variation and are not treated
as improvements.

## Memory

`Process max RSS` is the fresh worker's high-water RSS at the end of timed
execution, captured before the parity check constructs a packed dense copy.
The bracket contains the minimum and IQR across five workers.

The final column is deliberately backend-specific:

- CPU exposes no allocator `memory_stats()`, so it reports the compiled
  executable's `peak_memory_in_bytes` estimate.
- MPS exposes no executable memory analysis, so it reports the allocator's
  process-lifetime `peak_bytes_in_use`, which includes the startup burn,
  compilation, and execution.

These two columns must not be compared across backends.

| Backend | State layout | Process max RSS median [min, IQR] (MiB) | RSS delta | Backend-specific peak median (MiB) | Peak delta |
|---|---:|---:|---:|---:|---:|
| CPU | dense | 298.359 [295.484, 7.188] | — | 0.688030 | — |
| CPU | 2-leaf | 299.719 [291.734, 8.617] | +0.5% | 0.688038 | +0.001% |
| CPU | 4-leaf | 310.172 [303.766, 6.391] | +4.0% | 0.688053 | +0.003% |
| CPU | 16-leaf | 322.234 [320.219, 3.812] | +8.0% | 0.688145 | +0.017% |
| safe MPS | dense | 215.109 [214.922, 0.383] | — | 37.216949 | — |
| safe MPS | 2-leaf | 215.859 [215.328, 0.578] | +0.35% | 37.216297 | -0.002% |
| safe MPS | 4-leaf | 215.734 [215.391, 0.586] | +0.29% | 37.216297 | -0.002% |
| safe MPS | 16-leaf | 218.078 [217.266, 1.078] | +1.38% | 37.341949 | +0.336% |

The sub-0.01% peak differences are counter/estimate granularity, not evidence
that a structured representation saves memory.

## Correctness gate

- Exact dense-versus-2/4/16-leaf checksum parity passed independently on CPU
  and MPS. The checksum covers marginal likelihood, final particles after
  repacking, log weights, ancestors, ESS, and evidence increments.
- The CPU SHA-256 was
  `7af56a326f5b9b83c87441445998f222b925fd6b62ee0d0923ac2542d8012895`;
  the MPS SHA-256 was
  `76533e97d266383b42a2ae01bf6f543f13042438ff335fc4969f0a954d8ca697`.
- CPU and MPS have different checksums, as expected from backend-specific
  float32 reduction behavior; cross-backend bit parity was not required.
- Evidence increments summed to the marginal log-likelihood within
  `3.0518e-05` on CPU and exactly at the recorded float32 value on MPS.
- Every requested worker used its requested backend, and all 40 workers
  completed successfully.

## Workload and method

| Item | Value |
|---|---|
| Algorithm | `bootstrap_filter`, forced resampling every step (`resampling_threshold=1.1`) |
| Size | `N=10,000`, `T=100`, total latent `D=16` |
| Layouts | one `(N,16)` array; tuples of 2 × `(N,8)`, 4 × `(N,4)`, or 16 × `(N,1)` leaves |
| History | `store_history=False`; final particle cloud only, full scalar traces |
| Initial state | One keyed `jr.normal((N, D), float32)` draw, split without changing values |
| Transition | Deterministic elementwise `state + 0.01`, applied once per leaf; transition key unused |
| Observation | Scalar Gaussian-style log weight using the first state coordinate |
| Correctness | Same key and mathematical state; exact within-backend output checksum |
| Isolation | Five fresh processes per backend/layout; 40 workers total |
| Timing | One first execution/warm-up, then seven `jax.block_until_ready`-fenced runs per worker |
| Primary statistic | Median of five fresh-process medians |
| Ordering | Seeded pseudorandom cell order, seed `20260719` |
| Compilation | Backend-startup burn first; persistent JAX compilation cache disabled |
| MPS dispatch | Safe/default; `JAX_MPS_ASYNC_DISPATCH` unset |
| Precision | float32; JAX x64 disabled |

The likelihood uses one coordinate so all layouts follow identical weighting
and ancestor paths. All 16 dimensions are still initialized, transitioned,
resampled, carried through `lax.scan`, and returned. Splitting the affine
transition creates one operation per leaf, so the result includes realistic
callback fragmentation as well as smcx's tree gather/carry/output overhead.

## Environment

| Item | Value |
|---|---|
| Date | 2026-07-19 |
| Hardware | Apple M3 Pro, 12 logical CPUs, 36 GiB unified memory |
| OS | macOS 26.2, arm64 |
| Power | AC power; battery charged at 100% |
| Thermal | No thermal or performance warning recorded by `pmset` |
| Python | 3.13.9 |
| JAX / jaxlib | 0.10.2 / 0.10.2 |
| jax-mps | 0.10.9 |
| NumPy | 2.5.1 |
| smcx metadata | 1.2.1 |
| Source commit | `5d74ca6a849ada3c0b3d6b3b54fd5798489b0c5c` |
| Production-source SHA-256 | `66661002cf4d3ef647dd3d1e37df2a7d5248ac7a2f7734d43c889b20230876c4` |
| Working tree | Production source clean at the commit; documentation and benchmark artifacts were uncommitted during measurement |

## Limitations

- This is one machine and one `(N, T, D)` point. It does not establish a
  universal per-leaf cost model.
- `store_history=False` isolates the live cloud. Full-history runtime and peak
  memory were not measured.
- The callbacks are deliberately light and the transition ignores its random
  key. Models with per-leaf linear algebra, stochastic transitions, or a
  likelihood that combines every leaf can have different compiler behavior.
- Tuple PyTrees were measured. Registered custom nodes and Equinox modules
  were not part of this timing run.
- MPS is experimental. Async dispatch was not measured.
- Compile measurements disable JAX's persistent cache but cannot disable every
  operating-system or Metal pipeline cache. First execution is reported to
  expose MPS work deferred past `compile()`.
- RSS, CPU executable analysis, and MPS allocator peaks measure different
  scopes. Only within-metric, within-backend layout comparisons are valid.
- Five process blocks quantify ordinary variation but are not a confidence
  interval and were all collected in one session.

## Reproduce

```bash
uv run python benchmarks/pytree_state/benchmark.py \
  --platforms cpu mps \
  --blocks 5 \
  --n 10000 \
  --timesteps 100 \
  --repeats 7 \
  --warmups 1 \
  --seed 20260719 \
  --output /tmp/2026-07-19-pytree-state.json
```

The runner emits the full per-cell JSON record at the requested output path;
the tables above are the authoritative committed results.
