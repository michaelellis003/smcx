# Native MLX versus jax-mps — results

Profile: `full`. Balanced-order seed: 20260715.

## Environment

| Item | Value |
|---|---|
| Machine | Apple M3 Pro, 36 GB, macOS 26.2 (25C56); AC power, idle |
| Native | mlx 0.32.0, MLX GPU (float32), Python 3.13.9 |
| Compatibility | jax 0.10.2, jaxlib 0.10.2, jax-mps 0.10.9; `JAX_PLATFORMS=mps` |
| Dispatch arms | safe (async unset) and async (`JAX_MPS_ASYNC_DISPATCH=1`) |
| Design | 5 fresh-process blocks x 7 timed repeats; seeded balanced order (seed 20260715) |
| Primary estimate | median of the five per-process medians |
| Interval | 95% paired percentile bootstrap, 10,000 resamples, seed 20260715 |
| jax-mps binary | `libpjrt_plugin_mps.dylib` sha256 `d5845ce1…`; vendored `mlx.metallib` 125,453,832 B sha256 `b9fe8879…` |

The vendored `mlx.metallib` is recorded by hash; it is not asserted to be a
released MLX version. Where the safe and async arms both pass correctness, the
report quotes the faster one, which favors jax-mps and is disclosed here.


## Verdict

Native SMC ecosystem case: **mixed** (LGSSM-PF persistent: True; supporting motifs: 1/4; strong: True).

The negative controls below calibrate the harness and never count toward this verdict.

## SMC workloads

### gather_scatter

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 10000 | 0.000229959 | jax_mps_sync | 0.000200917 | 0.38 / 0.87 / 0.95 | 0.83 |
| 100000 | 0.000274333 | jax_mps_sync | 0.000243 | 0.61 / 0.89 / 1.05 | 0.83 |
| 1000000 | 0.000583458 | jax_mps_async | 0.000712625 | 0.76 / 1.22 / 1.96 | 0.83 |

Persistent native advantage: N=100000: ratio lower bound 0.61 < 1.5; N=1000000: ratio lower bound 0.76 < 1.5.

### lgssm_pf

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 10000 | 0.0343591 | jax_mps_async | 0.195705 | 5.65 / 5.70 / 5.86 | 0.08 |
| 100000 | 0.050251 | jax_mps_async | 1.20092 | 23.26 / 23.90 / 27.15 | 0.21 |
| 1000000 | 0.207978 | jax_mps_async | 12.449 | 38.85 / 59.86 / 62.62 | 0.42 |

Persistent native advantage: persistent native advantage.

### random

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 10000 | 0.000534917 | jax_mps_async | 0.000707292 | 1.18 / 1.32 / 3.70 | 0.04 |
| 1000000 | 0.000595625 | jax_mps_async | 0.00649371 | 6.78 / 10.90 / 19.25 | 0.03 |
| 10000000 | 0.00238558 | jax_mps_sync | 0.067905 | 28.11 / 28.46 / 28.82 | 0.05 |

Persistent native advantage: persistent native advantage.

### scan

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 10000 | 0.00178292 | jax_mps_sync | 0.00135354 | 0.55 / 0.76 / 0.90 | 9.22 |
| 100000 | 0.00251275 | jax_mps_sync | 0.00208983 | 0.51 / 0.83 / 1.09 | 12.50 |
| 1000000 | 0.00735375 | jax_mps_sync | 0.00423837 | 0.53 / 0.58 / 0.80 | 16.50 |

Persistent native advantage: N=100000: ratio lower bound 0.51 < 1.5; N=100000: peak memory over the 1.25x budget; N=1000000: ratio lower bound 0.53 < 1.5; N=1000000: peak memory over the 1.25x budget.

### systematic

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 10000 | 0.000267708 | jax_mps_sync | 0.00124229 | 0.95 / 4.64 / 6.44 | 0.09 |
| 100000 | 0.000890625 | jax_mps_async | 0.00332004 | 3.06 / 3.73 / 5.47 | 0.07 |
| 1000000 | 0.00249258 | — | — | native cell missing or failed correctness; no jax-mps arm passed correctness | — |

Persistent native advantage: N=1000000: native failed correctness.

## Negative controls

### eltwise_reduce

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 10000 | 0.000232792 | jax_mps_sync | 0.000323 | 0.40 / 1.39 / 2.23 | 1.00 |
| 1000000 | 0.0003955 | jax_mps_async | 0.000395 | 0.52 / 1.00 / 2.62 | 1.00 |
| 10000000 | 0.00136946 | jax_mps_sync | 0.00140779 | 0.83 / 1.03 / 1.27 | 1.00 |

### matmul

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | mem ratio |
|---|---|---|---|---|---|
| 256 | 0.000319666 | jax_mps_async | 0.000267708 | 0.40 / 0.84 / 0.99 | 1.00 |
| 1024 | 0.00194625 | jax_mps_async | 0.00110775 | 0.49 / 0.57 / 1.41 | 1.00 |
| 2048 | 0.00369888 | jax_mps_sync | 0.00367167 | 0.99 / 0.99 / 1.00 | 1.00 |

## Missing or failed cells

None: every registered cell produced a valid result.

## Reading

The pre-registered verdict is **mixed**: the native SMC ecosystem case is
"supported" only when at least two of the four translation-sensitive motifs
join LGSSM-PF in showing a persistent native advantage, and here just one
(RANDOM) does. SYSTEMATIC would be the natural second, but its N=10^6 cell
fails the frozen correctness gate on every float32 backend, native and jax-mps
alike, so it cannot count. The mechanism is recorded below.

The result is not neutral for the workload that motivates the port. The
end-to-end particle filter shows a strong, growing native advantage: the ratio
of jax-mps to native MLX median time rises from 5.7x at N=10^4 to 23.9x at
10^5 to 59.9x at 10^6, and native MLX holds peak memory to a fraction of the
compatibility arm (0.08-0.42x). RANDOM scales the same way, 1.3x to 28.5x.
Both clear the "strong" bar, a bootstrap lower bound of at least 3.0 at the two
largest sizes.

The losses inform the decision as much as the wins. SCAN runs faster under
jax-mps at every size (native 0.55-0.83x), and native MLX uses 9-16x the peak
memory: the loop-over-one-compiled-step pattern materializes more than JAX's
whole-loop `lax.scan`. GATHER-SCATTER sits near parity (0.87-1.22x). The
negative controls land near 1.0 (ELTWISE-REDUCE 1.0-1.4x, MATMUL 0.57-1.0x),
which is the point of including them: jax-mps matches direct MLX on dense and
fused work, so the large SMC-motif gaps are not an artifact of an unfair
harness.

For the build-native question, the reading is specific. A native MLX SMC
library buys a large win on the sequential particle filter that widens with
particle count, and a similar win on RNG-bound work. It does not buy a uniform
win: scan-shaped control flow and gather are competitive or better under
jax-mps today, and the Python-loop scan carries a memory cost worth addressing
before it ships.

## Systematic correctness at N=10^6

At the largest size, SYSTEMATIC fails the deterministic gate on `mlx_gpu`,
`mlx_cpu`, and both jax-mps arms; only `jax_cpu` passes. The three GPU/MPS
arms return the identical wrong checksum (-3.526573), which locates the cause
in shared float32 arithmetic rather than any one backend. The motif computes a
cumulative sum of N uniform weights, each near 1/N, then a right `searchsorted`
against fixed queries. By N=10^6 the float32 CDF has lost enough precision near
the top that a handful of ancestor indices differ from the float64 oracle;
because the particles are `linspace(-2, 2)`, an off-by-one index is a large
per-element error, so the `rtol=5e-5` comparison fails.

This is a float32 limit of standalone systematic resampling at high N, shared
symmetrically by native MLX and jax-mps, not a native-versus-compatibility
signal. The protocol fixes the tolerance before measurement and forbids
relaxing it afterward, so the cell is retained as failed and SYSTEMATIC is
excluded from the verdict. The production smcx resampler keeps ancestor indices
monotone through the gather and is a separate code path; this benchmark motif
is a bare cumsum/searchsorted/gather kernel.

## Translation audit

For each persistent-gap workload, the JAX StableHLO was captured at a small and
a large size on the safe jax-mps arm; the graph is size-invariant. LGSSM-PF
lowers to 1189 StableHLO operations across 32 kinds (266 `constant`, 248
`broadcast_in_dim`, 165 `add`, and a Threefry chain of `xor`/`shift_right_logical`/
`or`); RANDOM to 290 across 22. jax-mps executes this whole StableHLO graph
through its op-patched MLX dispatch, while native smcx runs one compiled step in
a Python loop. The op census is consistent with a materialization-and-overhead
mechanism, but the plugin exposes no compiled-executable text
(`compile().as_text()` returns none), so this stays a plausible mechanism, not a
proven compiler-causality claim. The trace bundles with full StableHLO text
and provenance are committed under
`benchmarks/results/2026-07-16-native-vs-jax-mps/traces/`.

## Reproduction

```bash
# Full matrix (525 fresh processes; AC power, idle machine):
uv run python benchmarks/native_vs_jax_mps/run.py --profile full --output-dir <dir>
# Render this report:
uv run python benchmarks/native_vs_jax_mps/report.py <dir>
# Re-run one cell (example): systematic, N=10^6, safe jax-mps arm
JAX_PLATFORMS=mps uv run --no-project --python 3.13 \
  --with jax==0.10.2 --with jaxlib==0.10.2 --with jax-mps==0.10.9 \
  python benchmarks/native_vs_jax_mps/jax_worker.py --arm jax_mps_sync \
  --block 0 --repeats 7 --size 1000000 --warmups 1 --workload systematic
# Capture StableHLO/IR for a workload:
#   add --capture-ir to the jax_worker command above.
```

The merged per-process JSON (manifest plus all 525 records) is committed as
`benchmarks/results/2026-07-16-native-vs-jax-mps/merged.json`, and the four
StableHLO/provenance trace bundles under the sibling `traces/` directory. The
verdict machinery, correctness gates, and balanced ordering are covered by
`tests/test_native_vs_jax_mps_benchmark.py` and
`tests/test_native_vs_jax_mps_report.py`.
