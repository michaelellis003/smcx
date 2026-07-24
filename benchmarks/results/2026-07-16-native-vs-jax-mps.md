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
proven compiler-causality claim. The trace bundles with full StableHLO text and
provenance were committed with this report and remain available in the
[immutable result archive][native-results-archive].

## Archived reproduction

The commands below reproduce the run from a checkout of immutable commit
[`ac9572d`][native-harness-archive]. The executable harness is no longer part
of the current source tree.

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

The raw artifacts are preserved at that immutable commit. The former
benchmark-specific tests are likewise historical and remain in the
[pre-migration test archive][native-tests-archive]; they do not validate
current smcx. The audited SHA256 values of the removed raw artifacts are:

- `merged.json`
  `c79f3b8d9669fc474919a8fe62b10053cf5d8c106cfbf95525ae95b696b7cac8`
- `tuned_lgssm_nohist.json`
  `9c651b9ec7653d7e4df1ba1727a1215dc038c204ae03a557f8642f01080e5595`
- `traces/lgssm_pf_n1000000_jax_mps_sync_ir.json`
  `8010a4d7963c27615a24eef360e85b47dc9a6f926af20cb9f19b02af21bcfb1a`
- `traces/lgssm_pf_n10000_jax_mps_sync_ir.json`
  `8bbee4a7c17809ac3fc278e7ed178d89fca52afa85ea9897b2ddedae9988bcfe`
- `traces/random_n10000000_jax_mps_sync_ir.json`
  `afd7afc8aacea6085a0ac93119426c3f0d78a86db79b73b66b63f93fa1f550b0`
- `traces/random_n10000_jax_mps_sync_ir.json`
  `16ee2a6283cc8957f19d6f0823a6cc124912680f5f18ea3798afe81e177ef982`

## Addendum — 2026-07-16: tuned-JAX counter-experiment and scan correction

*This post-hoc check leaves the original text unchanged and corrects how two
of its numbers should be read. Report-only: nothing here enters the
pre-registered verdict.*

### The tuned filter

The JAX arm above is a naive port:
full history forces the compiler to materialize five (T, N) arrays, and the
`lax.cond` resampling branch blocks optimization. We built the strongest fair
JAX filter used for the counter-experiment (`lgssm_pf_nohist`): unconditional
systematic resampling with no `lax.cond`, a scan that emits no per-step
outputs, and only the marginal log-likelihood returned. The native arm was
changed identically (`smcx.bootstrap_filter` with `resampling_threshold=1.0`,
`store_history=False`), so both sides do the same work. Design: 3 blocks x 7
repeats per arm and size, R=20 Kalman gate on block 0; every gate passed.
This is lighter than the main matrix (3 blocks, not 5) and resamples every
step where the primary arm resamples conditionally, so the two experiments
measure related but not identical filters.

| N | native median (s) | best jax arm | jax median (s) | ratio low/est/high | native/jax peak | frozen-arm ratio |
|---|---|---|---|---|---|---|
| 10000 | 0.0294 | jax_mps_async | 0.1513 | 5.11 / 5.14 / 5.19 | 0.50 | 5.70 |
| 100000 | 0.1057 | jax_mps_async | 0.1605 | 1.51 / 1.52 / 1.54 | 0.11 | 23.90 |
| 1000000 | 0.2613 | jax_mps_sync | 1.5189 | 5.77 / 5.81 / 5.89 | 0.10 | 59.86 |

Two conclusions follow. The bulk of the headline gap above was the
implementation, not the substrate: at N=10^6 the tuned arm takes jax-mps from
12.45 s to 1.52 s, and the ratio falls from 59.9 to 5.8. And the advantage
that remains is real: native MLX wins at every size with tight intervals,
2-10x less peak memory, and all correctness gates passed. The dip to 1.5x at
N=10^5 reflects jax-mps's roughly 0.15 s per-run floor, which dominates until
the arrays are large enough to hide it; native has no such floor. Quote this
experiment as "roughly 1.5-6x with less memory on the strongest fair JAX
implementation we could write," not the 60x above, whenever the comparison is
about backends rather than about shipped libraries. The raw records and summary
remain available as `tuned_lgssm_nohist.json` in the
[immutable result archive][native-results-archive].

### The scan memory number was our artifact

The Reading section above flags native SCAN peak memory at 9-16x jax-mps.
That is a defect in the benchmark motif, not in MLX: the committed kernel
calls `mx.async_eval` on every step, a cadence the project's own MLX
constraints document bans at large N precisely because it pins every
intermediate. Re-measured at N=10^6 with the library's recommended
eval-every-4 cadence, peak memory falls from 200.7 MB to 12.0 MB, a factor of
16.7 that matches the reported blowup. The SCAN timing comparison stands; the
memory column for SCAN should be disregarded. The motif is retained unchanged
because the protocol froze it, and this note is the correction.

[native-harness-archive]: https://github.com/michaelellis003/smcx/tree/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/benchmarks/native_vs_jax_mps
[native-results-archive]: https://github.com/michaelellis003/smcx/tree/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/benchmarks/results/2026-07-16-native-vs-jax-mps
[native-tests-archive]: https://github.com/michaelellis003/smcx/tree/9bba1c57281a363fe69a71b3f108a2996bf03a18/tests
