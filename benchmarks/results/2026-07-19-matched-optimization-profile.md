# Matched optimization profile — 2026-07-19

The matched campaign supports one production optimization: carrying the
already-computed effective sample size (ESS) through bootstrap and guided
filter scans reduces the isolated-worker, process-global jax-mps/MLX peak
counter by `35.27%--38.13%` and produces a smaller StableHLO graph without
changing seeded results. The baseline does **not** support a wall-clock,
lowering, compilation, or first-call speed claim. The representation campaign
supports only a workload-specific low-single-digit dense-CPU signal.
Campaign-order effects were as large as, and usually larger than, candidate
timing effects.

SMC² keeps the simpler reuse of an already-computed row log normalizer, but
the campaign does not establish a compiled performance effect. The extracted
tempering and SMC² JIT factories did not earn their extra complexity and were
removed after measurement. Cross-public-call callable caches had already been
rejected because mutable callbacks and distinct hash-equal callbacks could
silently reuse stale compiled behavior.

## Environment and source identity

- Hardware: Apple M3 Pro (`Mac15,7`, 12 logical CPUs), 36 GiB RAM.
- OS: macOS 26.2 (`25C56`). Every timing worker's snapshots recorded no
  thermal or performance warning. Baseline timing workers stayed on AC power.
  Seven representation A2 block-four cells recorded battery power and were
  excluded from timing.
- Python 3.13.9; JAX 0.10.2; jaxlib 0.10.2; jax-mps 0.10.10; NumPy 2.5.1;
  smcx 1.3.0.
- JAX x64 and the compilation cache were disabled. CPU used asynchronous
  dispatch; MPS used safe dispatch. Inputs were resident on the selected
  device and every measured output was fenced.
- Schedule: one full warm-up and seven fenced repeats in each fresh timing
  process; five independently scheduled process blocks per cell.
- Before source A: temporary measured checkout
  `085a38b1093c6acdf626a38a8dc590bf828a67d6`, source SHA-256
  `dd760c19d4a0beb058fcc160bf656a662a2e9384ee61cde2f4f827ca6372c125`.
  Its production source is baseline commit `47f1cb7`; it adds only the same
  R128 profiling protocol, harness, and tests used by B.
- Measured candidate source B: commit
  `87932a10aa6d05119759f77401b5ef891507cd57`, source SHA-256
  `e320380492f0ee9731710dd00fccf1c014d95ae24c62625548b05623231fb6bc`.
- Both sources used lock SHA-256
  `fbe167588280f38aed17f38e08ed5ddbba1222875ccb1fd1ea90ba3446343ff7`.
  `pyproject.toml`, `uv.lock`, and the profiling harness were byte-identical;
  recorded runtime versions and flags were identical.

No outside implementation was imported into the test suite, and no
benchmarking dependency was added.

## Comparison design and eligibility

The first whole-profile comparison showed a pronounced campaign phase shift,
so its artifacts were retained but no result was credited from that pair
alone. The baseline was then completed in `A1, B1, B2, A2` order. The
representation profile used the complementary `B1, A1, A2, B2` order. This
balances a first-order monotone order effect but does not make uncontrolled
machine-state variation disappear.

Each baseline campaign completed 120 timing cells and 22 validation workers;
all four campaigns therefore completed 480 timing cells and 88 validation
workers. There were no baseline exclusions or failures. Each representation
campaign completed 100 timing cells and 10 validation workers, for another
400 timing cells and 40 validation workers. All were complete and correct,
but representation A2 had seven registered timing-state exclusions, all in
block four: correlated/history-off PyTree CPU; correlated/history-off and
history-on PyTree MPS; and all four Liu–West cells. Only 13 of 20 reverse-order
five-block representation aggregates were fully timing eligible.

Across every A/B campaign, serialized correctness summaries, adaptive-work
records, and independent replicate-validation results were exactly equal.
The raw timing schema does not serialize every posterior leaf, so this is not
a claim of raw-artifact bitwise equality for complete outputs. Exact
fixed-key parity is separately covered by unit tests across resampling
thresholds, history modes, SMC² rejuvenation, and tempering callback cases.

The local campaign directories were:

- baseline: `smcx-opt-before-baseline-r128-20260719`,
  `smcx-opt-after-baseline-r128-20260719`,
  `smcx-opt-after-baseline-r128-20260719-b2`, and
  `smcx-opt-before-baseline-r128-20260719-a2`;
- representation: `smcx-opt-after-representation-r128-20260719`,
  `smcx-opt-before-representation-r128-20260719`,
  `smcx-opt-before-representation-r128-20260719-a2`, and
  `smcx-opt-after-representation-r128-20260719-b2`.

## Timing result: no defensible effect

For each cell, the order-balanced after/before estimand is

`exp(0.5 * ((log(B1) - log(A1)) + (log(B2) - log(A2))))`.

Positive percentages below mean candidate B was slower. Fifteen of the
sixteen production steady-state cells reversed direction between the two
order pairs. The only directionally consistent cell, Liu–West on MPS, changed
by just `+1.6%`, inside the much wider variation of source-neutral controls.

| Workload | Device | B1 / A1 | B2 / A2 | Balanced |
|---|---|---:|---:|---:|
| Auxiliary | CPU | +22.8% | -1.5% | +10.0% |
| Auxiliary | MPS | +2.1% | -3.6% | -0.8% |
| Bootstrap LGSSM | CPU | +16.7% | -5.6% | +4.9% |
| Bootstrap LGSSM | MPS | +17.4% | -6.4% | +4.8% |
| Bootstrap SV | CPU | +14.8% | -5.9% | +4.0% |
| Bootstrap SV | MPS | +8.2% | -15.4% | -4.3% |
| Guided LGSSM | CPU | +7.3% | -3.6% | +1.7% |
| Guided LGSSM | MPS | +12.9% | -4.4% | +3.9% |
| Liu–West | CPU | +21.0% | -17.4% | -0.0% |
| Liu–West | MPS | +0.3% | +3.0% | +1.6% |
| SMC² forced | CPU | +9.4% | -2.7% | +3.2% |
| SMC² forced | MPS | +12.3% | -0.5% | +5.7% |
| SMC² forward | CPU | +4.5% | -0.9% | +1.8% |
| SMC² forward | MPS | +12.9% | -0.6% | +5.9% |
| Tempering | CPU | +12.0% | -1.5% | +5.1% |
| Tempering | MPS | +5.8% | -2.5% | +1.6% |

Balanced steady-state changes among untouched controls spanned `-17.4%` to
`+9.3%`. Some controls agreed in direction across both order pairs while
showing mutually contradictory large effects, so direction agreement alone
is not an adequate gate here. For each outer-jittable baseline arm, the five
block-balanced ratios straddled one in each of lowering, backend compilation,
and first execution. Tempering and both SMC² regimes are host-controlled and
also reversed direction for every first-call comparison. No timing phase
supports a performance claim.

## Reproducible MPS memory result

Unlike timing, the MPS memory result repeated in every process block and both
opposite-order pairs. Values are campaign medians of the per-worker,
process-global jax-mps/MLX `peak_bytes_in_use` counter sampled immediately
after the measured calls at the baseline `N=10,000` cells.

| Workload | A peak, MiB | B peak, MiB | Change |
|---|---:|---:|---:|
| Bootstrap LGSSM | 7.4964 | 4.6379 | -38.13% |
| Bootstrap SV | 7.5110 | 4.6518 | -38.07% |
| Guided LGSSM | 7.3089 | 4.7310 | -35.27% |

Auxiliary, Liu–West, and all four resampler controls were unchanged. CPU
executable peak memory was unchanged; reported CPU temporary memory decreased
by only 184 bytes. Process RSS varied with campaign phase and is not credited.
The supported interpretation is specifically an isolated-worker jax-mps/MLX
peak-counter reduction, not a portable total-memory claim.

## StableHLO census

The registered CPU census covered all nine outer-jittable block-zero baseline
arms. Tempering and SMC² are host-controlled and were correctly excluded.

| Workload | A ops | B ops | A bytes | B bytes |
|---|---:|---:|---:|---:|
| Bootstrap LGSSM | 920 | 913 | 86,345 | 85,908 |
| Bootstrap SV | 916 | 909 | 85,881 | 85,444 |
| Guided LGSSM | 982 | 975 | 90,849 | 90,394 |

Each changed graph removed one `broadcast_in_dim`, two constants, one
exponential, two multiplies, and one subtract. Auxiliary, Liu–West, and every
resampler had byte-identical StableHLO and identical operation counts. This is
a syntactic compiler-IR census, not FLOPs or a device-cost model. It supports
the graph-simplification attribution but not a latency claim.

## Representation and retained history

All dense/PyTree tracking pairs retained exact correctness and adaptive work.
The source comparison had the same phase problem as baseline: identical-source
A2/A1 changes were `15.3%--24.8%`, nearly reproducing the apparent B1/A1
improvement, while B2/B1 ranged from `1.8%` faster to `0.3%` slower.
Report-median BAAB effects and exact block-paired effects disagreed
materially.

The clean reverse pair and block-paired BAAB both suggest a workload-specific
`2.2%--5.1%` improvement for dense CPU tracking. Dense MPS ranges from `1.7%`
faster to `0.6%` slower, PyTree effects are mixed, and Liu–West has no valid
reverse aggregate. This is a limited low-single-digit CPU signal, not support
for the original `15%--24%` apparent effect or a general runtime claim.

Within-campaign representation ratios avoid that source-order confound. The
following pool uses every AC-valid, exactly work-matched process-block pair
across all four campaigns.

| Device | Covariance | History | Pairs | Median PyTree/dense | IQR |
|---|---|---:|---:|---:|---:|
| CPU | correlated | off | 19 | 1.071 | 1.061--1.079 |
| CPU | correlated | on | 20 | 1.095 | 1.054--1.106 |
| CPU | diagonal | off | 20 | 1.070 | 1.060--1.075 |
| CPU | diagonal | on | 20 | 1.085 | 1.053--1.105 |
| MPS | correlated | off | 19 | 1.011 | 0.992--1.015 |
| MPS | correlated | on | 19 | 1.009 | 0.998--1.021 |
| MPS | diagonal | off | 20 | 1.004 | 0.996--1.014 |
| MPS | diagonal | on | 20 | 1.008 | 0.995--1.019 |

The two-leaf state therefore costs about `7%--9.5%` on CPU and about `0%--1%`
on MPS for this model. The CPU result includes the workload's leaf-concatenation
adapter and is not a universal PyTree tax. Selective PyTrees remain useful for
semantically composite or Rao–Blackwellized states; homogeneous states should
remain dense. Native JAX PyTrees are sufficient, so Equinox remains
unnecessary. Liu–West stays dense because covariance and shrinkage require a
Euclidean parameter vector.

Pooled AC-valid history-on/off medians remained modest:

| Workload | CPU | MPS |
|---|---:|---:|
| Dense tracking, correlated | 1.027 | 1.020 |
| Dense tracking, diagonal | 1.031 | 1.017 |
| PyTree tracking, correlated | 1.043 | 1.024 |
| PyTree tracking, diagonal | 1.046 | 1.015 |
| Liu–West | 1.011 | 1.034 |

The representation campaign independently reproduced the MPS memory result:

| State | Covariance | A1 / A2, MiB | B1 / B2, MiB | Reduction |
|---|---|---:|---:|---:|
| Dense | correlated | 8.030 / 7.879 | 4.972 / 4.972 | 37.5% |
| Dense | diagonal | 7.935 / 8.127 | 4.972 / 4.972 | 38.1% |
| PyTree | correlated | 8.296 / excluded | 4.972 / 4.972 | 40.1%¹ |
| PyTree | diagonal | 8.343 / 8.247 | 4.972 / 4.972 | 40.1% |

¹ B1/A1 only; A2's correlated-PyTree MPS aggregate failed its power-state
gate. The other three rows have eligible balanced reductions.

Every timing-eligible B history-off tracking block was within
`4.97243--4.97246 MiB`; every eligible A block was at least `7.87855 MiB`.
Tracking history-on medians rounded to `137.45 MiB` (raw
`137.448--138.152 MiB`). Liu–West medians rounded to `7.824 MiB` off and
`45.941 MiB` on (raw `7.824378--7.824424 MiB` and
`45.940873--46.510910 MiB`). These are consistent with output-dominated
history. CPU executable peaks were unchanged. After the optimization, dense
and PyTree history-off peaks are effectively identical.

## Resampler validation correction

The first A baseline used the then-registered 64-replicate resampler gate. One
systematic-CPU hashed projection at `N=100,000` exceeded its five-SE tolerance
by `1.14%`, a signed standardized discrepancy of `+5.057`; every other
projection and every algorithm-specific hard invariant passed. Exact
integration over all 99,999 systematic-grid boundaries matched the oracle.
Keeping all 64 keys and extending the predeclared prefix to `R=128`, `R=256`,
and `R=512` changed that discrepancy to `+1.493`, `-0.189`, and `+0.075`.

The prospective gate was therefore raised uniformly to `R=128`, and all four
baseline campaigns in this report passed. That R64 baseline campaign remains
failed and timing-ineligible; no seed was rerolled and no resampling production
code was changed. The shared-grid dependence of systematic
resampling makes marginal variance reasoning subtle, as discussed by Douc,
Cappé, and Moulines
([2005 preprint](https://arxiv.org/abs/cs/0507025)). Selecting replication
from required Monte Carlo precision follows Morris, White, and Crowther
([2019 DOI](https://doi.org/10.1002/sim.8086)).

## Why the timing claim was withheld

The observed whole-campaign drift is a standard systems-benchmark hazard, but
this evidence does not identify its physical cause. In particular, normal
power and thermal snapshots do not rule out scheduler, frequency, allocator,
or other transient system-state effects.

Kalibera and Jones describe modern systems as nondeterministic, recommend
repetition at the experimental level where variation arises, and report
effect-size confidence intervals
([2013 DOI](https://doi.org/10.1145/2464157.2464160)). Google Benchmark's
official guide provides randomized interleaving specifically to reduce the
impact of system-state changes
([user guide](https://github.com/google/benchmark/blob/main/docs/user_guide.md)).
Python's pyperf likewise uses multiple worker processes, warm-ups, stability
diagnostics, and explicit system tuning
([run guide](https://pyperf.readthedocs.io/en/stable/run_benchmark.html),
[system guide](https://pyperf.readthedocs.io/en/stable/system.html)). JAX's
official guidance requires separating first-call compilation from steady
execution, resident inputs, and `block_until_ready()` fencing
([benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)).
The harness already followed the JAX requirements and isolated cells in fresh
processes; the order reversal exposed a higher-level source of variation.

A future timing comparison should randomly interleave A/B variants within
each process-block/cell schedule instead of running whole-source campaigns,
retain independent process repetitions, report an effect-size interval, add
complete-output digests, and expose resampling-event counts for auxiliary and
Liu–West filters. Adding pyperf as a project dependency is not justified: the
specialized JAX/MPS harness already owns process isolation and device fencing,
and a generic runner would not remove Apple-silicon system-state variation.

## Source decisions

- **Retained:** ESS carry in bootstrap/guided filters. Seeded results and
  public APIs are unchanged; MPS allocator memory and CPU StableHLO both show
  reproducible improvements.
- **Retained as simplification only:** SMC² reuses the log normalizer returned
  by row normalization. No speed or memory effect is claimed.
- **Removed after measurement:** tempering and SMC² module-level JIT factory
  scaffolding. It showed no lifecycle benefit. Invocation-local JIT functions
  preserve callback freshness and the simpler ownership boundary.
- **Rejected before this campaign:** cross-public-call caches keyed by user
  callables. Regression tests demonstrate stale behavior for mutable callbacks
  and distinct callbacks with equal hashes and equality.
- **Retained:** typed public callback Protocols, exact-output regression tests,
  native JAX PyTree support, and the public warning that five tempering sweeps
  are a mutation budget rather than an accuracy guarantee.

The filter source underlying the supported memory and StableHLO claims is
unchanged from measured B. Later cleanup only changes the uncredited
tempering/SMC² factory structure and public documentation. There is no new
runtime dependency, API divergence, numerical default, copied outside code,
or licensing obligation.

## Reproduction

From a clean worktree at each recorded source:

```bash
uv run python -m benchmarks.profiling.run \
  --profile baseline --platforms cpu mps \
  --output-dir /tmp/smcx-profile-baseline
uv run python -m benchmarks.profiling.run \
  --profile representation --platforms cpu mps \
  --output-dir /tmp/smcx-profile-representation
uv run python -m benchmarks.profiling.report \
  --input-dir /tmp/smcx-profile-baseline \
  --output /tmp/smcx-profile-baseline.md --date 2026-07-19
uv run python -m benchmarks.profiling.census \
  --campaign-dir /tmp/smcx-profile-baseline --platform cpu \
  --output /tmp/smcx-profile-baseline-census-cpu.json
```

Run sources serially on an otherwise idle machine. For a new inferential
timing claim, use the interleaved A/B design above rather than reproducing only
one whole-campaign pair.
