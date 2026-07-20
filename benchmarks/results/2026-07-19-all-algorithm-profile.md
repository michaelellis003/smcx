# All-algorithm, model-regime, and scaling profile — 2026-07-19

Status: **measurement complete, with explicit exclusions**. Baseline,
filter-regime, and representation campaigns passed their registered gates.
The original scaling campaign retained 12 failed mathematical cells; their
60 timing workers remain ineligible even where a later diagnostic explains
the failure. In total, 290 of 302 mathematical cells, and 1,450 of 1,510
timing workers, are eligible in the four campaigns summarized here.

This is a descriptive profile on one Apple-silicon machine. It identifies
optimization candidates; it does not establish a portable backend ranking or
credit any optimization without a separately matched before/after campaign.

## Measurement metadata

| Field | Value |
|---|---|
| Hardware | Apple M3 Pro, 12 CPU cores, 36 GiB unified memory |
| OS | macOS 26.2 (25C56) |
| Power / thermal | AC power; no thermal or performance warning before timing, after timing, or after extraction |
| Python | 3.13.9 |
| JAX / jaxlib | 0.10.2 / 0.10.2 |
| jax-mps | 0.10.10, safe dispatch |
| NumPy | 2.5.1 |
| TFP nightly | 0.26.0.dev20260717 |
| smcx | 1.3.0 |
| Timing design | Five isolated process blocks; one warm-up and seven fenced repeats per block |
| Primary estimate | Median of five per-process steady medians; block IQR retained |
| Dispatch | CPU asynchronous; MPS safe; every complete result PyTree fenced |
| Runtime | Compilation cache and x64 disabled; backend fixed per worker |
| Seeds | order/inference `20260719`; independent validation `20260720` |

Initial keys, observations, and static arrays were placed before timing.
Transfers and synchronizations performed inside a public operation remain in
the measurement. This distinction matters for host-controlled `temper` and
`smc2`. The timing method follows JAX's
[asynchronous-dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
and [benchmarking](https://docs.jax.dev/en/latest/benchmarking.html) guidance.

### Campaign provenance

| Profile | Source | Timing / validation workers | Calls | Result |
|---|---|---:|---:|---|
| baseline | `6a3f724cc95da5da216a839ead3b13d432de4308` | 120 / 22 | 960 / 264 | 24/24 mathematical cells passed |
| filter-regimes | `2452bfa3348a0708e5279a0f91ca3ceb973dcefb` | 540 / 84 | 4,320 / 1,680 | 108/108 passed under amended gate |
| scaling | `2452bfa3348a0708e5279a0f91ca3ceb973dcefb` | 750 / 144 | 6,000 / 1,464 | 138/150 passed; 12 retained failures |
| representation | `651666b414ce3c16fa7398d4e190c433715bfb5a` | 100 / 10 | 800 / 184 | 20/20 passed |

The corresponding source SHA-256 values are
`6ebea32803d1eb885688f587bc740d021a63da4eef3fedea59e9fb6e6567ae5b`,
`4cbd8426428c246882aba8bd64f054b76ddacf970bfcde95ca51aa4541690ca4`,
and
`b8aebca1ae24b8cca5d279b2eb62b65cb8d89c08be8faa73e5b27c16b72f9167`.
All measured source trees were clean. Baseline, filter-regime, and scaling
manifests share lock SHA-256
`fbe167588280f38aed17f38e08ed5ddbba1222875ccb1fd1ea90ba3446343ff7`.

The local raw directories used for this report were:

- `/tmp/smcx-profiling-baseline-20260719-6a3f724-01010`;
- `/tmp/smcx-profiling-filter-regimes-20260719-2452bfa-01010`;
- `/tmp/smcx-profiling-scaling-20260719-2452bfa-01010`; and
- the campaign summarized in
  [`2026-07-19-representation-history-profile.md`](2026-07-19-representation-history-profile.md).

## Correctness and claim eligibility

The independent gates used float64 Kalman recurrences for linear-Gaussian
models, numerical quadrature for the unknown-AR model, closed-form Gaussian
tempering moments, and repeated partition moments plus scheme-specific hard
invariants for resamplers. They checked evidence where meaningful, final
means, raw second moments, shapes, dtypes, finiteness, normalized weights,
ESS bounds, and internal evidence identities. Stochastic volatility is
structural-only because no independent exact oracle was registered.

The filter-regime rerun treats threshold-zero bootstrap and auxiliary arms as
structural-only. Their earlier 20-replicate evidence gates failed because
ordinary importance sampling never reached the rare paths controlling the
likelihood mean: calibrated and sharp mean evidence ratios were about
`5.33e-17` and `6.35e-237`. Guided threshold-zero arms, adaptive arms, and
forced-resampling arms retained and passed their Kalman gates. This is a
validation-feasibility correction, not evidence that never-resampling gives
accurate inference at these settings.

### Scaling failures retained exactly

The original scaling result remains failed for these mathematical cells:

- residual, CPU, `N=100,000`: uniform and zero-tail weights;
- residual, MPS, `N=100,000`: zero-tail weights;
- systematic, MPS: `N=100,000` zero-tail, and `N=1,000,000`
  moderately uneven and zero-tail weights;
- temper, both backends: `(d=32, N=1,000)`,
  `(d=128, N=1,000)`, and `(d=128, N=10,000)`.

The six resampler failures occurred only in the eight-way hashed projection.
All 16 contiguous projections and every scheme-specific hard invariant
passed. With eight replicates and 2,304 tested coordinates,
`P(|t_7| > 5)=0.001565`, giving 3.61 expected null exceedances before
accounting for dependent coordinates. A one-time fixed-prefix diagnostic kept
the original eight keys and added 56 deterministic keys. All six cells then
passed, and their worst absolute standardized error fell from `5.12--7.40`
to `0.81--1.86`.

The first prospective validation revision therefore used 64 replicates. Its
key schedule preserved `split(key(20260720), 8)` exactly and extended it with
a tagged, prefix-stable `fold_in` sequence. At 64 replicates,
`P(|t_63| > 5)=4.857e-6`; the Bonferroni sum over 2,304 coordinates is
`0.0112`. These are validation-design diagnostics under a replicate-normal
approximation, not an independence claim. The original six cell timings are
still ineligible and are excluded below. This treatment follows the Monte
Carlo error guidance of Morris et al. (2019)
([DOI](https://doi.org/10.1002/sim.8086)). No resampler implementation was
changed in response to these false rejections.

The temper failures were different: CPU and MPS agreed on material particle
impoverishment. With five random-walk Metropolis sweeps per stage, the mean
within-cloud variance retained the following share of exact posterior
variance:

| Dimension | Particles | CPU | MPS |
|---:|---:|---:|---:|
| 32 | 1,000 | 88.0% | 87.6% |
| 128 | 1,000 | 19.7% | 20.2% |
| 128 | 10,000 | 34.5% | 34.9% |

The previously eligible `(d=32, N=10,000)` cell retained `91.5%` on CPU
and `91.0%` on MPS. Replaying its exact 12 keys under the new within-cloud
gate passed on both backends: the estimated variances were `0.3008` and
`0.2993` against `0.3289`, within registered five-SE tolerances of about
`0.050`. Its timing remains eligible, while the retained-variance diagnostic
prevents that decision from being read as an exactness claim.

At `d=128`, every raw second-moment error was negative and 116--122 of 128
coordinates failed. Evidence-ratio means of `1e8--7e9` were not rejected only
because their Monte Carlo error was larger; they are not affirmative evidence
of accuracy. The gate now also checks the replicated coordinate-averaged,
`ddof=1` within-cloud variance.

An untimed, fixed-key CPU tuning diagnostic showed that additional invariant
RWM sweeps move toward the correct cloud:

| Dimension | Particles | 5 sweeps | 20 sweeps | 50 sweeps |
|---:|---:|---:|---:|---:|
| 32 | 1,000 | 88.0% | 96.0% | — |
| 32 | 10,000 | 91.5% | 99.3% | — |
| 128 | 1,000 | 19.7% | 44.8% | 70.1% |
| 128 | 10,000 | 34.5% | 68.0% | 91.1% |

Those percentages are accuracy diagnostics, not timing results. Five sweeps
are a workload/default, not an accuracy guarantee. Changing the default or
mutation kernel changes fixed-key output and requires an ADR.

A further fixed-key CPU diagnostic at `d=128` used 128 sweeps. It retained
`89.16%` variance at `N=1,000` and `98.69%` at `N=10,000`; the mean gates
passed, but raw-second-moment and within-cloud-variance gates still failed.
Both arms averaged 13 temperature stages, with acceptance `0.269` and `0.240`.
This remains accuracy-only evidence: more local mutation helps substantially,
but even 128 sweeps did not validate both cells.

## Baseline across every shipped algorithm

Times are milliseconds. Parentheses contain the IQR across the five process
medians. A ratio is reported only when CPU and MPS had matched, observable
adaptive work.

| Workload | CPU, ms (IQR) | MPS, ms (IQR) | MPS/CPU | Work evidence |
|---|---:|---:|---:|---|
| auxiliary LGSSM | 39.670 (0.464) | 160.277 (1.057) | — | adaptive count hidden |
| bootstrap LGSSM | 37.083 (0.438) | 150.737 (0.978) | 4.065 | 44 resamples |
| guided LGSSM | 34.211 (0.193) | 115.292 (2.446) | 3.370 | 20 resamples |
| bootstrap SV | 159.363 (0.236) | 512.426 (18.204) | — | CPU 49, MPS 50 resamples |
| Liu--West | 63.917 (1.657) | 214.529 (14.813) | — | adaptive count hidden |
| multinomial | 2.069 (0.026) | 4.047 (0.076) | 1.956 | matched |
| residual | 2.799 (0.033) | 3.161 (0.460) | 1.129 | matched |
| stratified | 1.581 (0.027) | 3.455 (0.649) | 2.186 | matched |
| systematic | 1.258 (0.021) | 2.880 (0.403) | 2.289 | matched |
| SMC2 forced | 315.180 (3.185) | 1,137.277 (14.888) | 3.608 | 20 rejuvenations |
| SMC2 forward | 271.690 (1.772) | 318.271 (0.761) | 1.171 | zero rejuvenations |
| temper | 312.249 (2.391) | 451.357 (1.015) | 1.446 | six stages |

Standard-filter baseline settings are `N=10,000`, `T=100`, calibrated
observations, threshold `0.5`, and no history. SV uses `T=500`; Liu--West uses
one parameter coordinate. Resamplers use `N=100,000` moderately uneven
weights. SMC2 forward is `128 x 256`, `T=40`; forced is `32 x 64`, `T=20`.
Temper is `d=32`, `N=10,000`, with five RWM sweeps.

Cold lifecycle costs are separate from steady execution. Whole-filter
CPU lowering plus compilation was roughly `237--273 ms`, versus `59--64 ms`
on MPS; Liu--West was about `366--489 ms` versus `69--74 ms`. Lowering plus
compilation for most isolated resamplers totaled `73--103 ms` on CPU and
`24--45 ms` on MPS.
Multinomial was the CPU exception: backend compilation grew from about
`116 ms` at `N=10,000` to `294 ms` at `N=100,000` and `312 ms` at
`N=1,000,000`, while lowering stayed near `35--40 ms`. These are cold-process
costs with the persistent compilation cache deliberately disabled.

Host-controlled first calls also include lazily encountered compilation and
internal synchronization. Baseline first/steady times were `933/272 ms`
(CPU) and `514/318 ms` (MPS) for forward SMC2, `2,566/315 ms` and
`1,754/1,137 ms` for forced SMC2, and `1,192/312 ms` and `725/451 ms` for
tempering. MPS first-call advantage does not imply a steady-state advantage.

## Observation, proposal, resampling, and history regimes

The following filter-regime slice is `N=10,000`, `T=100`, threshold `0.5`,
and history off. Counts are exact where returned by the algorithm; auxiliary's
adaptive count is intentionally not inferred.

| Filter | Observation | CPU, ms | MPS, ms | Resamples |
|---|---|---:|---:|---:|
| bootstrap | diffuse | 33.186 | 115.197 | 18 |
| bootstrap | calibrated | 37.439 | 151.455 | 44 |
| bootstrap | sharp | 46.674 | 220.695 | 99 |
| auxiliary | diffuse | 34.995 | 121.563 | hidden |
| auxiliary | calibrated | 39.934 | 159.282 | hidden |
| auxiliary | sharp | 49.175 | 227.365 | hidden |
| guided | diffuse | 33.298 | 108.570 | 13 |
| guided | calibrated | 34.148 | 115.463 | 20 |
| guided | sharp | 32.832 | 105.757 | 11 |

The locally optimal guided proposal avoids the bootstrap filter's sharp-data
resampling collapse: 11 versus 99 resamples, with `0.703x` CPU time and
`0.479x` MPS time in that matched cell. This is evidence for choosing a
well-adapted proposal when one is available, not for changing bootstrap's
implementation.

Across exact-work filter cells, MPS/CPU ranged from `3.22x` to `4.79x` at
this size. Threshold-zero cells performed no resampling; threshold `1.1`
performed exactly 99 events. Adaptive auxiliary CPU/MPS ratios are withheld
because the event count is not observable without changing production code.

For exact-work history pairs, retaining scalar state history added
`0.2--3.4%` on CPU and `1.2--9.7%` on MPS. The memory change was much larger:
CPU executable peak rose from about `0.116 MiB` to `11.446 MiB`, while the MPS
allocator peak rose from `5.18--7.59 MiB` to about `34.46 MiB`. The detailed
tracking and Liu--West history results are reported separately in the
[representation/history profile](2026-07-19-representation-history-profile.md).

## Particle-count and model scaling

The LGSSM arms use `T=100`; SV uses `T=500`. Auxiliary and Liu--West force
99 resampling events. Bootstrap and guided use threshold `0.5` and matched
counts. Times are eligible steady medians in milliseconds.

| Workload | CPU: 1k / 10k / 100k | MPS: 1k / 10k / 100k | MPS/CPU at 100k |
|---|---:|---:|---:|
| auxiliary | 4.924 / 49.089 / 228.816 | 206.586 / 231.526 / 271.335 | 1.186 |
| bootstrap | 3.959 / 37.586 / 160.775 | 140.543 / 151.321 / 167.623 | 1.043 |
| guided | 3.608 / 34.488 / 133.484 | 111.471 / 114.917 / 131.451 | 0.985 |
| Liu--West, `d=1` | 6.583 / 73.171 / 363.821 | 220.393 / 247.524 / 393.216 | 1.081 |
| bootstrap SV | 17.017 / 159.529 / 609.945 | 481.799 / 524.216 / 564.511 | — |

SV's CPU and MPS runs performed 49 and 50 resamples, respectively, so its raw
times are reported but the cross-backend ratio is withheld. Its raw medians
cross by `N=100,000`; that is not an equal-work performance conclusion.

The full `1,000` to `100,000` descriptive power-law exponents were:

| Workload | CPU exponent | MPS exponent | Descriptive crossover |
|---|---:|---:|---|
| auxiliary | 0.834 | 0.059 | about 133k, extrapolated |
| bootstrap | 0.804 | 0.038 | about 107k, extrapolated |
| guided | 0.784 | 0.036 | about 97k, interpolated |
| Liu--West | 0.871 | 0.126 | about 117k, extrapolated |
| bootstrap SV | 0.777 | 0.034 | withheld: unequal work |

Crossovers are log-linear descriptions of these three sizes, with no formal
interval. Extrapolated values are unmeasured and must be confirmed around
`75k--150k`. At `N=100,000`, no-history MPS device peaks were about
`69--73 MiB` for every filter, including `T=500` SV. This suggests the particle
working set, not sequence length, controls device peak when history is off.

### Liu--West parameter dimension

The independent dimension axis uses `N=1,000`, `T=100`, threshold `1.1`, and
mathematically replicated coordinates preserving the same scalar AR target.

| Dimension | CPU, ms | MPS, ms | MPS device peak, MiB |
|---:|---:|---:|---:|
| 1 | 6.583 | 220.393 | 0.55 |
| 4 | 13.168 | 228.180 | 0.67 |
| 16 | 23.155 | 229.000 | 1.60 |
| 64 | 44.742 | 226.796 | 5.42 |

CPU time responds to dimension, while MPS time is essentially a `227 ms`
dispatch floor over this range. All dimension gates passed, including the
orthogonal parameter-spread check. This argues against small-problem MPS use,
not against general PyTree or multidimensional latent-state support.

### Isolated resamplers

Each value below is the median of mutually eligible weight-regime medians at
that size; failed regime/backend pairs are excluded from both sides. It is a
compact descriptive summary, not a pooled statistical estimate.

| Resampler | CPU: 10k / 100k / 1m, ms | MPS: 10k / 100k / 1m, ms | Crossover |
|---|---:|---:|---|
| multinomial | 0.483 / 2.077 / 13.73 | 2.288 / 3.886 / 13.15 | about 0.86m, interpolated |
| residual | 0.526 / 2.773 / 18.66 | 2.968 / 3.14 / 21.88 | none observed |
| stratified | 0.348 / 1.587 / 10.49 | 2.309 / 3.238 / 11.57 | about 1.44m, extrapolated |
| systematic | 0.212 / 1.267 / 8.73 | 2.132 / 3.278 / 10.75 | about 1.91m, extrapolated |

At one million particles, MPS multinomial was between parity and about 6%
faster across the four eligible regimes. No other isolated resampler crossed
in the measured range. Some small MPS cells had block IQRs of `25--42%` of
their median, while million-particle cells were tighter; the crossover values
are therefore routing hypotheses, not thresholds to encode.

At `N=1,000,000`, CPU executable analysis reported `7.63 MiB`, close to the
logical float32-weight plus int32-output footprint. The MPS process-global
allocator peak was `494--502 MiB`, roughly 65 times that logical footprint.
This is the strongest memory-attribution candidate in the campaign.

### SMC2

All forward cells had exactly zero rejuvenations, so their backend ratios are
matched. First and steady times are both end-to-end public-call times.

| `N_theta x N_x`, `T` | CPU first / steady, ms | MPS first / steady, ms | MPS/CPU steady | MPS peak, MiB |
|---|---:|---:|---:|---:|
| `32 x 64`, 20 | 856.509 / 232.582 | 375.999 / 176.377 | 0.758 | 0.38 |
| `128 x 256`, 40 | 963.542 / 272.691 | 529.523 / 318.848 | 1.169 | 7.54 |
| `512 x 512`, 100 | 1,238.290 / 622.693 | 1,077.056 / 865.338 | 1.390 | 66.54 |

MPS helps only the smallest forward steady cell and all three first calls.
The baseline forced-rejuvenation arm is a different regime: with exactly 20
rejuvenations, MPS was `3.608x` slower. The reversal points to host/device
phase and synchronization overhead as the next attribution target; macro
timing alone does not identify a production code line.

### Tempering

Only correctness-eligible cells are timed below. Both backends performed the
same number of adaptive temperature stages.

| Dimension | Particles | Stages | CPU first / steady, ms | MPS first / steady, ms | MPS peak, MiB |
|---:|---:|---:|---:|---:|---:|
| 4 | 1,000 | 2 | 1,042.633 / 222.432 | 380.060 / 125.173 | 0.80 |
| 4 | 10,000 | 2 | 1,028.875 / 235.319 | 382.305 / 128.686 | 10.30 |
| 32 | 10,000 | 6 | 1,208.226 / 314.054 | 714.581 / 448.981 | 70.10 |

MPS is about `0.55--0.56x` CPU at `d=4`, but `1.43x` CPU at the eligible
`d=32`, `N=10,000` cell. The failed `d=32`, `N=1,000` and all `d=128`
timings support no performance claim.

## Structured state and retained history

The four-coordinate controlled tracker compared a dense array with a two-leaf
position/velocity PyTree at `N=10,000`, `T=200`, two covariance regimes, and
history off/on. Matched PyTree/dense ratios were `1.047--1.068` on CPU and
`1.004--1.015` on MPS. PyTrees are therefore useful for semantic structure
and selectively generalized latent states, but are not a speed optimization
for this model. The CPU number includes the workload's leaf-flattening
adapter.

Tracking history cost `1.041--1.060x` on CPU and `1.012--1.028x` on MPS.
The retained result is `45.78 MiB`; CPU executable peak equals that floor and
MPS allocator peak is `137.45 MiB`. Liu--West state-plus-parameter history cost
`1.005x` on CPU and `1.030x` on MPS, with memory rising from `0.15` to
`15.26 MiB` in CPU executable analysis and from `7.82` to `45.94 MiB` in the
MPS allocator.

This profile uses JAX-native
[PyTrees](https://docs.jax.dev/en/latest/pytrees.html); Equinox is neither
needed nor used. jax-mps 0.10.10's native slice updates removed the much larger
0.10.9 history allocation, so the evidence does not justify an smcx-specific
history rewrite. See the
[jax-mps release](https://github.com/tillahoffmann/jax-mps/releases/tag/v0.10.10)
and upstream slice work in [#219](https://github.com/tillahoffmann/jax-mps/pull/219),
[#220](https://github.com/tillahoffmann/jax-mps/pull/220), and
[#222](https://github.com/tillahoffmann/jax-mps/pull/222).

## Memory-scope warning

Memory columns intentionally keep incompatible scopes separate:

- CPU executable peak is compiler analysis for one executable; it is not
  process RSS.
- MPS device peak is a process-global MLX allocator high-water mark in unified
  memory. jax-mps 0.10.10 does not implement PJRT executable memory analysis
  ([source](https://github.com/tillahoffmann/jax-mps/blob/v0.10.10/src/pjrt_plugin/pjrt_executable.cc#L260-L263)).
- Process maximum RSS includes Python, imports, JAX, and plugin state and is
  not an incremental workload allocation.
- Host-controlled `temper` and `smc2` have no single outer executable to
  analyze.

CPU executable and MPS allocator numbers must not be divided to claim a
backend memory ratio. Within-scope history and size trends remain useful.

## Optimization decisions supported by the evidence

1. **Fix tempering accuracy before optimizing its failed cells.** The current
   local RWM mutation is under-mixed at high dimension. Compare 20/50 sweeps
   and at least one justified alternative using accuracy per time. Any default
   or kernel change needs an ADR and fixed-key output-change handling.
2. **Profile million-particle MPS resampling next.** The `494--502 MiB`
   allocator peak and multinomial's late crossover justify an isolated
   primitive/allocator trace. Preserve the resampling API and distributional
   gates; the fixed-prefix result gives no reason to rewrite a correct kernel.
3. **Instrument SMC2 host phases without changing results.** Separate inner
   filters, ESS transfers, rejuvenation, and PMMH synchronization. The small
   forward win, large forward loss, and forced `3.608x` loss are not explained
   by particle count alone.
4. **Confirm filter crossover instead of encoding it.** Measure
   `N=75k--150k` at matched work and at least one higher-dimensional model.
   Current `97k--133k` values are descriptive interpolation/extrapolation.
5. **Treat proposal quality as an algorithm-selection lever.** The guided
   filter's sharp-data result is a real reduction in resampling work; a generic
   callback micro-optimization is unlikely to reproduce it.
6. **Keep the current dense and PyTree paths.** Selective structured-state
   support is useful and modestly costly. There is no evidence for a universal
   PyTree conversion or an Equinox dependency.
7. **Keep current history storage.** Runtime overhead is small and memory
   follows the requested retained output after the jax-mps floor fix. Optimize
   only if a new workload demonstrates a remaining allocation defect.
8. **Separate cold-start work from steady work.** CPU compilation, especially
   multinomial, deserves a cold-start investigation, but this campaign
   deliberately disabled persistent caching. It is not a hot-loop regression.

No current macro result identifies a safe production rewrite by itself. The
next optimization campaign must preregister a counterfactual, retain the same
correctness gate, and report matched before/after measurements.

## Scope and limitations

- Results are local-only for one M3 Pro, OS, JAX, and jax-mps version. CI must
  not assert these times.
- Five process medians support robust description, not a formal effect-size
  interval. Repeated-process design follows Kalibera and Jones (2013)
  ([DOI](https://doi.org/10.1145/2464157.2464160)).
- Models cover scalar linear Gaussian, four-coordinate controlled tracking,
  nonlinear stochastic volatility, unknown static AR parameters, and a
  conjugate static Gaussian target. They do not establish performance for
  every user callback or state dimension.
- Adaptive algorithms are compared only with observable matched work. Hidden
  counts and mismatched counts are explicitly withheld.
- StableHLO counts would be compiler-IR census data, not FLOPs. The later
  matched optimization campaign added that census and is reported separately;
  no trace was used here to turn macro observations into source attribution.
- Descriptive crossovers outside measured sizes are hypotheses.

## Sources, attribution, and licenses

Primary algorithm and model sources used to define the campaign are:

- Gordon, Salmond, and Smith (1993), bootstrap filtering
  ([DOI](https://doi.org/10.1049/ip-f-2.1993.0015));
- Pitt and Shephard (1999), auxiliary particle filtering
  ([DOI](https://doi.org/10.1080/01621459.1999.10474153));
- Doucet, Godsill, and Andrieu (2000), proposal-weight correction
  ([DOI](https://doi.org/10.1023/A:1008935410038));
- Liu and West (2001), joint state/parameter filtering
  ([DOI](https://doi.org/10.1007/978-1-4757-3437-9_10));
- Del Moral, Doucet, and Jasra (2006), SMC samplers
  ([DOI](https://doi.org/10.1111/j.1467-9868.2006.00553.x));
- Jasra et al. (2011), adaptive temperature selection
  ([DOI](https://doi.org/10.1111/j.1467-9469.2010.00723.x));
- Chopin, Jacob, and Papaspiliopoulos (2013), SMC2
  ([DOI](https://doi.org/10.1111/j.1467-9868.2012.01046.x));
- Andrieu, Doucet, and Holenstein (2010), particle MCMC and PMMH
  ([DOI](https://doi.org/10.1111/j.1467-9868.2009.00736.x));
- Douc, Cappe, and Moulines (2005), resampling schemes
  ([DOI](https://doi.org/10.1109/ISPA.2005.195385));
- Kong, Liu, and Wong (1994), normalized-weight ESS
  ([DOI](https://doi.org/10.1080/01621459.1994.10476469));
- Del Moral, Doucet, and Jasra (2012), adaptive resampling
  ([DOI](https://doi.org/10.3150/10-BEJ335));
- Kalman (1960), linear-Gaussian filtering
  ([DOI](https://doi.org/10.1115/1.3662552));
- Schweppe (1965), Gaussian innovations likelihood
  ([DOI](https://doi.org/10.1109/TIT.1965.1053737)); and
- Kim, Shephard, and Chib (1998), stochastic volatility
  ([DOI](https://doi.org/10.1111/1467-937X.00050)).

The temper interpretation also follows Roberts, Gelman, and Gilks (1997)
([DOI](https://doi.org/10.1214/aoap/1034625254)) and Beskos, Crisan, and Jasra
(2014) ([DOI](https://doi.org/10.1214/13-AAP951)). Evidence-scale checks for
standard particle filters follow Del Moral's *Feynman--Kac Formulae*
([DOI](https://doi.org/10.1007/978-1-4684-9393-1)) and Pitt et al. (2012)
([DOI](https://doi.org/10.1016/j.jeconom.2012.06.004)).

No implementation code was copied or translated from these papers. The
benchmark models and float64 oracles are independent implementations. The
deleted Dynamax adapter was original public-API glue against MIT-licensed
Dynamax 1.0.2; its exact boundary, result, and immutable
[MIT license](https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/LICENSE)
are preserved in the
[Dynamax validation report](2026-07-19-dynamax-integration-validation.md).
jax-mps source was inspected under
[Apache-2.0](https://github.com/tillahoffmann/jax-mps/blob/v0.10.10/LICENSE)
without porting code. No GPL or other red-line source was used.

## Reproduction commands

Run each profile sequentially on the measured source commit and an idle,
AC-powered M-series machine:

```bash
uv run python -m benchmarks.profiling.run \
  --profile baseline --platforms cpu mps \
  --output-dir /tmp/smcx-profiling-baseline-20260719-6a3f724-01010
uv run python -m benchmarks.profiling.run \
  --profile filter-regimes --platforms cpu mps \
  --output-dir /tmp/smcx-profiling-filter-regimes-20260719-2452bfa-01010
uv run python -m benchmarks.profiling.run \
  --profile scaling --platforms cpu mps \
  --output-dir /tmp/smcx-profiling-scaling-20260719-2452bfa-01010
uv run python -m benchmarks.profiling.run \
  --profile representation --platforms cpu mps \
  --output-dir /tmp/smcx-profiling-representation-20260719-651666b-01010
```

The later matched baseline exposed one more finite-replication rejection at
`R=64`. The complete fixed prefix passed at `R=128`, `R=256`, and `R=512`, so
the protocol prospectively raised every resampler cell to `R=128`. See
[`2026-07-19-matched-optimization-profile.md`](2026-07-19-matched-optimization-profile.md).
A present-day rerun therefore has a larger validation-call count than the
immutable baseline and scaling manifests summarized above.
