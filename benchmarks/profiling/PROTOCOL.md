# Current-JAX all-algorithm profiling protocol

*Pre-registered 2026-07-19 before implementation or measurement. Amendments
must be appended with a date and rationale; earlier text remains intact.*

## Question

How do the shipped smcx algorithms spend compile time, first-call time,
steady execution time, and memory across model and control-flow regimes, and
which measured costs are attributable to the inference shell, model callbacks,
resampling, history storage, structured state, or adaptive host control?

This campaign finds optimization candidates. It does not use timing as a
correctness signal and does not optimize production code in the same baseline
slice.

## Algorithms

The primary matrix exercises current public production entry points:

1. `bootstrap_filter`;
2. `auxiliary_filter`;
3. `guided_filter`;
4. `liu_west_filter`;
5. `temper`;
6. `smc2`, separately with rejuvenation disabled and forced;
7. `systematic`, `stratified`, `multinomial`, and `residual` as isolated
   resampling kernels.

`simulate` generates or checks model data but is not a primary timed inference
arm. Diagnostics are deferred to a separately pre-registered campaign because
their scale axes and output contracts differ from inference.

## Model workloads

### L1 — scalar linear Gaussian state-space model

```text
x_0 ~ Normal(m0, p0)
x_t | x_(t-1) ~ Normal(a * x_(t-1) + b * u_t, q)
y_t | x_t ~ Normal(x_t, r)
```

Fixed defaults are `a=0.9`, `b=0.25`, `m0=0`, `p0=1`, `q=0.2`, `T=100`,
and deterministic sinusoidal inputs. The observation regimes are:

- diffuse: `r=2.0`;
- calibrated: `r=0.3`;
- sharp: `r=0.03`, with one committed central outlier.

Bootstrap uses the transition prior. Auxiliary uses the exact predictive
potential `Normal(y_t; a*x_(t-1)+b*u_t, q+r)`. Guided uses the exact conditional
Gaussian proposal and evaluates the full `log g + log f - log q` correction.
An independent NumPy float64 Kalman recurrence supplies exact filtering moments
and log evidence.

### L2 — multivariate controlled tracking

A constant-velocity linear Gaussian model has `state_dim=4`,
`emission_dim=2`, `input_dim=2`, and `T=200`, with both diagonal and correlated
process/observation covariance regimes. It exercises dense matvecs, covariance
factorization in model callbacks, inputs, and larger per-particle state. A
NumPy float64 multivariate Kalman recurrence is the oracle.

An exactly equivalent representation arm splits position and velocity into a
two-leaf semantic PyTree. This arm is a representation comparison, not a claim
that PyTrees accelerate inference.

### N1 — stochastic volatility

```text
h_0 ~ Normal(mu, sigma / sqrt(1-rho^2))
h_t | h_(t-1) ~ Normal(mu + rho*(h_(t-1)-mu), sigma)
y_t | h_t ~ Normal(0, exp(h_t / 2))
```

Defaults are `mu=-0.5`, `rho=0.97`, `sigma=0.2`, and `T=500`. The committed
data include ordinary returns and a fixed outlier. This nonlinear,
non-Gaussian-in-state likelihood covers long scans and weight stress. It is a
bootstrap workload initially; auxiliary or guided variants require a separately
justified proposal amendment.

### P1 — unknown-AR state-space model

L1 is reused with unknown scalar `a` under a bounded prior represented as one
dense parameter coordinate. It drives Liu--West and SMC2. Liu--West uses the
predictive observation potential at the shrunk parameter. SMC2 runs:

- forward: `ess_threshold=0`;
- forced rejuvenation: `ess_threshold=1.1`, one PMMH step.

A dense float64 grid over `a` combines the Kalman likelihood and prior to give
a small-case parameter-evidence and posterior-mean oracle. The timed grid is
never part of the worker execution.

### G1 — conjugate Gaussian static target

```text
theta ~ Normal(0, I)
y | theta ~ Normal(theta, sigma_y^2 I)
```

The observed vector is fixed and deterministic. Dimensions are `4`, `32`, and
`128`; `sigma_y=0.7`. Closed-form marginal likelihood and posterior moments
anchor adaptive tempered SMC. A Bayesian logistic-regression workload may be
added only by dated amendment after G1 establishes the host/model split.

### D1 — optional Dynamax adapter

Dynamax 1.0.2 defines the exact same L1 parameters and emissions. Its public
`initial_distribution`, `transition_distribution`, and
`emission_distribution` methods are closed over to produce smcx bootstrap
callbacks. The local and adapter arms use identical keys, data, and inference
settings. This arm measures integration overhead; it is excluded when the
notebook dependency group is unavailable and never becomes a runtime or test
dependency.

The undocumented `dynamax.slds` particle filters are not validators or timed
production comparators.

## Profiles

All orderings use seed `20260719`.

### `smoke`

- One fresh-process block, one warm-up, one timed execution.
- CPU and safe MPS when requested.
- Small parameters chosen to finish quickly:
  - L1 standard filters: `N=128`, `T=20`, calibrated, no history;
  - N1 bootstrap: `N=128`, `T=30`, no history;
  - P1 Liu--West: `N=128`, `T=20`, no history;
  - P1 SMC2 forward/forced: `N_theta=8`, `N_x=16`, `T=8`;
  - G1 temper: `N=128`, `d=4`, one RWM step;
  - resamplers: `N=1,024`, moderately uneven weights.
- Smoke proves only that planning, isolation, backend selection, measurement,
  and correctness gates execute. It supports no timing conclusion.

### `baseline`

- Five fresh-process blocks, one warm-up, seven fenced steady executions.
- L1 standard filters: `N=10,000`, `T=100`, calibrated, no history,
  adaptive threshold `0.5`.
- N1 bootstrap: `N=10,000`, `T=500`, no history.
- P1 Liu--West: `N=10,000`, `T=100`, no history.
- P1 SMC2 forward: `N_theta=128`, `N_x=256`, `T=40`.
- P1 SMC2 forced rejuvenation: `N_theta=32`, `N_x=64`, `T=20`.
- G1 temper: `N=10,000`, `d=32`, five RWM steps.
- Resamplers: `N=100,000`, moderately uneven weights.

### `filter-regimes`

At `N=10,000`, compare bootstrap, auxiliary, and guided L1 across:

- diffuse, calibrated, and sharp observations;
- thresholds `0`, `0.5`, and `1.1`;
- history disabled and enabled.

This is 54 mathematical cells per backend before process blocks. It runs only
after baseline and may be narrowed only by a dated amendment based on a failed
correctness or resource feasibility gate, never based on a preferred result.

### `scaling`

- Standard filters and N1: `N in {1,000, 10,000, 100,000}`.
- Liu--West: `N in {1,000, 10,000, 100,000}` and a later independent
  parameter-dimension axis `{1, 4, 16, 64}` using mathematically replicated
  coordinates.
- Tempering: `N in {1,000, 10,000}` and `d in {4, 32, 128}`.
- SMC2 forward: `(N_theta,N_x,T)` in
  `{(32,64,20), (128,256,40), (512,512,100)}`; the largest cell is CPU/MPS
  resource-permitting and failure is retained rather than downscaled silently.
- Resamplers: `N in {10,000, 100,000, 1,000,000}` under uniform,
  moderately uneven, one-dominant, and zero-tail normalized weights.

### `representation` and `integration`

- Representation: L2 dense versus two-leaf state, `N=10,000`, `T=200`,
  history off/on; and P1 Liu--West, `N=10,000`, `T=100`, parameter dimension
  one, history off/on, with threshold `1.1` for equal discrete work.
- Integration: local L1 versus D1 adapter, `N=10,000`, `T=100`, bootstrap,
  no history. Lowered IR census is captured for both.

## Correctness gates

Timing is retained only with a visible correctness result; a failed cell is
never silently omitted.

Every successful cell must verify:

- requested and actual backend agree;
- full output PyTree is ready and all inspected leaves are finite;
- shapes match the registered workload;
- final log weights normalize within a float32-honest tolerance;
- `sum(log_evidence_increments) == marginal_loglik` when increments exist.

L1 uses independent replicated estimates against the float64 Kalman log
evidence. The preregistered gate is the existing MC-error-honest interval:
with replicate mean error `e`, sample standard deviation `s`, and replicate
count `R`, require

```text
-(3*s/sqrt(R) + 0.5*s^2) <= e <= 3*s/sqrt(R).
```

The asymmetric lower allowance reflects log-estimate bias. Baseline uses 20
replicates in block zero; smoke uses structural/invariant gates because it is
not inferential evidence.

G1 compares marginal likelihood and posterior means to closed form with a
tolerance derived as five estimator standard errors across independent
correctness replicates. P1 uses the grid oracle in untimed correctness runs.
Unit tests prove the exact L1 proposal and Kalman/grid calculations independently
of the benchmark workers.

## Timing and isolation

- One fresh process executes one workload/backend/block cell.
- The manifest and exact cell order are written before any worker starts.
- Completed raw cells are never overwritten; an interrupted campaign resumes.
- `JAX_ENABLE_COMPILATION_CACHE=false` for workers.
- `JAX_PLATFORMS` selects exactly `cpu` or `mps`; safe MPS clears
  `JAX_MPS_ASYNC_DISPATCH`.
- A trivial compiled operation removes one-time backend/plugin startup before
  measuring the workload lifecycle.
- `jax.block_until_ready` fences the complete posterior/result PyTree.
- Scan-based outer-jittable algorithms report Python lowering, backend
  compilation, first fenced execution, and steady fenced execution separately.
- `temper` and `smc2` are host-controlled public APIs. Their lowering and
  backend-compilation fields are null with reason `host_controlled`; first and
  repeated public-API calls are timed end to end. No synthetic outer `jit` is
  used to make their lifecycle appear comparable.
- Primary steady estimate: median of five fresh-process medians. Raw samples,
  minimum, quartiles, IQR, and MAD remain available.
- Timeouts and out-of-memory exits are data and remain failed raw cells.

JAX dispatch is asynchronous, so an unfenced measurement is invalid. Device
transfer is excluded by placing keys, observations, and static argument arrays
on the selected device before timing.

## Memory and adaptive work

Record process high-water RSS. Also record compiled executable memory analysis
or backend allocator counters when the backend implements them; incompatible
memory scopes are never compared.

Record algorithm work needed to interpret adaptive timings:

- sequence length, particle counts, dimensions, history mode, and threshold;
- mean/minimum ESS for filters;
- number of temperature stages and requested RWM sweeps for tempering;
- forward/forced regime, expected rejuvenation opportunities, and PMMH steps
  for SMC2;
- state leaf count and total scalar state dimension;
- resampler and weight regime.

## Phase attribution and profiler traces

Macro timing precedes tracing. After baseline, representative cells may add
untimed counterfactuals for callbacks only, resampling only, history, and state
representation. These use the same inputs and output fences.

Capture one StableHLO operation census for each outer-jittable baseline arm on
CPU, plus local-versus-Dynamax integration. Capture a JAX profiler trace only
for representative bottlenecks selected by baseline timing. Trace files are
local raw artifacts; dated findings record the command and interpretation.

## Reporting rules

- GPU numbers are local-only and never CI assertions.
- Every result report records date, hardware, OS, power/thermal state, Python,
  JAX, jaxlib, jax-mps, NumPy, smcx version/commit/source digest, exact profile,
  cell failures, warm-up, repeats, blocks, and dispatch mode.
- Smoke numbers are labeled non-inferential and do not support rankings.
- CPU/MPS comparisons use ratios only for matched mathematical cells.
- Adaptive algorithms are not compared without their work counters.
- History on/off ratios require equal non-history parameters and exact
  per-block adaptive-work counters; unmatched pairs are reported as exclusions.
- Correctness and accuracy-per-time accompany runtime where an oracle exists.
- No optimization is credited without a new matched before/after campaign and
  unchanged correctness gate.

## Amendments

### 2026-07-19 — pre-measurement workload clarifications

These clarifications were recorded before any inferential profile was run.

- The P1 prior is the implemented `Normal(0.9, 0.15^2)` prior, not a bounded
  distribution. The float64 oracle truncates numerical quadrature at eight
  prior standard deviations on each side; the omitted two-sided standard
  Normal tail mass is approximately `1.24e-15`. The earlier word "bounded"
  was imprecise.
- Resampling weights are constructed once outside timing in float32. Uniform
  is constant; moderately uneven is proportional to
  `exp(-linspace(0, 5, N))`; one-dominant assigns mass `0.9` to index zero and
  divides `0.1` equally over the remainder; zero-tail applies the moderately
  uneven construction to the first `N - floor(N/4)` entries and sets the
  final quarter to zero. Every vector is normalized after construction.
- L2 `correlated` uses integrated-white-acceleration process covariance and
  observation correlation `0.3`. L2 `diagonal` preserves the same marginal
  variances while setting all process and observation cross-covariances to
  zero. Representation crosses both regimes with history off/on, for eight
  L2 mathematical cells per backend. A later amendment adds two Liu--West
  history cells.
- L2 replicated Kalman evidence gates run on history-off arms. History-on arms
  retain structural/evidence-identity gates and the independently tested
  dense/PyTree equivalence, avoiding retention of twenty full `N x T`
  histories solely for an untimed check.
- The preregistered log-evidence interval above is superseded before
  inferential measurement. Particle-filter evidence estimates are unbiased on
  the evidence scale, whereas the stated `-s^2/2` log correction requires an
  unverified lognormal approximation. L1, L2, and D1 therefore gate
  `exp(log_Z_hat - log_Z_exact)` against one at five estimated standard
  errors, plus the committed float32 floor. Log-evidence errors remain
  diagnostics only. This also aligns the gate with the repository-wide
  five-SE moment-test standard.
- Inferential cells declare one of three validation levels. `structural`
  proves registered shapes, float32 leaves, finite values, normalized weights,
  ESS bounds, and internal evidence identities. `statistical` additionally
  tests repeated draws against a known sampling target. `oracle_accuracy`
  additionally compares evidence and available first and raw second posterior
  moments with an independent analytic, Kalman, or quadrature oracle. The last
  is evidence about algorithmic accuracy, not by itself proof that an
  approximate method such as Liu--West implements its kernel correctly.
- Every oracle-backed mathematical variant in baseline, filter-regimes,
  scaling, integration, and history-off representation runs its independent
  replicated gate in block zero on each requested backend. Stochastic
  volatility remains explicitly structural-only. Isolated resamplers use a
  repeated sampling-moment gate so a shape-correct but distributionally wrong
  kernel cannot become eligible.
- Float64 oracles consume the exact float32-rounded model constants, inputs,
  and observations seen by the timed operation. Tempering, Liu--West, and SMC2
  compare both posterior means and raw second moments. This prevents a
  collapsed or over-dispersed approximation from passing on its mean alone.
- The structured L2 arm is an end-to-end representation comparison. Its
  callbacks flatten the semantic position/velocity leaves for shared linear
  algebra, so the measured difference includes that adapter work as well as
  smcx PyTree handling. It is not an isolated core-PyTree microbenchmark.
- Timing and replicated validation run in separate campaign phases. All
  fresh-process timing cells finish before any independent replicate workers
  start, preventing validation heat from contaminating the next timing cell.
  Structural checks remain attached to timing output; immutable validation
  sidecars are then merged into the final raw envelope. Power and thermal
  snapshots are captured immediately before and after the measured interval.
- Workers discard inherited `JAX_*`, `XLA_*`, and common numerical-library
  thread-count variables before setting the preregistered backend, disabled
  compilation cache, and disabled x64 mode. The resulting explicit runtime
  flags are recorded with each result and StableHLO census.
- The independent Liu--West parameter-dimension axis uses
  `d in {1, 4, 16, 64}`, `N=1,000`, `T=100`, shrinkage `0.95`, no history,
  and threshold `1.1`. For coordinates
  `theta_j ~ Normal(mu, d * sigma^2)`, the AR coefficient is
  `a = mean(theta)`, so `a ~ Normal(mu, sigma^2)` at every dimension and the
  scalar P1 evidence remains the exact oracle. The joint posterior covariance
  is `d*sigma^2*I + (v - sigma^2)*11'`, where `v` is the scalar grid-posterior
  variance. Replicated validation gates evidence, the mean and raw second
  moment of `a`, and the parameter spread orthogonal to `a`. Forced resampling
  makes the discrete work count exactly `T - 1 = 99`, permitting matched
  CPU/MPS comparison without production instrumentation. These four arms
  bring `scaling` to 76 mathematical cells per backend.

### 2026-07-19 — pre-measurement oracle and structural gate completion

An implementation audit found two places where the registered validation
level was stronger than the workload gate that had been wired. This amendment
was recorded before inferential measurement and does not change the timed
matrix.

- L1, history-off L2 dense/PyTree, and D1 `oracle_accuracy` validation gates
  replicated final weighted filtered-state means and coordinatewise raw second
  moments in addition to evidence. The comparison uses the same independent
  Kalman recurrences and the exact float32-rounded model constants, inputs, and
  observations consumed by the profiling operation. The L2 PyTree is
  explicitly flattened in position-then-velocity oracle order.
- G1 temper and P1 SMC2 structural gates verify every registered output
  field's full shape and float32 dtype. They also enforce ESS bounds and trace
  alignment; G1 additionally requires a finite, positive, strictly increasing
  temperature schedule bounded by one and ending at one. Acceptance traces
  must remain in `[0, 1]`. These checks complete the already registered
  `structural` prerequisite and do not add a new timing or accuracy criterion.

### 2026-07-19 — pre-measurement execution-integrity completion

A final adversarial audit tightened campaign provenance and resource visibility
before any inferential measurement. These requirements do not change a timed
mathematical operation.

- A report reconstructs the complete registered plan from the profile,
  requested platform order, order seed, and the manifest's frozen Dynamax
  package identity. A self-consistent digest of an omitted, reordered, empty,
  or otherwise invented cell list is rejected. The preflight estimate is
  likewise recomputed from the cells rather than trusted as free-form
  metadata.
- `order_seed=20260719` controls only balanced process order. Model data use
  the documented base seed plus fixed model offsets; timed inference uses
  `20260719`; independent validation uses `20260720`. The full seed-role map
  is frozen in every manifest.
- Workers remove inherited `PYTHONPATH` and `PYTHONHOME`, disable the user site,
  and attest the exact sanitized runtime flags, device kind and ID, backend,
  and dispatch mode. Validation sidecars preserve their own source, package,
  host, runtime, and device provenance; the final record requires the timing
  and validation devices to match.
- The supervisor rejects output beneath `src/smcx` or
  `benchmarks/profiling`, freezes source/package/host identity before launch,
  and checks it again before every missing timing or validation worker. An
  identity drift aborts the campaign instead of becoming an ordinary failed
  cell.
- Inferential timing is eligible only when the snapshots immediately before
  timing, immediately after timing, and after structural/work extraction all
  report AC power and no macOS thermal or performance warning. The third
  snapshot detects post-timing diagnostic heat that could contaminate the next
  process. Resampler work extraction therefore retains only registered static
  counters and does not perform an unnecessary `unique` pass over up to one
  million ancestors.
- StableHLO censuses and profiler traces require a non-smoke campaign, a
  canonical raw member, exact manifest/result hashes, and a complete,
  correctness-passing, timing-eligible aggregate. Standalone or renamed JSON
  records cannot authorize either tool.
- The committed preflight reports exact worker-process counts, timed public
  calls, validation-replicate calls, the configured sequential timeout bound,
  per-workload/backend breakdowns, and maxima of every explicit numeric axis.
  It deliberately does not label adaptive inner steps or FLOPs as exact. With
  both backends and Dynamax 1.0.2 available, the registered profiles are:

  | Profile | Timing workers | Validation workers | Timed calls | Validation calls |
  |---|---:|---:|---:|---:|
  | smoke | 24 | 0 | 48 | 0 |
  | baseline | 120 | 22 | 960 | 264 |
  | filter-regimes | 540 | 108 | 4,320 | 2,160 |
  | scaling | 760 | 146 | 6,080 | 1,488 |
  | representation | 80 | 8 | 640 | 160 |
  | integration | 20 | 4 | 160 | 80 |

  The full registered campaign therefore schedules 1,832 fresh processes and
  16,360 public-workload calls. Profiles run sequentially on the shared SoC;
  no two performance workers or campaigns run concurrently.

### 2026-07-19 — pre-measurement statistical-gate hardening

A mathematical audit found four validation premises that needed to be made
explicit before inferential measurement. These changes do not alter a timed
operation or the registered cell matrix.

- Standard-filter structural checks cover every summary array and ancestor
  array, including full shape, float32/int32 dtype, finite values, and ancestor
  range. Tempering additionally enforces its public equal-weight output
  contract before unweighted posterior moments are evaluated.
- G1's analytic oracle consumes the exact float32 observation variance closed
  over by the likelihood callback, rather than independently rounding and then
  squaring its standard deviation.
- Resampler sampling evidence combines sixteen contiguous index partitions
  with a fixed independent eight-way affine-hash partition. Registered joint
  invariants additionally require monotone output from systematic, stratified,
  and multinomial kernels, the one-query-per-stratum CDF-discrepancy bound for
  systematic and stratified kernels, and deterministic floor counts for the
  residual kernel. These finite projections materially test marginal and some
  scheme-specific behavior; they do not prove a complete joint law or, by
  themselves, distinguish every pair of resampling schemes.
- The one-time D1 integration arm blocks on Dynamax's exact
  `marginal_log_prob` before timing and compares it with the independent Kalman
  oracle under an f32-honest tolerance. Dynamax uses `u[t]` to predict
  `x[t + 1]`, so this check shifts controls by one step to express L1's
  `u[t]`-for-`x[t]` convention; the timed smcx callbacks retain the original
  controls.

### 2026-07-19 — pre-measurement equal-work and isolation amendment

The final dry-run audit found that two algorithms hide their actual adaptive
resampling decision from their returned posterior. It also found that the
newly forced Liu--West particle-scaling cell at `N=1,000`, `d=1` was identical
to the first parameter-dimension cell. Before inferential measurement:

- Auxiliary and Liu--West particle-scaling cells use threshold `1.1`, making
  the event count exactly `T - 1`. Their adaptive threshold `0.5` baseline
  remains measurable, but CPU/MPS ratios stay withheld because its internal
  lookahead decision is not observable without changing production code.
- The local and Dynamax integration arms initially used threshold `0`, making
  the event count exactly zero and isolating callback integration overhead
  from random resampling-path differences. The post-failure correction below
  supersedes that setting.
- A hidden resampling decision is reported as exactly zero for threshold
  `<= 0` and exactly `T - 1` for threshold `> 1`; it remains null otherwise.
- The shared `N=1,000`, `d=1`, threshold-`1.1` Liu--West cell supplies both
  the particle and dimension axes. It is scheduled once, leaving 75 unique
  scaling cells per backend rather than 76 duplicate-containing entries.
- A nonblocking host-wide advisory lock covers each complete timing campaign,
  CPU profiler trace, or StableHLO census. Concurrent processes fail before
  device work rather than contaminating the Apple shared-SoC measurement.

### 2026-07-19 — post-failure integration validation correction

The first integration campaign showed that never resampling for the 100-step
LGSSM makes the evidence gate uninformative. All 20 replicates from both the
local and Dynamax callback arms produced nearly identical log-evidence
estimates, but their mean likelihood ratio was approximately `5.33e-17`
against the exact Kalman likelihood. No replicate sampled the rare tail that
determines the expectation, so the sample standard error collapsed with the
estimate and the registered five-standard-error gate correctly failed.

The integration arms therefore use threshold `1.1`, forcing the same `T - 1`
resampling decisions in both arms. This retains exact equal work and exercises
the complete callback stack while restoring a statistically useful likelihood
estimator. The failed never-resample campaign remains non-eligible evidence;
its timings are not used.

The corrected exact counts with both backends and Dynamax 1.0.2 are:

| Profile | Timing workers | Validation workers | Timed calls | Validation calls |
|---|---:|---:|---:|---:|
| smoke | 24 | 0 | 48 | 0 |
| baseline | 120 | 22 | 960 | 264 |
| filter-regimes | 540 | 108 | 4,320 | 2,160 |
| scaling | 750 | 144 | 6,000 | 1,464 |
| representation | 80 | 8 | 640 | 160 |
| integration | 20 | 4 | 160 | 80 |

The full registered campaign therefore schedules 1,820 fresh processes and
16,256 public-workload calls.

### 2026-07-19 — pre-measurement JAX dispatch and profiler boundary

JAX uses asynchronous dispatch, including on CPU. CPU records are therefore
labelled `asynchronous`; validity comes from fencing the complete returned
PyTree on every timed call, not from claiming synchronous dispatch. This
follows the official
[JAX asynchronous-dispatch documentation](https://docs.jax.dev/en/latest/async_dispatch.html).

The JAX profiler is restricted to selected CPU cells. jax-mps 0.10.10's PJRT
profiler extension is explicitly unsupported and returns errors for profiler
operations in its
[immutable implementation](https://github.com/tillahoffmann/jax-mps/blob/v0.10.10/src/pjrt_plugin/pjrt_profiler.cc#L57-L79).
It must not be described as a Metal device trace. The plugin separately
implements a process-global bounded `.gputrace` capture through
`JAX_MPS_GPU_CAPTURE`, as shown in its
[Metal executable source](https://github.com/tillahoffmann/jax-mps/blob/v0.10.10/src/pjrt_plugin/mlx_executable.cc#L456-L559).
That capture changes dispatch behavior, starts at the first eligible execute,
and counts PJRT dispatches rather than public host-shell calls. It is therefore
deferred until a baseline identifies one outer-jitted MPS bottleneck for a
separately preregistered, non-timing diagnostic capture.

Executable memory analysis and device allocator statistics remain separate
columns. jax-mps does not implement PJRT compiled-executable memory statistics
in
[v0.10.10](https://github.com/tillahoffmann/jax-mps/blob/v0.10.10/src/pjrt_plugin/pjrt_executable.cc#L260-L263),
while its device statistics are process-global MLX unified-memory counters.
Neither scope is silently substituted for or directly compared with the
other.

### 2026-07-19 — jax-mps floor correction

The first baseline and representation campaigns used jax-mps 0.10.9. The
representation result exposed the release's operand-wide
`dynamic_update_slice` lowering: a 45.8 MiB dense history reached 1.16 GiB of
Metal allocator use, and the two-leaf equivalent reached 1.52 GiB. jax-mps
0.10.10 was then available with native
[`slice_update`](https://github.com/tillahoffmann/jax-mps/pull/219) and
[`dynamic_slice`](https://github.com/tillahoffmann/jax-mps/pull/220)
lowerings. ADR-0026 raises the supported floor.

All 0.10.9 Metal timings remain diagnostic evidence for identifying the
backend defect, not current performance evidence. The complete baseline,
representation, integration, filter-regime, and scaling profiles restart
under 0.10.10. CPU/Metal ratios never mix dependency identities.

### 2026-07-19 — Liu--West history coverage completion

The history audit found that representation covered bootstrap state history
but not Liu--West's separate state and parameter histories. Representation
therefore adds one P1 pair at `N=10,000`, `T=100`, parameter dimension one,
and history off/on. Threshold `1.1` forces exactly `T - 1 = 99` resampling
decisions in both arms, so the matched comparison changes only retained
history. History-off receives the registered twelve-replicate oracle gate;
history-on remains structural-only to avoid retaining twelve additional full
histories for an untimed check.

The current exact counts with both backends and Dynamax 1.0.2 are:

| Profile | Timing workers | Validation workers | Timed calls | Validation calls |
|---|---:|---:|---:|---:|
| smoke | 24 | 0 | 48 | 0 |
| baseline | 120 | 22 | 960 | 264 |
| filter-regimes | 540 | 108 | 4,320 | 2,160 |
| scaling | 750 | 144 | 6,000 | 1,464 |
| representation | 100 | 10 | 800 | 184 |
| integration | 20 | 4 | 160 | 80 |

The full registered campaign therefore schedules 1,842 fresh processes and
16,440 public-workload calls. This table supersedes earlier count tables; it
does not revise their historical preregistration snapshots.

### 2026-07-19 — pre-measurement claim-scope and source audit

The source audit narrows several statements made earlier in this document:

- Only initial input placement is excluded from timing. Transfers and host
  synchronizations performed internally by a public operation remain included,
  especially for host-controlled `temper` and `smc2`. This supersedes the
  earlier unqualified sentence that device transfer is excluded and follows
  the official
  [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).
- Kalman results are analytic linear-Gaussian oracles evaluated in NumPy
  float64, exact for the specified model only up to floating-point rounding.
  The P1 grid adds documented numerical quadrature error.
- The D1 adapter maps L1 parameters into Dynamax 1.0.2; Dynamax does not define
  L1. Its comparison measures the complete callback-implementation delta,
  including TFP distribution construction, sampling, and density evaluation,
  rather than pure glue overhead. Dynamax's `marginal_log_prob` comparison
  validates this model/input mapping, not the smcx bootstrap-filter core,
  which both arms still execute. The immutable Dynamax sources show that
  [`marginal_log_prob` delegates to `lgssm_filter`](https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/linear_gaussian_ssm/models.py#L224-L240)
  and that
  [`u[t]` updates `y[t]` and predicts `x[t+1]`](https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/linear_gaussian_ssm/inference.py#L489-L511).
- StableHLO operation counts are a syntactic compiler-IR census, not FLOPs or
  a device-cost model. JAX documents these analyses as version- and
  backend-dependent debugging aids in its
  [AOT guide](https://docs.jax.dev/en/latest/aot.html#debug-information-and-analyses-when-available).
- Five fresh-process medians support robust descriptive ratios, not a formal
  effect-size interval. Kalibera and Jones (2013) motivate repeated-process
  benchmarking and uncertainty analysis
  ([DOI](https://doi.org/10.1145/2464157.2464160)).
- Evidence-scale unbiasedness motivates the replicated ratio-to-one gate only
  for standard particle filters with valid importance weights and
  conditionally unbiased resampling. It is not asserted for Liu--West or
  particle-adaptive tempering; those evidence comparisons remain empirical
  accuracy checks. See Del Moral's *Feynman--Kac Formulae*
  ([DOI](https://doi.org/10.1007/978-1-4684-9393-1)) and Pitt et al. (2012)
  for the auxiliary-filter likelihood result
  ([DOI](https://doi.org/10.1016/j.jeconom.2012.06.004)).
- Sorted multinomial ancestors are an smcx implementation contract resulting
  from its ordered-uniform construction, not a universal property of every
  multinomial-resampling implementation.

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
- Kong, Liu, and Wong (1994), the normalized-weight ESS heuristic
  ([DOI](https://doi.org/10.1080/01621459.1994.10476469));
- Del Moral, Doucet, and Jasra (2012), adaptive resampling
  ([DOI](https://doi.org/10.3150/10-BEJ335));
- Kalman (1960), linear-Gaussian filtering
  ([DOI](https://doi.org/10.1115/1.3662552));
- Schweppe (1965), Gaussian innovations likelihood
  ([DOI](https://doi.org/10.1109/TIT.1965.1053737)); and
- Kim, Shephard, and Chib (1998), the stochastic-volatility model
  ([DOI](https://doi.org/10.1111/1467-937X.00050)).

No implementation code was copied from these sources. The Dynamax adapter is
original public-API glue against MIT-licensed Dynamax 1.0.2, and the jax-mps
capability audit inspects Apache-2.0 source without porting it. No GPL or other
red-line source is required by this campaign.
