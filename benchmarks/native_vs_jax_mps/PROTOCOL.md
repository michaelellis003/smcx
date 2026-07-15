# Native MLX versus jax-mps protocol (pre-registered)

*Committed before benchmark implementation or measurement. Amendments retain
their original text and are dated. A full run made before an amendment cannot
be silently reinterpreted under the amended criterion.*

## Question

For representative Apple-silicon probabilistic-computing motifs, does a
direct MLX implementation have a persistent execution-time or memory advantage
over an equivalent JAX program executed by jax-mps, after compilation is
amortized and the compatibility backend receives its best documented dispatch
mode?

The experiment also measures cold-start cost, but a cold-start difference
alone cannot establish an execution advantage.

## Claims this protocol can support

The finite experiment can establish a versioned, workload-bounded result. It
can show that a measured gap persists after warm-up, scaling, correctness
gates, balanced process order, and documented jax-mps optimization modes. A
translation audit may attribute the gap to extra operations, synchronization,
materialization, or an unsupported optimization in the tested release.

The experiment cannot prove that jax-mps, or any compatibility backend, will
always be slower. jax-mps executes MLX operations and a cached one-to-one
lowering can theoretically match direct MLX. Direct MLX has a superset of the
available implementation choices because it need not preserve the JAX and
StableHLO contracts, but a particular direct implementation may still be
worse.

## Pinned environments

Primary comparison:

| Arm | Environment |
|---|---|
| Native | Python 3.13; mlx 0.32.0; MLX GPU |
| Compatibility, safe | Python 3.13; jax 0.10.2; jaxlib 0.10.2; jax-mps 0.10.9; `JAX_PLATFORMS=mps`; async dispatch unset |
| Compatibility, async | Same pins; `JAX_MPS_ASYNC_DISPATCH=1` |

Context-only arms are native MLX CPU and JAX CPU at the same package versions.
They do not determine the native-versus-compatibility verdict. jax-mps is
installed only through an isolated `uv run --no-project` environment. The
report records the resolved wheel hashes, jax-mps Git tag and commit, and the
fact that the wheel vendors its own MLX source snapshot; it does not describe
that snapshot as mlx 0.32.0 without evidence.

No persistent JAX compilation cache is enabled. Workers never import JAX and
MLX in the same process.

## Workloads

### Negative controls

These are intentionally favorable to one-to-one StableHLO-to-MLX lowering. If
jax-mps cannot approach native performance here, broad frontend or buffer costs
are a more plausible explanation than SMC-specific semantics.

1. **ELTWISE-REDUCE** — float32
   `sum(tanh(x) * sigmoid(x) + 0.1 * x**2)`, with
   N in {10^4, 10^6, 10^7}.
2. **MATMUL** — float32 square dense matrix multiplication followed by a
   scalar checksum, with D in {256, 1024, 2048}.

### Translation-sensitive motifs

3. **SCAN** — T=100 elementwise state transition and scalar reduction, with
   state width N in {10^4, 10^5, 10^6}. Both implementations use their native
   compiled scan/control-flow primitive and return only the final state and
   checksum.
4. **RANDOM** — standard-normal generation plus mean and variance, with
   N in {10^4, 10^6, 10^7}. Random streams need not match; distributional
   correctness does.
5. **GATHER-SCATTER** — deterministic indexed gather, update, and checksum,
   with N in {10^4, 10^5, 10^6}. Committed NumPy-generated indices are shared.
6. **SYSTEMATIC** — normalized positive weights, cumulative sum,
   searchsorted against a supplied scalar offset plus arange/N, and particle
   gather, with N in {10^4, 10^5, 10^6}. RNG is outside the timed function so
   the two programs receive identical weights, offset, and particles.

### End to end

7. **LGSSM-PF** — the committed T=100 scalar LGSSM data and a matched bootstrap
   particle filter at N in {10^4, 10^5, 10^6}. Float32, conditional resampling,
   threshold, full-history behavior, model vectorization, and output fencing
   must match. A no-history arm is report-only and runs only if identical
   semantics exist on both sides.

The supervisor refuses an unregistered workload name or size.

## Correctness before timing

A fast wrong result is a failed cell, never a speed result.

- Deterministic motifs compare their scalar checksum and, for smoke sizes,
  complete output arrays with a NumPy-f64 oracle. The fixed tolerance is
  `rtol=5e-5`, `atol=5e-6`; a workload may declare a tighter tolerance but not
  relax this one after measurement.
- RANDOM uses a committed seed per backend. For N samples, the sample mean
  must be within `5/sqrt(N)` of zero and sample variance within
  `5*sqrt(2/(N-1))` of one. The five-standard-error derivation is recorded in
  code. A failure at the committed seed is not re-rolled.
- LGSSM-PF uses R=20 independent keys per arm and the existing one-sided
  Kalman log-evidence gate from `benchmarks/PROTOCOL.md`. Correctness runs are
  separate from performance repetitions.
- NaN, infinity, backend fallback, or an unsupported operation is retained as
  a failed result. CPU fallback cannot count as MPS performance.

## Measurement design

Each workload/size/arm combination is one cell. Each cell has five independent
fresh-process blocks. Within each block:

1. construct deterministic inputs outside the timed region;
2. define and compile the function;
3. time the first call through an explicit device fence as `cold_s`;
4. run one untimed warm call and fence;
5. run seven timed calls, fencing every call;
6. record every duration, not only summaries;
7. record process RSS and backend-reported peak memory where available.

For every workload/size, process arms are run in a seeded balanced order: the
first block uses a deterministic shuffle and each subsequent block rotates it.
This prevents one backend from systematically receiving the coolest machine.
The full cell order is persisted before execution. The machine must be on AC
power with no intentional competing workload. Hardware, RAM, macOS, Python,
package versions, power status, and thermal warnings are recorded.

Primary steady-state estimate: median of the five per-process medians. The
report also includes all 35 timings, min, IQR, median absolute deviation, and a
95% percentile-bootstrap interval for the ratio of process medians using
10,000 resamples and committed seed 20260715. Cold time is summarized
separately.

## Comparison and verdict

For each workload/size, the adversarial compatibility result is the faster of
the safe and async jax-mps arms by primary median, provided that arm passes all
correctness gates. Choosing the faster arm favors jax-mps and is disclosed.

A workload has a **persistent native advantage** when, at both of its two
largest sizes:

- both sides pass correctness;
- the lower bound of the 95% bootstrap interval for
  `jax-mps time / native MLX time` is at least 1.5; and
- native MLX does not use more than 1.25 times the best available jax-mps peak
  memory, unless the report explicitly classifies the result as a
  speed-for-memory trade.

The **native SMC ecosystem case is supported** when LGSSM-PF has a persistent
native advantage and at least two of SCAN, RANDOM, GATHER-SCATTER, and
SYSTEMATIC do as well. It is **strongly supported** when the LGSSM-PF lower
confidence bound is at least 3.0 at N=10^5 and N=10^6.

It is **not supported** when the best correct jax-mps arm is within 1.2 times
native MLX at both large LGSSM-PF sizes, or wins either large size, unless a
pre-registered profiler check demonstrates CPU fallback or another invalid
comparison.

All other outcomes are mixed. Negative controls never count toward the native
ecosystem verdict; they calibrate whether the harness gives jax-mps a fair
opportunity to match direct MLX.

## Translation and trace audit

For one small and one large cell in every persistent-gap workload, retain:

- JAX StableHLO text and operation counts;
- jax-mps optimized IR using its documented dump facility;
- an optional Metal GPU trace for both arms;
- synchronization and dispatch-mode metadata.

Trace collection is outside timing. An architectural explanation is published
only when the retained artifacts show the mechanism; timing alone supports a
performance result, not a compiler-causality claim.

## Reproduction and reporting

The public command emits a manifest before running, raw JSON after every
process, a merged JSON document, and dated Markdown. Interrupted runs resume
without overwriting completed raw blocks. Reports list failed and missing
cells, and include the exact commands needed to rerun one cell.

CI validates schemas, deterministic statistics, balanced ordering, failure
retention, correctness gates, and tiny CPU worker execution. CI never asserts
timing and cannot attest Metal performance. Full results are committed only
after a local M-series run.

## Amendments

None.
