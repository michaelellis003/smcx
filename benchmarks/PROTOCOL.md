# Kill-test protocol (pre-registered)

*Committed before any benchmark code exists, so the success criterion
cannot drift toward whatever the numbers turn out to be. Changes to
this protocol before the verdict require a dated amendment section;
changes after the verdict require a new protocol file.*

## Question

Does MLX on Apple-silicon GPU beat the same algorithms on JAX-CPU —
the realistic alternative for a Mac-bound Bayesian — by enough to
justify an MLX-native SMC library?

## Workloads

All correctness-gated against the numpy-f64 Kalman oracle where
linear-Gaussian; seeded data generated once and committed to
`benchmarks/data/` (gitignored raw arrays, committed generation
script + hashes).

1. **LGSSM-1D**: scalar AR(1) + Gaussian emission, T=100. The
   correctness anchor; resampling/memory-bound.
2. **SV-1**: univariate stochastic volatility (nonlinear,
   non-Gaussian), T=500. Exercises per-particle density cost, where
   XLA CPU fusion is strong — the honest adversarial workload.
3. **TRACK-4**: d=4 linear-Gaussian tracking, diagonal emission
   covariance, T=200; plus one full-covariance variant, **report-only
   (not in the verdict)** — it exercises the full-covariance
   multivariate density path (precomputed `L⁻¹` matmul per design §7,
   no per-step CPU linalg).

Grid: N ∈ {10⁴, 10⁵, 10⁶} × the three workloads. The N=10⁴ row is
reported first — that is the regime typical filtering users occupy
and where dispatch overhead bites hardest.

## Success criterion (pre-registered)

A workload **counts** if MLX-GPU achieves ≥3× median wall-clock over
the JAX-CPU baseline at **both** N=10⁵ and N=10⁶, with both sides
passing that workload's correctness gate (a workload without a
passing gate on both sides can never count).

The thesis **holds** if at least two of the three workloads count.
It holds **weakly** (proceed, but reframe README claims) if exactly
one counts, or if any outcome matches none of the mapped cases —
every unmapped outcome is recorded as weak, never upgraded. It
**fails** (thesis dies, per README) if JAX-CPU is within 1.5×
everywhere, or MLX-GPU < 1.2× MLX-CPU everywhere after the profiling
check below.

## Correctness gates (run before any timing, at every gated (workload, N) cell, both libraries)

- **Kalman-oracle gate** (LGSSM-1D, TRACK-4):
  −(k·SD/√R + SD²/2) ≤ mean(log Ẑ) − log Z_Kalman ≤ k·SD/√R
  over R ≥ 20 independent keys (actual R recorded). One-sided Jensen
  budget: the SD²/2 downward allowance reflects E[log Ẑ] ≈
  log Z − Var/2; an *upward* deviation of that size is evidence of a
  bug and is not excused. **k = 3, fixed here** (normal approximation
  with SD estimated from the same R replicates; the harness derives
  the formula in comments but may not choose k). SD is computed over
  the R runs by the harness loop (`replicated_log_ml` where
  available — it is a v0.2 deliverable, so the harness carries its
  own loop for the kill test).
- **Cross-library gate** (SV-1, which has no Kalman oracle):
  |mean(log Ẑ_smcx) − mean(log Ẑ_smcjax)| ≤ 3·√(SD_x²/R + SD_j²/R)
  at matched N and algorithm over the same R ≥ 20 keys per side
  (Jensen biases cancel to first order at matched algorithm and N),
  plus structural invariants on both sides (increments sum to total;
  ESS ∈ [1, N]).

A fast wrong answer is a failure, not a win.

## Fairness rules

- JAX baseline pinned: x64 **disabled** (smcjax contains f64 literals
  that activate under x64 — record the smcjax commit), the entire
  filter jitted as one program, XLA CPU thread count recorded.
- MLX eval cadence is a harness parameter: per-step `mx.eval`,
  `mx.async_eval` + lagged blocking eval (lag k ∈ {2, 4, 8}), and
  pure `mx.async_eval` (reported with its peak memory, which
  research shows inflates ~3× at 10⁶). Report the best cadence per
  cell and the sweep itself.
- Warm-up: one full run per configuration before timing
  (`mx.compile` trace / XLA jit trace).
- Fencing: `mx.eval` + `mx.synchronize` / `block_until_ready` before
  stopping timers.
- ≥5 repeats: report median, min, and IQR — never median alone.
- MLX-GPU vs MLX-CPU also recorded (separates "MLX is fast" from
  "the GPU is doing the work").
- Peak memory both sides (`mx.get_peak_memory`; RSS for JAX).

## Results file header (mandatory)

Machine (chip, cores, RAM), macOS version, power source, thermal
state noted if throttling suspected (a multi-second N=10⁶ run on a
laptop can throttle), mlx/jax/jaxlib/smcjax versions + commits,
Python version, date. Results are dated markdown in
`benchmarks/results/`, committed.

## Interpretation guards

- If MLX-GPU ≈ MLX-CPU: profile (`mx.metal.start_capture`) before
  concluding — dispatch-bound ≠ thesis-dead.
- If JAX-CPU wins at N=10⁴ only: expected (dispatch overhead); the
  thesis was always about large N. Say so, don't bury it.
- The audit's 19× resample-kernel figure is a comparison against
  MLX's own CPU backend, not against XLA-CPU — never quote it as
  evidence for the thesis.

## Amendments

- 2026-07-14 (panel round 2, pre-code): pinned k=3; defined the SV-1
  cross-library gate; made the Jensen budget one-sided; quantified
  the verdict mapping (both-N rule, full-covariance report-only,
  ≈ defined as <1.2×, unmapped outcomes → weak).
- 2026-07-14 (performance research, pre-code): replaced the
  eval-every-k cadence arms (measured strictly dominated —
  `docs/research/mlx-performance.md`) with async+lag-k ∈ {2, 4, 8};
  pure-async arm retained but must report peak memory. Also: burn
  one throwaway compile before warm-up (first-process Metal JIT
  ≈ 68 ms) and capture one Xcode GPU trace per cell before accepting
  any "GPU ≈ CPU" verdict.
- 2026-07-15 (post-diagnosis): future runs execute a **fresh
  process per (workload, N) cell** — the in-process sweep was shown
  to accumulate state that inflated one cell by 1.8x
  (docs/research/perf-analysis.md).
- 2026-07-15 (pre-re-run): the 2026-07-14 run was contaminated
  (concurrent CPU workload; verdict provisional) — this re-run on an
  idle machine supersedes it. Added a `store_history=False`
  (ADR-0011) MLX arm, **report-only**: the verdict's primary
  comparison remains full-history on both sides (smcjax always
  materializes history), the lean arm documents the smcx-native
  path. Same datasets (hashes unchanged).
