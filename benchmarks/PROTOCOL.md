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
   covariance, T=200; plus one full-covariance variant to expose the
   CPU-pinned `mx.linalg` cost.

Grid: N ∈ {10⁴, 10⁵, 10⁶} × the three workloads. The N=10⁴ row is
reported first — that is the regime typical filtering users occupy
and where dispatch overhead bites hardest.

## Success criterion (pre-registered)

The thesis **holds** if MLX-GPU achieves ≥3× median wall-clock over
the JAX-CPU baseline at N ≥ 10⁵ on at least two of the three
workloads, with both sides passing the correctness gate. It holds
**weakly** (proceed, but reframe README claims) if ≥3× only at 10⁶ or
only on LGSSM-1D. It **fails** (thesis dies, per README) if JAX-CPU
is within 1.5× everywhere, or MLX-GPU ≈ MLX-CPU everywhere after the
profiling check below.

## Correctness gate (runs before any timing)

|mean(log Ẑ) − log Z_Kalman| ≤ k·SD/√R + SD²/2 over R ≥ 20 keys
(SD from `replicated_log_ml`; the SD²/2 term budgets the Jensen bias
of E[log Ẑ]; k and derivation in the harness source). Applied to
both libraries. A fast wrong answer is a failure, not a win.

## Fairness rules

- JAX baseline pinned: x64 **disabled** (smcjax contains f64 literals
  that activate under x64 — record the smcjax commit), the entire
  filter jitted as one program, XLA CPU thread count recorded.
- MLX eval cadence is a harness parameter: per-step `mx.eval`,
  eval-every-k (k ∈ {1, 5, 25}), and `mx.async_eval` with lagged
  degeneracy check. Report the best cadence per cell and the cadence
  sweep itself (it answers the design's open cadence question).
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
