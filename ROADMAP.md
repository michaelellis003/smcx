# Roadmap

*Last updated: 2026-07-16. Directional, not a promise; solo-maintained.
Themes here, tracking in GitHub issues/milestones. Design rationale:
`docs/design/v0-design.md`; decisions: `docs/adr/`.*

## Now (v0.1) — foundations and the kill test

The theme: prove the thesis before building breadth.

- [x] ~~Ratify ADR-0002 and ADR-0008~~ — both accepted 2026-07-14;
      v0.1 coding is unblocked
- [x] ~~`weights` module: log_normalize, normalize, ess, log_ess~~
      (2026-07-14, with ADR-0007 typing infra: jaxtyping + beartype
      hook + vendored mlx stubs)
- [x] ~~`resampling` module: systematic (counting kernel), stratified,
      multinomial, residual~~ (2026-07-14; shipped kernels measured
      2.2/1.0/0.9 ms at N=10⁶ GPU)
- [x] ~~Full-step microbenchmark confirming the async+lag-k cadence
      and kernel defaults on the real step~~ (subsumed by the
      kill-test runs, which time the real compiled step at
      10⁴–10⁶ particles across three workloads; cadence findings
      in docs/research/perf-analysis.md)
- [x] ~~FK core: FKModel protocol + generic loop~~ (2026-07-14)
- [x] ~~`bootstrap_filter` + `simulate` + containers + Protocols~~
      (2026-07-14; T=100 filter at N=10⁶ runs 3.25 ms/step GPU)
- [x] ~~Kalman oracle + LGSSM correctness suite~~ (2026-07-14;
      MC-calibrated gates incl. missing-obs and inputs-channel)
- [x] ~~`__all__` lock test~~ (2026-07-14)
- [x] ~~**Kill test**~~ (2026-07-14: **holds weakly**, 1/3 workloads
      ≥3×; all 15 gates pass; MLX-GPU faster in all 12 cells;
      `benchmarks/results/2026-07-14-kill-test.md`)
- [x] ~~`store_history=False` option~~ (2026-07-14, ADR-0011:
      O(T·N)→O(N) memory, bit-identical log-Z; kill-test re-run
      should sweep it on the SV/TRACK cells)
- [x] ~~File MLX issue: `categorical(num_samples=)` O(N·M) memory~~
      (2026-07-15: filed as ml-explore/mlx#3847 with a minimal repro
      and the inverse-CDF workaround; evidence in
      `docs/research/mlx-audit.md`)
- [x] ~~Open smcjax coordinated-change issues~~ — dissolved per
      ADR-0010 (smcjax frozen at e93d527 as benchmark baseline;
      smcx is the successor, single-repo governance)
- [x] ~~**Clean kill-test re-run**~~ (2026-07-15: **holds weakly**
      confirmed clean — 1/3 count, all 15 gates pass, MLX-GPU leads
      11/12 cells; store_history arm cuts memory 8–38× at unchanged
      speed; supersedes the contaminated 2026-07-14 run)
- [x] ~~Typing setup per ADR-0007 (jaxtyping + beartype hook + vendored
      `typings/mlx/core.pyi`)~~ (2026-07-14, landed with the weights
      module)
- [x] ~~SPEC 8 release-workflow hardening (Trusted Publishing,
      environment gate, attestations)~~ (2026-07-15: PyPI pending
      trusted publisher bound to release.yml + `release`
      environment with required reviewer; attestations on;
      public repo carries the infra skeleton with branch/tag
      rulesets and contributor gating — code itself unreleased,
      awaiting the explicit go)

## Next (v0.2) — the filter family and the flagship sampler

- [x] ~~`guided_filter`~~ (2026-07-15; ADR-0008 item 2, optimal-
      proposal variance reduction verified vs bootstrap)
- [x] ~~`auxiliary_filter` via twisted potentials~~ (2026-07-14;
      ADR-0002 mechanics, bit-exact skip-branch equivalence tested)
- [x] ~~Tempered SMC sampler with adaptive ESS-bisection schedule and
      particle-tuned RWM moves~~ (2026-07-15; exact conjugate
      evidence gate, compiled branchless sweeps)
- [x] ~~`distributions` module (~8 families) incl. Lanczos lgamma~~
      (2026-07-14, ADR-0012: 9 families, guarded chol_factor,
      MT gamma sampler)
- [x] ~~Diagnostics port from smcjax~~ (2026-07-15; 15 functions,
      tail_ess reimplemented quantile-based, Pareto-k semantics
      corrected, adaptive diagnose threshold)
- [x] ~~`liu_west_filter` (labeled approximate)~~ (2026-07-15; port
      complete — all 32 smcjax names present, full `__all__` lock
      direction active)
- [x] ~~GPU release gate~~ (2026-07-15: `gpu-smoke` CI job probes
      Metal and runs the full suite on GPU when available (warns
      loudly when not); release job gated on the `release`
      environment (created on the repo — required-reviewer rule
      activates when the repo goes public; Free-plan limitation on
      private repos); PR template gains the local-GPU checkbox.
      Metal-on-runner feasibility confirmed at first real CI run)
- [x] ~~Value-branch conditional resample~~ (2026-07-15;
      bit-identical, fairness-restoring)
- [x] ~~Batched-model fast path~~ (2026-07-15, ADR-0013)
- [x] ~~Kill-test re-run fresh-process-per-cell~~ (2026-07-15:
      **VERDICT HOLDS — 3/3 workloads, 3.4–7.8× at 10⁵–10⁶**, all
      gates pass; benchmarks/results/2026-07-15-kill-test-optimized.md)

## Later — ideas, ordered by thesis-fit

- Metropolis resampler (ratio-only, f32-safe beyond N≈10⁶; bias
  documented as PMMH-incompatible)
- Waste-free SMC (removes the MCMC-steps knob)
- [x] ~~SMC² (the (N_θ × N_x) nesting is the best unified-memory fit
  in the literature)~~ — **v1 implemented** (2026-07-15, ADR-0014):
  `smc2()` + `SMC2Posterior`, resident batched inner filters,
  data-tempering outer schedule, PMMH rejuvenation; 14 tests incl.
  exact Kalman-grid recovery, unbiased-evidence, bootstrap reduction;
  two numerics-review passes clean; `inner_step` compiled (the
  rejuvenation bottleneck). **Second kill test DONE** (2026-07-15):
  MLX-GPU vs MLX-CPU **~32–34×** at 0.26M–1.05M inner particles,
  correctness-gated; `benchmarks/results/2026-07-15-smc2-device-
  benchmark.md` + PROTOCOL amendment. Chopin's `particles` added as
  an external-authority baseline (independent implementation agrees
  on log Z; smcx-GPU ~60–119× faster, median-of-5, config caveat
  noted). Deferred
  (ADR-0014): adaptive N_x, the exchange step, guided inner engines.
- Differentiable resampling: Ścibior-Wood stop-gradient first; OT/DET
  as opt-in
- FFBSi smoothing (dense batched backward weights); fixed-lag comes
  cheap earlier
- `to_arviz()` InferenceData export; independent runs as chains
- CESS tempering; MALA/HMC moves once model grads are wired
- Island-mode resampling for N beyond a single population
- Benchmark suite tracking MLX releases (re-run audit + kill test)
- **Performance leadership (standing goal, 2026-07-16)**: keep native
  smcx the fastest SMC engine on Apple silicon at every N, under
  either frontend. Loop shell v2 (ADR-0016) made native fastest below
  10⁶; the measured 10⁶ residual vs a tuned jax-mps is substrate, not
  shell: the `mx.fast.metal_kernel` Python wrapper (~0.5 ms/call vs
  0.36 ms for the same kernel invoked from C++ — upstream-MLX issue
  candidate) and `mx.random.normal` vs a fused Philox (~2× at 10⁶;
  supersedes the perf-analysis "chasing RNG throughput" non-item,
  which predates this comparison). Evidence:
  `docs/research/2026-07-16-jax-mps-internals.md`.
- **jax-mps tracking (standing goal, 2026-07-16)**: re-run
  `benchmarks/native_vs_jax_mps` when jax-mps ships relevant fixes
  (filed: #215 T² scan history, #216 threefry registration; queued:
  the searchsorted kernel via their #203 mechanism decision) and
  contribute optimizations upstream when the gap is theirs to close.
  The comparison keeps both stacks honest; smcx's positioning does
  not depend on jax-mps staying slow.

*Sequencing note: the literature review's "v0 must-have" menu is
aspirational input; this file is the authoritative sequencing.*

## Non-goals

Standing scope guard — do not implement these; link here when closing
requests:

- General PPL, effect handlers, NUTS, or NumPyro feature parity
- PMMH / particle Gibbs as built-ins (we guarantee an unbiased
  evidence estimate Ẑ — E[exp(marginal_loglik)] = Z, the PMMH
  contract; log Ẑ itself is downward-biased — so smcx can be an
  inner engine)
- float64 on GPU, or emulating it
- OT/DET resampling as a default (O(N²), biased likelihood)
- SSP / Hilbert-ordered resampling (sequential-scan-shaped)
- Windows support; non-Apple hardware as a design target
- Distribution-framework ambitions (objects, bijectors, constraints)
