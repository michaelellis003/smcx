# Roadmap

*Last updated: 2026-07. Directional, not a promise; solo-maintained.
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
- [ ] Full-step microbenchmark confirming the async+lag-k cadence
      and kernel defaults on the real step (research numbers:
      docs/research/mlx-performance.md; scripts in
      benchmarks/exploratory/)
- [x] ~~FK core: FKModel protocol + generic loop~~ (2026-07-14)
- [x] ~~`bootstrap_filter` + `simulate` + containers + Protocols~~
      (2026-07-14; T=100 filter at N=10⁶ runs 3.25 ms/step GPU)
- [x] ~~Kalman oracle + LGSSM correctness suite~~ (2026-07-14;
      MC-calibrated gates incl. missing-obs and inputs-channel)
- [x] ~~`__all__` lock test~~ (2026-07-14)
- [x] ~~**Kill test**~~ (2026-07-14: **holds weakly**, 1/3 workloads
      ≥3×; all 15 gates pass; MLX-GPU faster in all 12 cells;
      `benchmarks/results/2026-07-14-kill-test.md`)
- [ ] `store_history=False` / trace option — **promoted from Later**:
      the kill test measured full-history materialization (12 GB at
      SV/10⁶) as the limiter on the two non-counting workloads;
      re-run SV/TRACK cells after it lands
- [ ] File MLX issue: `categorical(num_samples=)` O(N·M) memory
      (unreported upstream; evidence in `docs/research/mlx-audit.md`)
- [x] ~~Open smcjax coordinated-change issues~~ — dissolved per
      ADR-0010 (smcjax frozen at e93d527 as benchmark baseline;
      smcx is the successor, single-repo governance)
- [ ] **Clean kill-test re-run** on an idle machine (2026-07-14 run
      contaminated by a concurrent CPU workload — verdict flagged
      provisional) — after `store_history` lands
- [ ] Typing setup per ADR-0007 (jaxtyping + beartype hook + vendored
      `typings/mlx/core.pyi`)
- [ ] SPEC 8 release-workflow hardening (Trusted Publishing,
      environment gate, attestations)

## Next (v0.2) — the filter family and the flagship sampler

- [ ] `guided_filter` (new vs smcjax — coordinated backport candidate)
- [ ] `auxiliary_filter` via twisted potentials
- [ ] Tempered SMC sampler with adaptive ESS-bisection schedule and
      particle-tuned RWM moves
- [ ] `distributions` module (~8 families) incl. Lanczos lgamma
- [ ] Diagnostics port from smcjax (Pareto-k, tail-ESS, CRPS,
      diagnose, …)
- [ ] `liu_west_filter` (labeled approximate)
- [ ] GPU release gate: macos-arm64 runner MLX-GPU smoke job if
      feasible, else a mandatory local-M-series pre-merge suite run
      (CI is CPU-only; releases are automated — see AGENTS.md)

## Later — ideas, ordered by thesis-fit

- Metropolis resampler (ratio-only, f32-safe beyond N≈10⁶; bias
  documented as PMMH-incompatible)
- Waste-free SMC (removes the MCMC-steps knob)
- SMC² (the (N_θ × N_x) nesting is the best unified-memory fit in the
  literature)
- Differentiable resampling: Ścibior-Wood stop-gradient first; OT/DET
  as opt-in
- FFBSi smoothing (dense batched backward weights); fixed-lag comes
  cheap earlier
- `to_arviz()` InferenceData export; independent runs as chains
- CESS tempering; MALA/HMC moves once model grads are wired
- Island-mode resampling for N beyond a single population
- Benchmark suite tracking MLX releases (re-run audit + kill test)

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
