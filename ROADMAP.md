# Roadmap

*Last updated: 2026-07. Directional, not a promise; solo-maintained.
Themes here, tracking in GitHub issues/milestones. Design rationale:
`docs/design/v0-design.md`; decisions: `docs/adr/`.*

## Now (v0.1) — foundations and the kill test

The theme: prove the thesis before building breadth.

- [ ] `weights` module: log_normalize, normalize, ess, log_ess
- [ ] `resampling` module: systematic, stratified, multinomial,
      residual over the shared inverse-CDF kernel (ADR-0004)
- [ ] FK core: FKModel protocol + generic loop (ADR-0002)
- [ ] `bootstrap_filter` + `simulate` + containers
- [ ] Kalman oracle in tests (numpy f64) + LGSSM correctness suite
- [ ] **Kill test**: smcx (MLX GPU/CPU) vs smcjax (JAX CPU) at
      10⁴–10⁶ particles; verdict recorded in `benchmarks/results/`
      and README Status
- [ ] File MLX issue: `categorical(num_samples=)` O(N·M) memory
      (unreported upstream; evidence in `docs/research/mlx-audit.md`)
- [ ] Open smcjax coordinated-change issues (simulate fix,
      `resampling_criterion`, guided filter first)
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
- [ ] `__all__` parity lock test against smcjax's export list

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

## Non-goals

Standing scope guard — do not implement these; link here when closing
requests:

- General PPL, effect handlers, NUTS, or NumPyro feature parity
- PMMH / particle Gibbs as built-ins (we guarantee unbiased log-Z so
  smcx can be an inner engine)
- float64 on GPU, or emulating it
- OT/DET resampling as a default (O(N²), biased likelihood)
- SSP / Hilbert-ordered resampling (sequential-scan-shaped)
- Windows support; non-Apple hardware as a design target
- Distribution-framework ambitions (objects, bijectors, constraints)
