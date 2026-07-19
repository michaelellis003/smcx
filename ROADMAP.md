# Roadmap

*Last updated: 2026-07-19. Directional, not a promise; solo-maintained.
Themes here, tracking in GitHub issues. Decisions: `docs/adr/`.
Non-goals at the bottom are a standing scope guard.*

smcx v1.0 shipped 2026-07-17: four particle filters (bootstrap,
guided, auxiliary, Liu-West), adaptive tempered SMC, SMC², four
resamplers, a diagnostics suite, O(N)-memory filtering via
`store_history=False`, CPU/CUDA/TPU through stock JAX and
Apple-silicon GPUs through the optional jax-mps backend. Everything
below builds on that base. The v0 history (the MLX-era design,
benchmarks, and the pivot) lives in the ADRs.

## Now — diagnostics depth and ecosystem interop

The theme: extend the two things that differentiate smcx — the
diagnostics suite and the plain-callable model boundary — and fix
the small carry-overs the 2026-07 library review surfaced.

- [ ] All-algorithm correctness-first profiling campaign: CPU versus
      jax-mps across linear/nonlinear, parameter-learning, tempering,
      SMC², resampling, dense/PyTree representation, and optional Dynamax
      callback workloads; use the measured bottlenecks to prioritize fixes.
- [ ] Correct the Pareto-k warning text: weight variance is infinite
      for all k ≥ 0.5; 0.7 is the practical-reliability threshold
      (PSIS rate results), not the infinite-variance boundary.
- [ ] `reconstruct_trajectories`: genealogy tracing through the
      stored ancestor arrays (TFP and `particles` both ship this;
      we store the ancestry and offer nothing to walk it).
- [ ] Single-run log-ML variance estimators (Chan & Lai 2013,
      Lee & Whiteley 2018): Monte Carlo variance from one run's
      genealogy instead of `replicated_log_ml`'s R repeat runs.
- [x] ~~Exogenous-inputs channel (ADR-0022): explicit per-step
      covariates for controlled and covariate-driven models across
      all filters and simulation.~~
- [x] ~~Structured latent-state PyTrees (ADR-0024) in the bootstrap,
      auxiliary, and guided filters plus simulation, including joint
      genealogy and posterior-predictive operations.~~
- [ ] `to_arviz()` InferenceData export (ADR-0020): the single
      reporting bridge — independent runs as chains, weighted clouds
      resampled to draws — so ArviZ owns plots, R-hat, and posterior
      exploration and smcx never grows a reporting layer.
- [ ] Dynamax interop recipe: a documented `from_dynamax` pattern
      plus a worked Rao-Blackwellized particle filter example
      (per-particle Kalman statistics in the state, Dynamax KF
      update in the callbacks) — the recipe answers dynamax #112 and
      #271, where blackjax was found unsuitable for state-space
      filtering. Adapters produce callables; no model classes
      (ADR-0019).

## Next — smoothing and sampler upgrades

- [ ] Fixed-lag smoothing (cheap once trajectories reconstruct).
- [ ] FFBSi smoothing — the largest genuine algorithm gap against
      `particles`; needs `log_transition_fn`, which `guided_filter`
      already defines.
- [ ] Resampling criterion as a callable (TFP's
      `resample_criterion_fn` pattern), generalizing the ESS-fraction
      threshold at zero cost to current users.
- [ ] Waste-free SMC for `temper` (Dau & Chopin 2022; blackjax and
      `particles` both carry it).
- [ ] UKF-proposal particle filter recipe (dynamax #272 shape):
      Dynamax UKF step as `guided_filter`'s proposal.
- [x] ~~jax-mps CI leg: `SMCX_TEST_PLATFORM=mps` as a scheduled or
      best-effort job on macOS runners (they expose a paravirtual
      Metal device).~~
- [ ] Thesis-notebook Metal appendix once a jax-mps release ships
      the scan-history fixes (#219/#220): large-N f32 filtering on
      the GPU, f64 oracle checks staying on CPU.

## Later — ideas, ordered by fit

- Metropolis resampler (ratio-only, f32-safe beyond N≈10⁶; bias
  documented as PMMH-incompatible).
- Differentiable resampling: Ścibior-Wood stop-gradient first;
  OT/DET only ever as opt-in.
- CESS-based tempering schedules; MALA/HMC move kernels in `temper`
  once model grads are wired.
- SMC² follow-ups deferred by ADR-0014: adaptive N_x, the exchange
  step, guided inner engines.
- Island-mode resampling for N beyond a single population.
- SQMC (sequential quasi-Monte Carlo) — `particles`' signature
  algorithm; niche but well-specified.
- Iterated filtering (IF2/MOP) for maximum-likelihood estimation —
  pypomp's territory; a scope expansion that needs its own ADR
  discussion before any code.
- **jax-mps tracking (standing)**: re-run the CPU-vs-Metal benchmark
  when jax-mps ships relevant fixes (#219/#220 merged upstream;
  searchsorted contribution queued on their #203 mechanism
  decision), and contribute optimizations upstream when the gap is
  theirs to close. Performance claims about Apple silicon stay
  measured, never assumed.

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
- Windows as a tested platform (JAX itself is the constraint)
- Distribution-framework ambitions (objects, bijectors, constraints)
- Model classes or a model zoo of any kind (ADR-0019): smcx consumes
  models as JAX callables — user closures or thin adapters over model
  libraries such as Dynamax — and never defines them
- Plotting or visualization of any kind (ADR-0020): reporting
  delegates to ArviZ through `to_arviz()`; diagnostics stay in-library
  only when they consume SMC-native structures
