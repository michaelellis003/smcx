# Architecture Decision Records

Decisions that shape smcx's structure, dependencies, public API, or
numerics. Accepted ADRs are immutable: to reverse one, write a new ADR
that supersedes it. Template: [0000](0000-template.md) — it also
documents when an ADR is (and isn't) required.

| # | Decision | Status |
|---|---|---|
| [0001](0001-mlx-native-sibling-of-smcjax.md) | MLX-native library with API parity to smcjax | accepted |
| [0002](0002-feynman-kac-core-beneath-flat-api.md) | Feynman-Kac core beneath the smcjax-compatible flat API | accepted |
| [0003](0003-float32-log-domain-numerics-policy.md) | float32 log-domain numerics policy with a CPU-f64 diagnostics hatch | accepted |
| [0004](0004-native-inverse-cdf-resamplers.md) | Native inverse-CDF resamplers with in-library searchsorted | accepted |
| [0005](0005-explicit-rng-keys.md) | Explicit splittable RNG keys on every stochastic function | accepted |
| [0006](0006-vendor-with-tracking-issue-upstream-policy.md) | Vendor missing substrate capabilities, track upstream, never block | accepted |
| [0007](0007-jaxtyping-beartype-vendored-stubs.md) | jaxtyping annotations, beartype test-time enforcement, vendored MLX stubs | accepted |
| [0008](0008-api-divergences-from-smcjax-v0.md) | v0 public API divergences from smcjax, and the parity rule that bounds them | accepted |
| [0009](0009-resampler-kernel-selection.md) | Resampler kernel selection: counting for systematic, fused Metal binary search for the rest | accepted |
| [0010](0010-smcjax-frozen-baseline.md) | smcx is the successor; smcjax becomes a frozen benchmark baseline | accepted |
| [0011](0011-store-history-option.md) | store_history option: O(final)-memory filtering | accepted |
| [0012](0012-distributions-module.md) | Distributions: flat functions, vendored special functions, guarded factorization | accepted |
| [0013](0013-batched-model-fast-path.md) | Batched-model fast path (`batched=True`) | accepted |
| [0014](0014-smc2-nested-parameter-inference.md) | SMC² for nested parameter inference | accepted |
| [0015](0015-native-mlx-versus-jax-mps-benchmark.md) | Benchmark native MLX against pinned jax-mps in isolated environments | accepted |
| [0016](0016-loop-shell-v2.md) | Loop shell v2: single-ESS steps and an N-aware sync policy | accepted |
| [0017](0017-systematic-via-bisect.md) | Systematic resampling via the fused right-bisect kernel | accepted |
| [0018](0018-pivot-to-jax-core.md) | Pivot: smcx becomes a JAX library with jax-mps as the Apple backend | accepted |
| [0019](0019-model-free-inference-engine.md) | smcx is a model-free inference engine | accepted |
| [0020](0020-diagnostics-boundary.md) | Diagnostics stay SMC-native; reporting delegates to ArviZ | accepted |
| [0021](0021-genealogy-variance-estimator.md) | Single-run log-ML variance via Lee-Whiteley Eve classes | accepted |
| [0022](0022-exogenous-inputs-channel.md) | Exogenous-inputs channel for filter callbacks | accepted |
| [0023](0023-one-time-external-validation.md) | External implementations are one-time isolated validators | accepted |
| [0024](0024-structured-latent-state-pytrees.md) | Structured latent-state PyTrees in standard particle filters | accepted |
| [0025](0025-monotone-multinomial-prefix-rounding.md) | Monotone float32 multinomial prefix rounding | accepted |
| [0026](0026-jax-mps-scan-history-floor.md) | jax-mps 0.10.10 floor for scan history | accepted |
| [0027](0027-arviz-bridge-contract.md) | ArviZ bridge exports seeded equal-weight reporting data | proposed |
| [0028](0028-streaming-filter-checkpoints.md) | Streaming bootstrap-filter checkpoints | proposed |
| [0029](0029-common-space-static-posterior-updating.md) | Common-space static posterior updating | proposed |
