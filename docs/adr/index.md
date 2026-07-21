# Architecture Decision Records

Decisions that shape smcx's structure, dependencies, public API, or
numerics. Accepted ADRs are immutable: to reverse one, write a new ADR
that supersedes it. Records 0001–0022 predate the public decision log;
their stable numbers are retained for references in code and documentation.

| # | Decision | Status |
|---|---|---|
| 0001 | MLX-native library with API parity to smcjax | accepted |
| 0002 | Feynman-Kac core beneath the smcjax-compatible flat API | accepted |
| 0003 | float32 log-domain numerics policy with a CPU-f64 diagnostics hatch | accepted |
| 0004 | Native inverse-CDF resamplers with in-library searchsorted | accepted |
| 0005 | Explicit splittable RNG keys on every stochastic function | accepted |
| 0006 | Vendor missing substrate capabilities, track upstream, never block | accepted |
| 0007 | jaxtyping annotations, beartype test-time enforcement, vendored MLX stubs | accepted |
| 0008 | v0 public API divergences from smcjax, and the parity rule that bounds them | accepted |
| 0009 | Resampler kernel selection: counting for systematic, fused Metal binary search for the rest | accepted |
| 0010 | smcx is the successor; smcjax becomes a frozen benchmark baseline | accepted |
| 0011 | store_history option: O(final)-memory filtering | accepted |
| 0012 | Distributions: flat functions, vendored special functions, guarded factorization | accepted |
| 0013 | Batched-model fast path (`batched=True`) | accepted |
| 0014 | SMC² for nested parameter inference | accepted |
| 0015 | Benchmark native MLX against pinned jax-mps in isolated environments | accepted |
| 0016 | Loop shell v2: single-ESS steps and an N-aware sync policy | accepted |
| 0017 | Systematic resampling via the fused right-bisect kernel | accepted |
| 0018 | Pivot: smcx becomes a JAX library with jax-mps as the Apple backend | accepted |
| 0019 | smcx is a model-free inference engine | accepted |
| 0020 | Diagnostics stay SMC-native; reporting delegates to ArviZ | accepted |
| 0021 | Single-run log-ML variance via Lee-Whiteley Eve classes | accepted |
| 0022 | Exogenous-inputs channel for filter callbacks | accepted |
| [0023](0023-one-time-external-validation.md) | External implementations are one-time isolated validators | accepted |
| [0024](0024-structured-latent-state-pytrees.md) | Structured latent-state PyTrees in standard particle filters | accepted |
| [0025](0025-monotone-multinomial-prefix-rounding.md) | Monotone float32 multinomial prefix rounding | accepted |
| [0026](0026-jax-mps-scan-history-floor.md) | jax-mps 0.10.10 floor for scan history | accepted |
| [0027](0027-arviz-bridge-contract.md) | ArviZ bridge exports seeded equal-weight reporting data | accepted |
| [0028](0028-streaming-filter-checkpoints.md) | Streaming bootstrap-filter checkpoints | accepted |
| [0029](0029-common-space-static-posterior-updating.md) | Common-space static posterior updating | proposed |
| [0030](0030-native-conditionally-linear-gaussian-rbpf.md) | Native square-root conditionally linear-Gaussian RBPF | proposed |
| [0031](0031-mps-bootstrap-update-containment.md) | MPS host-loop containment for bootstrap chunk updates | accepted |
