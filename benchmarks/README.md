# Benchmarks

Benchmarks are local scientific evidence, not CI gates. Dated results describe
the package version, hardware, and measurement design used for that run; they
are not portable performance guarantees.

## Current local benchmarks

- The [all-algorithm profiling protocol](profiling/PROTOCOL.md) covers current
  JAX entry points across model, control-flow, and representation regimes. Its
  reports include the
  [all-algorithm profile](results/2026-07-19-all-algorithm-profile.md),
  [matched optimization profile][matched-profile], and
  [representation/history profile][representation-profile].
- The [structured-state PyTree runner](pytree_state/benchmark.py) measures the
  cost of dense and structured latent states. Its authoritative result is the
  [dated PyTree report](results/2026-07-19-pytree-state-benchmark.md).

Follow each protocol's correctness gates and environment controls. Keep raw
outputs local and commit only a concise dated report when a result supports a
public claim or an engineering decision.

## Historical evidence

The [MLX kill-test protocol](PROTOCOL.md), its
[result sequence](results/2026-07-14-kill-test.md), the
[SMC² device result](results/2026-07-15-smc2-device-benchmark.md), and the
[native-MLX versus jax-mps protocol](native_vs_jax_mps/PROTOCOL.md) and
[result](results/2026-07-16-native-vs-jax-mps.md) record decisions made before
smcx replaced its MLX core with JAX. These timings do not describe current
smcx performance.

The [one-time Dynamax integration result][dynamax-result] is retained after its
temporary adapter was removed. Completed MLX-era scripts remain available in
the immutable [kill-test][killtest-source], [SMC²][smc2-source], and
[exploratory][exploratory-source] source archives. They target the former MLX
implementation and are not current reproduction commands.

[matched-profile]: results/2026-07-19-matched-optimization-profile.md
[representation-profile]: results/2026-07-19-representation-history-profile.md
[dynamax-result]: results/2026-07-19-dynamax-integration-validation.md
[killtest-source]: https://github.com/michaelellis003/smcx/tree/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/benchmarks/killtest
[smc2-source]: https://github.com/michaelellis003/smcx/tree/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/benchmarks/smc2
[exploratory-source]: https://github.com/michaelellis003/smcx/tree/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd/benchmarks/exploratory
