# One-time Dynamax integration validation — 2026-07-19

Status: **complete, correct, and performance-eligible**. The temporary adapter
and profiling arm were removed after this campaign; only this evidence and the
dependency-free permanent fixtures remain.

## Metadata

| Field | Value |
|---|---|
| Hardware | Apple M3 Pro, 12 CPU cores, 36 GiB unified memory |
| OS | macOS 26.2 (25C56) |
| Power / thermal | AC power; no thermal or performance warning before, after, or after extraction for every cell |
| Python | 3.13.9 |
| JAX / jaxlib | 0.10.2 / 0.10.2 |
| jax-mps | 0.10.10, safe dispatch |
| Dynamax | 1.0.2 |
| NumPy | 2.5.1 |
| smcx | 1.3.0 |
| Source | `651666b414ce3c16fa7398d4e190c433715bfb5a`, clean, source SHA-256 `b8aebca1ae24b8cca5d279b2eb62b65cb8d89c08be8faa73e5b27c16b72f9167` |
| Workload | Scalar LGSSM, `N=10,000`, `T=100`, no history, threshold `1.1` |
| Timing design | Five isolated process blocks; one warm-up and seven fenced repeats per block |
| Primary estimate | Median of five per-process steady medians |
| Order / inference / validation seeds | `20260719` / `20260719` / `20260720` |

The threshold forces exactly 99 resampling decisions in every arm, so callback
comparisons have equal registered discrete work. Twenty timing workers and
four independent validation workers executed 160 timed calls and 80 validation
replicates. All 20 timing cells completed and passed.

## Results

| Callbacks | Backend | Steady median, ms | Block IQR, ms | Lower, ms | Compile, ms | First, ms | Process RSS, MiB | Device peak, MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| local | CPU | 46.288 | 0.029 | 59.836 | 193.287 | 51.651 | 304.00 | — |
| Dynamax | CPU | 49.782 | 0.240 | 78.179 | 192.018 | 55.814 | 400.02 | — |
| local | MPS | 219.289 | 0.476 | 44.952 | 12.129 | 247.968 | 202.34 | 7.59 |
| Dynamax | MPS | 212.541 | 0.911 | 65.481 | 15.475 | 246.211 | 317.62 | 7.54 |

The measured Dynamax/local steady ratio was `1.075` on CPU and `0.969` on
MPS. The MPS difference is a result for this workload, not evidence of a
general speed advantage. The adapter increased process RSS by about 96 MiB on
CPU and 115 MiB on MPS, and increased lowering time on both backends.

## Accuracy gates

Every arm passed structural invariants and an independent 20-replicate,
five-estimator-SE Kalman gate for the likelihood ratio, final filtered mean,
and final raw second moment.

| Callbacks | Backend | Mean evidence ratio | Evidence tolerance | Maximum mean error / tolerance | Maximum second-moment error / tolerance |
|---|---:|---:|---:|---:|---:|
| local | CPU | 0.9492 | 0.1027 | 0.0007077 / 0.004345 | 0.0002075 / 0.001981 |
| local | MPS | 1.0036 | 0.07983 | 0.0004497 / 0.004041 | 0.0006769 / 0.002389 |
| Dynamax | CPU | 1.0128 | 0.1034 | 0.0004925 / 0.005343 | 0.0005958 / 0.002385 |
| Dynamax | MPS | 0.9834 | 0.1194 | 0.0003837 / 0.003720 | 0.0005127 / 0.002209 |

Dynamax's own `marginal_log_prob` result also matched the independent float64
Kalman oracle: absolute error `2.3036e-5`, below the registered f32-honest
tolerance `2.5089e-4`.

## Interpretation and source boundary

Dynamax models can be adapted to smcx's callable boundary, but the measured
overhead and import footprint do not justify making Dynamax a core smcx
dependency. It remains appropriate for user-side model authoring and examples.
The temporary adapter used only public methods; no Dynamax code was copied or
translated.

Authoritative immutable sources:

- Dynamax 1.0.2 model API and `marginal_log_prob` delegation:
  <https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/linear_gaussian_ssm/models.py#L173-L240>
- Dynamax LGSSM input convention used to align `u[t]`:
  <https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/linear_gaussian_ssm/inference.py#L489-L511>
- Dynamax MIT license, Copyright 2022 Probabilistic machine learning:
  <https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/LICENSE>

Reproduction command used before deleting the one-time adapter:

```bash
uv run python -m benchmarks.profiling.run \
  --profile integration --platforms cpu mps \
  --output-dir /tmp/smcx-profiling-integration-20260719-651666b-01010
```
