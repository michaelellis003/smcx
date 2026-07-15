# SMC² device benchmark — MLX-GPU vs MLX-CPU, 2026-07-15

**The second kill test (ADR-0014). Result: the GPU wins decisively —
~32–34× at 0.26M–1.05M inner particles, far above the 3.4–7.8× of the
plain-filtering kill test.** SMC²'s (N_θ × N_x) tensor is the densest,
most batch-shaped workload in the SMC literature, and it is exactly
where unified memory pays off most — as ADR-0014 predicted.

## Setup

| | |
|---|---|
| Hardware | Apple M3 Pro, 36 GB unified memory |
| OS | macOS 26.2 |
| mlx | 0.32.0 |
| Model | LGSSM, unknown AR coefficient a (exact Kalman-grid reference) |
| SMC² config | `ess_threshold=0.5`, `num_pmmh_steps=3`, T=100 |
| Timing | warm-up compile first, median of 5, `mx.synchronize` fenced |
| Isolation | fresh process per cell (compile/device state cannot leak) |
| Correctness gate | \|log Ẑ − exact log Z\| < 0.5 on **both** devices |

The comparison isolates the hardware: identical smcx code, only the
MLX default device changes. This is the cleanest apples-to-apples
form of the thesis test — no second implementation to confound it.

## Results

| N_θ | N_x | inner particles | GPU | CPU | speedup | gate |
|----:|----:|----:|----:|----:|----:|:--:|
| 512  | 512  | 0.26M | 0.57 s | 19.30 s | **33.7×** | PASS |
| 1024 | 1024 | 1.05M | 2.85 s | 92.01 s | **32.3×** | PASS |

Correctness (log Ẑ vs exact log Z = −143.53): GPU −143.41 / −143.47,
CPU within the same band — the fast GPU runs are numerically sound,
not fast-but-wrong.

## Reading the result

The plain particle-filter kill test counted 3.4–7.8× because a
single filter's per-step work is modest and partly dispatch-bound at
these sizes. SMC² multiplies the work by N_θ independent inner
filters advanced as one batched step, so the GPU runs near its
throughput ceiling while the MLX CPU backend serializes the same
tensor ops — hence ~5× the filtering advantage. The
`inner_step` compile (the rejuvenation bottleneck) is in place; the
outer orchestration stays host-side (the ESS-triggered rejuvenation
is data-dependent, like the tempering schedule).

## Still open (ADR-0014 follow-ups)

- **External baseline**: Chopin's `particles` (CPU) as an
  independent-implementation reference. The smcx-CPU number already
  controls for the algorithm; `particles` would add external
  authority. Deferred (new dependency + model port).
- Adaptive N_x and the exchange step; guided inner engines.
- Outer-loop async/lag-k cadence (the forward pass is not the
  bottleneck, so this is low-priority).

Reproduce: `uv run python benchmarks/smc2/bench_smc2.py`.
