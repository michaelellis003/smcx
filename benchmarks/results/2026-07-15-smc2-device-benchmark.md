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

## Results — the hardware isolation (primary)

| N_θ | N_x | inner particles | GPU | CPU | speedup | gate |
|----:|----:|----:|----:|----:|----:|:--:|
| 512  | 512  | 0.26M | 0.57 s | 19.30 s | **33.7×** | PASS |
| 1024 | 1024 | 1.05M | 2.85 s | 92.01 s | **32.3×** | PASS |

Correctness (log Ẑ vs exact log Z = −143.53): GPU −143.41 / −143.47,
CPU within the same band — the fast GPU runs are numerically sound,
not fast-but-wrong.

## External-authority baseline: Chopin's `particles` (ADR-0014)

`particles` (nchopin/particles) is the reference SMC² implementation
from Chopin & Papaspiliopoulos (2020). It runs the same LGSSM and the
same data on CPU. Its primary value here is an **independent
correctness cross-check**: three separate implementations converge on
the exact log-evidence.

| N_θ = N_x | smcx-GPU | smcx-CPU | particles-CPU | log Ẑ (exact −143.53) |
|----:|----:|----:|----:|----:|
| 512  | 0.57 s | 19.30 s | 67.6 s  | smcx −143.4, particles −143.56 |
| 1024 | 2.85 s | 92.01 s | 170.3 s | smcx −143.5, particles −143.55 |

All three land on the exact log Z within Monte-Carlo error — smcx is
correct, confirmed by an implementation that shares no code with it.
All times are **medians of 5** (AGENTS.md), on both sides — each
`particles` run is 70–170 s. A first single-run pass read the 512
cell 41 % high (95.8 s vs the 67.6 s median), which is why the
external timings are medianed like everything else.

On speed, smcx-GPU runs **~60–119× faster** than `particles` and
smcx-CPU **~1.9–3.5× faster on the same hardware**. Read these as a
ballpark, not a controlled implementation contest: `particles`'
SMC² default is **waste-free with a length-10 move chain**, heavier
per rejuvenation than smcx's 3 PMMH steps, so some of the gap is
algorithm configuration, not implementation or hardware. The clean,
fully-controlled number is the smcx-GPU-vs-smcx-CPU 32–34× above
(identical code and algorithm, device the only variable).

Reproduce (isolated env — `particles` pins numpy<2, conflicting with
smcx): `uv run --no-project --with 'particles>=0.4' python
benchmarks/smc2/particles_side.py 512 512 100 5` (last arg = reps).

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

- Adaptive N_x and the exchange step; guided inner engines.
- Outer-loop async/lag-k cadence (the forward pass is not the
  bottleneck, so this is low-priority).
- A config-matched `particles` run (non-waste-free, 3-step move) for
  a tighter implementation comparison — the current baseline uses
  the `particles` default, which the table notes.

Reproduce: `uv run python benchmarks/smc2/bench_smc2.py`.
