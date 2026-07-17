# Adversarial review: native MLX versus jax-mps

*Reviewer stance: I contribute to jax-mps and I use it in production. My prior
is that a JAX program executed through jax-mps should, once the plugin matures,
run within a small constant factor of hand-written MLX, because jax-mps lowers
StableHLO to the same MLX kernels the native library calls. A large measured
gap is, to me, evidence of an immature lowering or an unfair JAX arm until
proven otherwise. I read the report, the workers, and the raw JSON with that
prior in mind.*

## Summary judgment

The study is careful and, on its own terms, honest. It does not claim what it
cannot support, the correctness gates are real, and the negative controls are
the right idea. I accept the headline as a statement about *this* jax-mps
release and *this* JAX implementation on *this* machine: native MLX is far
faster on the sequential filter and on RNG-bound reductions, and the gap grows
with N. What I do not accept, and what the authors are careful not to assert,
is that the gap reflects a property of MLX that jax-mps cannot close. Every
mechanism the report points to is, in principle, a plugin-maturity issue. The
paper's own protocol concedes this. So my review is less a rebuttal than a
demand for the evidence that would move the claim from "jax-mps is slower today"
to "jax-mps is structurally slower here."

## What I concede immediately

Three design choices remove my easy objections.

First, compilation is not in the number. The report times the first call
separately, and the raw data shows why that matters little at scale: for
LGSSM-PF at N=10^6 the cold time is 12.90 s against a warm median of 12.70 s, a
ratio of 1.0. My usual complaint, "you are measuring my compile time," is dead
on arrival. The gap is steady state.

Second, the harness is not rigged against me. On the negative controls jax-mps
sits at parity or ahead: MATMUL at D=1024 runs faster under jax-mps
(native/compat ratio 0.57), and ELTWISE-REDUCE is within noise of 1.0 across
sizes. If the harness systematically punished my backend, the controls would
show it. They do not. That is the single most persuasive part of the study,
because it means the SMC-motif gaps are localized, not global.

Third, the faster of my two dispatch modes is always the one reported, so I
cannot claim I was denied my best configuration.

## Where I push back

### 1. You wrote my filter, and you say so

The protocol labels the JAX arm "deliberately adversarial" and notes it is a
benchmark-local implementation, not a jax-mps-tuned one. This is the crux. The
native arm is `smcx.bootstrap_filter`, a library the authors have optimized for
months. The JAX arm is a hand-written `lax.scan` over a `lax.cond` resampling
branch, written by the same authors, and not reviewed by anyone who tunes
jax-mps. A filter I would write for this backend might avoid `lax.cond`
entirely (unconditional systematic resampling is standard and removes the
branch), might replace `searchsorted` with an inversion that vectorizes better,
and might restructure the history output to avoid materializing five arrays of
shape (T, N) per step. None of that was tried. Comparing a tuned native
library against a naive port of its algorithm is not a like-for-like contest,
and the authors know it, which is why the verdict is hedged.

### 2. jax-mps calls MLX; a 60x gap is removable by construction

This is my strongest theoretical objection and the report half-agrees with it.
jax-mps does not reimplement linear algebra. It patches JAX primitives to MLX
operations and ships the same `mlx.metallib` the native path uses. In the limit
of a one-to-one lowering with fused intermediates, my execution *is* the native
execution plus bookkeeping. A factor of 60 is therefore not a law of physics;
it is a measure of how much bookkeeping the current lowering leaves in. The
StableHLO census the authors themselves captured supports reading it this way:
1189 operations for LGSSM-PF, of which 266 are constants and 248 are
`broadcast_in_dim`. A mature lowering folds constants and elides broadcasts that
MLX would fuse. The op count is thus as much an indictment of the present
lowering as of anything intrinsic.

### 3. But async dispatch does not help, which complicates my own story

Honesty requires me to report a number that hurts my case. If the gap were
dominated by host-side per-op dispatch and synchronization, enabling
`JAX_MPS_ASYNC_DISPATCH` should shrink it. It does not: for LGSSM-PF the
async-to-safe ratio is 1.00, 1.00, and 1.02 across the three sizes, and for
RANDOM at N=10^7 it is 1.00. Whatever is costing 12.7 s at N=10^6, roughly
127 ms per timestep against the native 2 ms, is on-device work, not a dispatch
queue I can drain by going asynchronous. That pushes the explanation toward
per-operation kernel launches, redundant materialization, or an inefficient
`searchsorted`/`scan` lowering, and away from "just flip a flag." It is still
plausibly fixable, but not trivially, and not by me in an afternoon.

### 4. The memory comparison is not apples to apples

The report quotes native peak at 3604 MB against jax-mps 8624 MB for LGSSM-PF
at N=10^6 and reads it as a 2.4x native advantage. These counters come from
different allocators with different definitions: MLX's `get_peak_memory` over
its unified buffer pool versus the plugin's `peak_bytes_in_use`. A 2.4x
difference probably survives the definitional mismatch, but the smaller memory
claims elsewhere (ratios near 0.8 to 1.0 on the controls) are within the range
where I would not trust the comparison without a common measurement method. The
memory story is suggestive, not settled.

### 5. Variance at the largest size is high

The bootstrap interval for LGSSM-PF at N=10^6 runs from 38.85 to 62.62 with a
point estimate of 59.86. The direction is not in doubt, but a spread that wide
across five process medians says the largest cell is thermally or
scheduling-sensitive, and the specific multiplier I would quote in a paper is
unstable. Five blocks is thin for a number this noisy.

### 6. RNG semantics may not be equal work

Both sides draw counter-based normals, but JAX is committed to Threefry with a
specific counter layout, visible in the StableHLO as a chain of `xor` and
`shift_right_logical`. MLX's generator need not match those bits. If the two
libraries are not performing the same amount of RNG arithmetic per sample, then
part of the LGSSM and RANDOM gap is a difference in random-number cost, not in
execution efficiency. The five-standard-error moment gate checks that the
*distribution* is right; it does not check that the two paths did equal work to
get there.

## What survives my skepticism

After all of that, some things I cannot argue away.

The gap is real, steady state, correctness-gated, and localized to the SMC
motifs while the controls sit at parity. The direction never reverses on
LGSSM-PF or RANDOM, and the lower confidence bound is enormous even at its
weakest. Whatever the mechanism, a practitioner choosing a tool *today* for a
particle filter on Apple silicon would be materially slower and hungrier for
memory using jax-mps, by a margin no amount of my special pleading closes at the
current release. "Immature, therefore fixable" is a promise about the future,
not a defense of the present.

## What would actually convince me

To move from "jax-mps is slower today" to "MLX has an advantage jax-mps cannot
overcome," I would need at least the following, and the harness the authors
built can support most of it:

1. **A jax-mps-tuned filter.** Let a jax-mps contributor rewrite the JAX arm:
   unconditional systematic resampling to kill the `lax.cond`, an inversion
   resampler that avoids `searchsorted`, and a history layout that does not
   materialize five (T, N) arrays. If the gap holds within a small factor after
   that, the "you hobbled me" objection is dead. This is the single most
   important experiment and it is currently missing.

2. **A per-operation profile, not an op count.** The StableHLO census shows how
   many operations exist, not where the 12.7 s goes. A Metal capture attributing
   time to `searchsorted`, the RNG chain, the gather, and the scan boundary
   would tell us whether the cost is intrinsic materialization or a slow kernel
   the plugin could replace. The report captures the IR but explicitly declines
   the causal claim for lack of this profile. I agree with that restraint, and I
   want the profile.

3. **A mechanism the JAX contract forces.** The convincing version of the
   authors' thesis is not "MLX is faster," it is "the StableHLO/JAX contract
   forces jax-mps to materialize intermediates that MLX's lazy graph fuses, and
   it cannot stop doing so without breaking JAX semantics." The 248
   `broadcast_in_dim` and the functional array-update model are candidates. Show
   me one intermediate that jax-mps must allocate *because* it is JAX-compatible
   and that MLX provably need not, and you have an argument about a limit rather
   than a lag.

4. **A common memory counter** and **more blocks at N=10^6** to tighten the
   interval.

## Reviewer verdict

Accept as an honest, well-instrumented measurement of the present state.
Reject, or rather do not even entertain, any reading in which this proves a
permanent MLX advantage; the authors do not make that reading, and the async
and control results actively discipline it in both directions. The study earns
a strong practical conclusion (choose native MLX today for filters and RNG on
this hardware) and leaves the interesting scientific question open (is the gap
a property of the JAX contract or of a young plugin?). I would sign off on the
first and ask for the tuned-JAX-arm experiment before I let anyone write the
second down as settled.
