# smcx workbench architecture

*Publication note (2026-08-04): this is the accepted 2026-07-27
snapshot, committed as the durable record of the ratified direction.
It is a dated design document, not current API reference — several
constraints have since moved by their own recorded decisions (the
jax upper bound was removed in #390; the 2.0 cut shipped as v2.0.0
on 2026-07-28; the §13 questions were resolved in later decision
records), and the dated amendment
`2026-08-01-static-target-population-moves.md` narrows §8/§10's
one-loop rule for population-adaptive static-target samplers. The
guardrails in §10 bind every PR unchanged.*

*Status: THE course-keeping guide. Date: 2026-07-27. Ratified
direction: Michael decided (a) smcx's identity is the researcher's
SMC workbench, and (b) smcjax compatibility is retired entirely —
nobody used it, and cleanliness outranks conformance. Every design
decision and every PR is checked against §10's guardrails. Details
ratify individually as ADRs (ADR-0023 records the identity; the
supporting survey and duplication evidence live in
`2026-07-27-model-interface-and-fk-redesign.md`). This document
supersedes `v0-design.md` (historical, MLX era) as the architecture
reference.*

## 1. Identity, goals, constraints

smcx is **the researcher's SMC workbench**: one library where a
researcher can run standard state-space inference in a line, then
open the hood and replace any single component — proposal, potential,
resampler, criterion, mutation kernel, schedule, linearization rule —
without rewriting the machinery around it.

**Goals**

- G1. Every named algorithm is a thin composition of exchangeable,
  documented components. Nothing is reachable only as a monolith.
- G2. A custom method reuses the loop, primitives, diagnostics, and
  export. Writing a new filter means writing the parts that are new.
- G3. Correct-by-construction core: the classic SMC bug surface
  (weight branch rules, evidence increments, resample ordering)
  exists in exactly one place.
- G4. Everything jit/vmap/grad-transparent; f32-safe on Metal.
- G5. Honest diagnostics with statistically calibrated tests.

**Non-goals** (standing; unchanged by the redesign)

- A PPL, effect handlers, NUTS, distributions, or a model zoo
  (ADR-0019 stands: models are records of user callables).
- smcjax conformance of any kind — names, signatures, semantics
  (decision 2026-07-27; supersedes ADR-0010's frozen-baseline role).
  smcjax remains a historical reference only.
- float64 on GPU; emulating it.
- Bitwise-exact arithmetic in Monte Carlo diagnostics beyond a
  documented error bound (proportionality policy, §7).

**Constraints**

- jax 0.10 line; jax-mps `[metal]` extra; f32 on Metal (no f64).
- smcx v1.x is on PyPI. Compatibility follows NEP 23 pragmatism, not
  parity: additive in 1.x, one clean break at 2.0 (§9).
- Releases are dispatch-only (2026-07-27): merging never publishes;
  semantic-release batches unreleased commits into one aggregate
  version per manual dispatch. Slices must still be independently
  green and releasable.
- Metal containment for jax-mps (#38) is temporary and must stay
  quarantined (§8).

## 2. The component inventory (the spine)

The workbench is organized around what a researcher wants to swap.
Every row is a first-class concept with a typed contract, a default,
and a swap test.

| Component | Contract | Default | Layer |
|---|---|---|---|
| Initial sampler | model field | — (user) | model |
| Transition / proposal kernel | model field / FK `m` | transition prior | model / FK |
| Potential (weight function) | FK `log_g` | observation density | FK |
| Look-ahead twist (APF) | FK `log_eta` | none | FK |
| Resampler | `(key, weights, n) -> idx` | `systematic` | primitives |
| Resampling criterion | `(log_w, ess, t) -> bool` | ESS < 0.5N | loop |
| Whole step kernel | init/step -> record | — | runner |
| Tempering schedule | next-phi callback | ESS bisection | temper |
| Tempering mutation | init/step pair | adaptive RWM | temper |
| SMC² rejuvenation | same mutation contract | PMMH | smc2 |
| Gaussian linearization | strategy record | — (explicit) | gaussian |
| Storage policy | `store_history` | full | loop |

## 3. Layered architecture

```
L3  Named algorithms (thin, model-first)
    bootstrap_filter · guided_filter · auxiliary_filter ·
    liu_west_filter · temper · smc2 · kalman_filter ·
    gaussian_filter (EKF/UKF via strategies) · rts_smoother
      │  each: derive components → run shared core → posterior
L2  Model boundary + derivations
    StateSpaceModel (samplers/densities record)
    GaussianModel   (moments/Jacobians record)
    bootstrap_fk · guided_fk · auxiliary_fk  (model → FeynmanKac)
    taylor_order1 · unscented                (linearization records)
L1  Generic cores (each written once)
    run_smc(fk, ...)         — the FK loop: conditional resample →
                               mutate → reweight; branch weight rule;
                               evidence increments at reweight;
                               init-as-if-resampled; Neumaier carry;
                               degeneracy policy; history assembly
    run_particle_filter(...) — kernel escape hatch BELOW FK: caller
                               owns the whole step, loop owns
                               alignment/accumulation/containers
    _gaussian_core           — predict/update recursion over a
                               linearization strategy; shared
                               gain/whitening/logdet kernel
L0  Primitives
    weights (log_normalize, ess) · resampling (4 kernels + shared
    CDF guards — sole implementation, smc2 imports it) ·
    _numerics (Neumaier) · one canonical weighted-CDF for
    quantiles AND tail-ESS (§7)
```

A researcher enters at any layer: L3 to run, L2 to swap a component,
L1 with a custom `FeynmanKac` or step kernel, L0 to build something
we did not anticipate.

## 4. Model boundary (L2)

```python
class StateSpaceModel(NamedTuple):
    sample_initial: Callable      # (key, params, input_0) -> state
    sample_transition: Callable   # (key, state, params, input_t) -> state
    log_observation: Callable     # (emission, state, params, input_t) -> scalar
    log_transition: Callable | None = None
    sample_proposal: Callable | None = None   # sees emission
    log_proposal: Callable | None = None
    log_lookahead: Callable | None = None     # APF eta
    sample_emission: Callable | None = None   # simulate / predictive

class GaussianModel(NamedTuple):
    transition_mean: Callable     # (state, params, input_t) -> state_mean
    observation_mean: Callable
    transition_cov: Array | Callable
    observation_cov: Array | Callable
    transition_jacobian: Callable | None = None   # EKF strategy needs
    observation_jacobian: Callable | None = None
```

Conventions (binding, enforced by review against §10):

- `params` is an explicit pytree argument threaded by the library.
  Library code never closes over params. Consequences: `jax.grad`
  w.r.t. params through filters; no retrace on param change; no
  user-written binding factory anywhere.
- `input_t` is always present, `None` when absent. One protocol per
  role; `None` is static pytree structure (no runtime forks).
- Optional capability = field is `None`. An algorithm that needs a
  missing field raises a named error at entry. No signature
  inspection, ever.
- Per-particle semantics; the library vmaps. Only `sample_initial`
  sees the whole cloud (via the derivation, not the model).
- Construction-boundary validation: `validate_model(model, params,
  emissions[, inputs])` runs the structural checks once, eagerly,
  always (§6). Model callables' outputs are trusted thereafter.
- Decorators as binding are rejected (survey: zero precedents;
  definition-time binding cannot see call-time params).

Both records are also the interop story: a Dynamax adapter is a
function producing a `StateSpaceModel`/`GaussianModel` (ADR-0019's
"thin adapters that produce callables", upgraded to records).

## 5. Public API target (2.0 surface)

Monte Carlo family:

```python
posterior = smcx.bootstrap_filter(key, model, params, emissions,
                                  num_particles=4_096)
# all config keyword-only:
#   inputs, resampling_fn, resampling_criterion, store_history

fk = smcx.auxiliary_fk(model, params, emissions)      # or custom FK
posterior = smcx.run_smc(key, fk, num_particles=4_096)

posterior = smcx.run_particle_filter(key, init_fn, step_fn, emissions)
```

Static-target and joint inference:

```python
smcx.temper(key, sample_prior, log_prior, log_likelihood,
            num_particles=..., *, schedule_fn=None,
            mutation_init_fn=None, mutation_step_fn=None, ...)
smcx.smc2(key, model, params_prior, emissions, *, num_theta, num_x,
          rejuvenation_init_fn=None, rejuvenation_step_fn=None, ...)
```

Gaussian family:

```python
smcx.kalman_filter(...)                                # exact linear
smcx.gaussian_filter(key=None, model, params, emissions, *,
                     method=smcx.taylor_order1())      # EKF
smcx.gaussian_filter(..., method=smcx.unscented(1.0, 2.0, 0.0))
smcx.rts_smoother(posterior, model, params, *, method=...)
```

Rules for the surface:

- Model-first signatures; everything after the data is keyword-only.
- One container family satisfying one `ParticleFilterResult`
  protocol; `TemperedPosterior` gains `log_evidence_increments` and
  `store_history` support (parity freeze was the only reason not to).
- Diagnostics/reporting names unchanged;
  `posterior_predictive_sample` consumes `(model, params, posterior)`
  and absorbs the Liu-West parameter-aware variant (the #174
  FutureWarning already points here).
- The 1.x bag-of-callbacks forms survive until 2.0 as wrappers, then
  are removed. No long deprecation theater: usage is near zero;
  weeks, not quarters.

## 6. Validation architecture

- **Boundary, once, eagerly, always.** Structural validation
  (shapes, dtypes, tree signatures, covariance domains) happens at
  model/FK/entry construction on concrete values. It does not
  depend on eager-vs-jit calling convention — this removes the
  current two-regime contract.
- **Interiors are clean.** No data-dependent validators inside scan
  bodies or vmapped callbacks. Traced paths keep structure-only
  checks (free at trace time).
- **Outputs of our own layers are trusted.** L1 never re-validates
  what L2 produced *within one call graph*. A public entry point
  that accepts a protocol object of unknown provenance (for example
  `rts_smoother` taking a caller-constructible posterior) is a
  boundary and validates — verified 2026-07-27: its batched
  covariance check is one vectorized pass at the same O(T d^3)
  order as the smoother, with tolerances engineered against false
  rejection of internal roundoff; it stays.
- **Degeneracy policy is uniform and lives in the loop:** per-step
  normalizer finiteness gates the result identically for every FK
  derivation (today's four filters disagree); eager raise via
  `DegenerateWeightsError`, `-inf` propagation under jit
  (documented, unchanged).
- Budget: validation stays under ~10% of core-module lines. The
  campaign's validators shrink into `validate_model` and entry
  checks; per-call re-validation is deleted with the wrappers.

## 7. Numerics policy (delta from v0/ADR-0003, which stands)

Carried forward unchanged: log-domain weights across boundaries,
max-shifted logsumexp, Neumaier-compensated evidence, explicit keys,
sub-1 clamp on resampling grids, f32-calibrated test tolerances.

New decisions:

- **One canonical weighted empirical CDF** (compact positive
  support → directional midpoint axes → interpolation) shared by
  weighted quantiles AND tail-ESS. VERIFIED 2026-07-28: the #169/#218
  work already unified the conventions — the "frozen dual convention"
  concern described the pre-#169 state — and a pinning test now
  proves the equality (tail masks reconstructed from public weighted
  quantiles reproduce tail_ess). No 2.0 break needed here.
- **Proportionality rule for diagnostics:** a Monte Carlo diagnostic
  with O(N^-1/2) statistical error carries a documented numerical
  error bound, not bitwise exactness. Exact/slow paths may exist
  only host-side, gated on concrete inputs; nothing expensive may
  hide in a traced `lax.cond` that vmap turns into both-branches
  execution. The CRPS bignum path is refactored under this rule
  (host-gate now; candidate for retirement at 2.0).
- Gaussian domains: PD only where a factorization actually occurs
  (UKF process noise returns to PSD); subnormal-entry rejection is
  dropped from the public domain; the unscented guard becomes the
  invariant it protects (`beta >= alpha**2`).

## 8. Performance and platform policy

- `lax.scan` cores for all fixed-schedule loops; host-driven shells
  only where schedules are adaptive (`temper`, `smc2` outer).
- Metal containment (#38) is quarantined behind `_filter_scan` and
  the streaming shell. L1/L2 code never branches on platform. The
  #38 removal gate (fixed jax-mps release + reproducer pass) deletes
  containment without touching the layers.
- Carry memory: scan carries hold state, not records; history comes
  from scan stacking (`store_history=False` keeps O(N)).
  The runner's record-in-carry duplication is fixed by this rule.
- Streaming API (`bootstrap_init/step/update`) remains supported,
  backed by the same FK step function, its MPS loop confined to the
  shell.
- SMC² inner replay is O(T²) by construction (Chopin et al.);
  documented, not "fixed".

## 9. Testing and compatibility

- **Rewiring gate:** every internal rewiring slice (S1–S3 below)
  must reproduce fixed-key outputs bitwise on CPU for all existing
  filters. The existing key-schedule tests are the gate; they may be
  *replaced* only at 2.0 with an ADR.
- Statistical gates use derived-SE thresholds (the #241/#242
  ownership rules are policy: no raw tolerances, no
  investigation-artifact assertions, smallest input that kills the
  target bug).
- Oracles: exact Kalman for LGSSM; `Fraction`/f64 oracles where
  exactness is claimed; cross-library statistical equivalence only
  (no seeded-output comparisons — key streams are ours alone now).
- Every swap point has a test that swaps it (a custom resampler,
  criterion, mutation, schedule, FK, strategy).
- 2.0 break list (single ADR when cut): remove bag-of-callbacks
  wrappers and `WithInput` protocols; unify weighted-CDF; container
  unification (`TemperedPosterior`); predictive consolidation;
  any renames. Nothing else breaks.

## 10. Guardrails (check every PR against these)

1. **One loop.** A sequential importance-resampling variant is a
   `FeynmanKac` + `run_smc`. A new hand-rolled scan driver in a PR
   is a design failure: extend the loop once, or reject the feature.
2. **No arity forks.** No `...WithInput` twins, no signature
   inspection, no dispatch on callback arity. Variation is data
   (`None` fields, `None` input), never signature shape.
3. **Params are data.** Library code never closes over params.
4. **Validate at boundaries only.** Once, eagerly, at construction/
   entry. Never per-step, never per-element, never differently under
   jit. Our layers trust our layers.
5. **Every swap point: contract + default + swap test.**
6. **Diagnostics stay cheap under transforms.** No traced branch
   whose expensive side executes under vmap; exact paths gate
   host-side or do not exist.
7. **One numerics spine.** Log-domain, compensated sums, one
   weighted-CDF. Deviations need an ADR.
8. **Model-free forever** (ADR-0019): records of callables; no model
   classes, distributions, or zoos.
9. **Containment stays quarantined.** Platform workarounds live in
   named shims with a removal gate; layers are platform-blind.
10. **Slices ship clean.** <400 lines, one concern, independently
    green, fixed-key gates for rewiring, ADR before surface change,
    docs updated in the same PR as the surface they describe.
11. **No parity ghosts.** "smcjax did it this way" is not a reason.
    If a wart exists only for parity, delete it at the next boundary.

## 11. Migration plan (slices, each < 400 lines, each shippable)

| Slice | Type | Content | Gate |
|---|---|---|---|
| S0 | fix | UKF Q to PSD (shipped, PR #245); CRPS vmap cost documented (PR #248 — structural fix on the 2.0 list); tempering stall diagnostics; Liu-West kernel note. Retracted on verification: the UKF guard is the exact PSD boundary (comment added in #245) and the RTS posterior check is legitimate boundary validation. Keyword-only args move to S9 | existing suite + new regressions |
| S1 | feat | `fk.py`: `FeynmanKac`, `run_smc`; bootstrap rewired through it | bitwise fixed-key equality |
| S2 | refactor | guided + auxiliary rewired; degeneracy policy unified | bitwise fixed-key equality |
| S3 | decision | RESOLVED 2026-07-28: liu_west keeps its bespoke driver in 1.x — its three-way key split and weight-aware cloud-level kernel move cannot meet the bitwise gate through the two-way-split loop; rewiring folds into S9 with a documented key-schedule break | recorded |
| S4 | feat | `model.py`: `StateSpaceModel`, `validate_model`, `*_fk` derivations, model-first entry points | new tests; old paths untouched |
| S5 | feat | `gaussian_filter` + `taylor_order1`/`unscented`; EKF/UKF become wrappers; RTS takes strategies | numerical equality vs current EKF/UKF |
| S6 | feat | temper `schedule_fn`; smc2 pluggable rejuvenation; smc2 imports shared resampling | swap tests |
| S7 | fix | RESOLVED 2026-07-28: CDF convention verified already unified (pinning test added); tail_ess Python T×D loop replaced with nested vmap (bitwise-stable per the exact #241 contracts); runner record-in-carry memory fix | exact contracts + fixed-key suites |
| S8 | docs | guides rewritten workbench-first ("build your own filter" is chapter one); README; AGENTS.md refresh; `make_*_callbacks` recipes deleted | strict docs build |
| S9 | feat! | the 2.0 cut (§9 break list) | full suite + migration notes |

Order: S0 may go first (identity-neutral fixes). S1–S3 before S4 so
the model layer lands on the shared loop, not on four drivers.
Docs (S8) may interleave; each surface PR updates its own docs
regardless (guardrail 10).

## 12. Documentation debt to retire (tracked files, via PRs)

Contradicts this architecture today: `docs/guides/custom-models.md`
(the factory recipes), the smcjax-parity `__all__` lock in
`tests/test_init.py`, README positioning, any surviving
"parity with smcjax" phrasing in guides/CHANGELOG intro. Local-only
docs (CLAUDE.md, AGENTS.md, ADR statuses) were updated 2026-07-27
alongside this document.

## 13. Open questions for Michael

1. 2.0 timing: cut as soon as S1–S7 land (my recommendation — usage
   is near zero, carrying two surfaces has real cost), or hold 2.0
   until the workbench docs (S8) have been public for a while?
2. `temper`'s prior/likelihood remain bare callables (it has no
   state-space structure). Comfortable, or should a `StaticTarget`
   record mirror the model records for uniformity?
3. Keep `smcjax` checkout references in research docs as history
   (my recommendation), or scrub entirely?
