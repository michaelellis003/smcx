# Naming conventions and the 3.0 alignment list

*Status: accepted 2026-08-04. A dated amendment to
`2026-07-27-workbench-architecture.md`, codifying the naming grammar
the public surface already follows and recording the breaking
alignment items reserved for 3.0. The governing principle is the
Principle of Least Astonishment: the algorithm families behave
uniformly, so a contract learned on one entry point transfers to its
siblings, and new behavior added to one family member is specified
for the whole family in the same decision record. An August 2026
audit of all 85 exports found the grammar below already holds, with
the exceptions catalogued in the alignment list.*

## The conventions (binding for new surface)

- Filters are `<algorithm>_filter`; moment smoothers are
  `<x>_smoother`.
- Draw-producing operations carry the noun their literature uses:
  `posterior_sample`, `posterior_predictive_sample`, and
  `backward_simulation` (Godsill's term — matching the literature the
  reader expects outranks internal uniformity, and this sentence is
  the recorded reason).
- Result records are `<Domain><Operation>Posterior` when an algorithm
  has separate operations, and `<Algorithm>Posterior` when the
  algorithm is the operation (`SMC2Posterior`, `IBISPosterior`,
  `LiuWestPosterior`).
- Structural protocols are `<Consumes>Result`.
- Strategy and family factories are lowercase functions returning
  CapWords records (`unscented()` returns `Unscented`).
- Resamplers are bare method adjectives; counts are `num_*`; the key
  is first and named `key`; configuration is keyword-only.
- Module files: algorithm modules carry the algorithm name; domain
  modules carry a topic noun or gerund.
- No new entry point adds a callback-arity fork or a new spelling of
  an existing configuration concept.

## The 3.0 alignment list (breaking; accumulate, do not implement)

1. One vocabulary for the marginal likelihood: record fields say
   `marginal_loglik` while diagnostics say `log_ml_*`; pick one and
   alias-deprecate the other.
2. `TemperedPosterior` is the only past-participle record; align with
   the record grammar.
3. `temper` is the lone verb among noun drivers and the one
   file/function mismatch (`tempering.py`); rename or record the
   exception permanently.
4. `pareto_k_diagnostic` carries a suffix no other diagnostic has.
5. `TemperedPosterior.ess` means the pre-resampling selection ESS
   while `IBISPosterior.ess` and `SMC2Posterior.ess` are post-move;
   rename the tempered field to `selection_ess`.
6. One evolution-resupply vocabulary across the retrospective family
   (`transition_matrix`/`transition_covariance`/
   `scale_free_transition_covariance`/`discount`), with one shared
   docstring fragment for the cannot-verify warning.
7. The callback-arity purge (`WithInput` protocol pairs collapse to
   single-arity, always-present `input_t`).
8. The invariant-move knob unifies: `num_pmmh_steps=1` (floor zero)
   versus `num_mcmc_steps=5` (floor one) becomes one name, one floor,
   one default.
9. The ESS-knob triple unifies: `resampling_threshold`
   (float-or-callable), `ess_threshold` (float-only), and
   `target_ess` (bounded domain) converge where semantics allow.
10. The callback `input_t` grammar unifies (presence-dispatch in the
    filters, always-present-`None` in ibis and the model records,
    absent in smc2 — smc2 gaining `inputs` support may land earlier
    as its own additive change).
11. `TemperedPosterior` becomes consumable by the parameter
    diagnostics (field naming and a history axis) and gains
    `store_history`.
12. `liu_west_filter`'s transitional `resampling_threshold=None`
    default flips as already FutureWarned.
13. Count keywords unify (`num_draws` is keyword-only where counts
    elsewhere are positional) along with any remaining count-message
    drift.
14. The streaming `bootstrap_step`/`bootstrap_update` criterion
    restriction (no callable criteria without a time index) is either
    lifted or recorded as a permanent contract.
15. `DGLMFamily` gains the optional `sample_emission` capability
    field (added 2026-08-05 by the ADR-0036 amendment; blocked in
    2.x by the released four-field-sequence contract, so
    `dglm_forecast_sample` takes it standalone until then).

Nothing on this list may be implemented before the 3.0 cut, and the
list may only grow through a dated decision record.
