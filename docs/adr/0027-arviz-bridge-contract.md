# 0027. ArviZ bridge exports seeded equal-weight reporting data

Date: 2026-07-20 | Status: accepted | Supersedes: — | Superseded-by: —

## Context

ADR-0020 assigns reporting to ArviZ while smcx retains diagnostics that
consume SMC-native structures. The supported Python matrix resolves
ArviZ 0.23.4 on Python 3.11 and ArviZ 1.x on Python 3.12 and later.
Those generations have neither a common constructor nor a common return
type: legacy ``arviz.from_dict`` returns ``InferenceData``, while
``arviz_base.from_dict`` returns an ``xarray.DataTree``.

The ArviZ schema has no particle-weight representation. It does
standardize ``unconstrained_posterior``, but ArviZ 0.23.4 was verified
on Python 3.11 to silently drop that group when passed to
``arviz.from_dict`` as an extra keyword.

## Options considered

- Support both generations behind a lazy adapter. This preserves smcx's
  Python support, but the adapter must account for different APIs.
- Require Python 3.12 and ArviZ 1.x for reporting. This simplifies the
  adapter, but excludes a supported smcx interpreter for no numerical
  reason.
- Make ArviZ a core dependency. This gives one always-present output
  stack, but adds xarray and plotting dependencies to the inference
  engine.

## Decision

We will expose this public contract:

```python
def to_arviz(
    posteriors,
    *,
    key,
    num_draws=None,
    var_names=None,
    dims=None,
    emissions=None,
    unconstrained=None,
):
    ...
```

``posteriors`` accepts one ``TemperedPosterior`` or
``ParticleFilterPosterior``, or a nonempty homogeneous sequence of
shape-compatible posteriors. One posterior is one chain; sequence item
``i`` is chain ``i``. ``key`` is always required. ``num_draws`` must be
positive and defaults to the source particle count.

For a particle-filter posterior, each time-indexed weighted cloud is
systematically resampled using smcx's ADR-0004 kernel. Keys are split in
chain-major, time-minor order. The result has dimensions
``(chain, draw, time, ...)`` and represents filtering marginals at each
time, not joint trajectories. For an equal-weight tempered posterior,
the source order is retained when its particle count equals
``num_draws``; otherwise it is uniformly resampled with the same explicit
key discipline.

The adapter builds one canonical nested dictionary of groups. On ArviZ
1.x it passes the complete dictionary to ``arviz_base.from_dict``. On
ArviZ 0.23.4 it passes constructor-supported groups to
``arviz.from_dict`` and attaches ``unconstrained_posterior``, when
present, through the public ``InferenceData.add_groups`` API. The return
value is the installed generation's native ``InferenceData`` or
``DataTree`` object.

ArviZ will be declared as the optional ``smcx[arviz]`` extra with a
minimum of 0.23.4 and imported only inside ``to_arviz``. A missing extra
raises ``ImportError`` with an instruction to install ``smcx[arviz]``;
``import smcx`` never imports ArviZ.

The groups follow these rules:

- ``posterior`` contains decoded, equal-weight draws.
- ``sample_stats`` contains normalized source ``log_weights`` and the
  applicable ESS, Pareto-k, acceptance, temperature, and conditional
  log-evidence traces. ``log_weights`` is an extra schema-tolerated
  variable because ArviZ defines no weight field.
- Each run's total ``marginal_loglik`` is stored, in chain order, in the
  posterior dataset attributes. No evidence group is invented.
- ``observed_data`` contains optional shared ``emissions``.
- ``unconstrained_posterior`` contains optional u-space values aligned
  with the source particles and selected by the same resampling indices.
  The caller owns any codec and supplies already-decoded posterior values;
  smcx defines no codec.

``sample_stats`` uses a singleton ``draw`` axis for these run-level
quantities. Filter weights have dimensions
``(chain, 1, particle, time)`` and tempered weights have dimensions
``(chain, 1, particle)``. Time and stage traces use
``(chain, 1, time)`` or ``(chain, 1, stage)``. Thus ``log_weights``
preserves the raw source cloud on an explicit ``particle`` axis; it is
not gathered by the posterior resampling indices and remains independent
of ``num_draws``.

A dense state is named ``theta`` unless overridden. Structured PyTree
leaves use their tree paths joined with ``.``. ``var_names`` maps those
paths to output names, and ``dims`` maps output names to event-dimension
names. Unnamed event axes become ``<name>_dim_0``,
``<name>_dim_1``, and so on.

## Consequences

- The inference engine's core dependency and import footprint remain
  unchanged; reporting conversion and host transfer occur only on demand.
- Users see the generation-native ArviZ return type for their Python
  environment, so documentation and tests must cover both forms.
- Filtering exports cannot be interpreted as smoothing trajectories.
  Joint draws remain available separately through
  ``reconstruct_trajectories`` and may receive a later reporting bridge.
- Weight and SMC trace variables are deliberate ``sample_stats``
  extensions rather than claims of new ArviZ schema fields.
