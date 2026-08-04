# Export to ArviZ

Install the optional dependency. The example below uses the result and
observations from the [quickstart](quickstart.md).

```bash
pip install "smcx[arviz]"
```

```python
import jax.random as jr
import smcx

result = smcx.to_arviz(
    posterior,
    key=jr.key(7),
    num_draws=1_000,
    emissions=observations,
)
```

`posterior` above can be a `ParticleFilterPosterior` or
`TemperedPosterior`. Pass a sequence of independent results to represent
multiple chains. `num_draws` controls how many equal-weight draws are
resampled from each particle cloud; the key makes that resampling
reproducible.

Adaptive tempered runs may require different numbers of stages. Their
stage-wise diagnostics are padded with `NaN` to the longest run, and the
Boolean `particle_diagnostics.stage_valid` mask identifies each run's
recorded stages.

Particle-filter export requires the default `store_history=True`. A
final-only result keeps the full ESS and evidence traces but only one particle
cloud, so those arrays do not share an ArviZ `time` dimension. Rerun with full
history before exporting.

## Parameter posteriors

`to_arviz` does not consume `LiuWestPosterior`, `SMC2Posterior`, or
`IBISPosterior`. ArviZ draws are equal-weight by construction — every
summary and plot treats the `draw` dimension as unweighted — so a
weighted cloud must be resampled before export. For these records the
resampling has a second degree of freedom: each time slice is a
different posterior, with the final slice conditioning on all data and
earlier slices on `y[:t+1]`. Which slice to export is an analysis
decision, so smcx keeps both choices explicit instead of resampling on
your behalf.

To export the full-data parameter posterior, resample the final cloud
to equal weights and add a chain axis:

```python
import arviz
import jax.numpy as jnp
import jax.random as jr
import smcx

weights = jnp.exp(posterior.filtered_log_weights[-1])
indices = smcx.systematic(jr.key(3), weights, 1_000)
draws = posterior.filtered_params[-1][indices]
idata = arviz.from_dict({"posterior": {"params": draws[None]}})
```

Here `draws[None]` supplies the leading chain dimension. The mapping
form is the ArviZ 1.x signature; on 0.x, pass the group as a keyword
instead: `arviz.from_dict(posterior={"params": draws[None]})`. For
several independent
runs, stack the per-run `draws` along a new first axis so each run
becomes one chain. This is the same convention PyMC follows for its
SMC sampler, which resamples to equal weights before constructing
`InferenceData`.

## Groups

| Group or attribute | Contents |
| --- | --- |
| `posterior` | Resampled particle values |
| `particle_diagnostics` | Source log weights and algorithm diagnostics |
| `observed_data` | Emissions, when supplied |
| `unconstrained_posterior` | Aligned u-space values, when supplied |
| `posterior.attrs["marginal_loglik"]` | Evidence estimate for each run |

Particle-filter output has dimensions `(chain, draw, time, ...)`. Each time
slice is a filtering marginal $p(x_t \mid y_{0:t})$; draws with the same index
across time do not form a joint trajectory. `reconstruct_trajectories` gives
genealogy paths whose summaries use the final filtering weights, while
`backward_simulation` gives equal-weight draws from a discrete FFBS
approximation. `to_arviz` exports the filtering marginals; it does not consume
`ParticleSmootherPosterior`.

The `particle_diagnostics` group describes the source particle clouds, not
the resampled posterior draws. Its leading dimension is `run`, followed by
`time` for particle filters or `stage` for tempered SMC; source log weights
also have a `particle` dimension. These values are stored once per run rather
than repeated across `draw`.

Structured particle states use their PyTree paths as variable names. Supply
`var_names` to rename them and `dims` to label event dimensions. Values passed
through `unconstrained=` follow the same resampling indices as the constrained
particles. Raw dotted paths must be unambiguous before aliases are applied,
and aliases must resolve to unique names within each group. Variable names
cannot shadow sample or event dimensions. Event labels must be unique within
a variable and one size when shared in a group. Posterior and unconstrained
schemas may differ in rank or extent; particle labels do not affect
`observed_data`.

ArviZ 0.x returns `InferenceData`; ArviZ 1.x returns `DataTree`. smcx dispatches
to the constructor available in the installed generation:
[`arviz.from_dict` for 0.23.4][arviz-023] or
[`arviz_base.from_dict` for 1.x][arviz-1].
ArviZ is distributed under the
[Apache License 2.0](https://github.com/arviz-devs/arviz/blob/main/LICENSE).

[arviz-023]: https://python.arviz.org/en/v0.23.4/api/generated/arviz.from_dict.html
[arviz-1]: https://python.arviz.org/en/stable/api/generated/arviz.from_dict.html
