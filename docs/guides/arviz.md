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

## Groups

| Group or attribute | Contents |
| --- | --- |
| `posterior` | Resampled particle values |
| `sample_stats` | Source log weights and algorithm diagnostics |
| `observed_data` | Emissions, when supplied |
| `unconstrained_posterior` | Aligned u-space values, when supplied |
| `posterior.attrs["marginal_loglik"]` | Evidence estimate for each run |

Particle-filter output has dimensions `(chain, draw, time, ...)`. Each time
slice is a filtering marginal $p(x_t \mid y_{0:t})$; draws with the same index
across time do not form a joint trajectory. Use `reconstruct_trajectories`
when ancestry is needed.

Structured particle states use their PyTree paths as variable names. Supply
`var_names` to rename them and `dims` to label event dimensions. Values passed
through `unconstrained=` follow the same resampling indices as the constrained
particles.

ArviZ 0.x returns `InferenceData`; ArviZ 1.x returns `DataTree`. smcx dispatches
to the constructor available in the installed generation:
[`arviz.from_dict` for 0.23.4][arviz-023] or
[`arviz_base.from_dict` for 1.x][arviz-1].
ArviZ is distributed under the
[Apache License 2.0](https://github.com/arviz-devs/arviz/blob/main/LICENSE).

[arviz-023]: https://python.arviz.org/en/v0.23.4/api/generated/arviz.from_dict.html
[arviz-1]: https://python.arviz.org/en/stable/api/generated/arviz.from_dict.html
