# smcx

smcx is a [JAX](https://github.com/jax-ml/jax) library for Sequential
Monte Carlo: particle filters, adaptive tempered SMC, and SMC² with a
small, function-oriented API. It runs on CPU, CUDA, and TPU through JAX,
and on Apple-silicon GPUs through the optional
[jax-mps](https://github.com/tillahoffmann/jax-mps) backend.

Features include:

- bootstrap, auxiliary, guided, and Liu–West particle filters;
- adaptive tempered SMC and nested SMC² parameter inference;
- systematic, stratified, multinomial, and residual resampling;
- filtering diagnostics, scoring rules, trajectory reconstruction, and
  ArviZ export; and
- structured latent-state PyTrees and explicit time-varying inputs.

smcx supplies inference algorithms, not model or distribution classes.
Models are ordinary JAX callables, so they can be written directly or
adapted from libraries such as
[Dynamax](https://github.com/probml/dynamax).

## Installation

smcx requires Python 3.11 or later.

```bash
pip install smcx
```

Install the optional extras for Apple-silicon GPU execution or ArviZ
reporting with:

```bash
pip install "smcx[metal]"
pip install "smcx[arviz]"
```

The `metal` extra uses jax-mps and is available on macOS arm64. Metal is
float32-only; releases are tested on a physical M-series GPU as well as
on CPU.

## Documentation

The [documentation](https://michaelellis003.github.io/smcx/) includes a
[quickstart](https://michaelellis003.github.io/smcx/guides/quickstart/),
guides for
[custom models](https://michaelellis003.github.io/smcx/guides/custom-models/)
and [ArviZ reporting](https://michaelellis003.github.io/smcx/guides/arviz/),
and the complete
[API reference](https://michaelellis003.github.io/smcx/api/smcx/).

## Quick example

```python
import jax.numpy as jnp
import jax.random as jr

import smcx

a, q, r = 0.9, 0.5, 0.3


def initial_sampler(key, num_particles):
    return jr.normal(key, (num_particles, 1))


def transition_sampler(key, state):
    return a * state + jnp.sqrt(q) * jr.normal(key, state.shape)


def log_observation(y, state):
    error = y[0] - state[0]
    return -0.5 * (jnp.log(2 * jnp.pi * r) + error**2 / r)


observations = jnp.array([0.2, -0.1, 0.4, 0.7, 0.3])[:, None]
posterior = smcx.bootstrap_filter(
    jr.key(0),
    initial_sampler,
    transition_sampler,
    log_observation,
    observations,
    num_particles=10_000,
)

posterior.marginal_loglik
smcx.weighted_mean(posterior)
smcx.diagnose(posterior)
```

Callbacks describe one particle; smcx vectorizes them over the cloud.
Every stochastic operation takes an explicit PRNG key, and posterior
containers are JAX PyTrees.

## Citation

If smcx contributes to academic work, cite the release used. The
repository's **Cite this repository** menu is generated from
[`CITATION.cff`](https://github.com/michaelellis003/smcx/blob/main/CITATION.cff)
and provides BibTeX and APA entries.

## Sources and attribution

The broader Feynman–Kac architecture follows Chopin and
Papaspiliopoulos's
[*An Introduction to Sequential Monte Carlo*](https://doi.org/10.1007/978-3-030-47845-2).
The implemented methods draw on these primary sources:

- Particle filters: [Gordon, Salmond, and Smith (1993)](https://doi.org/10.1049/ip-f-2.1993.0015),
  [Pitt and Shephard (1999)](https://doi.org/10.1080/01621459.1999.10474153),
  [Doucet, Godsill, and Andrieu (2000)](https://doi.org/10.1023/A:1008935410038),
  and [Liu and West (2001)](https://doi.org/10.1007/978-1-4757-3437-9_10).
- Static and parameter inference:
  [Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x)
  and [Chopin, Jacob, and Papaspiliopoulos (2013)](https://doi.org/10.1111/j.1467-9868.2012.01046.x).
- Resampling and diagnostics:
  [Douc, Cappé, and Moulines (2005)](https://doi.org/10.1109/ISPA.2005.195385),
  [Lee and Whiteley (2018)](https://doi.org/10.1093/biomet/asy028),
  [Zhang and Stephens (2009)](https://doi.org/10.1198/TECH.2009.08017),
  and [Vehtari et al. (2024)](https://jmlr.org/papers/v25/19-556.html).
- Scoring rules:
  [Matheson and Winkler (1976)](https://doi.org/10.1287/mnsc.22.10.1087)
  and [Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437).
- Reporting: [ArviZ](https://doi.org/10.21105/joss.01143).

## Contributing

Contributions are welcome. See
[`CONTRIBUTING.md`](https://github.com/michaelellis003/smcx/blob/main/CONTRIBUTING.md)
for the development setup and pull-request conventions.

## License

smcx is distributed under the
[Apache License 2.0](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
