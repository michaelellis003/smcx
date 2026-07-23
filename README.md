# smcx

smcx is a [JAX](https://github.com/jax-ml/jax) library for state-space
inference: exact linear-Gaussian filtering and smoothing, first-order
nonlinear Gaussian filtering, particle filters, adaptive tempered SMC,
and SMC² with a small, function-oriented API. It runs on CPU, CUDA, and
TPU through JAX, and on Apple-silicon GPUs through the optional
[jax-mps](https://github.com/tillahoffmann/jax-mps) backend.

Features include:

- exact linear-Gaussian Kalman filtering and RTS smoothing;
- extended Kalman filtering with explicit, replaceable Jacobian callbacks;
- bootstrap, auxiliary, guided, and Liu–West particle filters;
- a public runner for caller-owned particle-filter kernels;
- adaptive tempered SMC and nested SMC² parameter inference;
- systematic, stratified, multinomial, and residual resampling;
- filtering diagnostics, scoring rules, trajectory reconstruction, and
  ArviZ export; and
- structured latent-state PyTrees and explicit time-varying inputs.

smcx supplies inference algorithms, not model or distribution classes.
Linear-Gaussian models are dense arrays. Nonlinear Gaussian and particle
models use ordinary JAX callables, so model functions, Jacobians, proposals,
and other algorithmic pieces can be replaced independently.
Filtering and smoothing remain separate functions joined by typed
posterior containers, allowing research code to replace one stage
without subclassing or rerunning the other.

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
guides for [custom models and custom particle filters][custom-models]
and [ArviZ reporting](https://michaelellis003.github.io/smcx/guides/arviz/),
and the complete
[API reference](https://michaelellis003.github.io/smcx/api/smcx/).

[custom-models]:
  https://michaelellis003.github.io/smcx/guides/custom-models/

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
The caller-owned particle-filter runner was informed by the functional
state/information protocol in
[BlackJAX 1.6.2](https://github.com/blackjax-devs/blackjax/releases/tag/1.6.2)
and the separation of orchestration from history in
[particles 0.4](https://github.com/nchopin/particles/releases/tag/v0.4).
These are design credits; no code was copied or translated.
The implemented methods draw on these primary sources:

- Exact linear-Gaussian state estimation:
  [Kalman (1960)](https://doi.org/10.1115/1.3662552) and
  [Rauch, Tung, and Striebel (1965)](https://doi.org/10.2514/3.3166).
- First-order nonlinear Gaussian filtering:
  [Schmidt (1966)](https://doi.org/10.1016/B978-1-4831-6716-9.50011-4).
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

### Numerical validation references

The linear Kalman and RTS outputs are independently validated against
[Dynamax 1.0.2](https://github.com/probml/dynamax/releases/tag/1.0.2)
and
[statsmodels 0.14.6](https://github.com/statsmodels/statsmodels/releases/tag/v0.14.6);
the details are recorded with the
[frozen linear fixture](tests/_kalman_reference.py).

The extended Kalman outputs are independently validated against
[Stone Soup 1.9.1](https://github.com/dstl/Stone-Soup/releases/tag/v1.9.1),
cross-checked with Dynamax 1.0.2, and checked against
[SciPy 1.18.0](https://github.com/scipy/scipy/releases/tag/v1.18.0)
innovation log densities. Exact commits, environments, licenses, and
observed differences are recorded with the
[frozen nonlinear fixture](tests/_extended_kalman_reference.py).

These projects are numerical comparison implementations, not code
lineage; no implementation code was copied or translated.

## Contributing

Contributions are welcome. See
[`CONTRIBUTING.md`](https://github.com/michaelellis003/smcx/blob/main/CONTRIBUTING.md)
for the development setup and pull-request conventions.

## License

smcx is distributed under the
[Apache License 2.0](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
