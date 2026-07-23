# smcx

smcx provides state-space inference algorithms for JAX: exact
linear-Gaussian filtering and smoothing, particle filters, adaptive
tempered SMC, and SMC². It is a function-oriented inference library,
not a modeling framework.

## Installation

smcx requires Python 3.11 or later.

```bash
pip install smcx
```

Optional extras add Apple-silicon GPU execution or ArviZ reporting:

```bash
pip install "smcx[metal]"
pip install "smcx[arviz]"
```

## Start here

- [Quickstart](guides/quickstart.md) establishes an exact Kalman baseline,
  then builds, diagnoses, and improves a particle filter.
- [Filtering tutorial](tutorials/filtering.md) runs a complete example and
  plots its filtering intervals and effective sample size.
- [Custom models](guides/custom-models.md) explains the callback boundary,
  structured latent states, time-varying inputs, and an optional Equinox
  representation.
- [Stochastic volatility](guides/stochastic-volatility.md) learns a static
  parameter online with the Liu–West filter.
- [ArviZ reporting](guides/arviz.md) exports weighted particle output for
  downstream analysis.
- The [API reference](api/smcx/index.md) documents every public function and
  posterior container from its source docstring.

## Model boundary

Linear-Gaussian models enter as dense arrays. Particle-model callbacks
describe one particle; smcx vectorizes them over the cloud. Every
stochastic operation takes an explicit PRNG key. Bootstrap, auxiliary,
and guided filters can carry nonempty latent-state PyTrees, and every
filter accepts an explicit sequence of time-varying inputs.

The Kalman filter and RTS smoother are separate functions connected by a
typed Gaussian posterior. Particle algorithms similarly expose model,
proposal, and resampling callbacks. These boundaries allow research code
to replace supported pieces without adopting a class hierarchy.

CPU, CUDA, and TPU use stock JAX. The optional `metal` extra uses jax-mps on
macOS arm64; Metal is float32-only.

## Citation and license

See the repository
[citation and source record](https://github.com/michaelellis003/smcx#citation)
and [`CITATION.cff`](https://github.com/michaelellis003/smcx/blob/main/CITATION.cff)
for citation metadata. smcx is distributed under the
[Apache License 2.0](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
