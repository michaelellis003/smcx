# smcx

Sequential inference in JAX: Kalman filters, particle filters, and
sequential Monte Carlo, with model code decoupled from inference code.

When a state-space model is linear and Gaussian with known
covariances, the Kalman filter computes the exact posterior by a
two-step recursion. Relaxing linearity keeps a Gaussian approximation
alive (the extended and unscented filters); asking to *learn* a noise
covariance, or observing through a discrete or heavy-tailed model,
breaks the recursion outright. Particle filters carry the posterior as
a weighted sample cloud instead, and sequential Monte Carlo extends
the same machinery to tempering and joint state-parameter inference.
smcx implements all of these methods. The
[introduction](guides/sequential-inference.md) develops them one
assumption at a time, with the equations and references.

Models enter as records of pure JAX callables with explicitly threaded
parameters; every particle algorithm is a Feynman–Kac derivation over
one generic loop; resamplers, criteria, proposals, potentials,
look-ahead twists, mutation kernels, temperature schedules, and
Gaussian linearization strategies are all caller-replaceable.

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
- [Custom models](guides/custom-models.md) covers the model record,
  Feynman–Kac derivations, structured latent states, time-varying
  inputs, and an optional Equinox representation.
- [Stochastic volatility](guides/stochastic-volatility.md) learns a static
  parameter online with the Liu–West filter.
- [ArviZ reporting](guides/arviz.md) exports weighted particle output for
  downstream analysis.
- The [API reference](api/smcx/index.md) documents every public function and
  posterior container from its source docstring.

## Backends and precision

CPU, CUDA, and TPU use stock JAX. The optional `metal` extra uses jax-mps on
macOS 14 or later on arm64; Metal is float32-only. Every stochastic
operation takes an explicit PRNG key.

## Citation and license

See the repository
[citation record](https://github.com/michaelellis003/smcx#citation)
and [`CITATION.cff`](https://github.com/michaelellis003/smcx/blob/main/CITATION.cff)
for citation metadata. smcx is distributed under the
[Apache License 2.0](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
