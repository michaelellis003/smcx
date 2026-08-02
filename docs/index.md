# smcx

Sequential inference in JAX: Kalman filters, particle filters, and
sequential Monte Carlo, with model code decoupled from inference code.

When a state-space model is linear and Gaussian with every model
matrix known (coefficients and covariances alike), the only unknowns
are the latent states. The Kalman filter computes their exact
filtering posterior by a two-step recursion. The RTS smoother revises
every estimate from the complete record, while exact joint trajectory
draws retain dependence across time. Relaxing linearity keeps a Gaussian
approximation alive (the extended and unscented filters and smoothers).
Outside special conjugate cases, unknown noise parameters or non-Gaussian
observations break
the Gaussian closed form. smcx implements an exact variance-scaled DLM with
retrospective smoothing and approximate conjugate/linear-Bayes DGLM filtering
and retrospective state-moment smoothing for specific observation families.
Particle filters carry the posterior as a weighted sample cloud and cover
general nonlinear or non-Gaussian
models. Full particle-filter histories support approximate genealogy paths
and backward-simulated joint trajectories. Broader sequential Monte Carlo
methods target other distribution sequences, including tempered paths for
static parameters, exact-likelihood IBIS updates, and nested state-parameter
inference.
smcx implements all of these methods. The
[introduction](guides/sequential-inference.md) develops them one
assumption at a time, with the equations and references.

Models are supplied as plain JAX callbacks. `StateSpaceModel` groups
the particle-model callbacks for reuse across the bootstrap, guided,
and auxiliary Feynman–Kac derivations over one shared loop, while the
named filters provide the shortest one-call interface; Liu–West,
tempered SMC, IBIS, SMC², and the caller-owned runner have their own
drivers. Resamplers, criteria, proposals, potentials, look-ahead
twists, mutation kernels, temperature schedules, and Gaussian
linearization strategies are all caller-replaceable, with parameters
as explicitly threaded PyTrees on the record path.

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
- [Custom models](guides/custom-models.md) covers nonlinear Gaussian filters
  and smoothers, particle smoother composition, the model record,
  Feynman–Kac derivations, structured latent states, time-varying inputs, and
  an optional Equinox representation.
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
