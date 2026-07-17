# smcx

[![CI](https://github.com/michaelellis003/smcx/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/smcx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/smcx)](https://pypi.org/project/smcx/)
[![License](https://img.shields.io/github/license/michaelellis003/smcx)](LICENSE)

Sequential Monte Carlo in [JAX](https://github.com/jax-ml/jax): particle
filters, adaptive tempered SMC, and SMC² with a small, flat API. Runs on
CPU, CUDA, and TPU through stock JAX, and on Apple-silicon GPUs through
the optional [jax-mps](https://github.com/tillahoffmann/jax-mps) backend.

## Install

```bash
pip install smcx            # CPU / CUDA / TPU via your jax install
pip install "smcx[metal]"   # + jax-mps for Apple-silicon GPUs
```

## What's in the box

- **Filters**: `bootstrap_filter`, `guided_filter` (general g·f/q
  proposal weights), `auxiliary_filter` (twisted potentials), and
  `liu_west_filter` (joint state–parameter, labeled approximate).
- **Static targets**: `temper` — adaptive tempered SMC with an
  ESS-bisection schedule and covariance-adapted random-walk moves.
- **Parameter inference**: `smc2` — nested SMC² with vmapped inner
  filters and PMMH rejuvenation.
- **Resampling**: systematic, stratified, multinomial, and residual —
  one contract, log-domain weights throughout, float32-safe query
  grids.
- **Diagnostics**: ESS traces, quantile tail-ESS, Pareto-k
  reliability, CRPS, cumulative log score, Bayes factors,
  posterior-predictive sampling, and a one-call `diagnose` summary.
- `store_history=False` on every filter drops memory from O(T·N) to
  O(N) with a bit-identical evidence estimate.

Every sampler is validated against exact references — Kalman oracles
for the filters, conjugate evidence for tempering, grid-integrated
posteriors for SMC² — with Monte-Carlo-calibrated gates, not loose
tolerances.

## Quick start

```python
import jax.numpy as jnp
import jax.random as jr

import smcx

# A 1-D linear-Gaussian state-space model.
A, Q, R = 0.9, 0.5, 0.3


def init(key, n):
    return jr.normal(key, (n, 1))


def transition(key, z):
    return A * z + jnp.sqrt(Q) * jr.normal(key, z.shape)


def log_observation(y, z):
    return -0.5 * (jnp.log(2 * jnp.pi * R) + (y[0] - z[0]) ** 2 / R)


def emission(key, z):
    return z + jnp.sqrt(R) * jr.normal(key, z.shape)


_, emissions = smcx.simulate(
    jr.key(1),
    lambda key: init(key, 1)[0],
    transition,
    emission,
    num_timesteps=100,
)

post = smcx.bootstrap_filter(
    jr.key(0),
    init,
    transition,
    log_observation,
    emissions,
    num_particles=10_000,
)
post.marginal_loglik  # unbiased evidence estimate (log-domain)
smcx.diagnose(post)  # ESS / diversity / Pareto-k health summary
```

Callbacks are per-particle; smcx vmaps them internally. Everything
takes an explicit PRNG key, and posteriors are NamedTuples — ordinary
JAX pytrees.

smcx is deliberately just the inference engine: it defines no model
classes and no distributions. Models enter as JAX callables — your
own closures, or thin wrappers around a model library such as
[Dynamax](https://github.com/probml/dynamax) (the test suite itself
uses Dynamax models this way).

## Apple silicon

The `[metal]` extra runs the same code on M-series GPUs via jax-mps.
Filter correctness on Metal is gate-verified in this repository's test
suite (`SMCX_TEST_PLATFORM=mps` runs it on the GPU), and several of the
performance fixes that make the backend fast for SMC-shaped workloads
were contributed upstream from this project (jax-mps #215, #216, #220).
Metal is float32-only; the suite runs float64 on CPU and float32 on
Metal automatically.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/michaelellis003/smcx.git
cd smcx
uv sync
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

A `Makefile` covers common tasks:

```bash
make test        # lint + pytest
make lint        # ruff check, format check, license headers, ty
make format      # add license headers, ruff format, ruff fix
make docs        # build docs
```

Releases are automated: `python-semantic-release` reads conventional
commits on merge to main, bumps the version, tags, and publishes.

## Acknowledgments

smcx's design draws on the SMC ecosystem:
[particles](https://github.com/nchopin/particles) and Chopin &
Papaspiliopoulos's *An Introduction to Sequential Monte Carlo* (the
Feynman-Kac architecture),
[BlackJAX](https://github.com/blackjax-devs/blackjax) (the resampling
contract), [Dynamax](https://github.com/probml/dynamax) (container
conventions), TensorFlow Probability (criterion/trace hooks), and
design lessons from PyMC, FilterPy, pfilter, pyfilter, Stone Soup,
pomp, nimbleSMC, and ArviZ. See `CITATION.cff` for formal references.

## License

Apache-2.0
