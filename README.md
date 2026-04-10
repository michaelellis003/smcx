# smcjax

[![CI](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelellis003/smcjax/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/smcjax)](https://pypi.org/project/smcjax/)
[![License](https://img.shields.io/github/license/michaelellis003/smcjax)](LICENSE)

Sequential Monte Carlo and particle filtering in JAX.

**smcjax** is a JAX implementation of the methods developed in my
[master's thesis](https://github.com/michaelellis003/sequential-monte-carlo-hmm)
on sequential inference for Hidden Markov Models (University of
Arkansas, 2018). It extends [Dynamax](https://github.com/probml/dynamax)
and [BlackJAX](https://github.com/blackjax-devs/blackjax) with particle
filters and Bayesian workflow diagnostics that neither library provides.
All filters are JIT-compiled via `jax.lax.scan` and GPU-ready.

## Features

- **Bootstrap (SIR) particle filter** — Gordon *et al.* (1993)
- **Auxiliary particle filter** — Pitt & Shephard (1999)
- **Liu-West filter** — joint state-parameter estimation via kernel
  density smoothing (Liu & West, 2001)
- **Forward simulation** — generate trajectories from state-space models
- **Diagnostics** — weighted mean/variance/quantiles, parameter
  summaries, ESS traces, particle diversity, per-step log evidence
  increments, replicated log-ML, log Bayes factors, CRPS
- **4 resampling schemes** (via BlackJAX): systematic, stratified,
  multinomial, residual
- **Conditional resampling** with configurable ESS threshold
- All functions are `jit`- and `vmap`-compatible
- Type annotations via [jaxtyping](https://github.com/google/jaxtyping)

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) installed

## Installation

```bash
pip install smcjax
```

Or from source:

```bash
git clone https://github.com/michaelellis003/smcjax.git
cd smcjax
uv sync
```

Install the pre-commit hooks (one-time setup):

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

## Quick example

```python
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

from smcjax import bootstrap_filter, weighted_mean, log_ml_increments

# Define a 1-D linear Gaussian state space model
m0, P0 = jnp.array([0.0]), jnp.array([[1.0]])
F, Q = jnp.array([[0.9]]), jnp.array([[0.25]])
H, R = jnp.array([[1.0]]), jnp.array([[1.0]])

chol_P0 = jnp.linalg.cholesky(P0)
chol_Q = jnp.linalg.cholesky(Q)


def initial_sampler(key, n):
    return m0 + jr.normal(key, (n, 1)) @ chol_P0.T


def transition_sampler(key, state):
    mean = (F @ state[:, None]).squeeze(-1)
    return mean + jr.normal(key, (1,)) @ chol_Q.T


def log_observation_fn(emission, state):
    mean = (H @ state[:, None]).squeeze(-1)
    return jstats.multivariate_normal.logpdf(emission, mean, R)


# Simulate some data
key = jr.PRNGKey(0)
T = 100
emissions = jr.normal(key, (T, 1))

# Run the bootstrap particle filter
posterior = bootstrap_filter(
    key=jr.PRNGKey(1),
    initial_sampler=initial_sampler,
    transition_sampler=transition_sampler,
    log_observation_fn=log_observation_fn,
    emissions=emissions,
    num_particles=1_000,
)

print(f"Log marginal likelihood: {posterior.marginal_loglik:.2f}")
print(f"Particles shape: {posterior.filtered_particles.shape}")
print(f"Mean ESS: {posterior.ess.mean():.1f}")

# Diagnostics
means = weighted_mean(posterior)
increments = log_ml_increments(posterior)
```

## Architecture

```
src/smcjax/
    __init__.py          # Public API (re-exports BlackJAX ESS & resampling)
    types.py             # PRNGKeyT, Scalar (matches Dynamax)
    containers.py        # ParticleState, ParticleFilterPosterior, LiuWestPosterior
    weights.py           # log_normalize, normalize
    bootstrap.py         # Bootstrap (SIR) particle filter
    auxiliary.py         # Auxiliary particle filter (Pitt & Shephard 1999)
    liu_west.py          # Liu-West filter for joint state-parameter estimation
    simulate.py          # Forward simulation from state-space models
    diagnostics.py       # Posterior summaries, model comparison, scoring rules
```

ESS and resampling (systematic, stratified, multinomial, residual) are
provided by [BlackJAX](https://github.com/blackjax-devs/blackjax) and
re-exported from `smcjax` for convenience.

## Cross-validation

All filters are tested against reference libraries:

| Module | Reference | Validation |
|--------|-----------|------------|
| `bootstrap` | [Dynamax](https://github.com/probml/dynamax) Kalman filter | Log-ML within 5% of exact |
| `auxiliary` | Dynamax Kalman filter | Log-ML within 5% of exact |
| `auxiliary` | Bootstrap (flat auxiliary = bootstrap) | Log-ML within 3 nats |
| `liu_west` | Auxiliary filter (fixed params) | Log-ML within 5 nats |

## Notebooks

The `notebooks/` directory contains a thesis-style Bayesian workflow
reproduction using a Hidden Markov Model with unknown parameters,
demonstrating the pipeline: simulation, Liu-West filtering, parameter
recovery, model comparison via log Bayes factors, and CRPS evaluation.

## Roadmap

| Phase | What | Status |
|-------|------|--------|
| 1 | Bootstrap particle filter | Done |
| 2 | Auxiliary particle filter | Done |
| 3 | Forward simulation + diagnostics | Done |
| 4 | Liu-West filter + model comparison | Done |
| 5 | EKF/UKF proposal particle filters | Planned |
| 6 | Particle MCMC (PMMH) | Planned |

## Development

A `Makefile` collects the common development tasks:

```bash
make test        # lint + pytest
make lint        # ruff check, format check, license headers, ty
make format      # add license headers, ruff format, ruff fix
make license     # add missing license headers
make docs        # build documentation
make serve-docs  # serve documentation locally
make install     # uv sync
make clean       # git clean (preserves .venv)
```

## How releases work

Releases are fully automated. When a commit lands on `main` and CI
passes, `python-semantic-release` inspects the commit history to
determine whether a version bump is warranted:

- `fix: ...` produces a patch release
- `feat: ...` produces a minor release
- A `BREAKING CHANGE` footer or `!` suffix produces a major release

## License

Apache-2.0. See [LICENSE](LICENSE) for the full text.
