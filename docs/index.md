# smcx

Sequential Monte Carlo for Apple silicon, built on
[MLX](https://github.com/ml-explore/mlx).

smcx runs particle filters and SMC samplers on the GPU of an
M-series Mac. It targets the one thing that hardware does that a
discrete GPU cannot: CPU and GPU share memory, so the host-side
work that particle methods lean on — resampling bookkeeping,
degeneracy checks, full particle histories, diagnostics mid-filter
— costs no transfer. The algorithms are the throughput-shaped,
float32-tolerant, embarrassingly parallel half of Bayesian
computation, which is where this trade pays off. The
[thesis section of the README](https://github.com/michaelellis003/smcx#thesis)
makes the full argument.

## Install

```bash
pip install smcx        # or: uv add smcx
```

The GPU paths need Apple silicon and a recent macOS. The library
imports and runs on Linux and Intel Macs too (CPU backend), which
is what CI exercises; the performance story is Metal-specific.

## A first filter

Track a latent AR(1) state from noisy observations with a bootstrap
particle filter. Everything is log-domain and keyed — the two
non-negotiables of the MLX substrate.

```python
import math
import mlx.core as mx
import smcx

rho, q_sd, r_sd = 0.95, 0.3, 0.7
key_sim, key_filt = mx.random.split(mx.random.key(0))

def initial_sampler(key, n):
    return mx.random.normal((n, 1), key=key)

def transition_sampler(key, state):
    return rho * state + q_sd * mx.random.normal(state.shape, key=key)

def emission_sampler(key, state):
    return state + r_sd * mx.random.normal(state.shape, key=key)

# A log-density for the observation model p(y | state).
def log_observation_fn(y, state):
    z = (y[0] - state[0]) / r_sd
    return -0.5 * z * z - math.log(r_sd * math.sqrt(2 * math.pi))

# Draw a synthetic series, then filter it.
states, observations = smcx.simulate(
    key_sim, initial_sampler, transition_sampler, emission_sampler,
    num_timesteps=100,
)

posterior = smcx.bootstrap_filter(
    key_filt, initial_sampler, transition_sampler, log_observation_fn,
    observations, num_particles=10_000,
)

print(posterior.marginal_loglik.item())    # log p(y_{1:T})
print(smcx.weighted_mean(posterior).shape)  # (100, 1) filtered means
```

The returned `posterior` carries the filtered particles, their
log-weights, the per-step effective sample size, and a running
marginal log-likelihood. `smcx.diagnose(posterior)` summarizes
filter health in one call.

## Where to go next

- [Quickstart](guides/quickstart.md) — the same model end to end:
  simulate, filter, read diagnostics, and switch to a guided
  proposal to cut weight variance.
- [Stochastic volatility](guides/stochastic-volatility.md) — a
  worked example that learns a static parameter online with the
  Liu-West filter.
- [API reference](api/) — every public function, generated from the
  source docstrings.

## Developing smcx

Contributions follow a test-first loop; see
[CONTRIBUTING](https://github.com/michaelellis003/smcx/blob/main/CONTRIBUTING.md).
In brief:

```bash
git clone https://github.com/michaelellis003/smcx.git
cd smcx
uv sync
uv run pre-commit install
uv run pytest
```

The `Makefile` collects the common tasks: `make test` (lint then
pytest), `make lint`, `make format`, `make docs`.

## License

Apache-2.0. See
[LICENSE](https://github.com/michaelellis003/smcx/blob/main/LICENSE).
