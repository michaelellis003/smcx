# smcx

Sequential Monte Carlo in [JAX](https://github.com/jax-ml/jax):
particle filters, adaptive tempered SMC, and SMC² with a small, flat
API. Runs on CPU, CUDA, and TPU through stock JAX, and on
Apple-silicon GPUs through the optional
[jax-mps](https://github.com/tillahoffmann/jax-mps) backend.

## Install

```bash
pip install smcx            # CPU / CUDA / TPU via your jax install
pip install "smcx[metal]"   # + jax-mps for Apple-silicon GPUs
```

## A first filter

Track a latent AR(1) state from noisy observations with a bootstrap
particle filter. Model closures are per particle — smcx `vmap`s them
internally — and every sampler takes an explicit PRNG key.

```python
import math
import jax.random as jr
import smcx

rho, q_sd, r_sd = 0.95, 0.3, 0.7
key_sim, key_filt = jr.split(jr.key(0))


def initial_sampler(key, n):
    return jr.normal(key, (n, 1))


def transition_sampler(key, state):
    return rho * state + q_sd * jr.normal(key, state.shape)


def emission_sampler(key, state):
    return state + r_sd * jr.normal(key, state.shape)


# A log-density for the observation model p(y | state).
def log_observation_fn(y, state):
    z = (y[0] - state[0]) / r_sd
    return -0.5 * z * z - math.log(r_sd * math.sqrt(2 * math.pi))


# Draw a synthetic series, then filter it.
states, observations = smcx.simulate(
    key_sim,
    lambda key: initial_sampler(key, 1)[0],
    transition_sampler,
    emission_sampler,
    num_timesteps=100,
)

posterior = smcx.bootstrap_filter(
    key_filt,
    initial_sampler,
    transition_sampler,
    log_observation_fn,
    observations,
    num_particles=10_000,
)

print(posterior.marginal_loglik.item())  # log p(y_{1:T})
print(smcx.weighted_mean(posterior).shape)  # (100, 1) filtered means
```

The returned `posterior` carries the filtered particles, their
log-weights, the per-step effective sample size, and a running
marginal log-likelihood. `smcx.diagnose(posterior)` summarizes
filter health in one call.

The bootstrap, auxiliary, and guided filters, plus `simulate`, can carry
a nonempty latent-state PyTree. Every particle-cloud leaf has leading
axis `N`, every history leaf has leading axes `(T, N)`, and one ancestor
selection is applied jointly across all leaves. Trajectory reconstruction
and posterior prediction preserve that structure. Euclidean summaries
such as `weighted_mean`, `tail_ess`, and `diagnose` require a dense state
history; select or project a leaf before using them on a structured
posterior. See the
[structured-state quickstart](guides/quickstart.md#carry-a-structured-latent-state).

Beyond filtering, `smcx.temper` targets a static posterior with
adaptive tempered SMC, and `smcx.smc2` nests particle filters inside
an SMC sampler for full state-and-parameter inference.

smcx is deliberately just the inference engine: no model classes, no
distributions. Models enter as JAX callables — your own closures, or
thin wrappers around a model library such as
[Dynamax](https://github.com/probml/dynamax).

## Where to go next

- [Quickstart](guides/quickstart.md) — the same model end to end:
  simulate, filter, read diagnostics, and switch to a guided
  proposal to cut evidence-estimate variance.
- [Stochastic volatility](guides/stochastic-volatility.md) — a
  worked example that learns a static parameter online with the
  Liu-West filter.
- [Regime-switching HMM](examples/thesis_hmm_bayesian_workflow.ipynb)
  — a full Bayesian workflow on S&P 500 returns: prior predictive
  checks, NUTS warm-start, Liu-West filtering, SBC, forecasting
  with proper scoring rules, and a float32 precision study.
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
