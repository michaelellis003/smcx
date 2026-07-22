# Quickstart

This example simulates a one-dimensional linear-Gaussian model, runs a
bootstrap filter, and then compares it with a guided proposal. Run the code
blocks in order in one Python session.

## The model

We use a stationary AR(1) latent state with Gaussian observations,

$$
z_t = \rho\,z_{t-1} + \sigma_q\,\varepsilon_t, \qquad
y_t = z_t + \sigma_r\,\eta_t,
$$

with $\rho = 0.95$, $\sigma_q = 0.3$, and $\sigma_r = 0.7$, so the
observation noise dominates the process noise and filtering has real
work to do. A model in smcx is a handful of closures, written
*per particle*: each takes one state array or PyTree and smcx `vmap`s it
over the particle cloud internally. Samplers take an explicit PRNG key;
the observation model returns a log-density.

```python
import math

import jax.random as jr
import numpy as np
import smcx

rho, q_sd, r_sd = 0.95, 0.3, 0.7
key_sim, key_filt = jr.split(jr.key(0))


def initial_sampler(key, n):
    return jr.normal(key, (n, 1))


def transition_sampler(key, state):
    return rho * state + q_sd * jr.normal(key, state.shape)


def emission_sampler(key, state):
    return state + r_sd * jr.normal(key, state.shape)


def log_observation_fn(y, state):
    z = (y[0] - state[0]) / r_sd
    return -0.5 * z * z - math.log(r_sd * math.sqrt(2 * math.pi))
```

The key threading is JAX's standard functional PRNG: no global seed,
no hidden state, and any run is reproducible from its key. smcx
splits the key it receives across time steps and particles for you.

## Simulate

`simulate` runs the same transition/emission closures forward to
produce a latent path and the observations we will filter. It draws
a single trajectory, so its initial sampler takes just a key — a
lambda adapts the filter-style sampler:

```python
states, observations = smcx.simulate(
    key_sim,
    lambda key: initial_sampler(key, 1)[0],
    transition_sampler,
    emission_sampler,
    num_timesteps=100,
)
```

## Filter

The bootstrap filter proposes from the transition and weights by the
observation density.

```python
posterior = smcx.bootstrap_filter(
    key_filt,
    initial_sampler,
    transition_sampler,
    log_observation_fn,
    observations,
    num_particles=10_000,
)

means = smcx.weighted_mean(posterior)
rmse = float(np.sqrt(np.mean((np.array(means) - np.array(states)) ** 2)))
print("marginal loglik:", round(posterior.marginal_loglik.item(), 1))
print("filter RMSE:", round(rmse, 3), "observation sd:", round(r_sd, 3))
```

For the fixed seed above, the filtered RMSE is about 0.369, compared with
an observation-noise scale of 0.7.

## Diagnose

`diagnose` returns a dictionary of health summaries and a list of
plain-language warnings. It runs host-side and is not meant for the
hot loop — call it after filtering.

```python
report = smcx.diagnose(posterior)
print("min ESS:", round(report["min_ess"], 1))
print("max Pareto-k:", round(report["max_pareto_k"], 2))
for w in report["warnings"]:
    print("warning:", w)
```

For this run, no warnings are returned. The diagnostics describe the particle
weights; they do not measure the Monte Carlo variance of the evidence estimate.

## Cut the variance with a guided proposal

The bootstrap proposal ignores the current observation. The guided
filter proposes from the *locally optimal* density
$p(z_t \mid z_{t-1}, y_t)$, which for this linear-Gaussian model is
available in closed form. The proposal precision is the sum of the
process and observation precisions:

```python
prop_var = 1.0 / (1.0 / q_sd**2 + 1.0 / r_sd**2)
prop_sd = math.sqrt(prop_var)


def proposal_sampler(key, state, y):
    mean = prop_var * (rho * state / q_sd**2 + y[0] / r_sd**2)
    return mean + prop_sd * jr.normal(key, state.shape)


def log_proposal_fn(y, new_state, old_state):
    mean = prop_var * (rho * old_state[0] / q_sd**2 + y[0] / r_sd**2)
    z = (new_state[0] - mean) / prop_sd
    return -0.5 * z * z - math.log(prop_sd * math.sqrt(2 * math.pi))


def log_transition_fn(new_state, old_state):
    z = (new_state[0] - rho * old_state[0]) / q_sd
    return -0.5 * z * z - math.log(q_sd * math.sqrt(2 * math.pi))


guided = smcx.guided_filter(
    key_filt,
    initial_sampler,
    proposal_sampler,
    log_proposal_fn,
    log_transition_fn,
    log_observation_fn,
    observations,
    num_particles=10_000,
)

g = smcx.diagnose(guided)
print("guided marginal loglik:", round(guided.marginal_loglik.item(), 1))
print("guided min ESS:", round(g["min_ess"], 1))
```

`replicated_log_ml` runs independent filters without retaining particle
histories:

```python
def boot_lml(key):
    return smcx.bootstrap_filter(
        key,
        initial_sampler,
        transition_sampler,
        log_observation_fn,
        observations,
        num_particles=10_000,
        store_history=False,
    ).marginal_loglik


def guided_lml(key):
    return smcx.guided_filter(
        key,
        initial_sampler,
        proposal_sampler,
        log_proposal_fn,
        log_transition_fn,
        log_observation_fn,
        observations,
        num_particles=10_000,
        store_history=False,
    ).marginal_loglik


lml_b = smcx.replicated_log_ml(jr.key(7), boot_lml, 20)
lml_g = smcx.replicated_log_ml(jr.key(7), guided_lml, 20)
print("bootstrap log-ML sd:", round(float(np.std(np.array(lml_b))), 3))
print("guided    log-ML sd:", round(float(np.std(np.array(lml_g))), 3))
```

With these seeds, the minimum ESS is about 1,122 for the bootstrap filter and
1,480 for the guided filter. Across the 20 replications above, the standard
deviation of the log-evidence estimate is 0.074 and 0.053 respectively. These
numbers describe this example, not a general performance guarantee.

## What next

- The [stochastic volatility guide](stochastic-volatility.md) adds an
  unknown static parameter and learns it online.
- The [custom-model guide](custom-models.md) covers structured latent states
  and time-varying inputs.
- `bootstrap_init`, `bootstrap_step`, and `bootstrap_update` support
  checkpointed or chunked filtering.
- Every function used here — `bootstrap_filter`, `guided_filter`,
  `simulate`, `diagnose`, `weighted_mean`, `replicated_log_ml` — has
  a full contract in the [API reference](../api/smcx/index.md).
