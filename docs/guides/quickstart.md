# Quickstart

This guide filters a one-dimensional linear-Gaussian state-space
model from end to end: simulate data, run a bootstrap filter, read
the diagnostics, then swap in a guided proposal and watch the weight
variance drop. Every code block runs as written; paste them in order
into one session.

## The model

We use a stationary AR(1) latent state with Gaussian observations,

$$
z_t = \rho\,z_{t-1} + \sigma_q\,\varepsilon_t, \qquad
y_t = z_t + \sigma_r\,\eta_t,
$$

with $\rho = 0.95$, $\sigma_q = 0.3$, and $\sigma_r = 0.7$, so the
observation noise dominates the process noise and filtering has real
work to do. A model in smcx is a handful of closures. Samplers take
an explicit key and return arrays; the observation model returns a
log-density.

```python
import math
import mlx.core as mx
import numpy as np
import smcx

rho, q_sd, r_sd = 0.95, 0.3, 0.7
key_sim, key_filt = mx.random.split(mx.random.key(0))

def initial_sampler(key, n):
    return mx.random.normal((n, 1), key=key)

def transition_sampler(key, state):
    return rho * state + q_sd * mx.random.normal(state.shape, key=key)

def emission_sampler(key, state):
    return state + r_sd * mx.random.normal(state.shape, key=key)

def log_observation_fn(y, state):
    z = (y[0] - state[0]) / r_sd
    return -0.5 * z * z - math.log(r_sd * math.sqrt(2 * math.pi))
```

The key threading is load-bearing, not ceremony: random routines
inside a compiled MLX step are frozen unless you pass a key
explicitly, so smcx makes the key an argument everywhere.

## Simulate

`simulate` runs the same initial/transition/emission closures forward
to produce a latent path and the observations we will filter.

```python
states, observations = smcx.simulate(
    key_sim, initial_sampler, transition_sampler, emission_sampler,
    num_timesteps=100,
)
```

## Filter

The bootstrap filter proposes from the transition and weights by the
observation density. Ten thousand particles is comfortable; on an
M-series GPU this is a few milliseconds.

```python
posterior = smcx.bootstrap_filter(
    key_filt, initial_sampler, transition_sampler, log_observation_fn,
    observations, num_particles=10_000,
)

means = smcx.weighted_mean(posterior)
rmse = float(np.sqrt(np.mean((np.array(means) - np.array(states)) ** 2)))
print("marginal loglik:", round(posterior.marginal_loglik.item(), 1))
print("filter RMSE:", round(rmse, 3),
      "vs obs-only RMSE:", round(r_sd, 3))
```

The filtered mean tracks the latent state at an RMSE well below the
observation noise $\sigma_r = 0.7$ — the filter is extracting signal,
not echoing the data.

## Diagnose

`diagnose` returns a dictionary of health summaries and a list of
plain-language warnings. It runs host-side in float64 (a deliberate
escape from the float32 GPU stream) and is not meant for the hot
loop — call it after filtering.

```python
report = smcx.diagnose(posterior)
print("min ESS:", round(report["min_ess"], 1))
print("max Pareto-k:", round(report["max_pareto_k"], 2))
for w in report["warnings"]:
    print("warning:", w)
```

The effective sample size dips below ten percent of $N$ at a few
steps — the sharp observations occasionally concentrate the weights
— but the Pareto-$k$ tail index of the importance weights stays well
under the 0.7 reliability threshold. That is the signature of a
filter that is working but leaving variance-reduction on the table.

## Cut the variance with a guided proposal

The bootstrap proposal ignores the current observation, so at a sharp
likelihood most proposed particles land in low-weight regions. The
guided filter proposes from the *locally optimal* density
$p(z_t \mid z_{t-1}, y_t)$, which for this linear-Gaussian model is
available in closed form. The proposal precision is the sum of the
process and observation precisions:

```python
prop_var = 1.0 / (1.0 / q_sd**2 + 1.0 / r_sd**2)
prop_sd = math.sqrt(prop_var)

def proposal_sampler(key, state, y):
    mean = prop_var * (rho * state / q_sd**2 + y[0] / r_sd**2)
    return mean + prop_sd * mx.random.normal(state.shape, key=key)

def log_proposal_fn(y, new_state, old_state):
    mean = prop_var * (rho * old_state[0] / q_sd**2 + y[0] / r_sd**2)
    z = (new_state[0] - mean) / prop_sd
    return -0.5 * z * z - math.log(prop_sd * math.sqrt(2 * math.pi))

def log_transition_fn(new_state, old_state):
    z = (new_state[0] - rho * old_state[0]) / q_sd
    return -0.5 * z * z - math.log(q_sd * math.sqrt(2 * math.pi))

guided = smcx.guided_filter(
    key_filt, initial_sampler, proposal_sampler, log_proposal_fn,
    log_transition_fn, log_observation_fn, observations,
    num_particles=10_000,
)

g = smcx.diagnose(guided)
print("guided min ESS:", round(g["min_ess"], 1))
print("guided max Pareto-k:", round(g["max_pareto_k"], 2))
print("guided warnings:", g["warnings"])
```

The guided filter more than doubles the minimum effective sample
size and drops the worst-case Pareto-$k$ by a wide margin — the
warnings clear — while returning a marginal likelihood statistically
indistinguishable from the bootstrap estimate. Same target, tighter
weights. That is the whole point of a better proposal, and smcx keeps
the two filters behind the same call shape so switching costs one
function name and three densities.

## What next

- The [stochastic volatility guide](stochastic-volatility.md) adds an
  unknown static parameter and learns it online.
- Every function used here — `bootstrap_filter`, `guided_filter`,
  `simulate`, `diagnose`, `weighted_mean` — has a full contract in
  the [API reference](../api/).
