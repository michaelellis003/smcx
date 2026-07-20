# Quickstart

This guide filters a one-dimensional linear-Gaussian state-space
model from end to end: simulate data, run a bootstrap filter, read
the diagnostics, then swap in a guided proposal and watch the
evidence-estimate variance drop. Every code block runs as written;
paste them in order into one session.

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
from typing import NamedTuple

import jax
import jax.numpy as jnp
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
observation density. Ten thousand particles is comfortable: the whole
time loop is one `lax.scan`, compiled once.

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
print("filter RMSE:", round(rmse, 3), "vs obs-only RMSE:", round(r_sd, 3))
```

The filtered RMSE comes out near 0.37, roughly half the observation
noise $\sigma_r = 0.7$ — the filter is extracting signal, not echoing
the data.

## Filter a stream in chunks

Use a checkpoint when observations arrive incrementally. Initialization
consumes the first observation; every later observation has its own explicit
key, so changing chunk boundaries cannot reorder randomness. Here the same
series is processed in two chunks:

```python
step_root, init_key = jr.split(key_filt)
step_keys = jr.split(step_root, observations.shape[0] - 1)
checkpoint, _ = smcx.bootstrap_init(
    init_key,
    initial_sampler,
    log_observation_fn,
    observations[0],
    num_particles=10_000,
)
checkpoint, early = smcx.bootstrap_update(
    step_keys[:49],
    checkpoint,
    transition_sampler,
    log_observation_fn,
    observations[1:50],
)
checkpoint, late = smcx.bootstrap_update(
    step_keys[49:],
    checkpoint,
    transition_sampler,
    log_observation_fn,
    observations[50:],
)
```

`early` and `late` contain only their chunk histories and conditional
log-evidence. The checkpoint retains the live particles, normalized weights,
ESS, and compensated cumulative evidence. Use `bootstrap_step` with one key
and observation for event-at-a-time processing. Input-aware models pass
`input_t=inputs[0]` to initialization and the aligned input slice to each
update.

## Carry a structured latent state

The bootstrap, auxiliary, and guided filters can carry any nonempty JAX
PyTree of arrays. This is useful when a particle contains several values
with different shapes—for example, a nonlinear state together with the
conditional mean and covariance from a Kalman update. The initial
sampler adds the same leading particle axis to every leaf; callbacks see
one particle with that axis removed.

```python
class CompositeState(NamedTuple):
    signal: jax.Array
    conditional_mean: jax.Array
    conditional_covariance: jax.Array


def structured_initial(key, n):
    signal = jr.normal(key, (n, 1))
    return CompositeState(
        signal=signal,
        conditional_mean=jnp.zeros((n, 2)),
        conditional_covariance=jnp.broadcast_to(jnp.eye(2), (n, 2, 2)),
    )


def structured_transition(key, state):
    signal = rho * state.signal + q_sd * jr.normal(key, state.signal.shape)
    mean = 0.9 * state.conditional_mean + 0.1 * signal
    return CompositeState(signal, mean, state.conditional_covariance)


def structured_log_observation(y, state):
    z = (y[0] - state.signal[0]) / r_sd
    return -0.5 * z * z - math.log(r_sd * math.sqrt(2 * math.pi))


structured = smcx.bootstrap_filter(
    jr.key(2),
    structured_initial,
    structured_transition,
    structured_log_observation,
    observations,
    num_particles=10_000,
)

print(structured.filtered_particles.signal.shape)  # (100, 10000, 1)
print(structured.filtered_particles.conditional_covariance.shape)
# (100, 10000, 2, 2)
```

The tree structure, leaf event shapes, and dtypes stay fixed for the
whole run. smcx applies each resampling decision jointly to every leaf,
so the signal and its conditional moments cannot lose particle identity.
`smcx.reconstruct_trajectories(structured)` returns the same tree with
the same leaf shapes.

Diagnostics that need Euclidean state arithmetic intentionally accept a
dense `(T, N, D)` history. Select or project the state you want to
summarize and reuse the posterior metadata:

```python
signal_posterior = structured._replace(
    filtered_particles=structured.filtered_particles.signal
)
signal_means = smcx.weighted_mean(signal_posterior)
```

Liu–West, tempered SMC, and SMC² remain dense because their parameter
proposals require an explicit Euclidean geometry. Equinox modules and
other registered PyTree classes work at the callable boundary, but smcx
does not require Equinox or define model classes of its own.

## Add time-varying inputs

Controlled dynamics and covariate-driven observations use the
keyword-only `inputs` channel. An input sequence has shape `(T, U)`;
a scalar sequence `(T,)` becomes `(T, 1)`. At time zero, `inputs[0]`
reaches the initial-state and observation callbacks. At each later
time t, `inputs[t]` reaches the transition into t and the observation
at t. Every input-aware callback takes `input_t` last, even when that
callback does not use it.

Here a known control shifts the latent dynamics. The same input-aware
model generates and filters the data:

```python
controls = jnp.sin(jnp.linspace(0.0, 4.0 * jnp.pi, 100))[:, None]
control_scale = 0.2
key_control_sim, key_control_filt = jr.split(jr.key(1))


def controlled_initial(key, input_0):
    return jr.normal(key, (1,)) + control_scale * input_0


def controlled_initial_cloud(key, n, input_0):
    return jr.normal(key, (n, 1)) + control_scale * input_0


def controlled_transition(key, state, input_t):
    noise = q_sd * jr.normal(key, state.shape)
    return rho * state + control_scale * input_t + noise


def controlled_emission(key, state, input_t):
    del input_t
    return state + r_sd * jr.normal(key, state.shape)


def controlled_log_observation(y, state, input_t):
    del input_t
    z = (y[0] - state[0]) / r_sd
    return -0.5 * z * z - math.log(r_sd * math.sqrt(2 * math.pi))


controlled_states, controlled_observations = smcx.simulate(
    key_control_sim,
    controlled_initial,
    controlled_transition,
    controlled_emission,
    num_timesteps=100,
    inputs=controls,
)
controlled_posterior = smcx.bootstrap_filter(
    key_control_filt,
    controlled_initial_cloud,
    controlled_transition,
    controlled_log_observation,
    controlled_observations,
    num_particles=10_000,
    inputs=controls,
)
```

The auxiliary and guided filters follow the same input-last rule.
Liu–West keeps parameters first, so its callbacks end in
`(..., params, input_t)`; its parameter initializer remains
`(key, num_particles)`.

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

Both diagnostics come back clean: the effective sample size stays
above ten percent of $N$ at every step, and the Pareto-$k$ tail
index of the importance weights sits far below the 0.7 reliability
threshold, so no warnings fire. Clean weight diagnostics do not make
the bootstrap proposal free, though — ignoring the current
observation still costs variance in the evidence estimate, and that
is where a better proposal earns its keep.

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

The guided filter lifts the worst-case ESS by about a third while
returning the same marginal likelihood. The gain is real but bounded
by the model itself: with $\sigma_r \approx 2\sigma_q$ the optimal
proposal's standard deviation (0.28) barely differs from the
transition's (0.3), so no proposal choice can make the weights flat
here. Where the improvement shows clearly is in the variance of the
evidence estimate. `replicated_log_ml` vmaps the whole filter over
independent keys, and `store_history=False` keeps the replicated run
at O(N) memory:

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

The guided filter cuts the standard deviation of the log-evidence
estimate from about 0.074 to 0.053 — half the variance for the price
of three extra densities. Same target, tighter estimate; and smcx
keeps the two filters behind the same call shape, so switching costs
one function name.

## What next

- The [stochastic volatility guide](stochastic-volatility.md) adds an
  unknown static parameter and learns it online.
- Every function used here — `bootstrap_filter`, `guided_filter`,
  `simulate`, `diagnose`, `weighted_mean`, `replicated_log_ml` — has
  a full contract in the [API reference](../api/).
