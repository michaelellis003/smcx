---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Filtering a simulated series

This tutorial simulates a noisy time series and recovers its latent state
with a bootstrap particle filter. The documentation build runs every code
cell and stops if one fails, so the output below belongs to the installed
version of smcx.

Running the source locally also requires Matplotlib and IPython.

The latent process is a stationary AR(1),

$$
z_t = \rho z_{t-1} + \sigma_q \varepsilon_t,
\qquad
y_t = z_t + \sigma_r \eta_t,
$$

where both innovations are standard normal. Here $\sigma_r > \sigma_q$, so
individual observations are noisier than the state innovations.

```python
import math

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

import smcx

rho = 0.95
q_sd = 0.3
r_sd = 0.7
initial_sd = q_sd / math.sqrt(1.0 - rho**2)
num_timesteps = 60
num_particles = 2_000
```

smcx model functions operate on one particle at a time. The library maps
them over the particle cloud and threads explicit JAX random keys through
the simulation and filter.

```python
def initial_sampler(key, n):
    return initial_sd * jr.normal(key, (n, 1))


def transition_sampler(key, state):
    noise = q_sd * jr.normal(key, state.shape)
    return rho * state + noise


def emission_sampler(key, state):
    noise = r_sd * jr.normal(key, state.shape)
    return state + noise


def log_observation_fn(observation, state):
    residual = (observation[0] - state[0]) / r_sd
    normalizer = math.log(r_sd * math.sqrt(2.0 * math.pi))
    return -0.5 * residual**2 - normalizer
```

Split one root key so that simulation and inference use independent random
streams. The simulation retains its latent states because they let us inspect
this one fixed run; real data would provide only `observations`.

```python
key_simulation, key_filter = jr.split(jr.key(2026))

states, observations = smcx.simulate(
    key_simulation,
    lambda key: initial_sampler(key, 1)[0],
    transition_sampler,
    emission_sampler,
    num_timesteps=num_timesteps,
)

posterior = smcx.bootstrap_filter(
    key_filter,
    initial_sampler,
    transition_sampler,
    log_observation_fn,
    observations,
    num_particles=num_particles,
)
```

Weighted summaries preserve the unequal particle weights. Here the interval
uses the 5th and 95th weighted quantiles at each time step.

```python
state = np.asarray(states)[:, 0]
observation = np.asarray(observations)[:, 0]
mean = np.asarray(smcx.weighted_mean(posterior))[:, 0]
interval = np.asarray(
    smcx.weighted_quantile(posterior, jnp.array([0.05, 0.95]))
)[:, :, 0]
ess_fraction = np.asarray(posterior.ess) / num_particles

rmse = float(np.sqrt(np.mean((mean - state) ** 2)))
print(f"marginal log likelihood: {posterior.marginal_loglik.item():.2f}")
print(f"filtered-mean RMSE: {rmse:.3f}")
print(f"minimum ESS / N: {ess_fraction.min():.3f}")
```

```python
from base64 import b64encode
from io import BytesIO

from IPython.display import Image, display

time = np.arange(num_timesteps)
figure, (state_axis, ess_axis) = plt.subplots(
    2,
    1,
    figsize=(8, 5.5),
    sharex=True,
    gridspec_kw={"height_ratios": (3, 1)},
    constrained_layout=True,
)

state_axis.scatter(time, observation, color="0.65", s=12, label="observation")
state_axis.plot(time, state, color="black", linewidth=1.3, label="latent state")
state_axis.plot(time, mean, color="C0", linewidth=1.6, label="filtered mean")
state_axis.fill_between(
    time,
    interval[:, 0],
    interval[:, 1],
    color="C0",
    alpha=0.2,
    label="central 90% interval",
)
state_axis.set_ylabel("state")
state_axis.legend(frameon=False, ncols=2)

ess_axis.plot(time, ess_fraction, color="C1")
ess_axis.axhline(
    0.5,
    color="0.4",
    linestyle="--",
    linewidth=1,
    label="next-step threshold",
)
ess_axis.set(xlabel="time", ylabel="ESS / N", ylim=(0, 1.02))
ess_axis.legend(frameon=False, loc="lower right")

image_data = BytesIO()
figure.savefig(image_data, format="png", dpi=120)
plt.close(figure)
encoded_image = b64encode(image_data.getvalue()).decode()
image_url = "data:image/png;base64," + encoded_image
display(
    Image(
        url=image_url,
        alt=(
            "Noisy observations, simulated latent state, filtered mean, and "
            "central 90% filtering interval above; ESS fraction and the "
            "next-step resampling threshold below."
        ),
    )
)
```

Each vertical slice of the shaded band approximates the filtering marginal
$p(z_t \mid y_{0:t})$. It is not a smoothing interval or a simultaneous band
for the entire path. The displayed truth and RMSE describe this simulated
series only; they are not a coverage study. ESS is recorded after weighting
by $y_t$; a value below 0.5 tells the filter to resample before propagating
particles to $t + 1$.
