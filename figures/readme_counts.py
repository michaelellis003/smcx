# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Generate the README particle-filter figure.

Simulates the README's Poisson count model (log-intensity AR(1) with
rho = 0.9, sigma = 0.4), runs the bootstrap filter, and plots the
observed counts with the filtered intensity and a 90% band. Rerun
from the repository root when the API or the model changes:

    JAX_PLATFORMS=cpu uv run python docs/figures/readme_counts.py
"""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

import smcx

RHO, SIGMA, T = 0.9, 0.4, 60


def sample_initial(key, params, input_0):
    """Draw the initial log-intensity."""
    return jr.normal(key, (1,))


def sample_transition(key, state, params, input_t):
    """Advance the log-intensity one step."""
    noise = params["sigma"] * jr.normal(key, state.shape)
    return params["rho"] * state + noise


def log_observation(count, state, params, input_t):
    """Poisson log density up to a count-only constant."""
    return count[0] * state[0] - jnp.exp(state[0])


def simulate(key):
    """Simulate the latent path and the observed counts."""
    keys = jr.split(key, 2 * T + 1)
    state = float(jr.normal(keys[0]))
    states, counts = [], []
    for t in range(T):
        state = RHO * state + SIGMA * float(jr.normal(keys[1 + 2 * t]))
        rate = jnp.exp(state)
        count = int(jr.poisson(keys[2 + 2 * t], rate))
        states.append(state)
        counts.append(count)
    return jnp.asarray(states), jnp.asarray(counts, dtype=jnp.int32)


def main():
    """Run the bootstrap filter and write the README figure."""
    _, counts = simulate(jr.key(3))
    model = smcx.StateSpaceModel(
        sample_initial=sample_initial,
        sample_transition=sample_transition,
        log_observation=log_observation,
    )
    params = {"rho": jnp.asarray(RHO), "sigma": jnp.asarray(SIGMA)}
    fk = smcx.bootstrap_fk(model, params, counts[:, None])
    posterior = smcx.run_smc(jr.key(0), fk, num_particles=8_192)

    # Quantiles of the log-intensity map exactly to quantiles of the
    # intensity through exp; the mean would not.
    quantiles = smcx.weighted_quantile(posterior, jnp.array([0.05, 0.5, 0.95]))
    lower = quantiles[:, 0, 0]
    median = quantiles[:, 1, 0]
    upper = quantiles[:, 2, 0]

    time = jnp.arange(T)
    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=110)
    ax.fill_between(
        time,
        jnp.exp(lower),
        jnp.exp(upper),
        color="#4878cf",
        alpha=0.25,
        linewidth=0,
        label="90% band, latent intensity",
    )
    ax.plot(
        time,
        jnp.exp(median),
        color="#4878cf",
        linewidth=1.8,
        label="latent intensity, filtered median",
    )
    ax.scatter(
        time,
        counts,
        color="#333333",
        s=14,
        zorder=3,
        label="observed counts",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("count")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        "docs/figures/readme-counts.png",
        facecolor="white",
        bbox_inches="tight",
    )
    print("wrote docs/figures/readme-counts.png")


if __name__ == "__main__":
    main()
