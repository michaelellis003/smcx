# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
r"""Forward simulation from a state-space model.

Generates a single trajectory of latent states and observed emissions
by drawing from the initial, transition, and emission distributions
sequentially.  Uses the same callback interface as the particle
filters so that model definitions are reusable.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

from collections.abc import Callable

import jax.random as jr
from jax import lax
from jaxtyping import Array, Float

from smcjax._utils import _prepend
from smcjax.types import PRNGKeyT


def simulate(
    key: PRNGKeyT,
    initial_sampler: Callable,
    transition_sampler: Callable,
    emission_sampler: Callable,
    num_timesteps: int,
) -> tuple[
    Float[Array, 'ntime state_dim'],
    Float[Array, 'ntime emission_dim'],
]:
    r"""Simulate a single trajectory from a state-space model.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key) -> state`` that draws a
            single sample from the initial state distribution
            :math:`p(z_1)`.  Unlike the filter interface, this draws
            *one* sample (no ``num_particles`` argument).
        transition_sampler: Function ``(key, state) -> state`` that
            draws from the transition distribution
            :math:`p(z_t \mid z_{t-1})`.
        emission_sampler: Function ``(key, state) -> emission`` that
            draws from the emission distribution
            :math:`p(y_t \mid z_t)`.
        num_timesteps: Number of time steps :math:`T` to simulate.

    Returns:
        A tuple ``(states, emissions)`` where *states* has shape
        ``(T, state_dim)`` and *emissions* has shape
        ``(T, emission_dim)``.
    """
    k_init, k_rest = jr.split(key)

    # --- t = 0 --------------------------------------------------------------
    k_z0, k_y0 = jr.split(k_init)
    z_0 = initial_sampler(k_z0)
    y_0 = emission_sampler(k_y0, z_0)

    # --- Scan body for t = 1, ..., T-1 --------------------------------------
    def _step(
        z_prev: Array,
        step_key: PRNGKeyT,
    ) -> tuple[Array, tuple[Array, Array]]:
        k_z, k_y = jr.split(step_key)
        z_t = transition_sampler(k_z, z_prev)
        y_t = emission_sampler(k_y, z_t)
        return z_t, (z_t, y_t)

    step_keys = jr.split(k_rest, num_timesteps - 1)
    _, (states_rest, emissions_rest) = lax.scan(_step, z_0, step_keys)

    # --- Combine t=0 with t=1..T-1 ------------------------------------------
    all_states = _prepend(z_0, states_rest)
    all_emissions = _prepend(y_0, emissions_rest)

    return all_states, all_emissions
