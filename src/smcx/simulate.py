# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: typed callback protocols and aligned exogenous inputs.

r"""Forward simulation from a state-space model.

Generates a single trajectory of latent states and observed emissions
by drawing from the initial, transition, and emission distributions
sequentially.  Uses the same callback interface as the particle
filters so that model definitions are reusable.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

from typing import cast

import jax.random as jr
from jax import lax
from jaxtyping import Array, Float

from smcx._utils import _canonicalize_inputs, _prepend
from smcx.types import (
    EmissionSampler,
    EmissionSamplerWithInput,
    InputSequence,
    PRNGKeyT,
    SingleInitialSampler,
    SingleInitialSamplerWithInput,
    TransitionSampler,
    TransitionSamplerWithInput,
)


def simulate(
    key: PRNGKeyT,
    initial_sampler: SingleInitialSampler | SingleInitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    emission_sampler: EmissionSampler | EmissionSamplerWithInput,
    num_timesteps: int,
    *,
    inputs: InputSequence | None = None,
) -> tuple[
    Float[Array, "ntime state_dim"],
    Float[Array, "ntime emission_dim"],
]:
    r"""Simulate a single trajectory from a state-space model.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key[, input_0]) -> state`` that
            draws a single sample from the initial state distribution
            :math:`p(z_1)`.  Unlike the filter interface, this draws
            *one* sample (no ``num_particles`` argument).
        transition_sampler: Function ``(key, state[, input_t]) -> state`` that
            draws from the transition distribution
            :math:`p(z_t \mid z_{t-1})`.
        emission_sampler: Function ``(key, state[, input_t]) -> emission`` that
            draws from the emission distribution
            :math:`p(y_t \mid z_t)`.
        num_timesteps: Number of time steps :math:`T` to simulate.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Input zero reaches initialization and the first
            emission; each later input reaches its aligned transition and
            emission.

    Returns:
        A tuple ``(states, emissions)`` where *states* has shape
        ``(T, state_dim)`` and *emissions* has shape
        ``(T, emission_dim)``.

    Raises:
        ValueError: ``inputs`` is not rank one or two, or its leading
            dimension does not equal ``num_timesteps``.
    """
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    k_init, k_rest = jr.split(key)

    # --- t = 0 --------------------------------------------------------------
    k_z0, k_y0 = jr.split(k_init)
    step_keys = jr.split(k_rest, num_timesteps - 1)
    if inputs_arr is None:
        initial_fn = cast(SingleInitialSampler, initial_sampler)
        transition_fn = cast(TransitionSampler, transition_sampler)
        emission_fn = cast(EmissionSampler, emission_sampler)
        z_0 = initial_fn(k_z0)
        y_0 = emission_fn(k_y0, z_0)

        def _step(
            z_prev: Array,
            step_key: PRNGKeyT,
        ) -> tuple[Array, tuple[Array, Array]]:
            k_z, k_y = jr.split(step_key)
            z_t = transition_fn(k_z, z_prev)
            y_t = emission_fn(k_y, z_t)
            return z_t, (z_t, y_t)

        _, (states_rest, emissions_rest) = lax.scan(_step, z_0, step_keys)
    else:
        initial_fn_u = cast(SingleInitialSamplerWithInput, initial_sampler)
        transition_fn_u = cast(TransitionSamplerWithInput, transition_sampler)
        emission_fn_u = cast(EmissionSamplerWithInput, emission_sampler)
        z_0 = initial_fn_u(k_z0, inputs_arr[0])
        y_0 = emission_fn_u(k_y0, z_0, inputs_arr[0])

        def _step_with_input(
            z_prev: Array,
            args: tuple[PRNGKeyT, Float[Array, " input_dim"]],
        ) -> tuple[Array, tuple[Array, Array]]:
            step_key, input_t = args
            k_z, k_y = jr.split(step_key)
            z_t = transition_fn_u(k_z, z_prev, input_t)
            y_t = emission_fn_u(k_y, z_t, input_t)
            return z_t, (z_t, y_t)

        _, (states_rest, emissions_rest) = lax.scan(
            _step_with_input,
            z_0,
            (step_keys, inputs_arr[1:]),
        )

    # --- Combine t=0 with t=1..T-1 ------------------------------------------
    all_states = _prepend(z_0, states_rest)
    all_emissions = _prepend(y_0, emissions_rest)

    return all_states, all_emissions
