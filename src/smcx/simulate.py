# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: typed callback protocols, aligned exogenous inputs,
# and structured latent state.

r"""Forward simulation from a state-space model.

Generates a single trajectory of latent states and observed emissions
by drawing from the initial, transition, and emission distributions
sequentially. Simulation shares the particle filters' latent-state,
transition, and input conventions. Its initial and emission callbacks
draw one trajectory rather than a particle cloud or log-density.

The implementation expresses the full time-loop as one `jax.lax.scan`.
"""

from typing import cast

import jax.random as jr
from jax import lax
from jaxtyping import Array, Shaped

from smcx._utils import (
    _canonicalize_emission,
    _canonicalize_inputs,
    _prepend,
    _prepend_state_history,
    _validate_initial_state,
    _validate_state_tree,
)
from smcx.types import (
    EmissionSampler,
    EmissionSamplerWithInput,
    InputSequence,
    ModelInput,
    PRNGKeyT,
    SingleInitialSampler,
    SingleInitialSamplerWithInput,
    StateHistory,
    StateTree,
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
    StateHistory,
    Shaped[Array, "ntime emission_dim"],
]:
    r"""Simulate a single trajectory from a state-space model.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key[, input_0]) -> state`` that
            draws one state from $p(z_1)$. The state may be a dense
            array or a nonempty PyTree of arrays.
        transition_sampler: Function ``(key, state[, input_t]) -> state`` that
            draws from $p(z_t \mid z_{t-1})$ and preserves the
            state's PyTree structure, leaf shapes, and dtypes.
        emission_sampler: Function ``(key, state[, input_t]) -> emission`` that
            draws from the emission distribution $p(y_t \mid z_t)$. It
            returns a scalar or nonempty vector with dtype fixed across
            time. Scalars become length-one vectors.
        num_timesteps: Number of time steps $T$ to simulate.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)`` and a nonempty event. Input zero reaches
            initialization and the first
            emission; each later input reaches its aligned transition and
            emission.

    Returns:
        A tuple ``(states, emissions)``. ``states`` preserves the latent
        PyTree and adds a leading time axis to every leaf; a dense state
        therefore has shape ``(T, state_dim)``. ``emissions`` has shape
        ``(T, emission_dim)``.

    Raises:
        ValueError: ``num_timesteps`` is less than one, inputs are malformed,
            the initial state tree is empty, a transition changes the state
            structure, or an emission has an invalid shape or changes dtype.
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1; got {num_timesteps}")
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
        state_signature = _validate_initial_state(
            z_0, name="initial_sampler output"
        )
        y_0 = _canonicalize_emission(
            emission_fn(k_y0, z_0),
            name="emission_sampler output",
        )
        emission_signature = _validate_initial_state(
            y_0,
            name="emission_sampler output",
        )

        def _step(
            z_prev: StateTree,
            step_key: PRNGKeyT,
        ) -> tuple[StateTree, tuple[StateTree, Array]]:
            k_z, k_y = jr.split(step_key)
            z_t = transition_fn(k_z, z_prev)
            _validate_state_tree(
                z_t,
                state_signature,
                name="transition_sampler output",
            )
            y_t = _canonicalize_emission(
                emission_fn(k_y, z_t),
                name="emission_sampler output",
            )
            _validate_state_tree(
                y_t,
                emission_signature,
                name="emission_sampler output",
            )
            return z_t, (z_t, y_t)

        _, (states_rest, emissions_rest) = lax.scan(_step, z_0, step_keys)
    else:
        initial_fn_u = cast(SingleInitialSamplerWithInput, initial_sampler)
        transition_fn_u = cast(TransitionSamplerWithInput, transition_sampler)
        emission_fn_u = cast(EmissionSamplerWithInput, emission_sampler)
        z_0 = initial_fn_u(k_z0, inputs_arr[0])
        state_signature = _validate_initial_state(
            z_0, name="initial_sampler output"
        )
        y_0 = _canonicalize_emission(
            emission_fn_u(k_y0, z_0, inputs_arr[0]),
            name="emission_sampler output",
        )
        emission_signature = _validate_initial_state(
            y_0,
            name="emission_sampler output",
        )

        def _step_with_input(
            z_prev: StateTree,
            args: tuple[PRNGKeyT, ModelInput],
        ) -> tuple[StateTree, tuple[StateTree, Array]]:
            step_key, input_t = args
            k_z, k_y = jr.split(step_key)
            z_t = transition_fn_u(k_z, z_prev, input_t)
            _validate_state_tree(
                z_t,
                state_signature,
                name="transition_sampler output",
            )
            y_t = _canonicalize_emission(
                emission_fn_u(k_y, z_t, input_t),
                name="emission_sampler output",
            )
            _validate_state_tree(
                y_t,
                emission_signature,
                name="emission_sampler output",
            )
            return z_t, (z_t, y_t)

        _, (states_rest, emissions_rest) = lax.scan(
            _step_with_input,
            z_0,
            (step_keys, inputs_arr[1:]),
        )

    # --- Combine t=0 with t=1..T-1 ------------------------------------------
    all_states = _prepend_state_history(z_0, states_rest)
    all_emissions = _prepend(y_0, emissions_rest)

    return all_states, all_emissions
