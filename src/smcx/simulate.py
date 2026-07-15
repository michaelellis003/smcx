# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Forward simulation from a state-space model.

Accepts BOTH initial-sampler arities (ADR-0008 item 3): smcjax's
single-draw ``(key) -> state`` form and the filters' cloud-level
``(key, num_particles) -> particles`` form, detected via
``inspect.signature`` (undecidable signatures are treated as the
cloud-level form). Model closures are therefore reusable between
``simulate`` and the filters — including input-driven models.
"""

from typing import Any

import mlx.core as mx

from smcx import _utils
from smcx.types import (
    EmissionSampler,
    EmissionSamplerWithInput,
    InitialSampler,
    KeyT,
    SingleInitialSampler,
    TransitionSampler,
    TransitionSamplerWithInput,
)


def _draw_initial(key: KeyT, initial_sampler) -> mx.array:
    n = _utils.num_positional_params(initial_sampler)
    if n == 1:
        return initial_sampler(key)
    return initial_sampler(key, 1)[0]


def simulate(
    key: KeyT,
    initial_sampler: InitialSampler | SingleInitialSampler,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    emission_sampler: EmissionSampler | EmissionSamplerWithInput,
    num_timesteps: int,
    *,
    inputs: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Simulate one latent trajectory and its emissions.

    Args:
        key: PRNG key.
        initial_sampler: ``(key) -> state`` or ``(key, n) -> cloud``
            (both accepted; ADR-0008).
        transition_sampler: ``(key, state[, input_t]) -> state``.
        emission_sampler: ``(key, state[, input_t]) -> emission``.
        num_timesteps: Number of steps T.
        inputs: Optional per-step inputs, leading dimension T;
            ``inputs[t]`` feeds the transition into t and the
            emission at t (same alignment as the filters).

    Returns:
        ``(states, emissions)`` with leading dimension T.
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1; got {num_timesteps}")
    has_inputs = inputs is not None
    _utils.check_callback_arity(
        transition_sampler, "transition_sampler", 2, has_inputs
    )
    _utils.check_callback_arity(
        emission_sampler, "emission_sampler", 2, has_inputs
    )
    # ty: the arity is dispatched on `inputs` at runtime, which ty
    # cannot narrow across the union (same pattern as the filters).
    trans_any: Any = transition_sampler
    emit_any: Any = emission_sampler
    keys = mx.random.split(key, 2 * num_timesteps)
    state = _draw_initial(keys[0], initial_sampler)
    states = [state]
    if inputs is not None:
        inputs_arr = _utils.canonicalize_inputs(inputs, num_timesteps)
        emissions = [emit_any(keys[1], state, inputs_arr[0])]
        for t in range(1, num_timesteps):
            state = trans_any(keys[2 * t], state, inputs_arr[t])
            states.append(state)
            emissions.append(emit_any(keys[2 * t + 1], state, inputs_arr[t]))
    else:
        emissions = [emit_any(keys[1], state)]
        for t in range(1, num_timesteps):
            state = trans_any(keys[2 * t], state)
            states.append(state)
            emissions.append(emit_any(keys[2 * t + 1], state))
    return mx.stack(states), mx.stack(emissions)
