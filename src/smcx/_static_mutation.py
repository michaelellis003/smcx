# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Private custom-mutation machinery shared by static-target SMC."""

from typing import cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Bool, Float

from smcx._static_smc import _is_valid_acceptance_rate
from smcx.types import (
    PRNGKeyT,
    StaticLogDensity,
    StaticMutation,
    StaticMutationInfo,
    StaticMutationInitFn,
    StaticMutationState,
    StaticMutationStepFn,
)


def _resolve_static_mutation(
    mutation: StaticMutation | None,
    mutation_init_fn: StaticMutationInitFn | None,
    mutation_step_fn: StaticMutationStepFn | None,
) -> tuple[StaticMutationInitFn | None, StaticMutationStepFn | None]:
    """Resolve the mutation record and the legacy pair into one pair."""
    if mutation is None:
        return mutation_init_fn, mutation_step_fn
    if mutation_init_fn is not None or mutation_step_fn is not None:
        raise ValueError(
            "supply mutation or the mutation_init_fn/mutation_step_fn "
            "pair, not both"
        )
    if mutation.init is None or mutation.step is None:
        raise ValueError("StaticMutation members must both be callables")
    return mutation.init, mutation.step


def _mutation_position(
    state: object,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
    *,
    source: str,
) -> Array:
    """Validate and return a structural mutation state's position."""
    if not hasattr(state, "position"):
        raise ValueError(f"{source} must return state with a position field")
    position = cast(StaticMutationState, state).position
    if not hasattr(position, "shape") or not hasattr(position, "dtype"):
        raise ValueError(f"{source} state.position must be a JAX array")
    if tuple(position.shape) != expected_shape:
        raise ValueError(
            f"{source} state.position must have shape {expected_shape}; "
            f"got {position.shape}"
        )
    if position.dtype != expected_dtype:
        raise ValueError(
            f"{source} state.position must have dtype {expected_dtype}; "
            f"got {position.dtype}"
        )
    return position


def _mutation_acceptance_rate(info: object) -> Array:
    """Validate and return one structural mutation diagnostic."""
    if not hasattr(info, "acceptance_rate"):
        raise ValueError(
            "mutation_step_fn info must have an acceptance_rate field"
        )
    value = cast(StaticMutationInfo, info).acceptance_rate
    try:
        rate: Array = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "mutation_step_fn acceptance_rate must be a scalar float"
        ) from error
    if rate.ndim != 0 or not jnp.issubdtype(rate.dtype, jnp.floating):
        raise ValueError(
            "mutation_step_fn acceptance_rate must be a scalar float; "
            f"got shape {rate.shape} and dtype {rate.dtype}"
        )
    return rate


def _run_custom_mutation_sweep(
    key: PRNGKeyT,
    particles: Float[Array, "num_particles state_dim"],
    logdensity_fn: StaticLogDensity,
    *,
    num_steps: int,
    initialize: StaticMutationInitFn,
    mutate: StaticMutationStepFn,
) -> tuple[
    Float[Array, "num_particles state_dim"],
    Float[Array, ""],
    Bool[Array, ""],
]:
    """Run one traceable, fixed-count caller-owned mutation sweep."""
    num_particles, state_dim = particles.shape

    def initialize_one(position):
        state = initialize(position, logdensity_fn)
        _mutation_position(
            state,
            (state_dim,),
            particles.dtype,
            source="mutation_init_fn",
        )
        return state

    states = vmap(initialize_one)(particles)
    sweep_keys = jr.split(key, num_steps)

    def apply_sweep(states, sweep_key):
        particle_keys = jr.split(sweep_key, num_particles)

        def apply_one(particle_key, state):
            next_state, info = mutate(particle_key, state, logdensity_fn)
            _mutation_position(
                next_state,
                (state_dim,),
                particles.dtype,
                source="mutation_step_fn",
            )
            rate = _mutation_acceptance_rate(info)
            return next_state, (rate, _is_valid_acceptance_rate(rate))

        return vmap(apply_one)(particle_keys, states)

    states, (acceptance_rates, valid_rates) = lax.scan(
        apply_sweep,
        states,
        sweep_keys,
    )
    positions = cast(StaticMutationState, states).position
    mean_dtype = (
        jnp.float32
        if jnp.finfo(acceptance_rates.dtype).bits < 32
        else acceptance_rates.dtype
    )
    mean_acceptance_rate = jnp.mean(acceptance_rates.astype(mean_dtype)).astype(
        acceptance_rates.dtype
    )
    return positions, mean_acceptance_rate, jnp.all(valid_rates)
