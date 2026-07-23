# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Execution for caller-owned particle-filter kernels."""

from typing import cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array, Float

from smcx._utils import (
    _canonicalize_inputs,
    _particle_time_axis,
    _prepend,
    _prepend_particle_history,
    _raise_if_degenerate,
)
from smcx.containers import ParticleFilterPosterior, ParticleFilterRecord
from smcx.types import (
    FilterCarry,
    InputSequence,
    ParticleFilterInitFn,
    ParticleFilterInitFnWithInput,
    ParticleFilterStepFn,
    ParticleFilterStepFnWithInput,
    PRNGKeyT,
)
from smcx.weights import ess as compute_ess


def _neumaier_add(
    total: Array, correction: Array, value: Array
) -> tuple[Array, Array]:
    """Add one value while retaining a Neumaier correction."""
    updated = total + value
    correction = correction + jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, correction


def run_particle_filter(
    key: PRNGKeyT,
    initialize: ParticleFilterInitFn | ParticleFilterInitFnWithInput,
    step: ParticleFilterStepFn | ParticleFilterStepFnWithInput,
    emissions: Float[Array, "ntime emission_dim"],
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    """Run a caller-owned particle-filter initialization and step kernel."""
    num_timesteps = emissions.shape[0]
    if num_timesteps == 0:
        raise ValueError("emissions must contain at least one row")
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    step_key_root, init_key = jr.split(key)
    time_0 = jnp.asarray(0, dtype=jnp.int32)
    if inputs_arr is None:
        init_fn = cast(ParticleFilterInitFn, initialize)
        carry_0, record_0 = init_fn(time_0, emissions[0], init_key)
    else:
        init_fn_u = cast(ParticleFilterInitFnWithInput, initialize)
        carry_0, record_0 = init_fn_u(
            time_0, emissions[0], inputs_arr[0], init_key
        )

    increment_0 = jnp.asarray(record_0.log_evidence_increment)
    correction_0 = jnp.zeros_like(increment_0)
    ess_0 = jnp.asarray(compute_ess(record_0.log_weights))
    step_keys = jr.split(step_key_root, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    scan_inputs = (
        (time_indices, emissions[1:], step_keys)
        if inputs_arr is None
        else (time_indices, emissions[1:], inputs_arr[1:], step_keys)
    )

    def advance(
        runner_carry: tuple[
            FilterCarry,
            ParticleFilterRecord,
            Array,
            Array,
        ],
        args: tuple[Array, ...],
    ):
        carry, _previous_record, total, correction = runner_carry
        if inputs_arr is None:
            time_index, emission_t, key_t = args
            step_fn = cast(ParticleFilterStepFn, step)
            next_carry, record = step_fn(carry, time_index, emission_t, key_t)
        else:
            time_index, emission_t, input_t, key_t = args
            step_fn_u = cast(ParticleFilterStepFnWithInput, step)
            next_carry, record = step_fn_u(
                carry, time_index, emission_t, input_t, key_t
            )
        increment = jnp.asarray(record.log_evidence_increment)
        total, correction = _neumaier_add(total, correction, increment)
        ess_t = jnp.asarray(compute_ess(record.log_weights))
        traces = ess_t, increment
        if store_history:
            output = (
                record.particles,
                record.log_weights,
                record.ancestors,
                *traces,
            )
        else:
            output = traces
        return (next_carry, record, total, correction), output

    (final_carry, final_record, total, correction), outputs = lax.scan(
        advance,
        (carry_0, record_0, increment_0, correction_0),
        scan_inputs,
    )
    del final_carry
    if store_history:
        particles, log_weights, ancestors, ess_rest, increments_rest = outputs
        all_particles = _prepend_particle_history(record_0.particles, particles)
        all_log_weights = _prepend(record_0.log_weights, log_weights)
        all_ancestors = _prepend(record_0.ancestors, ancestors)
    else:
        ess_rest, increments_rest = outputs
        all_particles = _particle_time_axis(final_record.particles)
        all_log_weights = final_record.log_weights[None]
        all_ancestors = final_record.ancestors[None]
    all_ess = _prepend(ess_0, ess_rest)
    all_increments = _prepend(increment_0, increments_rest)
    marginal_loglik = total + correction
    _raise_if_degenerate(marginal_loglik)
    return ParticleFilterPosterior(
        marginal_loglik=marginal_loglik,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_weights,
        ancestors=all_ancestors,
        ess=all_ess,
        log_evidence_increments=all_increments,
    )
