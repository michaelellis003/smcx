# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Execution for caller-owned particle-filter kernels."""

from typing import NamedTuple, cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree
from jaxtyping import Array

from smcx._utils import (
    _canonicalize_inputs,
    _particle_time_axis,
    _prepend,
    _prepend_particle_history,
    _raise_if_degenerate,
    _TreeSignature,
    _validate_particle_cloud,
    _validate_state_tree,
)
from smcx.containers import ParticleFilterPosterior, ParticleFilterRecord
from smcx.types import (
    EmissionSequence,
    FilterCarry,
    InputSequence,
    ParticleFilterInitFn,
    ParticleFilterInitFnWithInput,
    ParticleFilterStepFn,
    ParticleFilterStepFnWithInput,
    PRNGKeyT,
)
from smcx.weights import ess as compute_ess


class _RecordSignature(NamedTuple):
    """Static structure shared by every callback record."""

    particles: _TreeSignature
    num_particles: int
    log_weights_dtype: object
    ancestors_dtype: object
    increment_dtype: object


def _validate_record(
    record: object,
    *,
    name: str,
    expected: _RecordSignature | None = None,
) -> tuple[ParticleFilterRecord, _RecordSignature]:
    """Validate and canonicalize one callback record."""
    if not isinstance(record, ParticleFilterRecord):
        raise TypeError(f"{name} must be a ParticleFilterRecord")
    log_weights = jnp.asarray(record.log_weights)
    if log_weights.ndim != 1:
        raise ValueError(f"{name} log_weights must be rank 1")
    num_particles = log_weights.shape[0]
    if num_particles == 0:
        raise ValueError(f"{name} must contain at least one particle")
    if not jnp.issubdtype(log_weights.dtype, jnp.floating):
        raise ValueError(f"{name} log_weights must be floating")
    particle_signature = _validate_particle_cloud(
        record.particles,
        num_particles,
        name=f"{name} particles",
    )
    ancestors = jnp.asarray(record.ancestors)
    if ancestors.ndim != 1:
        raise ValueError(f"{name} ancestors must be rank 1")
    if ancestors.shape[0] != num_particles:
        raise ValueError(
            f"{name} ancestors must have length num_particles={num_particles}; "
            f"got {ancestors.shape[0]}"
        )
    if not jnp.issubdtype(ancestors.dtype, jnp.integer):
        raise ValueError(f"{name} ancestors must be integer")
    increment = jnp.asarray(record.log_evidence_increment)
    if increment.ndim != 0:
        raise ValueError(f"{name} log_evidence_increment must be scalar")
    if not jnp.issubdtype(increment.dtype, jnp.floating):
        raise ValueError(f"{name} log_evidence_increment must be floating")
    signature = _RecordSignature(
        particle_signature,
        num_particles,
        log_weights.dtype,
        ancestors.dtype,
        increment.dtype,
    )
    if expected is not None:
        sample = tree.map(lambda leaf: leaf[0], record.particles)
        _validate_state_tree(
            sample,
            expected.particles,
            name=f"{name} particles",
        )
        if num_particles != expected.num_particles:
            raise ValueError(
                f"{name} must preserve num_particles={expected.num_particles}; "
                f"got {num_particles}"
            )
        for field, actual_dtype, expected_dtype in (
            ("log_weights", log_weights.dtype, expected.log_weights_dtype),
            ("ancestors", ancestors.dtype, expected.ancestors_dtype),
            (
                "log_evidence_increment",
                increment.dtype,
                expected.increment_dtype,
            ),
        ):
            if actual_dtype != expected_dtype:
                raise ValueError(
                    f"{name} {field} must preserve dtype {expected_dtype}; "
                    f"got {actual_dtype}"
                )
    canonical = ParticleFilterRecord(
        record.particles,
        log_weights,
        ancestors,
        increment,
    )
    return canonical, signature


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
    emissions: EmissionSequence,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    """Run caller-owned particle-filter initialization and step kernels.

    This is the extension boundary for particle methods not represented by a
    built-in filter. The callbacks own selection, mutation, weighting, and
    each evidence increment. The runner owns time, input, and key alignment;
    structural validation; ESS; compensated evidence accumulation; history
    policy; and posterior construction.

    Input-free callbacks have these scan-shaped signatures::

        initialize(time_index, emission_t, key_t) -> (carry, record)
        step(carry, time_index, emission_t, key_t) -> (carry, record)

    Input-aware callbacks insert ``input_t`` immediately before ``key_t``.
    ``record`` must be :class:`~smcx.containers.ParticleFilterRecord`.
    Its log weights must already be normalized and its ancestor indices must
    be in range; these are data-dependent callback preconditions rather than
    conditions the compiled scan can raise on.

    Args:
        key: Root JAX PRNG key. The runner splits off initialization once,
            then pre-splits one untouched key for every later callback.
        initialize: Pure time-zero kernel, with or without an input argument.
        step: Pure scan-shaped kernel, with the same input convention as
            ``initialize``. Its carry may be any JAX-compatible PyTree whose
            structure, leaf shapes, and dtypes remain fixed.
        emissions: Observation array with shape ``(T, emission_dim)`` and
            at least one row.
        inputs: Optional inputs with shape ``(T, input_dim)`` or ``(T,)``.
            Rank-one inputs become ``(T, 1)``.
        store_history: If False, retain only the final particle, weight, and
            ancestor record while keeping full ESS and evidence traces.

    Returns:
        Standard particle-filter posterior. The algorithm-specific carry is
        internal execution state and is not returned.

    Raises:
        TypeError: A callback does not return ``ParticleFilterRecord``.
        ValueError: Emissions, inputs, or a callback record are structurally
            invalid, or a later record changes its particle or dtype contract.
        DegenerateWeightsError: Eager evidence accumulation ends at NaN or
            negative infinity.
    """
    if emissions.ndim != 2:
        raise ValueError(
            "emissions must have shape (T, emission_dim); "
            f"got ndim={emissions.ndim}"
        )
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

    record_0, record_signature = _validate_record(
        record_0,
        name="initial record",
    )
    increment_0 = record_0.log_evidence_increment
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
        record, _ = _validate_record(
            record,
            name="step record",
            expected=record_signature,
        )
        increment = record.log_evidence_increment
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
