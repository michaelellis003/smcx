# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Private host stage shared by static-target SMC algorithms."""

import math
from typing import NamedTuple, Protocol, runtime_checkable

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from smcx._numerics import _neumaier_add
from smcx._utils import (
    _gather_particles,
    _raise_if_degenerate,
    _raise_invalid_ancestors,
    _resampling_decision,
    _validate_ancestors,
)
from smcx.types import (
    ParticleCloud,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


class _StaticSMCState(NamedTuple):
    """Resident population and compensated evidence state."""

    population: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    log_evidence: Float[Array, ""]
    log_evidence_correction: Float[Array, ""]


class _StaticMoveResult(NamedTuple):
    """Target-specific population move output."""

    population: ParticleCloud
    acceptance_rate: Float[Array, ""]
    acceptance_valid: Bool[Array, ""]


class _StaticStageDiagnostics(NamedTuple):
    """Diagnostics defined by one correction-selection-move stage."""

    log_evidence_increment: Float[Array, ""]
    selection_ess: Float[Array, ""]
    ess: Float[Array, ""]
    acceptance_rate: Float[Array, ""]
    resampled: Bool[Array, ""]


class _StaticSMCCorrection(NamedTuple):
    """Pure correction and compensated-evidence result."""

    log_weights: Float[Array, " num_particles"]
    log_evidence: Float[Array, ""]
    log_evidence_correction: Float[Array, ""]
    log_evidence_increment: Float[Array, ""]
    selection_ess: Float[Array, ""]


@runtime_checkable
class _StaticMoveFn(Protocol):
    """Move selected seeds while retaining the corrected source cloud."""

    def __call__(
        self,
        key: PRNGKeyT,
        corrected_population: ParticleCloud,
        corrected_log_weights: Float[Array, " num_particles"],
        selected_population: ParticleCloud,
        /,
    ) -> _StaticMoveResult: ...  # pragma: no cover


def _static_smc_correction(
    log_weights: Float[Array, " num_particles"],
    log_increment: Float[Array, " num_particles"],
    log_evidence: Float[Array, ""],
    log_evidence_correction: Float[Array, ""],
) -> _StaticSMCCorrection:
    """Correct normalized weights and update compensated evidence."""
    corrected, increment = log_normalize(log_weights + log_increment)
    selection_ess = jnp.asarray(compute_ess(corrected))
    log_evidence, log_evidence_correction = _neumaier_add(
        log_evidence,
        log_evidence_correction,
        increment,
    )
    return _StaticSMCCorrection(
        log_weights=corrected,
        log_evidence=log_evidence,
        log_evidence_correction=log_evidence_correction,
        log_evidence_increment=jnp.asarray(increment),
        selection_ess=selection_ess,
    )


def _validated_acceptance_rate(
    result: _StaticMoveResult,
) -> Float[Array, ""]:
    """Validate a move's scalar acceptance diagnostic at the host gate."""
    rate: Array = jnp.asarray(result.acceptance_rate)
    if rate.ndim != 0 or not jnp.issubdtype(rate.dtype, jnp.floating):
        raise ValueError(
            "mutation_step_fn acceptance_rate must be a scalar float"
        )
    acceptance_valid: Array = jnp.asarray(result.acceptance_valid)
    if acceptance_valid.ndim != 0 or not jnp.issubdtype(
        acceptance_valid.dtype, jnp.bool_
    ):
        raise ValueError("move acceptance_valid must be a scalar Boolean")
    bit_width = jnp.finfo(rate.dtype).bits
    if bit_width < 16:
        nonnegative = rate.astype(jnp.float32) >= 0.0
    else:
        unsigned_dtype = {
            16: jnp.uint16,
            32: jnp.uint32,
            64: jnp.uint64,
        }[bit_width]
        bits = lax.bitcast_convert_type(rate, unsigned_dtype)
        sign_mask = jnp.asarray(1 << (bit_width - 1), dtype=unsigned_dtype)
        nonnegative = ((bits & sign_mask) == 0) | (bits == sign_mask)
    in_domain = jnp.isfinite(rate) & nonnegative & (rate <= 1.0)
    if not bool(acceptance_valid & in_domain):
        raise ValueError(
            "mutation_step_fn acceptance_rate must be finite and in [0, 1]"
        )
    return rate


def _static_smc_stage(
    state: _StaticSMCState,
    log_increment: Float[Array, " num_particles"],
    resampling_key: PRNGKeyT,
    mutation_key: PRNGKeyT,
    *,
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion | None,
    time_index: Int[Array, ""] | None,
    move_fn: _StaticMoveFn,
) -> tuple[_StaticSMCState, _StaticStageDiagnostics]:
    """Apply one host-driven correction-selection-move stage."""
    corrected = _static_smc_correction(
        state.log_weights,
        log_increment,
        state.log_evidence,
        state.log_evidence_correction,
    )
    _raise_if_degenerate(corrected.log_evidence_increment)
    num_particles = state.log_weights.shape[0]
    if resampling_threshold is None:
        do_resample = jnp.asarray(True)
    else:
        do_resample = _resampling_decision(
            corrected.log_weights,
            corrected.selection_ess,
            resampling_threshold,
            num_particles,
            time_index,
        )

    if bool(do_resample):
        ancestors, invalid_ancestors = _validate_ancestors(
            resampling_fn(
                resampling_key,
                jnp.exp(corrected.log_weights),
                num_particles,
            ),
            num_particles,
            num_particles,
        )
        _raise_invalid_ancestors(invalid_ancestors, num_particles)
        selected = _gather_particles(state.population, ancestors)
        moved = move_fn(
            mutation_key,
            state.population,
            corrected.log_weights,
            selected,
        )
        population = moved.population
        acceptance_rate = _validated_acceptance_rate(moved)
        log_weights = jnp.full((num_particles,), -math.log(num_particles))
    else:
        population = state.population
        acceptance_rate = jnp.zeros((), dtype=state.log_weights.dtype)
        log_weights = corrected.log_weights

    next_state = _StaticSMCState(
        population=population,
        log_weights=log_weights,
        log_evidence=corrected.log_evidence,
        log_evidence_correction=corrected.log_evidence_correction,
    )
    diagnostics = _StaticStageDiagnostics(
        log_evidence_increment=corrected.log_evidence_increment,
        selection_ess=corrected.selection_ess,
        ess=jnp.asarray(compute_ess(log_weights)),
        acceptance_rate=acceptance_rate,
        resampled=do_resample,
    )
    return next_state, diagnostics
