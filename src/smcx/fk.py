# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Feynman-Kac core: one generic sequential Monte Carlo loop.

A `FeynmanKac` record describes one particle algorithm over one
observation sequence: an initial law, a per-particle mutation kernel,
and a per-particle log-potential, with the per-step data (emissions,
inputs, or anything else) carried as a context PyTree. `run_smc` is
the single loop beneath the named filters: conditional resample,
mutate, reweight, with the evidence increment taken at the reweight
stage and the branch weight rule

``log_w = where(resampled, log_g, log_w + log_g)``,

Neumaier-compensated evidence, and init-as-if-resampled at time zero
[Chopin and Papaspiliopoulos, 2020]. Encoding the loop once makes the
classic auxiliary-filter weight and increment bugs structurally
impossible (ADR-0002, reaffirmed for the workbench identity by
ADR-0023).

This module is internal while the named filters migrate onto it; the
public model-record derivations arrive with the model layer.
"""

import math
import operator
from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import tree, vmap
from jaxtyping import Array, Bool, Float, Int

from smcx._numerics import _neumaier_add
from smcx._utils import (
    _conditional_resample,
    _filter_scan,
    _gather_particles,
    _particle_time_axis,
    _prepend,
    _prepend_particle_history,
    _raise_if_degenerate,
    _raise_invalid_ancestors,
    _validate_log_density_batch,
    _validate_particle_cloud,
    _validate_resampling_threshold,
    _validate_state_tree,
)
from smcx.containers import ParticleFilterPosterior, ParticleState
from smcx.resampling import systematic
from smcx.types import (
    ParticleCloud,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
    StateTree,
)
from smcx.weights import _center_log_batch, log_normalize
from smcx.weights import ess as compute_ess


class CallbackNames(NamedTuple):
    """Names used in structural error messages for one derivation."""

    m0: str = "m0 output"
    m: str = "mutation output"
    log_g: str = "log_g"
    log_eta: str = "log_eta"


class FeynmanKac(NamedTuple):
    """One particle algorithm bound to one observation sequence.

    Attributes:
        m0: Whole-cloud initial sampler
            ``(key, num_particles, context_0) -> particles``.
        m: Per-particle mutation kernel
            ``(key, parent_state, context_t) -> state``; vmapped by the
            loop and required to preserve the state PyTree contract.
        log_g: Per-particle log-potential
            ``(parent_state, state, context_t) -> scalar``; vmapped by
            the loop. At time zero the parent is the state itself.
        contexts: PyTree of arrays with a shared leading time axis;
            entry ``t`` is passed to every callback at time ``t``.
        names: Error-message names for the callbacks, so a derivation
            reports contract violations in the vocabulary of the API
            the user actually called.
        log_eta: Optional per-particle look-ahead twist
            ``(state, context_t) -> scalar`` (auxiliary filters).
            First-stage weights ``W * eta`` drive the resampling
            decision and draw; on resampled steps the potential is
            corrected by ``-log_eta[ancestor]`` and the evidence
            increment gains the first-stage normalizer; on skipped
            steps eta is not applied anywhere.
        log_g_batch: Optional cloud-level potential
            ``(parents, particles, context_t) -> (num_particles,)``
            that owns its own log-density validation, for composite
            potentials whose parts carry distinct error names. When
            set it replaces the vmapped ``log_g`` at every step after
            time zero; ``log_g`` still weights initialization.
    """

    m0: Callable[[PRNGKeyT, int, Any], ParticleCloud]
    m: Callable[[PRNGKeyT, StateTree, Any], StateTree]
    log_g: Callable[[StateTree, StateTree, Any], Float[Array, ""]]
    contexts: Any
    names: CallbackNames = CallbackNames()
    log_eta: Callable[[StateTree, Any], Float[Array, ""]] | None = None
    log_g_batch: Callable[[ParticleCloud, ParticleCloud, Any], Array] | None = (
        None
    )


class _StepResult(NamedTuple):
    """Normalized particle update emitted by one generic step."""

    particles: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    log_evidence_increment: Float[Array, ""]
    do_resample: Bool[Array, ""]
    invalid_resampling: Bool[Array, ""]
    normalizers_finite: Bool[Array, ""]


def _fk_step(
    step_key: PRNGKeyT,
    state: ParticleState,
    current_ess: Float[Array, ""],
    fk: FeynmanKac,
    context_t: Any,
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion,
    state_signature: Any,
    time_index: Int[Array, ""] | None,
) -> _StepResult:
    """Resample conditionally, mutate, and reweight one cloud.

    With a look-ahead twist, first-stage weights ``W * eta`` drive the
    resampling decision and draw; the decision otherwise reads the
    carried weights and ESS. Both branches are resolved statically at
    trace time from the derivation, never per step.
    """
    num_particles = state.log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resample_key, mutation_key = jr.split(step_key)
    log_eta_fn = fk.log_eta
    if log_eta_fn is None:
        centered_aux = None
        decision_weights = state.log_weights
        decision_ess = current_ess
        log_first_sum = None
    else:
        log_aux = vmap(lambda z: log_eta_fn(z, context_t))(state.particles)
        _validate_log_density_batch(
            log_aux, num_particles, name=fk.names.log_eta
        )
        # Center the twist before it meets the carried normalized
        # weights (2026-08-06 review, P1): a large common offset would
        # otherwise absorb their relative information. The twist's
        # constant cancels from the two-stage evidence identity, so
        # the centered first-stage normalizer is used directly below.
        # Resolving the promotion first keeps a weakly-typed callback
        # output from strengthening through the centering reduction.
        log_aux = log_aux.astype(jnp.result_type(state.log_weights, log_aux))
        centered_aux, _ = _center_log_batch(log_aux)
        log_first_stage = state.log_weights + centered_aux
        decision_weights, log_first_sum = log_normalize(log_first_stage)
        decision_ess = jnp.asarray(compute_ess(decision_weights))
    do_resample, ancestors, invalid_resampling = _conditional_resample(
        resample_key,
        decision_weights,
        decision_ess,
        resampling_fn,
        resampling_threshold,
        num_particles,
        identity,
        time_index,
    )
    parents = _gather_particles(state.particles, ancestors)
    particle_keys = jr.split(mutation_key, num_particles)
    propagated = vmap(lambda key, parent: fk.m(key, parent, context_t))(
        particle_keys, parents
    )
    # Validate the mutated state before any potential consumes it, so a
    # contract drift reports the mutation kernel by name instead of
    # surfacing as a PyTree mismatch inside a density callback.
    sample = tree.map(lambda leaf: leaf[0], propagated)
    _validate_state_tree(sample, state_signature, name=fk.names.m)
    if fk.log_g_batch is None:
        log_g = vmap(lambda parent, moved: fk.log_g(parent, moved, context_t))(
            parents, propagated
        )
        _validate_log_density_batch(log_g, num_particles, name=fk.names.log_g)
    else:
        log_g = fk.log_g_batch(parents, propagated, context_t)

    log_n = jnp.asarray(math.log(num_particles))
    # The potential is centered like the twist above; its shift rejoins
    # the stage normalizer, so the increment algebra is unchanged while
    # the weight combination becomes exact under a constant offset.
    log_g_array = jnp.asarray(log_g)
    log_g_array = log_g_array.astype(
        jnp.result_type(state.log_weights, log_g_array)
    )
    centered_g, log_g_shift = _center_log_batch(log_g_array)
    resampled_weights = (
        centered_g
        if centered_aux is None
        else centered_g - centered_aux[ancestors]
    )
    log_w_unnorm = jnp.where(
        do_resample,
        resampled_weights,
        state.log_weights + centered_g,
    )
    log_w_norm, centered_sum = log_normalize(log_w_unnorm)
    log_sum = centered_sum + jnp.squeeze(log_g_shift, axis=-1)
    if log_first_sum is None:
        log_ev_inc = jnp.where(do_resample, log_sum - log_n, log_sum)
        normalizers_finite = jnp.isfinite(log_sum)
    else:
        log_ev_inc = jnp.where(
            do_resample,
            log_first_sum + log_sum - log_n,
            log_sum,
        )
        normalizers_finite = jnp.isfinite(log_first_sum) & jnp.isfinite(log_sum)
    ess_t = jnp.asarray(compute_ess(log_w_norm))
    return _StepResult(
        propagated,
        log_w_norm,
        ancestors,
        ess_t,
        log_ev_inc,
        do_resample,
        invalid_resampling,
        normalizers_finite,
    )


def run_smc(
    key: PRNGKeyT,
    fk: FeynmanKac,
    num_particles: int,
    *,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    store_history: bool = True,
    gate_stage_normalizers: bool = True,
) -> ParticleFilterPosterior:
    """Run the generic conditional-resample/mutate/reweight loop.

    Args:
        key: JAX PRNG key; split once for initialization, then per step.
        fk: Feynman-Kac record; its context PyTree's leading axis sets
            the number of time steps.
        num_particles: Number of particles.
        resampling_fn: Resampling algorithm
            ``(key, weights, num_samples) -> indices``.
        resampling_threshold: ESS fraction or JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
        store_history: When False, particle, weight, and ancestor
            histories keep only the final step while ESS and evidence
            traces stay full.
        gate_stage_normalizers: When True (the default), a nonfinite
            stage normalizer at any time step is degenerate for the
            whole run — the uniform policy of the named filters: an
            eager raise, and an exactly ``-inf`` marginal under
            ``jax.jit``. False
            restores the stage-local view in which only the retained
            normalizers gate the result; a degenerate intermediate
            stage then passes silently, so prefer the default.

    Returns:
        `smcx.containers.ParticleFilterPosterior` for the full sweep.

    Raises:
        DegenerateWeightsError: A weight stage cannot be normalized
            (eager only; under ``jax.jit`` the nonfinite marginal
            propagates).
        ValueError: A callback output or resampler output violates its
            structural contract.
    """
    if isinstance(num_particles, bool):
        raise ValueError(
            "num_particles must be a positive integer, not a boolean"
        )
    try:
        num_particles = operator.index(num_particles)
    except TypeError as error:
        raise ValueError(
            f"num_particles must be a positive integer; got {num_particles!r}"
        ) from error
    if num_particles < 1:
        raise ValueError(
            f"num_particles must be a positive integer; got {num_particles}"
        )
    _validate_resampling_threshold(resampling_threshold)
    context_leaves = tree.leaves(fk.contexts)
    if not context_leaves:
        raise ValueError("fk.contexts must have at least one array leaf")
    leading_lengths = {jnp.shape(leaf)[:1] for leaf in context_leaves}
    if len(leading_lengths) != 1 or () in leading_lengths:
        raise ValueError(
            "every fk.contexts leaf must share one leading time length; "
            f"got shapes {sorted(jnp.shape(leaf) for leaf in context_leaves)}"
        )
    num_timesteps = context_leaves[0].shape[0]
    if num_timesteps == 0:
        raise ValueError(
            "fk.contexts must cover at least one time step; "
            "got an empty leading axis"
        )
    key, init_key = jr.split(key)
    log_n = jnp.asarray(math.log(num_particles))
    context_0 = tree.map(lambda leaf: leaf[0], fk.contexts)

    particles_0 = fk.m0(init_key, num_particles, context_0)
    state_signature = _validate_particle_cloud(
        particles_0,
        num_particles,
        name=fk.names.m0,
    )
    log_g_0 = vmap(lambda state: fk.log_g(state, state, context_0))(particles_0)
    _validate_log_density_batch(log_g_0, num_particles, name=fk.names.log_g)
    log_w_0, log_sum_0 = log_normalize(log_g_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0: Array = jnp.asarray(compute_ess(log_w_0))
    identity_ancestors = jnp.arange(num_particles, dtype=jnp.int32)
    init_state = ParticleState(
        particles=particles_0,
        log_weights=log_w_0,
        log_marginal_likelihood=log_ev_0,
    )

    def _step(carry, args):
        state, current_ess, correction, _prev_ancestors, invalid_seen = carry
        step_key, context_t, time_index = args
        result = _fk_step(
            step_key,
            state,
            current_ess,
            fk,
            context_t,
            resampling_fn,
            resampling_threshold,
            state_signature,
            time_index,
        )
        log_evidence, correction = _neumaier_add(
            jnp.asarray(state.log_marginal_likelihood),
            correction,
            result.log_evidence_increment,
        )
        new_state = ParticleState(
            particles=result.particles,
            log_weights=result.log_weights,
            log_marginal_likelihood=log_evidence,
        )
        next_carry = (
            new_state,
            result.ess,
            correction,
            result.ancestors,
            invalid_seen | result.invalid_resampling,
        )
        if store_history:
            return next_carry, (
                result.particles,
                result.log_weights,
                result.ancestors,
                result.ess,
                result.log_evidence_increment,
                result.normalizers_finite,
            )
        # Final-only mode: ancestors ride the carry (O(N)); the scan
        # stacks just the scalar traces.
        return next_carry, (
            result.ess,
            result.log_evidence_increment,
            result.normalizers_finite,
        )

    step_keys = jr.split(key, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    later_contexts = tree.map(lambda leaf: leaf[1:], fk.contexts)
    scan_inputs = (step_keys, later_contexts, time_indices)
    init_carry = (
        init_state,
        ess_0,
        jnp.zeros_like(log_ev_0),
        identity_ancestors,
        jnp.asarray(False),
    )
    if store_history:
        (
            (final_state, _, final_correction, _, invalid),
            (
                particles_rest,
                log_w_rest,
                ancestors_rest,
                ess_rest,
                log_ev_inc_rest,
                normalizers_rest,
            ),
        ) = _filter_scan(_step, init_carry, scan_inputs)
        all_particles = _prepend_particle_history(particles_0, particles_rest)
        all_log_w = _prepend(log_w_0, log_w_rest)
        all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    else:
        (
            (final_state, _, final_correction, final_ancestors, invalid),
            (ess_rest, log_ev_inc_rest, normalizers_rest),
        ) = _filter_scan(_step, init_carry, scan_inputs)
        all_particles = _particle_time_axis(final_state.particles)
        all_log_w = final_state.log_weights[None]
        all_ancestors = final_ancestors[None]
    all_ess = _prepend(ess_0, ess_rest)
    all_log_ev_inc = _prepend(jnp.asarray(log_ev_0), log_ev_inc_rest)
    final_log_evidence = final_state.log_marginal_likelihood + final_correction
    if gate_stage_normalizers:
        # Every stage normalizer must be finite for the marginal to be
        # meaningful. A degenerate stage is a valid zero likelihood, so
        # the gated value is exactly -inf (raised eagerly below; under a
        # user jit it propagates so outer pseudo-marginal algorithms can
        # treat it as rejection rather than NaN-poisoned acceptance
        # arithmetic).
        all_finite = jnp.isfinite(log_ev_0) & jnp.all(normalizers_rest)
        final_log_evidence = jnp.where(
            all_finite,
            final_log_evidence,
            -jnp.inf,
        )

    _raise_invalid_ancestors(invalid, num_particles)
    _raise_if_degenerate(final_log_evidence, tree.leaves(fk.contexts)[0])

    return ParticleFilterPosterior(
        marginal_loglik=final_log_evidence,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=all_ess,
        log_evidence_increments=all_log_ev_inc,
    )
