# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Guided (proposal-based) particle filter.

The guided filter propagates through a user proposal
$q(z_t \mid z_{t-1}, y_t)$ — which, unlike the bootstrap
transition prior, can see the current observation — and corrects with
the general importance weight
$w \propto g(y_t \mid z_t)\, f(z_t \mid z_{t-1}) /
q(z_t \mid z_{t-1}, y_t)$ [Doucet, Godsill & Andrieu, 2000].
Approximate proposals (EKF/UKF/Laplace) MUST use this general
formula — the predictive-likelihood shortcut is exact only for the
locally optimal proposal. With ``q = f`` the filter reduces to
bootstrap (same key stream, agreement to floating-point tolerance —
the ``f/q`` cancellation is mathematical, not bitwise; tested).
"""

import math
from functools import partial
from typing import NamedTuple, cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Float, Int

from smcx._numerics import _neumaier_add
from smcx._utils import (
    _canonicalize_inputs,
    _conditional_resample,
    _gather_particles,
    _init_standard,
    _particle_time_axis,
    _prepend,
    _prepend_particle_history,
    _raise_if_degenerate,
    _TreeSignature,
    _validate_filter_inputs,
    _validate_log_density_batch,
    _validate_state_tree,
)
from smcx.containers import ParticleFilterPosterior, ParticleState
from smcx.resampling import systematic
from smcx.types import (
    InitialSampler,
    InitialSamplerWithInput,
    InputSequence,
    LogObservationFn,
    LogObservationFnWithInput,
    LogProposalFn,
    LogProposalFnWithInput,
    LogTransitionFn,
    LogTransitionFnWithInput,
    ParticleCloud,
    PRNGKeyT,
    ProposalSampler,
    ProposalSamplerWithInput,
    ResamplingCriterion,
    ResamplingFn,
    StateTree,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


class _GuidedCarry(NamedTuple):
    state: ParticleState
    ess: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]
    ancestors: Int[Array, " num_particles"]


class _GuidedStepInput(NamedTuple):
    emission: Float[Array, " emission_dim"]
    model_input: Float[Array, " input_dim"] | None
    time_index: Int[Array, ""]


class _GuidedStepOutput(NamedTuple):
    particles: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    log_evidence_increment: Float[Array, ""]


def _guided_step(
    carry: _GuidedCarry,
    inputs_t: _GuidedStepInput,
    key_t: PRNGKeyT,
    *,
    proposal_sampler: ProposalSampler | ProposalSamplerWithInput,
    log_proposal_fn: LogProposalFn | LogProposalFnWithInput,
    log_transition_fn: LogTransitionFn | LogTransitionFnWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion,
    state_signature: _TreeSignature,
    log_num_particles: Float[Array, ""],
) -> tuple[_GuidedCarry, _GuidedStepOutput]:
    """Resample, propose, and reweight one guided particle cloud."""
    state, current_ess, correction, _ = carry
    y_t, input_t, time_index = inputs_t
    num_particles = state.log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resampling_key, proposal_key = jr.split(key_t)
    do_resample, ancestors = _conditional_resample(
        resampling_key,
        state.log_weights,
        current_ess,
        resampling_fn,
        resampling_threshold,
        num_particles,
        identity,
        time_index,
    )
    parents = _gather_particles(state.particles, ancestors)
    keys = jr.split(proposal_key, num_particles)

    if input_t is None:
        proposal_fn = cast(ProposalSampler, proposal_sampler)

        def propose(key_i: PRNGKeyT, state_i: StateTree) -> StateTree:
            next_state = proposal_fn(key_i, state_i, y_t)
            _validate_state_tree(
                next_state,
                state_signature,
                name="proposal_sampler output",
            )
            return next_state

        propagated = vmap(propose)(keys, parents)
        observation_fn = cast(LogObservationFn, log_observation_fn)
        transition_fn = cast(LogTransitionFn, log_transition_fn)
        proposal_density = cast(LogProposalFn, log_proposal_fn)
        log_g = vmap(lambda z: observation_fn(y_t, z))(propagated)
        log_f = vmap(transition_fn)(propagated, parents)
        log_q = vmap(lambda z_new, z_old: proposal_density(y_t, z_new, z_old))(
            propagated, parents
        )
    else:
        proposal_fn_u = cast(ProposalSamplerWithInput, proposal_sampler)

        def propose_with_input(
            key_i: PRNGKeyT, state_i: StateTree
        ) -> StateTree:
            next_state = proposal_fn_u(key_i, state_i, y_t, input_t)
            _validate_state_tree(
                next_state,
                state_signature,
                name="proposal_sampler output",
            )
            return next_state

        propagated = vmap(propose_with_input)(keys, parents)
        observation_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
        transition_fn_u = cast(LogTransitionFnWithInput, log_transition_fn)
        proposal_density_u = cast(LogProposalFnWithInput, log_proposal_fn)
        log_g = vmap(lambda z: observation_fn_u(y_t, z, input_t))(propagated)
        log_f = vmap(transition_fn_u, in_axes=(0, 0, None))(
            propagated, parents, input_t
        )
        log_q = vmap(
            lambda z_new, z_old: proposal_density_u(y_t, z_new, z_old, input_t)
        )(propagated, parents)

    for name, values in (
        ("log_observation_fn", log_g),
        ("log_transition_fn", log_f),
        ("log_proposal_fn", log_q),
    ):
        _validate_log_density_batch(
            cast(Array, values), num_particles, name=name
        )
    log_w_step = log_g + log_f - log_q
    log_w_unnorm = jnp.where(
        do_resample,
        log_w_step,
        state.log_weights + log_w_step,
    )
    log_w_norm, log_sum = log_normalize(log_w_unnorm)
    log_ev_inc = jnp.where(do_resample, log_sum - log_num_particles, log_sum)
    log_evidence, correction = _neumaier_add(
        jnp.asarray(state.log_marginal_likelihood),
        correction,
        log_ev_inc,
    )
    next_state = ParticleState(
        particles=propagated,
        log_weights=log_w_norm,
        log_marginal_likelihood=log_evidence,
    )
    ess_t: Array = jnp.asarray(compute_ess(log_w_norm))
    next_carry = _GuidedCarry(next_state, ess_t, correction, ancestors)
    output = _GuidedStepOutput(
        propagated, log_w_norm, ancestors, ess_t, log_ev_inc
    )
    return next_carry, output


def guided_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    proposal_sampler: ProposalSampler | ProposalSamplerWithInput,
    log_proposal_fn: LogProposalFn | LogProposalFnWithInput,
    log_transition_fn: LogTransitionFn | LogTransitionFnWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a guided particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: ``(key, num_particles[, input_0]) -> particles``
            drawing from $p(z_1)$. ``particles`` may be a dense
            array or a nonempty PyTree whose array leaves all have leading
            size ``num_particles``.
        proposal_sampler: Per-particle
            ``(key, z_prev, y_t[, input_t]) -> z_t``
            drawing from $q(z_t \mid z_{t-1}, y_t)$. It receives
            one particle PyTree and must preserve its structure, leaf
            shapes, and dtypes. smcx ``vmap``-s it internally.
        log_proposal_fn: Per-particle
            ``(y_t, z_t, z_prev[, input_t]) -> scalar`` log proposal
            density with at least float32 precision.
        log_transition_fn: Per-particle
            ``(z_t, z_prev[, input_t]) -> scalar`` log transition
            density $\log f$ with at least float32 precision.
        log_observation_fn: Per-particle
            ``(y_t, z_t[, input_t]) -> scalar`` log observation density
            $\log g$ with at least float32 precision.
        emissions: Observations with leading time dimension.
        num_particles: Number of particles $N$.
        resampling_fn: Resampler with signature
            ``(key, weights, num_samples) -> indices``.
        resampling_threshold: ESS fraction, or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            The callback receives carried weights and ESS at the zero-based
            emission indices 1 through T - 1.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Input zero reaches initialization; each later
            input reaches every guided callback at that time step.
        store_history: When False, the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

    Returns:
        `smcx.containers.ParticleFilterPosterior`. Structured
        particle histories preserve the state PyTree and add ``(T, N)``
        to every leaf.

    Raises:
        DegenerateWeightsError: All weights collapsed (eager execution
            only; under ``jax.jit`` the ``-inf`` marginal propagates).
        ValueError: Inputs are malformed, a criterion result is not a scalar
            Boolean, the initial state tree is empty or has a wrong leading
            axis, a proposal changes its state contract, or a log-density
            callback output is malformed.
    """
    num_timesteps = _validate_filter_inputs(emissions, num_particles)
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    key, init_key = jr.split(key)
    log_n = jnp.asarray(math.log(num_particles))

    # --- t = 0: observation-only weighting ---------------------------------
    (
        particles_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity_ancestors,
        init_state,
        state_signature,
    ) = (
        _init_standard(
            init_key,
            initial_sampler,
            log_observation_fn,
            emissions[0],
            num_particles,
            log_n,
        )
        if inputs_arr is None
        else _init_standard(
            init_key,
            initial_sampler,
            log_observation_fn,
            emissions[0],
            num_particles,
            log_n,
            inputs_arr[0],
        )
    )

    step_keys = jr.split(key, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    model_inputs = None if inputs_arr is None else inputs_arr[1:]
    step_inputs = _GuidedStepInput(emissions[1:], model_inputs, time_indices)
    scan_inputs = (step_inputs, step_keys)
    step = partial(
        _guided_step,
        proposal_sampler=proposal_sampler,
        log_proposal_fn=log_proposal_fn,
        log_transition_fn=log_transition_fn,
        log_observation_fn=log_observation_fn,
        resampling_fn=resampling_fn,
        resampling_threshold=resampling_threshold,
        state_signature=state_signature,
        log_num_particles=log_n,
    )

    def scan_step(
        carry: _GuidedCarry,
        inputs_and_key: tuple[_GuidedStepInput, PRNGKeyT],
    ) -> tuple[_GuidedCarry, _GuidedStepOutput]:
        inputs_t, key_t = inputs_and_key
        return step(carry, inputs_t, key_t)

    init_carry = _GuidedCarry(
        init_state,
        ess_0,
        jnp.zeros_like(log_ev_0),
        identity_ancestors,
    )
    if store_history:
        final_carry, outputs = lax.scan(scan_step, init_carry, scan_inputs)
        all_particles = _prepend_particle_history(
            particles_0, outputs.particles
        )
        all_log_w = _prepend(log_w_0, outputs.log_weights)
        all_ancestors = _prepend(identity_ancestors, outputs.ancestors)
        ess_rest = outputs.ess
        log_ev_inc_rest = outputs.log_evidence_increment
    else:

        def final_only_step(
            carry: _GuidedCarry,
            inputs_and_key: tuple[_GuidedStepInput, PRNGKeyT],
        ) -> tuple[_GuidedCarry, tuple[Array, Array]]:
            next_carry, output = scan_step(carry, inputs_and_key)
            return next_carry, (output.ess, output.log_evidence_increment)

        final_carry, (ess_rest, log_ev_inc_rest) = lax.scan(
            final_only_step,
            init_carry,
            scan_inputs,
        )
        all_particles = _particle_time_axis(final_carry.state.particles)
        all_log_w = final_carry.state.log_weights[None]
        all_ancestors = final_carry.ancestors[None]
    final_state = final_carry.state
    all_ess = _prepend(jnp.asarray(ess_0), ess_rest)
    all_log_ev_inc = _prepend(jnp.asarray(log_ev_0), log_ev_inc_rest)
    final_log_evidence = (
        final_state.log_marginal_likelihood
        + final_carry.log_evidence_compensation
    )

    _raise_if_degenerate(final_log_evidence)
    return ParticleFilterPosterior(
        marginal_loglik=final_log_evidence,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=all_ess,
        log_evidence_increments=all_log_ev_inc,
    )
