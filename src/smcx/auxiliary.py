# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling and validation, typed callback
# protocols, exogenous inputs, structured state, and optional history storage.

r"""Auxiliary particle filter (Pitt & Shephard, 1999).

The auxiliary particle filter (APF) improves on the bootstrap filter by
using a *look-ahead* step that biases resampling towards particles
likely to match the next observation **before** propagation.

At each time step the APF:

1. **First-stage weights** — combines the current normalised weights
   with the look-ahead log-density
   $\log g(y_{t+1} \mid x_t^i)$ to form first-stage weights.
2. **Resamples** (conditionally on ESS) using the first-stage weights.
3. **Propagates** resampled particles through the transition prior.
4. **Second-stage weights** — corrects for the look-ahead bias:
   $w_t^{(2)} = p(y_{t+1} \mid x_{t+1}^i) /
   g(y_{t+1} \mid x_t^{a_i})$.

When ``log_auxiliary_fn`` returns zero for all inputs, the APF
reduces to the bootstrap filter.

The implementation uses `jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

import math
from typing import NamedTuple, cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree, vmap
from jaxtyping import Array, Float, Int

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
    ParticleCloud,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
    TransitionSampler,
    TransitionSamplerWithInput,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


class _AuxiliaryStepCarry(NamedTuple):
    """Dynamic state carried between auxiliary-filter scan steps."""

    state: ParticleState
    ancestors: Int[Array, " num_particles"]


class _AuxiliaryStepInput(NamedTuple):
    """Observation-aligned inputs for one auxiliary-filter step."""

    emission: Float[Array, " emission_dim"]
    input_t: Float[Array, " input_dim"] | None
    time_index: Int[Array, ""]


class _AuxiliaryStepOutput(NamedTuple):
    """Normalized particle record emitted by one auxiliary-filter step."""

    particles: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    log_evidence_increment: Float[Array, ""]


def _auxiliary_step(
    carry: _AuxiliaryStepCarry,
    inputs_t: _AuxiliaryStepInput,
    key_t: PRNGKeyT,
    *,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    log_auxiliary_fn: LogObservationFn | LogObservationFnWithInput,
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion,
    log_num_particles: Float[Array, ""],
    state_signature: _TreeSignature,
) -> tuple[_AuxiliaryStepCarry, _AuxiliaryStepOutput]:
    """Apply one pure auxiliary-filter update."""
    state = carry.state
    emission_t, input_t, time_index = inputs_t
    num_particles = state.log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resample_key, transition_key = jr.split(key_t)

    if input_t is None:
        auxiliary_fn = cast(LogObservationFn, log_auxiliary_fn)
        log_aux = cast(
            Array,
            vmap(auxiliary_fn, in_axes=(None, 0))(emission_t, state.particles),
        )
    else:
        auxiliary_fn_u = cast(LogObservationFnWithInput, log_auxiliary_fn)
        log_aux = cast(
            Array,
            vmap(auxiliary_fn_u, in_axes=(None, 0, None))(
                emission_t, state.particles, input_t
            ),
        )
    _validate_log_density_batch(
        log_aux,
        num_particles,
        name="log_auxiliary_fn",
    )
    log_first_stage = state.log_weights + log_aux
    log_first_norm, log_first_sum = log_normalize(log_first_stage)
    first_ess = jnp.asarray(compute_ess(log_first_norm))

    do_resample, ancestors = _conditional_resample(
        resample_key,
        log_first_norm,
        first_ess,
        resampling_fn,
        resampling_threshold,
        num_particles,
        identity,
        time_index,
    )
    resampled_particles = _gather_particles(state.particles, ancestors)
    log_aux_ancestors = log_aux[ancestors]
    particle_keys = jr.split(transition_key, num_particles)

    if input_t is None:
        transition_fn = cast(TransitionSampler, transition_sampler)
        observation_fn = cast(LogObservationFn, log_observation_fn)
        propagated = vmap(transition_fn)(particle_keys, resampled_particles)
        log_obs = vmap(observation_fn, in_axes=(None, 0))(
            emission_t, propagated
        )
    else:
        transition_fn_u = cast(TransitionSamplerWithInput, transition_sampler)
        observation_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
        propagated = vmap(transition_fn_u, in_axes=(0, 0, None))(
            particle_keys, resampled_particles, input_t
        )
        log_obs = vmap(observation_fn_u, in_axes=(None, 0, None))(
            emission_t, propagated, input_t
        )
    sample = tree.map(lambda leaf: leaf[0], propagated)
    _validate_state_tree(
        sample, state_signature, name="transition_sampler output"
    )
    _validate_log_density_batch(
        cast(Array, log_obs),
        num_particles,
        name="log_observation_fn",
    )

    log_second_stage = log_obs - log_aux_ancestors
    log_w_unnorm = jnp.where(
        do_resample,
        log_second_stage,
        state.log_weights + log_obs,
    )
    log_w_norm, log_sum = log_normalize(log_w_unnorm)
    log_ev_inc = jnp.where(
        do_resample,
        log_first_sum + log_sum - log_num_particles,
        log_sum,
    )
    new_state = ParticleState(
        particles=propagated,
        log_weights=log_w_norm,
        log_marginal_likelihood=(state.log_marginal_likelihood + log_ev_inc),
    )
    ess_t = jnp.asarray(compute_ess(log_w_norm))
    new_carry = _AuxiliaryStepCarry(new_state, ancestors)
    output = _AuxiliaryStepOutput(
        propagated,
        log_w_norm,
        ancestors,
        ess_t,
        log_ev_inc,
    )
    return new_carry, output


def auxiliary_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    log_auxiliary_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run an auxiliary particle filter (Pitt & Shephard, 1999).

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws from $p(z_1)$. ``particles`` may
            be a dense array or a nonempty PyTree whose array leaves all
            have leading size ``num_particles``.
        transition_sampler: Function ``(key, state[, input_t]) -> state`` that
            draws from $p(z_t \mid z_{t-1})$. It receives one
            particle PyTree and must preserve its structure, leaf shapes,
            and dtypes. smcx ``vmap``-s it internally.
        log_observation_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            observation log-density $\log p(y_t \mid z_t)$.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        log_auxiliary_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            look-ahead log-density
            $\log g(y_{t+1} \mid x_t)$.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.  When this returns zero for all
            inputs the APF reduces to the bootstrap filter.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles $N$.
        resampling_fn: Resampling algorithm matching the Blackjax
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to `smcx.resampling.systematic`.
        resampling_threshold: ESS fraction (e.g. 0.5 means resample when
            ``ESS < 0.5 * N``), or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            The callback receives the first-stage weights and ESS at the
            zero-based emission indices 1 through T - 1.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Input zero reaches initialization; each later
            input reaches the transition, observation, and auxiliary
            callbacks aligned at the same time step.
        store_history: When False, the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

    Returns:
        `smcx.containers.ParticleFilterPosterior` containing
        filtered particles, log weights, ancestor indices, the marginal
        log-likelihood estimate, and ESS trace. Structured particle
        histories preserve the state PyTree and add ``(T, N)`` to every
        leaf.

    Raises:
        ValueError: Inputs are malformed, a criterion result is not a scalar
            Boolean, the initial state tree is empty or has a wrong leading
            axis, a transition changes its state contract, or a log-density
            callback output is malformed.
    """
    num_timesteps = _validate_filter_inputs(emissions, num_particles)
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    key, init_key = jr.split(key)
    log_n = jnp.asarray(math.log(num_particles))

    # --- Initialise at t=0 -------------------------------------------------
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

    # --- Scan body for t = 1, ..., T-1 -------------------------------------
    def _step(carry: _AuxiliaryStepCarry, args: tuple[Array, ...]):
        if inputs_arr is None:
            step_key, y_t, time_index = args
            input_t = None
        else:
            step_key, y_t, input_t, time_index = args
        next_carry, output = _auxiliary_step(
            carry,
            _AuxiliaryStepInput(y_t, input_t, time_index),
            step_key,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation_fn,
            log_auxiliary_fn=log_auxiliary_fn,
            resampling_fn=resampling_fn,
            resampling_threshold=resampling_threshold,
            log_num_particles=log_n,
            state_signature=state_signature,
        )
        if store_history:
            return next_carry, output
        # In final-only mode, ancestors ride the carry (O(N)) and the
        # scan stacks just the scalar traces.
        return next_carry, (output.ess, output.log_evidence_increment)

    # Run the scan over t = 1 ... T-1
    step_keys = jr.split(key, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    scan_inputs = (
        (step_keys, emissions[1:], time_indices)
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:], time_indices)
    )
    init_carry = _AuxiliaryStepCarry(init_state, identity_ancestors)
    if store_history:
        final_carry, outputs = lax.scan(_step, init_carry, scan_inputs)
        all_particles = _prepend_particle_history(
            particles_0, outputs.particles
        )
        all_log_w = _prepend(log_w_0, outputs.log_weights)
        all_ancestors = _prepend(identity_ancestors, outputs.ancestors)
        ess_rest = outputs.ess
        log_ev_inc_rest = outputs.log_evidence_increment
    else:
        final_carry, (ess_rest, log_ev_inc_rest) = lax.scan(
            _step, init_carry, scan_inputs
        )
        all_particles = _particle_time_axis(final_carry.state.particles)
        all_log_w = final_carry.state.log_weights[None]
        all_ancestors = final_carry.ancestors[None]
    final_state = final_carry.state
    ess_0_arr: Array = jnp.asarray(ess_0)
    all_ess = _prepend(ess_0_arr, ess_rest)
    all_log_ev_inc = _prepend(jnp.asarray(log_ev_0), log_ev_inc_rest)

    _raise_if_degenerate(final_state.log_marginal_likelihood)

    return ParticleFilterPosterior(
        marginal_loglik=final_state.log_marginal_likelihood,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=all_ess,
        log_evidence_increments=all_log_ev_inc,
    )
