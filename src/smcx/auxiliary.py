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
   :math:`\log g(y_{t+1} \mid x_t^i)` to form first-stage weights.
2. **Resamples** (conditionally on ESS) using the first-stage weights.
3. **Propagates** resampled particles through the transition prior.
4. **Second-stage weights** — corrects for the look-ahead bias:
   :math:`w_t^{(2)} = p(y_{t+1} \mid x_{t+1}^i) / g(y_{t+1} \mid x_t^{a_i})`.

When ``log_auxiliary_fn`` returns zero for all inputs, the APF
reduces to the bootstrap filter.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

import math
from typing import cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Float

from smcx._utils import (
    _canonicalize_inputs,
    _conditional_resample,
    _gather_particles,
    _init_standard,
    _particle_time_axis,
    _prepend,
    _prepend_particle_history,
    _raise_if_degenerate,
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
    PRNGKeyT,
    ResamplingFn,
    TransitionSampler,
    TransitionSamplerWithInput,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


def auxiliary_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    log_auxiliary_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run an auxiliary particle filter (Pitt & Shephard, 1999).

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws from :math:`p(z_1)`. ``particles`` may
            be a dense array or a nonempty PyTree whose array leaves all
            have leading size ``num_particles``.
        transition_sampler: Function ``(key, state[, input_t]) -> state`` that
            draws from :math:`p(z_t \mid z_{t-1})`. It receives one
            particle PyTree and must preserve its structure, leaf shapes,
            and dtypes. smcx ``vmap``-s it internally.
        log_observation_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            observation log-density :math:`\log p(y_t \mid z_t)`.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        log_auxiliary_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            look-ahead log-density
            :math:`\log g(y_{t+1} \mid x_t)`.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.  When this returns zero for all
            inputs the APF reduces to the bootstrap filter.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles :math:`N`.
        resampling_fn: Resampling algorithm matching the Blackjax
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to :func:`~smcx.resampling.systematic`.
        resampling_threshold: Fraction of ``num_particles`` below
            which resampling is triggered (e.g. 0.5 means resample
            when ``ESS < 0.5 * N``).
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Input zero reaches initialization; each later
            input reaches the transition, observation, and auxiliary
            callbacks aligned at the same time step.
        store_history: When False (ADR-0011), the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

    Returns:
        :class:`~smcx.containers.ParticleFilterPosterior` containing
        filtered particles, log weights, ancestor indices, the marginal
        log-likelihood estimate, and ESS trace. Structured particle
        histories preserve the state PyTree and add ``(T, N)`` to every
        leaf.

    Raises:
        ValueError: Inputs are malformed, the initial state tree is empty
            or has a wrong leading axis, or a transition changes the state
            structure, leaf shape, or dtype.
    """
    inputs_arr = (
        None
        if inputs is None
        else _canonicalize_inputs(inputs, emissions.shape[0])
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
    def _step(
        carry: tuple[ParticleState, Array],
        args: tuple[Array, ...],
    ):
        state, _prev_ancestors = carry
        if inputs_arr is None:
            step_key, y_t = args
            input_t = None
        else:
            step_key, y_t, input_t = args
        k1, k2 = jr.split(step_key)
        # Invariant: state.log_weights are normalized (logsumexp = 0).

        # 1. First-stage weights: combine current weights with
        #    look-ahead g(y_{t+1} | x_t)
        if input_t is None:
            auxiliary_fn = cast(LogObservationFn, log_auxiliary_fn)
            log_aux = cast(
                Array,
                vmap(lambda z: auxiliary_fn(y_t, z))(state.particles),
            )
        else:
            auxiliary_fn_u = cast(LogObservationFnWithInput, log_auxiliary_fn)
            log_aux = cast(
                Array,
                vmap(lambda z: auxiliary_fn_u(y_t, z, input_t))(
                    state.particles
                ),
            )
        log_first_stage = state.log_weights + log_aux

        # Normalise first-stage weights for resampling
        log_first_norm, log_first_sum = log_normalize(log_first_stage)

        # 2. Conditionally resample using first-stage weights
        threshold = resampling_threshold * num_particles
        do_resample, ancestors = _conditional_resample(
            k1,
            log_first_norm,
            resampling_fn,
            threshold,
            num_particles,
            identity_ancestors,
        )
        resampled_particles = _gather_particles(state.particles, ancestors)

        # Store the look-ahead values for ancestors (needed for
        # second-stage correction)
        log_aux_ancestors = log_aux[ancestors]

        # 3. Propagate through transition
        keys = jr.split(k2, num_particles)
        if input_t is None:
            transition_fn = cast(TransitionSampler, transition_sampler)

            def _propagate(key_i, state_i):
                next_state = transition_fn(key_i, state_i)
                _validate_state_tree(
                    next_state,
                    state_signature,
                    name="transition_sampler output",
                )
                return next_state

            propagated = vmap(_propagate)(keys, resampled_particles)
        else:
            transition_fn_u = cast(
                TransitionSamplerWithInput, transition_sampler
            )

            def _propagate_with_input(key_i, state_i):
                next_state = transition_fn_u(key_i, state_i, input_t)
                _validate_state_tree(
                    next_state,
                    state_signature,
                    name="transition_sampler output",
                )
                return next_state

            propagated = vmap(_propagate_with_input)(keys, resampled_particles)

        # 4. Second-stage weights: observation / look-ahead adjustment
        if input_t is None:
            observation_fn = cast(LogObservationFn, log_observation_fn)
            log_obs = vmap(lambda z: observation_fn(y_t, z))(propagated)
        else:
            observation_fn_u = cast(
                LogObservationFnWithInput, log_observation_fn
            )
            log_obs = vmap(lambda z: observation_fn_u(y_t, z, input_t))(
                propagated
            )
        log_second_stage = log_obs - log_aux_ancestors

        # Compute evidence increment and normalize.
        # If resampled: first-stage weights were used for resampling,
        #   the evidence increment is the product of two factors:
        #   (a) E_W[g] = sum_i W_i * g_i  (first-stage normaliser)
        #   (b) (1/N) sum_j w_j^(2)       (mean second-stage weight)
        # If not resampled: standard importance weighting,
        #   increment = logsumexp(log_w_old + log_obs)
        log_w_unnorm = jnp.where(
            do_resample,
            log_second_stage,
            state.log_weights + log_obs,
        )
        log_w_norm, log_sum = log_normalize(log_w_unnorm)

        # log_first_sum = logsumexp(log_w_norm_old + log_aux)
        #   = log(sum W_i g_i) = log E_W[g]  (no 1/N needed)
        # log_sum for second stage = logsumexp(log_second_stage)
        #   so mean = log_sum - log_n
        log_ev_inc_resample = log_first_sum + log_sum - log_n
        log_ev_inc_no_resample = log_sum
        log_ev_inc = jnp.where(
            do_resample, log_ev_inc_resample, log_ev_inc_no_resample
        )

        new_state = ParticleState(
            particles=propagated,
            log_weights=log_w_norm,
            log_marginal_likelihood=(
                state.log_marginal_likelihood + log_ev_inc
            ),
        )
        ess_t: Array = jnp.asarray(compute_ess(log_w_norm))
        if store_history:
            return (new_state, ancestors), (
                propagated,
                log_w_norm,
                ancestors,
                ess_t,
                log_ev_inc,
            )
        # Final-only mode (ADR-0011): ancestors ride the carry (O(N));
        # the scan stacks just the scalar traces.
        return (new_state, ancestors), (ess_t, log_ev_inc)

    # Run the scan over t = 1 ... T-1
    step_keys = jr.split(key, emissions.shape[0] - 1)
    scan_inputs = (
        (step_keys, emissions[1:])
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:])
    )
    init_carry = (init_state, identity_ancestors)
    if store_history:
        (
            (final_state, _),
            (
                particles_rest,
                log_w_rest,
                ancestors_rest,
                ess_rest,
                log_ev_inc_rest,
            ),
        ) = lax.scan(_step, init_carry, scan_inputs)
        all_particles = _prepend_particle_history(particles_0, particles_rest)
        all_log_w = _prepend(log_w_0, log_w_rest)
        all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    else:
        (
            (final_state, final_ancestors),
            (ess_rest, log_ev_inc_rest),
        ) = lax.scan(_step, init_carry, scan_inputs)
        all_particles = _particle_time_axis(final_state.particles)
        all_log_w = final_state.log_weights[None]
        all_ancestors = final_ancestors[None]
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
