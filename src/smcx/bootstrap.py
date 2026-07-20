# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling and validation, typed callback
# protocols, exogenous inputs, structured state, and optional history storage.

r"""Bootstrap (SIR) particle filter.

The bootstrap filter [Gordon *et al.*, 1993] is the simplest Sequential
Monte Carlo algorithm.  At each time step it:

1. **Resamples** (conditionally on ESS) to focus particles on
   high-likelihood regions.
2. **Propagates** particles through the transition prior.
3. **Weights** particles by the observation likelihood.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.
"""

import math
from typing import cast

import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap
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
    _TreeSignature,
    _validate_particle_cloud,
    _validate_state_tree,
)
from smcx.containers import (
    BootstrapCheckpoint,
    BootstrapStepInfo,
    ParticleFilterPosterior,
    ParticleState,
)
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


def _neumaier_add(total: Array, correction: Array, value: Array):
    """Add one value while retaining a Neumaier correction."""
    updated = total + value
    correction = correction + jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, correction


def bootstrap_init(
    init_key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    first_emission: Float[Array, " emission_dim"],
    num_particles: int,
    *,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[BootstrapCheckpoint, BootstrapStepInfo]:
    """Initialize a resumable bootstrap filter at the first observation."""
    log_n = jnp.asarray(math.log(num_particles))
    (
        _,
        _,
        log_ev_0,
        ess_0,
        identity,
        state,
        _,
    ) = _init_standard(
        init_key,
        initial_sampler,
        log_observation_fn,
        first_emission,
        num_particles,
        log_n,
        input_t,
    )
    ess_arr = jnp.asarray(ess_0)
    checkpoint = BootstrapCheckpoint(
        state=state,
        ess=ess_arr,
        log_evidence_compensation=jnp.zeros_like(log_ev_0),
    )
    info = BootstrapStepInfo(
        ancestors=identity,
        ess=ess_arr,
        resampled=jnp.asarray(False),
        log_evidence_increment=jnp.asarray(log_ev_0),
    )
    return checkpoint, info


def _bootstrap_step(
    step_key: PRNGKeyT,
    checkpoint: BootstrapCheckpoint,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emission_t: Float[Array, " emission_dim"],
    resampling_fn: ResamplingFn,
    resampling_threshold: float,
    input_t: Float[Array, " input_dim"] | None,
    state_signature: _TreeSignature,
) -> tuple[BootstrapCheckpoint, BootstrapStepInfo]:
    """Apply one pure bootstrap-filter update."""
    state = checkpoint.state
    num_particles = state.log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resample_key, transition_key = jr.split(step_key)
    do_resample, ancestors = _conditional_resample(
        resample_key,
        state.log_weights,
        checkpoint.ess,
        resampling_fn,
        resampling_threshold * num_particles,
        num_particles,
        identity,
    )
    resampled_particles = _gather_particles(state.particles, ancestors)
    particle_keys = jr.split(transition_key, num_particles)

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

        propagated = vmap(_propagate)(particle_keys, resampled_particles)
        observation_fn = cast(LogObservationFn, log_observation_fn)
        log_obs = vmap(lambda z: observation_fn(emission_t, z))(propagated)
    else:
        transition_fn_u = cast(TransitionSamplerWithInput, transition_sampler)

        def _propagate_with_input(key_i, state_i):
            next_state = transition_fn_u(key_i, state_i, input_t)
            _validate_state_tree(
                next_state,
                state_signature,
                name="transition_sampler output",
            )
            return next_state

        propagated = vmap(_propagate_with_input)(
            particle_keys, resampled_particles
        )
        observation_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
        log_obs = vmap(lambda z: observation_fn_u(emission_t, z, input_t))(
            propagated
        )

    log_w_unnorm = jnp.where(
        do_resample,
        log_obs,
        state.log_weights + log_obs,
    )
    log_w_norm, log_sum = log_normalize(log_w_unnorm)
    log_ev_inc = jnp.where(
        do_resample,
        log_sum - jnp.asarray(math.log(num_particles)),
        log_sum,
    )
    log_ev_sum, correction = _neumaier_add(
        jnp.asarray(state.log_marginal_likelihood),
        checkpoint.log_evidence_compensation,
        log_ev_inc,
    )
    ess_t = jnp.asarray(compute_ess(log_w_norm))
    new_checkpoint = BootstrapCheckpoint(
        state=ParticleState(propagated, log_w_norm, log_ev_sum),
        ess=ess_t,
        log_evidence_compensation=correction,
    )
    return new_checkpoint, BootstrapStepInfo(
        ancestors=ancestors,
        ess=ess_t,
        resampled=do_resample,
        log_evidence_increment=log_ev_inc,
    )


_compiled_bootstrap_step = jit(
    _bootstrap_step,
    static_argnums=(2, 3, 5, 6, 8),
)


def bootstrap_step(
    step_key: PRNGKeyT,
    checkpoint: BootstrapCheckpoint,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emission_t: Float[Array, " emission_dim"],
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
    *,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[BootstrapCheckpoint, BootstrapStepInfo]:
    """Advance a resumable bootstrap filter by one observation."""
    num_particles = checkpoint.state.log_weights.shape[0]
    state_signature = _validate_particle_cloud(
        checkpoint.state.particles,
        num_particles,
        name="checkpoint particles",
    )
    return _compiled_bootstrap_step(
        step_key,
        checkpoint,
        transition_sampler,
        log_observation_fn,
        emission_t,
        resampling_fn,
        resampling_threshold,
        input_t,
        state_signature,
    )


def bootstrap_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a bootstrap (SIR) particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws from :math:`p(z_1)`. ``particles`` may
            be a dense array or a nonempty PyTree whose array leaves all
            have leading size ``num_particles``.
        transition_sampler: Function ``(key, state[, input_t]) -> state``
            drawing from :math:`p(z_t \mid z_{t-1})`. It receives one
            particle PyTree and must preserve its structure, leaf shapes,
            and dtypes. smcx ``vmap``-s it internally.
        log_observation_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            observation log-density :math:`\log p(y_t \mid z_t)`.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles :math:`N`.
        resampling_fn: Resampling algorithm matching the Blackjax
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to :func:`~smcx.resampling.systematic`.
        resampling_threshold: Fraction of ``num_particles`` below which
            resampling is triggered (e.g. 0.5 means resample when
            ``ESS < 0.5 * N``).
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. The latter becomes ``(T, 1)``. ``inputs[0]``
            reaches the initial sampler and observation callback;
            ``inputs[t]`` then reaches the transition into t and the
            observation at t.
        store_history: When False (ADR-0011), the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full —
            dropping memory from O(T*N) to O(N).

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
        carry: tuple[ParticleState, Array, Array],
        args: tuple[Array, ...],
    ):
        state, current_ess, _prev_ancestors = carry
        if inputs_arr is None:
            step_key, y_t = args
            input_t = None
        else:
            step_key, y_t, input_t = args
        k1, k2 = jr.split(step_key)
        # Invariant: state.log_weights are normalized (logsumexp = 0).

        # 1. Conditionally resample
        threshold = resampling_threshold * num_particles
        do_resample, ancestors = _conditional_resample(
            k1,
            state.log_weights,
            current_ess,
            resampling_fn,
            threshold,
            num_particles,
            identity_ancestors,
        )
        resampled_particles = _gather_particles(state.particles, ancestors)

        # 2. Propagate through transition
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

        # 3. Weight by observation likelihood
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

        # Compute evidence increment and normalize.
        # If resampled: weights were reset to uniform (1/N), so
        #   increment = logsumexp(log_obs) - log(N)
        # If not resampled: old normalized weights W_i sum to 1, so
        #   increment = logsumexp(log_W + log_obs)
        log_w_unnorm = jnp.where(
            do_resample,
            log_obs,
            state.log_weights + log_obs,
        )
        log_w_norm, log_sum = log_normalize(log_w_unnorm)
        log_ev_inc = jnp.where(
            do_resample,
            log_sum - log_n,
            log_sum,
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
            return (new_state, ess_t, ancestors), (
                propagated,
                log_w_norm,
                ancestors,
                ess_t,
                log_ev_inc,
            )
        # Final-only mode: ancestors ride the carry (O(N)), the scan
        # stacks just the scalar traces.
        return (new_state, ess_t, ancestors), (ess_t, log_ev_inc)

    # Run the scan over t = 1 ... T-1
    step_keys = jr.split(key, emissions.shape[0] - 1)
    scan_inputs = (
        (step_keys, emissions[1:])
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:])
    )
    init_carry = (init_state, ess_0, identity_ancestors)
    if store_history:
        (
            (final_state, _, _),
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
            (final_state, _, final_ancestors),
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
