# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Guided (proposal-based) particle filter.

The guided filter propagates through a user proposal
:math:`q(z_t \mid z_{t-1}, y_t)` — which, unlike the bootstrap
transition prior, can see the current observation — and corrects with
the general importance weight
:math:`w \propto g(y_t \mid z_t)\, f(z_t \mid z_{t-1}) /
q(z_t \mid z_{t-1}, y_t)` [Doucet, Godsill & Andrieu, 2000].
Approximate proposals (EKF/UKF/Laplace) MUST use this general
formula — the predictive-likelihood shortcut is exact only for the
locally optimal proposal. With ``q = f`` the filter reduces to
bootstrap (same key stream, agreement to floating-point tolerance —
the ``f/q`` cancellation is mathematical, not bitwise; tested).
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
    _init_standard,
    _prepend,
    _raise_if_degenerate,
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
    PRNGKeyT,
    ProposalSampler,
    ProposalSamplerWithInput,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


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
    resampling_threshold: float = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a guided particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: ``(key, num_particles[, input_0]) -> particles``
            drawing from :math:`p(z_1)`.
        proposal_sampler: Per-particle
            ``(key, z_prev, y_t[, input_t]) -> z_t``
            drawing from the proposal
            :math:`q(z_t \mid z_{t-1}, y_t)`.
        log_proposal_fn: Per-particle
            ``(y_t, z_t, z_prev[, input_t]) -> scalar`` log proposal
            density.
        log_transition_fn: Per-particle
            ``(z_t, z_prev[, input_t]) -> scalar`` log transition
            density :math:`\log f`.
        log_observation_fn: Per-particle
            ``(y_t, z_t[, input_t]) -> scalar`` log observation density
            :math:`\log g`.
        emissions: Observations with leading time dimension.
        num_particles: Number of particles :math:`N`.
        resampling_fn: ADR-0004 contract resampler
            ``(key, weights, num_samples) -> indices``.
        resampling_threshold: Resample when
            ``ESS < resampling_threshold * N``.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Input zero reaches initialization; each later
            input reaches every guided callback at that time step.
        store_history: When False (ADR-0011), the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

    Returns:
        :class:`~smcx.containers.ParticleFilterPosterior`.

    Raises:
        DegenerateWeightsError: All weights collapsed (eager execution
            only; under ``jax.jit`` the ``-inf`` marginal propagates).
        ValueError: ``inputs`` is not rank one or two, or its leading
            dimension does not match ``emissions``.
    """
    inputs_arr = (
        None
        if inputs is None
        else _canonicalize_inputs(inputs, emissions.shape[0])
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

        # 1. Conditionally resample on the carried weights.
        threshold = resampling_threshold * num_particles
        do_resample, ancestors = _conditional_resample(
            k1,
            state.log_weights,
            resampling_fn,
            threshold,
            num_particles,
            identity_ancestors,
        )
        parents = state.particles[ancestors]

        # 2. Propagate through the proposal (sees y_t).
        keys = jr.split(k2, num_particles)
        if input_t is None:
            proposal_fn = cast(ProposalSampler, proposal_sampler)
            propagated = vmap(lambda k, z: proposal_fn(k, z, y_t))(
                keys, parents
            )
        else:
            proposal_fn_u = cast(ProposalSamplerWithInput, proposal_sampler)
            propagated = vmap(proposal_fn_u, in_axes=(0, 0, None, None))(
                keys, parents, y_t, input_t
            )

        # 3. General guided weight: log g + log f - log q.
        if input_t is None:
            observation_fn = cast(LogObservationFn, log_observation_fn)
            transition_fn = cast(LogTransitionFn, log_transition_fn)
            proposal_density = cast(LogProposalFn, log_proposal_fn)
            log_g = vmap(lambda z: observation_fn(y_t, z))(propagated)
            log_f = vmap(transition_fn)(propagated, parents)
            log_q = vmap(
                lambda z_new, z_old: proposal_density(y_t, z_new, z_old)
            )(propagated, parents)
        else:
            observation_fn_u = cast(
                LogObservationFnWithInput, log_observation_fn
            )
            transition_fn_u = cast(LogTransitionFnWithInput, log_transition_fn)
            proposal_density_u = cast(LogProposalFnWithInput, log_proposal_fn)
            log_g = vmap(lambda z: observation_fn_u(y_t, z, input_t))(
                propagated
            )
            log_f = vmap(transition_fn_u, in_axes=(0, 0, None))(
                propagated, parents, input_t
            )
            log_q = vmap(
                lambda z_new, z_old: proposal_density_u(
                    y_t, z_new, z_old, input_t
                )
            )(propagated, parents)
        log_w_step = log_g + log_f - log_q

        log_w_unnorm = jnp.where(
            do_resample,
            log_w_step,
            state.log_weights + log_w_step,
        )
        log_w_norm, log_sum = log_normalize(log_w_unnorm)
        log_ev_inc = jnp.where(do_resample, log_sum - log_n, log_sum)

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
        all_particles = _prepend(particles_0, particles_rest)
        all_log_w = _prepend(log_w_0, log_w_rest)
        all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    else:
        (
            (final_state, final_ancestors),
            (ess_rest, log_ev_inc_rest),
        ) = lax.scan(_step, init_carry, scan_inputs)
        all_particles = final_state.particles[None]
        all_log_w = final_state.log_weights[None]
        all_ancestors = final_ancestors[None]
    all_ess = _prepend(jnp.asarray(ess_0), ess_rest)
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
