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
from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Float

from smcx._utils import (
    _conditional_resample,
    _init_standard,
    _prepend,
    _raise_if_degenerate,
)
from smcx.containers import ParticleFilterPosterior, ParticleState
from smcx.resampling import systematic
from smcx.types import PRNGKeyT
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


def guided_filter(
    key: PRNGKeyT,
    initial_sampler: Callable,
    proposal_sampler: Callable,
    log_proposal_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: Callable = systematic,
    resampling_threshold: float = 0.5,
    *,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a guided particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: ``(key, num_particles) -> particles`` drawing
            from :math:`p(z_1)` (t=0 is weighted by the observation
            only, as in the bootstrap filter).
        proposal_sampler: Per-particle ``(key, z_prev, y_t) -> z_t``
            drawing from the proposal
            :math:`q(z_t \mid z_{t-1}, y_t)`.
        log_proposal_fn: Per-particle ``(y_t, z_t, z_prev) -> scalar``
            log proposal density.
        log_transition_fn: Per-particle ``(z_t, z_prev) -> scalar``
            log transition density :math:`\log f`.
        log_observation_fn: Per-particle ``(y_t, z_t) -> scalar`` log
            observation density :math:`\log g`.
        emissions: Observations with leading time dimension.
        num_particles: Number of particles :math:`N`.
        resampling_fn: ADR-0004 contract resampler
            ``(key, weights, num_samples) -> indices``.
        resampling_threshold: Resample when
            ``ESS < resampling_threshold * N``.
        store_history: When False (ADR-0011), the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

    Returns:
        :class:`~smcx.containers.ParticleFilterPosterior`.

    Raises:
        DegenerateWeightsError: All weights collapsed (eager execution
            only; under ``jax.jit`` the ``-inf`` marginal propagates).
    """
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
    ) = _init_standard(
        init_key,
        initial_sampler,
        log_observation_fn,
        emissions[0],
        num_particles,
        log_n,
    )

    # --- Scan body for t = 1, ..., T-1 -------------------------------------
    def _step(
        carry: tuple[ParticleState, Array],
        args: tuple[PRNGKeyT, Float[Array, " emission_dim"]],
    ):
        (state, _prev_ancestors), (step_key, y_t) = carry, args
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
        propagated = vmap(lambda k, z: proposal_sampler(k, z, y_t))(
            keys, parents
        )

        # 3. General guided weight: log g + log f - log q.
        log_g = vmap(lambda z: log_observation_fn(y_t, z))(propagated)
        log_f = vmap(log_transition_fn)(propagated, parents)
        log_q = vmap(lambda z_new, z_old: log_proposal_fn(y_t, z_new, z_old))(
            propagated, parents
        )
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
        ) = lax.scan(_step, init_carry, (step_keys, emissions[1:]))
        all_particles = _prepend(particles_0, particles_rest)
        all_log_w = _prepend(log_w_0, log_w_rest)
        all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    else:
        (
            (final_state, final_ancestors),
            (ess_rest, log_ev_inc_rest),
        ) = lax.scan(_step, init_carry, (step_keys, emissions[1:]))
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
