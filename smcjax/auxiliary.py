# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
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

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
from blackjax.smc.ess import ess as compute_ess
from blackjax.smc.resampling import systematic
from jax import lax, vmap
from jaxtyping import Array, Float

from smcjax._utils import (
    _conditional_resample,
    _init_standard,
    _prepend,
)
from smcjax.containers import ParticleFilterPosterior, ParticleState
from smcjax.types import PRNGKeyT
from smcjax.weights import log_normalize


def auxiliary_filter(
    key: PRNGKeyT,
    initial_sampler: Callable,
    transition_sampler: Callable,
    log_observation_fn: Callable,
    log_auxiliary_fn: Callable,
    emissions: Float[Array, 'ntime emission_dim'],
    num_particles: int,
    resampling_fn: Callable = systematic,
    resampling_threshold: float = 0.5,
) -> ParticleFilterPosterior:
    r"""Run an auxiliary particle filter (Pitt & Shephard, 1999).

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles) -> particles``
            that draws from the initial state distribution
            :math:`p(z_1)`.
        transition_sampler: Function ``(key, state) -> state`` that
            draws from the transition distribution
            :math:`p(z_t \mid z_{t-1})`.  Will be ``vmap``-ped over
            the particle dimension internally.
        log_observation_fn: Function
            ``(emission, state) -> log_prob`` that evaluates the
            observation log-density :math:`\log p(y_t \mid z_t)`.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        log_auxiliary_fn: Function
            ``(emission, state) -> log_prob`` that evaluates the
            look-ahead log-density
            :math:`\log g(y_{t+1} \mid x_t)`.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.  When this returns zero for all
            inputs the APF reduces to the bootstrap filter.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles :math:`N`.
        resampling_fn: Resampling algorithm matching the Blackjax
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to :func:`~blackjax.smc.resampling.systematic`.
        resampling_threshold: Fraction of ``num_particles`` below
            which resampling is triggered (e.g. 0.5 means resample
            when ``ESS < 0.5 * N``).

    Returns:
        :class:`~smcjax.containers.ParticleFilterPosterior` containing
        filtered particles, log weights, ancestor indices, the
        marginal log-likelihood estimate, and ESS trace.
    """
    key, init_key = jr.split(key)
    log_n = jnp.log(jnp.asarray(num_particles, dtype=jnp.float64))

    # --- Initialise at t=0 -------------------------------------------------
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
        carry: ParticleState,
        args: tuple[PRNGKeyT, Float[Array, ' emission_dim']],
    ) -> tuple[ParticleState, tuple[Array, Array, Array, Array, Array]]:
        state, (step_key, y_t) = carry, args
        k1, k2 = jr.split(step_key)
        # Invariant: state.log_weights are normalized (logsumexp = 0).

        # 1. First-stage weights: combine current weights with
        #    look-ahead g(y_{t+1} | x_t)
        log_aux = vmap(lambda z: log_auxiliary_fn(y_t, z))(state.particles)
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
        resampled_particles = state.particles[ancestors]

        # Store the look-ahead values for ancestors (needed for
        # second-stage correction)
        log_aux_ancestors = log_aux[ancestors]

        # 3. Propagate through transition
        keys = jr.split(k2, num_particles)
        propagated = vmap(transition_sampler)(keys, resampled_particles)

        # 4. Second-stage weights: observation / look-ahead adjustment
        log_obs = vmap(lambda z: log_observation_fn(y_t, z))(propagated)
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
        return new_state, (
            propagated,
            log_w_norm,
            ancestors,
            ess_t,
            log_ev_inc,
        )

    # Run the scan over t = 1 ... T-1
    step_keys = jr.split(key, emissions.shape[0] - 1)
    (
        final_state,
        (
            particles_rest,
            log_w_rest,
            ancestors_rest,
            ess_rest,
            log_ev_inc_rest,
        ),
    ) = lax.scan(_step, init_state, (step_keys, emissions[1:]))

    # --- Combine t=0 with t=1..T-1 -----------------------------------------
    all_particles = _prepend(particles_0, particles_rest)
    all_log_w = _prepend(log_w_0, log_w_rest)
    all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    ess_0_arr: Array = jnp.asarray(ess_0)
    all_ess = _prepend(ess_0_arr, ess_rest)
    all_log_ev_inc = _prepend(jnp.asarray(log_ev_0), log_ev_inc_rest)

    return ParticleFilterPosterior(
        marginal_loglik=final_state.log_marginal_likelihood,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=all_ess,
        log_evidence_increments=all_log_ev_inc,
    )
