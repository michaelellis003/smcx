# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Shared private helpers for particle filters.

These utilities are extracted from the individual filter modules to
eliminate duplication.  They are not part of the public API.
"""

from collections.abc import Callable

import jax.numpy as jnp
from blackjax.smc.ess import ess as compute_ess
from jax import lax, vmap
from jaxtyping import Array, Float, Int

from smcjax.containers import ParticleState
from smcjax.types import PRNGKeyT
from smcjax.weights import log_normalize, normalize


def _prepend(first: Array, rest: Array) -> Array:
    """Prepend a single leading slice to an array along axis 0.

    Args:
        first: Array of shape ``(...)``.
        rest: Array of shape ``(T, ...)``.

    Returns:
        Concatenated array of shape ``(T+1, ...)``.
    """
    return jnp.concatenate([jnp.expand_dims(first, 0), rest], axis=0)


def _weighted_quantile_1d(
    particles: Float[Array, ' num_particles'],
    weights: Float[Array, ' num_particles'],
    q: Float[Array, ' num_quantiles'],
) -> Float[Array, ' num_quantiles']:
    """Compute weighted quantiles for a single 1-D vector.

    Sorts particles, builds a midpoint CDF from the normalised
    weights, and interpolates at the requested quantile levels.

    Args:
        particles: Particle values for one dimension.
        weights: Normalised weights (sum to one).
        q: Quantile levels in [0, 1].

    Returns:
        Interpolated quantile values.
    """
    sort_idx = jnp.argsort(particles)
    p_sorted = particles[sort_idx]
    w_sorted = weights[sort_idx]
    cum_w = jnp.cumsum(w_sorted)
    # Midpoint CDF: centre each particle's mass in its interval
    # so that zero-weight particles don't create flat regions.
    mid_cdf = (jnp.concatenate([jnp.zeros(1), cum_w[:-1]]) + cum_w) / 2
    # Tiny tiebreaker ensures strict monotonicity for jnp.interp.
    n = p_sorted.shape[0]
    eps = jnp.arange(n, dtype=p_sorted.dtype) * 1e-12
    return jnp.interp(q, mid_cdf + eps, p_sorted)


def _init_standard(
    init_key: PRNGKeyT,
    initial_sampler: Callable,
    log_observation_fn: Callable,
    first_emission: Array,
    num_particles: int,
    log_n: Array,
) -> tuple[Array, Array, Array, Array, Array, ParticleState]:
    """Initialise a standard (bootstrap/auxiliary) filter at t=0.

    Samples from the prior, weights by the first observation, and
    builds the initial :class:`ParticleState`.

    Args:
        init_key: PRNG key for initialisation.
        initial_sampler: State prior sampler ``(key, N) -> particles``.
        log_observation_fn: Observation log-density
            ``(emission, state) -> log_prob``.
        first_emission: First observation y_0.
        num_particles: Number of particles N.
        log_n: Precomputed ``log(N)`` as a float64 array.

    Returns:
        Tuple of ``(particles_0, log_w_0, log_ev_0, ess_0,
        identity_ancestors, init_state)``.
    """
    particles_0 = initial_sampler(init_key, num_particles)
    log_obs_0 = vmap(lambda z: log_observation_fn(first_emission, z))(
        particles_0
    )
    log_w_0, log_sum_0 = log_normalize(log_obs_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0: Array = jnp.asarray(compute_ess(log_w_0))
    identity_ancestors = jnp.arange(num_particles, dtype=jnp.int32)

    init_state = ParticleState(
        particles=particles_0,
        log_weights=log_w_0,
        log_marginal_likelihood=log_ev_0,
    )
    return (
        particles_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity_ancestors,
        init_state,
    )


def _conditional_resample(
    key: PRNGKeyT,
    log_weights: Float[Array, ' num_particles'],
    resampling_fn: Callable,
    threshold: float,
    num_particles: int,
    identity: Int[Array, ' num_particles'],
) -> tuple[Array, Int[Array, ' num_particles']]:
    """Conditionally resample particles based on ESS.

    Computes the effective sample size of the given log weights
    and resamples only when ESS falls below the threshold.

    Args:
        key: PRNG key for resampling.
        log_weights: Normalised log weights (logsumexp = 0).
        resampling_fn: Blackjax-compatible resampling function.
        threshold: Absolute ESS threshold (e.g. ``0.5 * N``).
        num_particles: Number of particles N.
        identity: Identity ancestor indices ``arange(N)``.

    Returns:
        Tuple of ``(do_resample, ancestors)`` where *do_resample*
        is a boolean scalar and *ancestors* are the resampled (or
        identity) indices.
    """
    cur_ess = compute_ess(log_weights)
    w_norm = normalize(log_weights)
    do_resample: Array = jnp.asarray(cur_ess < threshold)
    ancestors = lax.cond(
        do_resample,
        lambda: resampling_fn(key, w_norm, num_particles),
        lambda: identity,
    )
    return do_resample, ancestors
