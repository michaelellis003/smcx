# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Shared private helpers for particle filters.

These utilities are extracted from the individual filter modules to
eliminate duplication.  They are not part of the public API.
"""

from typing import cast

import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float, Int

from smcx.containers import ParticleState
from smcx.types import (
    InitialSampler,
    InitialSamplerWithInput,
    InputSequence,
    LogObservationFn,
    LogObservationFnWithInput,
    PRNGKeyT,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize, normalize


def _prepend(first: Array, rest: Array) -> Array:
    """Prepend a single leading slice to an array along axis 0.

    Args:
        first: Array of shape ``(...)``.
        rest: Array of shape ``(T, ...)``.

    Returns:
        Concatenated array of shape ``(T+1, ...)``.
    """
    return jnp.concatenate([jnp.expand_dims(first, 0), rest], axis=0)


def _canonicalize_inputs(
    inputs: InputSequence, num_timesteps: int
) -> Float[Array, "ntime input_dim"]:
    """Validate and canonicalize a per-step input sequence.

    Args:
        inputs: Input sequence with shape ``(T,)`` or ``(T, input_dim)``.
        num_timesteps: Expected leading dimension T.

    Returns:
        Input sequence with shape ``(T, input_dim)``.

    Raises:
        ValueError: The rank is not one or two, or the leading dimension
            does not equal ``num_timesteps``.
    """
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    if inputs.ndim != 2:
        raise ValueError(
            "inputs must have shape (T,) or (T, input_dim); "
            f"got ndim={inputs.ndim}"
        )
    if inputs.shape[0] != num_timesteps:
        raise ValueError(
            f"inputs must have leading dimension T={num_timesteps}; "
            f"got {inputs.shape[0]}"
        )
    return inputs


def _weighted_quantile_1d(
    particles: Float[Array, " num_particles"],
    weights: Float[Array, " num_particles"],
    q: Float[Array, " num_quantiles"],
) -> Float[Array, " num_quantiles"]:
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
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    first_emission: Array,
    num_particles: int,
    log_n: Array,
    input_t: Float[Array, " input_dim"] | None = None,
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
        log_n: Precomputed ``log(N)`` as a scalar array in the
            default float dtype.
        input_t: Optional input at t=0. When present, both callbacks
            receive it as their final argument.

    Returns:
        Tuple of ``(particles_0, log_w_0, log_ev_0, ess_0,
        identity_ancestors, init_state)``.
    """
    if input_t is None:
        init_fn = cast(InitialSampler, initial_sampler)
        obs_fn = cast(LogObservationFn, log_observation_fn)
        particles_0 = init_fn(init_key, num_particles)
        log_obs_0 = cast(
            Array, vmap(lambda z: obs_fn(first_emission, z))(particles_0)
        )
    else:
        init_fn_u = cast(InitialSamplerWithInput, initial_sampler)
        obs_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
        particles_0 = init_fn_u(init_key, num_particles, input_t)
        log_obs_0 = cast(
            Array,
            vmap(lambda z: obs_fn_u(first_emission, z, input_t))(particles_0),
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
    log_weights: Float[Array, " num_particles"],
    resampling_fn: ResamplingFn,
    threshold: float,
    num_particles: int,
    identity: Int[Array, " num_particles"],
) -> tuple[Array, Int[Array, " num_particles"]]:
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


def _raise_if_degenerate(marginal_loglik) -> None:
    """Raise :class:`DegenerateWeightsError` on a collapsed filter.

    Host-side check: fires only in eager execution. Under a user
    ``jax.jit`` the value is a tracer and the check is skipped — the
    ``-inf``/NaN marginal propagates instead (see the exception's
    docstring).
    """
    from jax.core import Tracer

    from smcx.exceptions import DegenerateWeightsError

    if isinstance(marginal_loglik, Tracer):
        return
    value = float(marginal_loglik)
    if value != value or value == float("-inf"):
        raise DegenerateWeightsError(
            f"all particle weights collapsed (marginal log-likelihood {value})"
        )
