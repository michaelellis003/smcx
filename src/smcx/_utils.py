# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling, structural validation,
# exogenous inputs, structured states, and checkpoint helpers.

"""Shared private helpers for particle filters.

These utilities are extracted from the individual filter modules to
eliminate duplication.  They are not part of the public API.
"""

from typing import NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
from jax import lax, tree, vmap
from jax.core import Tracer
from jax.tree_util import PyTreeDef, keystr
from jaxtyping import Array, Float, Int, PyTree, Shaped

from smcx.containers import ParticleState
from smcx.types import (
    InitialSampler,
    InitialSamplerWithInput,
    InputSequence,
    LogObservationFn,
    LogObservationFnWithInput,
    ParticleCloud,
    ParticleHistory,
    PRNGKeyT,
    ResamplingFn,
    StateHistory,
    StateTree,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize, normalize

_ParticleHistoryTail: TypeAlias = PyTree[
    Shaped[Array, "remaining_time num_particles ..."]
]
_StateHistoryTail: TypeAlias = PyTree[Shaped[Array, "remaining_time ..."]]
_SampledCloud: TypeAlias = PyTree[Shaped[Array, "num_samples ..."]]


def _prepend(first: Array, rest: Array) -> Array:
    """Prepend a single leading slice to an array along axis 0.

    Args:
        first: Array of shape ``(...)``.
        rest: Array of shape ``(T, ...)``.

    Returns:
        Concatenated array of shape ``(T+1, ...)``.
    """
    return jnp.concatenate([jnp.expand_dims(first, 0), rest], axis=0)


class _TreeSignature(NamedTuple):
    """Fixed dynamic-leaf contract for one latent state."""

    structure: PyTreeDef
    paths: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[object, ...]


def _array_tree_signature(value: object, *, name: str) -> _TreeSignature:
    """Validate a nonempty PyTree of JAX arrays and describe its leaves."""
    path_leaves, structure = tree.flatten_with_path(value)
    if not path_leaves:
        raise ValueError(f"{name} must be a nonempty PyTree of JAX arrays")

    paths: list[str] = []
    shapes: list[tuple[int, ...]] = []
    dtypes: list[object] = []
    for path, leaf in path_leaves:
        path_text = keystr(path) or "<root>"
        if not isinstance(leaf, (jax.Array, Tracer)):
            raise ValueError(
                f"{name} leaf {path_text} must be a JAX array; "
                f"got {type(leaf).__name__}"
            )
        paths.append(path_text)
        shapes.append(tuple(leaf.shape))
        dtypes.append(leaf.dtype)
    return _TreeSignature(
        structure=structure,
        paths=tuple(paths),
        shapes=tuple(shapes),
        dtypes=tuple(dtypes),
    )


def _validate_particle_cloud(
    particles: object,
    num_particles: int,
    *,
    name: str,
) -> _TreeSignature:
    """Validate a batched latent-state tree and return one-state metadata."""
    cloud = _array_tree_signature(particles, name=name)
    event_shapes: list[tuple[int, ...]] = []
    for path, shape in zip(cloud.paths, cloud.shapes, strict=True):
        if not shape:
            raise ValueError(
                f"{name} leaf {path} must have a leading particle axis"
            )
        if shape[0] != num_particles:
            raise ValueError(
                f"{name} leaf {path} must have leading dimension "
                f"num_particles={num_particles}; got {shape[0]}"
            )
        event_shapes.append(shape[1:])
    return _TreeSignature(
        structure=cloud.structure,
        paths=cloud.paths,
        shapes=tuple(event_shapes),
        dtypes=cloud.dtypes,
    )


def _validate_state_tree(
    state: object,
    expected: _TreeSignature,
    *,
    name: str,
) -> None:
    """Require an unbatched callback state to preserve its initial contract."""
    actual = _array_tree_signature(state, name=name)
    if actual.structure != expected.structure:
        raise ValueError(
            f"{name} must preserve the initial latent-state PyTree structure; "
            f"expected {expected.structure}, got {actual.structure}"
        )
    for path, shape, expected_shape, dtype, expected_dtype in zip(
        actual.paths,
        actual.shapes,
        expected.shapes,
        actual.dtypes,
        expected.dtypes,
        strict=True,
    ):
        if shape != expected_shape:
            raise ValueError(
                f"{name} leaf {path} must preserve shape {expected_shape}; "
                f"got {shape}"
            )
        if dtype != expected_dtype:
            raise ValueError(
                f"{name} leaf {path} must preserve dtype {expected_dtype}; "
                f"got {dtype}"
            )


def _validate_initial_state(state: object, *, name: str) -> _TreeSignature:
    """Validate one unbatched latent state and capture its fixed contract."""
    return _array_tree_signature(state, name=name)


def _gather_particles(
    particles: ParticleCloud,
    ancestors: Int[Array, " num_samples"],
) -> _SampledCloud:
    """Gather every state leaf with one shared ancestor index array."""
    return tree.map(lambda leaf: leaf[ancestors], particles)


def _prepend_particle_history(
    first: ParticleCloud,
    rest: _ParticleHistoryTail,
) -> ParticleHistory:
    """Prepend a particle cloud to every leaf of a scanned history."""
    return tree.map(_prepend, first, rest)


def _particle_time_axis(particles: ParticleCloud) -> ParticleHistory:
    """Add a length-one time axis to every particle-cloud leaf."""
    return tree.map(lambda leaf: leaf[None], particles)


def _prepend_state_history(
    first: StateTree,
    rest: _StateHistoryTail,
) -> StateHistory:
    """Prepend one state to every leaf of a simulated state history."""
    return tree.map(_prepend, first, rest)


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
) -> tuple[
    ParticleCloud,
    Array,
    Array,
    Array,
    Array,
    ParticleState,
    _TreeSignature,
]:
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
        identity_ancestors, init_state, state_signature)``.
    """
    if input_t is None:
        init_fn = cast(InitialSampler, initial_sampler)
        obs_fn = cast(LogObservationFn, log_observation_fn)
        particles_0 = init_fn(init_key, num_particles)
        state_signature = _validate_particle_cloud(
            particles_0,
            num_particles,
            name="initial_sampler output",
        )
        log_obs_0 = cast(
            Array, vmap(lambda z: obs_fn(first_emission, z))(particles_0)
        )
    else:
        init_fn_u = cast(InitialSamplerWithInput, initial_sampler)
        obs_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
        particles_0 = init_fn_u(init_key, num_particles, input_t)
        state_signature = _validate_particle_cloud(
            particles_0,
            num_particles,
            name="initial_sampler output",
        )
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
        state_signature,
    )


def _conditional_resample(
    key: PRNGKeyT,
    log_weights: Float[Array, " num_particles"],
    current_ess: Float[Array, ""],
    resampling_fn: ResamplingFn,
    threshold: float,
    num_particles: int,
    identity: Int[Array, " num_particles"],
) -> tuple[Array, Int[Array, " num_particles"]]:
    """Conditionally resample particles based on ESS.

    Resamples only when the precomputed effective sample size falls
    below the threshold.

    Args:
        key: PRNG key for resampling.
        log_weights: Normalised log weights (logsumexp = 0).
        current_ess: Effective sample size of ``log_weights``.
        resampling_fn: Blackjax-compatible resampling function.
        threshold: Absolute ESS threshold (e.g. ``0.5 * N``).
        num_particles: Number of particles N.
        identity: Identity ancestor indices ``arange(N)``.

    Returns:
        Tuple of ``(do_resample, ancestors)`` where *do_resample*
        is a boolean scalar and *ancestors* are the resampled (or
        identity) indices.
    """
    do_resample: Array = jnp.asarray(current_ess < threshold)
    ancestors = lax.cond(
        do_resample,
        lambda: resampling_fn(key, normalize(log_weights), num_particles),
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
