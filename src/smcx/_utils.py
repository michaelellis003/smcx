# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling, structural validation,
# exogenous inputs, structured states, and checkpoint helpers.

"""Shared private helpers for particle filters.

These utilities are extracted from the individual filter modules to
eliminate duplication.  They are not part of the public API.
"""

import math
from typing import Any, NamedTuple, TypeAlias, cast

import jax
import jax.numpy as jnp
from jax import lax, tree, vmap
from jax.core import Tracer
from jax.tree_util import PyTreeDef, keystr
from jaxtyping import Array, Bool, Float, Int, Int32, PyTree, Shaped

from smcx._numerics import _validate_minimum_float_precision
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
    ResamplingCriterion,
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


def _filter_scan(step: Any, carry: Any, xs: Any) -> tuple[Any, Any]:
    """Run a filter scan without the jax-mps history defect."""
    num_steps = tree.leaves(xs)[0].shape[0]
    if num_steps == 0:  # pragma: no branch - both paths are tested
        return lax.scan(step, carry, xs)

    def full_scan(current: Any, inputs: Any) -> tuple[Any, Any]:
        return lax.scan(step, current, inputs)

    def mps_scan(current: Any, inputs: Any) -> tuple[Any, Any]:
        # Remove this containment under smcx#38 after a fixed release.
        outputs: list[Any] = []
        for index in range(num_steps):
            step_input = tree.map(lambda value, i=index: value[i], inputs)
            step_xs = tree.map(lambda value: value[None], step_input)
            current, batched_output = lax.scan(step, current, step_xs)
            output = tree.map(lambda value: value[0], batched_output)
            outputs.append(output)
        return current, tree.map(lambda *values: jnp.stack(values), *outputs)

    leaves = tree.leaves((carry, xs))
    traced = any(isinstance(leaf, Tracer) for leaf in leaves)
    if traced:
        return lax.platform_dependent(
            carry,
            xs,
            mps=mps_scan,
            default=full_scan,
        )
    if (  # pragma: no cover - exercised by the physical Metal gate
        next(iter(leaves[0].devices())).platform == "mps"
    ):
        return mps_scan(carry, xs)
    return full_scan(carry, xs)


def _validate_numeric_ess_threshold(
    threshold: float,
    *,
    name: str,
) -> None:
    """Require a finite, nonnegative host-side ESS fraction."""
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            f"{name} must be finite and nonnegative; got {threshold}"
        )


def _validate_resampling_threshold(
    threshold: float | ResamplingCriterion,
) -> None:
    """Validate a numeric filter threshold without invoking a criterion."""
    if not callable(threshold):
        _validate_numeric_ess_threshold(threshold, name="resampling_threshold")


def _validate_filter_inputs(
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
) -> int:
    """Validate the common structural inputs of a particle filter."""
    if emissions.ndim != 2:
        raise ValueError(
            "emissions must have shape (T, emission_dim); "
            f"got ndim={emissions.ndim}"
        )
    num_timesteps = emissions.shape[0]
    if num_timesteps == 0:
        raise ValueError("emissions must contain at least one row")
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}")
    return num_timesteps


def _validate_log_density_batch(
    values: Array,
    num_particles: int,
    *,
    name: str,
) -> None:
    """Require one >=f32 log-density value per particle."""
    if values.shape != (num_particles,):
        raise ValueError(
            f"{name} output must have shape ({num_particles},); "
            f"got {values.shape}"
        )
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise ValueError(
            f"{name} output must have a floating dtype; got {values.dtype}"
        )
    _validate_minimum_float_precision(values, name=f"{name} output")


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


def _validate_emission(emission: object, *, name: str) -> _TreeSignature:
    """Require one nonempty floating emission vector."""
    if not isinstance(emission, (jax.Array, Tracer)):
        raise ValueError(
            f"{name} must be a JAX array with shape (emission_dim,)"
        )
    if emission.ndim != 1 or emission.shape[0] == 0:
        raise ValueError(
            f"{name} must have shape (emission_dim,) with emission_dim >= 1; "
            f"got {emission.shape}"
        )
    if not jnp.issubdtype(emission.dtype, jnp.floating):
        raise ValueError(
            f"{name} must have a floating dtype; got {emission.dtype}"
        )
    return _validate_initial_state(emission, name=name)


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


def _compact_positive_weight_support(
    values: Float[Array, " num_particles"],
    weights: Float[Array, " num_particles"],
) -> tuple[
    Float[Array, " num_particles"],
    Float[Array, " num_particles"],
    Int[Array, ""],
]:
    """Compact an already value-sorted support without changing its shape."""
    positive = weights > 0.0
    num_positive = jnp.sum(positive)
    # A stable boolean sort keeps positive values in their existing order
    # while avoiding a data-dependent output shape under JIT and jax-mps.
    order = jnp.argsort(~positive, stable=True)
    supported_values = values[order]
    supported_weights = weights[order]
    in_support = jnp.arange(values.shape[0]) < num_positive
    final_index = jnp.maximum(num_positive - 1, 0)
    padding_value = supported_values[final_index]
    supported_values = jnp.where(
        in_support,
        supported_values,
        padding_value,
    )
    supported_weights = jnp.where(
        in_support,
        supported_weights,
        0.0,
    )
    return supported_values, supported_weights, num_positive


def _weighted_quantile_1d(
    particles: Float[Array, " num_particles"],
    weights: Float[Array, " num_particles"],
    q: Float[Array, " num_quantiles"],
) -> Float[Array, " num_quantiles"]:
    """Compute weighted quantiles for a single 1-D vector.

    Sorts particles and interpolates on directional midpoint
    cumulative-weight axes. Lower quantiles accumulate from the minimum;
    upper quantiles accumulate from the maximum against ``1 - q`` so
    small positive upper-tail weights remain represented.

    Args:
        particles: Particle values for one dimension.
        weights: Normalised weights (sum to one).
        q: Quantile levels in [0, 1].

    Returns:
        Interpolated quantile values. A support with no positive mass
        returns NaN.

    References:
        Sterbenz, P. H. (1974). *Floating-Point Computation*.
        Prentice-Hall.
    """
    sort_idx = jnp.argsort(particles)
    p_sorted = particles[sort_idx]
    w_sorted = weights[sort_idx]
    p_supported, w_supported, num_positive = _compact_positive_weight_support(
        p_sorted, w_sorted
    )
    zero = jnp.zeros(1, dtype=w_supported.dtype)
    cum_w = jnp.cumsum(w_supported)
    lower_mid = (jnp.concatenate([zero, cum_w[:-1]]) + cum_w) / 2.0
    lower = jnp.interp(q * cum_w[-1], lower_mid, p_supported)
    half = jnp.asarray(0.5, dtype=q.dtype)
    forward_median = jnp.interp(
        half * cum_w[-1],
        lower_mid,
        p_supported,
    )

    p_descending = p_supported[::-1]
    w_descending = w_supported[::-1]
    reverse_cum_w = jnp.cumsum(w_descending)
    upper_mid = (
        jnp.concatenate([zero, reverse_cum_w[:-1]]) + reverse_cum_w
    ) / 2.0
    upper = jnp.interp(
        (jnp.ones_like(q) - q) * reverse_cum_w[-1],
        upper_mid,
        p_descending,
    )
    final_index = jnp.maximum(num_positive - 1, 0)
    final_support = p_supported[final_index]
    upper_clamped = jnp.minimum(
        jnp.maximum(upper, forward_median),
        final_support,
    )
    # The clamp corrects the represented primal splice; retain the
    # directional interpolation tangent with respect to q.
    upper = lax.stop_gradient(upper_clamped) + (
        upper - lax.stop_gradient(upper)
    )

    result = jnp.where(q < half, lower, upper)
    return jnp.where(num_positive > 0, result, jnp.nan)


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
    builds the initial `smcx.containers.ParticleState`.

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
    _validate_log_density_batch(
        log_obs_0,
        num_particles,
        name="log_observation_fn",
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
    resampling_threshold: float | ResamplingCriterion,
    num_particles: int,
    identity: Int[Array, " num_particles"],
    time_index: Int[Array, ""] | None = None,
) -> tuple[Array, Int[Array, " num_particles"], Bool[Array, ""]]:
    """Conditionally resample particles using a float or callback rule.

    A float resamples when the precomputed effective sample size falls
    below that fraction of the particle count.

    Args:
        key: PRNG key for resampling.
        log_weights: Normalised log weights (logsumexp = 0).
        current_ess: Effective sample size of ``log_weights``.
        resampling_fn: BlackJAX-compatible resampling function.
        resampling_threshold: ESS fraction or caller-owned criterion.
        num_particles: Number of particles N.
        identity: Identity ancestor indices ``arange(N)``.
        time_index: Zero-based emission index for a callable criterion.

    Returns:
        Tuple of ``(do_resample, ancestors, invalid_ancestors)`` where
        *do_resample* is a boolean scalar, *ancestors* are the resampled
        (or identity) indices, and *invalid_ancestors* is a scalar range
        failure flag for the eager shell.
    """
    if callable(resampling_threshold):
        if time_index is None:
            raise ValueError(
                "a callable resampling criterion requires a time index"
            )
        criterion = cast(ResamplingCriterion, resampling_threshold)
        raw_decision = criterion(log_weights, current_ess, time_index)
    else:
        raw_decision = current_ess < resampling_threshold * num_particles
    try:
        do_resample: Array = jnp.asarray(raw_decision)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "resampling criterion must return a scalar Boolean"
        ) from error
    if do_resample.ndim != 0 or not jnp.issubdtype(
        do_resample.dtype, jnp.bool_
    ):
        raise ValueError(
            "resampling criterion must return a scalar Boolean; "
            f"got shape {do_resample.shape} and dtype {do_resample.dtype}"
        )

    def resample() -> tuple[Int32[Array, " num_particles"], Array]:
        output = resampling_fn(key, normalize(log_weights), num_particles)
        return _validate_ancestors(output, num_particles, num_particles)

    ancestors, invalid_ancestors = lax.cond(
        do_resample,
        resample,
        lambda: (identity, jnp.asarray(False)),
    )
    return do_resample, ancestors, invalid_ancestors


def _validate_ancestors(
    output: object, num_samples: int, num_particles: int
) -> tuple[Int32[Array, " num_samples"], Bool[Array, ""]]:
    """Validate a caller-owned resampler and return its range-failure flag."""
    if not isinstance(output, (jax.Array, Tracer)):
        raise ValueError(
            "resampling_fn output must be a JAX array with dtype int32 "
            f"and shape ({num_samples},)"
        )
    if output.shape != (num_samples,):
        raise ValueError(
            f"resampling_fn output must have shape ({num_samples},); "
            f"got {output.shape}"
        )
    if output.dtype != jnp.dtype(jnp.int32):
        raise ValueError(
            f"resampling_fn output must have dtype int32; got {output.dtype}"
        )
    invalid = jnp.any((output < 0) | (output >= num_particles))
    return output, invalid


def _raise_invalid_ancestors(
    invalid: Bool[Array, ""], num_particles: int
) -> None:
    """Raise at an eager boundary when compiled ancestor validation failed."""
    if isinstance(invalid, Tracer):
        return
    if bool(invalid):
        raise ValueError(
            f"resampling_fn output entries must be in [0, {num_particles})"
        )


def _raise_if_degenerate(log_value) -> None:
    """Raise `smcx.DegenerateWeightsError` on a nonfinite checked value.

    This eager shell check rejects a nonfinite log normalizer or evidence
    state. Under a user ``jax.jit`` the value is a tracer, so the check is
    skipped and the nonfinite value propagates instead.
    """
    from jax.core import Tracer

    from smcx.exceptions import DegenerateWeightsError

    if isinstance(log_value, Tracer):
        return
    value = float(log_value)
    if not math.isfinite(value):
        raise DegenerateWeightsError(
            "particle weights or evidence cannot be normalized "
            f"(checked log value {value})"
        )
