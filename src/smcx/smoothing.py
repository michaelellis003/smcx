# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Particle fixed-interval smoothing."""

import operator
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeAlias

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree, vmap
from jaxtyping import Array, Bool, Float, Int32, PyTree, Shaped

from smcx._numerics import _validate_minimum_float_precision
from smcx._utils import (
    _canonicalize_inputs,
    _gather_particles,
    _raise_if_degenerate,
)
from smcx.containers import (
    LiuWestPosterior,
    ParticleFilterPosterior,
    ParticleSmootherPosterior,
)
from smcx.diagnostics import _validate_particle_result_axes
from smcx.resampling import multinomial
from smcx.types import (
    InputSequence,
    ModelInput,
    ModelLogTransition,
    ParticleCloud,
    PRNGKeyT,
)

_DrawCloud: TypeAlias = PyTree[Shaped[Array, "num_draws ..."]]
if TYPE_CHECKING:
    _PosteriorArgument: TypeAlias = ParticleFilterPosterior
    _LogTransitionArgument: TypeAlias = ModelLogTransition
else:
    _PosteriorArgument: TypeAlias = Any
    _LogTransitionArgument: TypeAlias = Any


def _safe_weights(
    logits: Float[Array, "*rows num_particles"],
) -> tuple[Array, Bool[Array, " *rows"]]:
    valid = (
        jnp.any(jnp.isfinite(logits), axis=-1)
        & ~jnp.any(jnp.isnan(logits), axis=-1)
        & ~jnp.any(jnp.isposinf(logits), axis=-1)
    )
    safe = jnp.where(valid[..., None], logits, jnp.zeros_like(logits))
    shifted = safe - jnp.max(safe, axis=-1, keepdims=True)
    return jnp.exp(shifted), valid


def _draw_one(
    key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
) -> Int32[Array, ""]:
    return multinomial(key, weights, 1)[0]


def _draw(
    logits: Float[Array, "num_draws num_particles"],
    particles: ParticleCloud,
    keys: Array,
) -> tuple[_DrawCloud, Int32[Array, " num_draws"], Bool[Array, ""]]:
    weights, valid = _safe_weights(logits)
    indices = vmap(_draw_one)(keys, weights)
    return _gather_particles(particles, indices), indices, jnp.all(valid)


def _draw_previous(
    next_states: _DrawCloud,
    particles: ParticleCloud,
    log_weights: Float[Array, " num_particles"],
    keys: Array,
    log_transition: _LogTransitionArgument,
    params: Any,
    input_t: ModelInput | None,
    *,
    validate: bool = False,
) -> tuple[_DrawCloud, Int32[Array, " num_draws"], Bool[Array, ""]]:
    over_previous = vmap(log_transition, in_axes=(None, 0, None, None))
    transition = jnp.asarray(
        vmap(over_previous, in_axes=(0, None, None, None))(
            next_states, particles, params, input_t
        )
    )
    if validate:
        shape = (tree.leaves(next_states)[0].shape[0], log_weights.shape[0])
        if transition.shape != shape or not jnp.issubdtype(
            transition.dtype, jnp.floating
        ):
            raise ValueError(
                f"log_transition batch must be floating with shape {shape}; "
                f"got {transition.shape} and {transition.dtype}"
            )
        _validate_minimum_float_precision(
            transition, name="log_transition output"
        )
    return _draw(transition + log_weights[None, :], particles, keys)


def _finish(
    trajectories: _DrawCloud,
    indices: Int32[Array, "num_draws ntime"],
    valid: Bool[Array, ""],
    dtype: jnp.dtype,
) -> ParticleSmootherPosterior:
    def _invalid(leaf: Array) -> Array:
        fill = jnp.nan if jnp.issubdtype(leaf.dtype, jnp.inexact) else 0
        return jnp.full_like(leaf, fill)

    trajectories = tree.map(
        lambda leaf: jnp.where(valid, leaf, _invalid(leaf)), trajectories
    )
    indices = jnp.where(valid, indices, jnp.full_like(indices, -1))
    checked = jnp.where(valid, jnp.zeros((), dtype=dtype), -jnp.inf)
    _raise_if_degenerate(checked)
    return ParticleSmootherPosterior(trajectories, indices)


def backward_simulation(
    key: PRNGKeyT,
    posterior: _PosteriorArgument,
    log_transition: _LogTransitionArgument,
    params: Any,
    *,
    num_draws: SupportsIndex,
    inputs: InputSequence | None = None,
) -> ParticleSmootherPosterior:
    r"""Draw particle FFBS trajectories from stored filtering clouds.

    Draws are independent and equally weighted under the O(T * num_draws * N)
    discrete FFBS approximation. Leaves are draw-major; each int32 index
    selects its value from the matching filtering cloud.

    Args:
        key: JAX PRNG key.
        posterior: Fixed-parameter filter result with full history.
        log_transition: Matching forward transition log density or mass,
            called as ``(state, prev_state, params, input_t)``.
        params: Explicit transition parameters.
        num_draws: Positive trajectory count, static under an outer ``jit``.
        inputs: Optional length-T inputs. Transition ``t`` to ``t+1`` consumes
            ``inputs[t+1]``; row zero is unused.

    Returns:
        Draw-major trajectories and their filtering-cloud indices.

    Raises:
        ValueError: An argument, history, or callback output is malformed.
        DegenerateWeightsError: A concrete terminal or backward categorical
            law cannot be normalized.

    Note:
        Liu--West needs particle-specific parameters and is excluded. Traced
        degeneracy uses all ``-1`` indices, inexact NaNs, and exact
        placeholders; NaN debugging may raise. Gradients cover gathered values,
        not probabilities or expectations. This is Algorithm 1 of Godsill,
        Doucet, and West (2004):
        https://www2.stat.duke.edu/~mw/MWextrapubs/Godsill2004.pdf
    """
    if isinstance(num_draws, bool):
        raise ValueError("num_draws must be a positive integer, not a boolean")
    try:
        num_draws = operator.index(num_draws)
    except TypeError as error:
        raise ValueError("num_draws must be a positive integer") from error
    if num_draws < 1:
        raise ValueError(f"num_draws must be positive; got {num_draws}")
    if isinstance(posterior, LiuWestPosterior):
        raise ValueError(
            "backward_simulation cannot ignore LiuWestPosterior.filtered_params"
        )
    if not isinstance(posterior, ParticleFilterPosterior):
        raise ValueError("posterior must be a ParticleFilterPosterior")

    ntime, _ = _validate_particle_result_axes(
        posterior, "backward_simulation", particles=True, full_history=True
    )
    log_weights = posterior.filtered_log_weights
    particle_history = posterior.filtered_particles
    _validate_minimum_float_precision(
        log_weights, name="posterior.filtered_log_weights"
    )
    inputs = None if inputs is None else _canonicalize_inputs(inputs, ntime)
    time_keys = jr.split(key, ntime)
    draw_keys = vmap(lambda value: jr.split(value, num_draws))(time_keys)
    terminal_logits = jnp.tile(log_weights[-1], (num_draws, 1))
    terminal_cloud = tree.map(lambda leaf: leaf[-1], particle_history)
    terminal, terminal_indices, terminal_valid = _draw(
        terminal_logits, terminal_cloud, draw_keys[-1]
    )

    if ntime == 1:
        trajectories = tree.map(lambda leaf: leaf[:, None], terminal)
        return _finish(
            trajectories,
            terminal_indices[:, None],
            terminal_valid,
            log_weights.dtype,
        )

    peeled_input = None if inputs is None else inputs[-1]
    peeled_cloud = tree.map(lambda leaf: leaf[-2], particle_history)
    peeled, peeled_indices, peeled_valid = _draw_previous(
        terminal,
        peeled_cloud,
        log_weights[-2],
        draw_keys[-2],
        log_transition,
        params,
        peeled_input,
        validate=True,
    )
    valid = terminal_valid & peeled_valid
    earlier_clouds = tree.map(lambda leaf: leaf[:-2], particle_history)
    earlier_inputs = None if inputs is None else inputs[1:-1]

    def _step(carry, step_inputs):
        next_states, all_valid = carry
        particles_t, log_weights_t, keys_t, input_t = step_inputs
        states, indices, row_valid = _draw_previous(
            next_states,
            particles_t,
            log_weights_t,
            keys_t,
            log_transition,
            params,
            input_t,
        )
        return (states, all_valid & row_valid), (states, indices)

    (_, valid), (earlier, earlier_indices) = lax.scan(
        _step,
        (peeled, valid),
        (
            earlier_clouds,
            log_weights[:-2],
            draw_keys[:-2],
            earlier_inputs,
        ),
        reverse=True,
    )
    time_major = tree.map(
        lambda a, b, c: jnp.concatenate((a, b[None], c[None])),
        earlier,
        peeled,
        terminal,
    )
    index_time = jnp.concatenate((
        earlier_indices,
        peeled_indices[None],
        terminal_indices[None],
    ))

    trajectories = tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), time_major)
    return _finish(
        trajectories,
        jnp.swapaxes(index_time, 0, 1),
        valid,
        log_weights.dtype,
    )
