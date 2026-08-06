# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Particle fixed-interval smoothing."""

import operator
from typing import TYPE_CHECKING, Any, SupportsIndex, TypeAlias

import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree, vmap
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Int32, PyTree, Shaped

from smcx._numerics import _validate_minimum_float_precision
from smcx._utils import (
    _canonicalize_inputs,
    _gather_particles,
    _raise_if_degenerate,
)
from smcx.containers import (
    LiuWestPosterior,
    ParticleFilterResult,
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
    _PosteriorArgument: TypeAlias = ParticleFilterResult
    _LogTransitionArgument: TypeAlias = ModelLogTransition
    _NumDrawsArgument: TypeAlias = SupportsIndex
else:
    _PosteriorArgument: TypeAlias = Any
    _LogTransitionArgument: TypeAlias = Any
    _NumDrawsArgument: TypeAlias = Any


def _safe_weights(
    logits: Float[Array, "*rows num_particles"],
) -> tuple[Array, Bool[Array, " *rows"]]:
    invalid = jnp.any(jnp.isnan(logits) | jnp.isposinf(logits), axis=-1)
    valid = jnp.any(jnp.isfinite(logits), axis=-1) & ~invalid
    safe = jnp.where(valid[..., None], logits, jnp.zeros_like(logits))
    shifted = safe - jnp.max(safe, axis=-1, keepdims=True)
    return jnp.exp(shifted), valid


def _draw(
    logits: Float[Array, "num_draws num_particles"],
    particles: ParticleCloud,
    keys: Array,
) -> tuple[_DrawCloud, Int32[Array, " num_draws"], Bool[Array, ""]]:
    weights, valid = _safe_weights(logits)
    indices = vmap(lambda k, w: multinomial(k, w, 1)[0])(keys, weights)
    return _gather_particles(particles, indices), indices, jnp.all(valid)


def _backward_logits(
    next_states: _DrawCloud,
    particles: ParticleCloud,
    log_weights: Float[Array, " num_particles"],
    log_transition: _LogTransitionArgument,
    params: Any,
    input_t: ModelInput | None,
    *,
    validate: bool = False,
) -> Float[Array, "num_draws num_particles"]:
    """Evaluate the backward transition matrix plus filtering weights."""

    def _as_transition_array(*args: Any) -> Array:
        output = log_transition(*args)
        try:
            return jnp.asarray(output)
        except (TypeError, ValueError) as error:
            raise ValueError("log_transition output is not a scalar") from error

    over_previous = vmap(_as_transition_array, in_axes=(None, 0, None, None))
    over_draws = vmap(over_previous, in_axes=(0, None, None, None))
    transition = over_draws(next_states, particles, params, input_t)
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
    return transition + log_weights[None, :]


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
    logits = _backward_logits(
        next_states,
        particles,
        log_weights,
        log_transition,
        params,
        input_t,
        validate=validate,
    )
    return _draw(logits, particles, keys)


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
    return ParticleSmootherPosterior(
        smoothed_trajectories=trajectories, backward_indices=indices
    )


def backward_simulation(
    key: PRNGKeyT,
    posterior: _PosteriorArgument,
    log_transition: _LogTransitionArgument,
    params: Any,
    *,
    num_draws: _NumDrawsArgument,
    inputs: InputSequence | None = None,
) -> ParticleSmootherPosterior:
    r"""Draw particle FFBS trajectories from stored filtering clouds.

    Conditional on the stored filtering record, rows are independent,
    equally weighted draws from its discrete FFBS approximation. Runtime is
    O(T * num_draws * N). Leaves are draw-major; leaf ``[j, t]`` is the state
    selected by ``backward_indices[j, t]`` from filtering cloud ``t``.

    Args:
        key: JAX PRNG key.
        posterior: Fixed-parameter filter result with full history — the
            nominal ``ParticleFilterPosterior`` or any caller-owned
            record satisfying `smcx.containers.ParticleFilterResult`
            whose stored log-weight rows are the normalized filtering
            weights of their stored clouds.
        log_transition: Forward-consistent transition log density or mass,
            called as ``(state, prev_state, params, input_t) -> scalar``.
        params: Explicit transition parameters.
        num_draws: Positive trajectory count, static under an outer ``jit``.
        inputs: Optional length-T inputs. Transition ``t`` to ``t+1`` consumes
            ``inputs[t+1]``; row zero is unused.

    Returns:
        Draw-major trajectories with leaf shape ``(num_draws, ntime, ...)``
        and filtering-cloud indices shaped ``(num_draws, ntime)``.

    Raises:
        ValueError: An argument, history, or callback output is malformed.
        DegenerateWeightsError: A concrete categorical law cannot normalize.

    Note:
        Liu--West needs particle-specific parameters and is excluded. Traced
        failure uses all ``-1`` indices, inexact NaNs, and exact placeholders;
        NaN debugging may raise. Gradients cover realized gathers, not
        probabilities or expectations. Keys split by time, then draw; reuse
        repeats on the same backend and code path. Cross-backend equality and
        draw-count prefix stability are not promised. Godsill, Doucet, and
        West (2004), Algorithm 1:
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
    if not isinstance(posterior, ParticleFilterResult):
        raise ValueError(
            "posterior must be a ParticleFilterResult: a record with "
            "marginal_loglik, filtered_particles, filtered_log_weights, "
            "ancestors, ess, and log_evidence_increments fields"
        )
    if not callable(log_transition):
        raise ValueError("log_transition must be callable")
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


def smoothing_weights(
    posterior: _PosteriorArgument,
    log_transition: _LogTransitionArgument,
    params: Any,
    *,
    inputs: InputSequence | None = None,
) -> Float[Array, "ntime num_particles"]:
    r"""Reweight stored filtering clouds toward the smoothing marginals.

    The deterministic backward companion of `backward_simulation`
    (Doucet, Godsill, and Andrieu 2000): starting from the terminal
    filtering weights, each earlier cloud is reweighted through the
    backward kernel,

    $$
    s_t^{(i)} \propto w_t^{(i)} \sum_j s_{t+1}^{(j)}
    \frac{f\bigl(x_{t+1}^{(j)} \mid x_t^{(i)}\bigr)}
    {\sum_k w_t^{(k)} f\bigl(x_{t+1}^{(j)} \mid x_t^{(k)}\bigr)},
    $$

    so row ``t`` of the result and filtering cloud ``t`` together form
    a weighted-cloud representation of $p(x_t \mid y_{1:T})$. Runtime
    is $O(T N^2)$ transition evaluations, and the computation is
    deterministic — no key. Weighted summaries consume the result by
    replacing the record's weights:
    ``posterior._replace(filtered_log_weights=smoothing_weights(...))``
    feeds `weighted_mean` and its siblings unchanged.

    Args:
        posterior: Fixed-parameter filter result with full history —
            the nominal ``ParticleFilterPosterior`` or any caller-owned
            record satisfying `smcx.containers.ParticleFilterResult`
            whose stored log-weight rows are the normalized filtering
            weights of their stored clouds.
        log_transition: Per-pair transition log-density
            ``(state, prev_state, params, input_t) -> scalar``,
            vmapped internally over both clouds — the
            `backward_simulation` contract, validated on its first
            batch at every sequence length.
        params: Model parameter PyTree passed to every evaluation.
        inputs: Optional exogenous inputs, shape ``(ntime,)`` or
            ``(ntime, input_dim)``; ``inputs[t]`` reaches the
            transition into step ``t``.

    Returns:
        Normalized log smoothing weights, shape
        ``(ntime, num_particles)``. Row ``ntime - 1`` equals the
        stored terminal filtering row.

    Raises:
        DegenerateWeightsError: A backward row is unnormalizable
            (NaN or positive-infinite transition output, or no finite
            mass). Raised eagerly; under a caller's ``jax.jit`` the
            result is NaN-filled instead (the `backward_simulation`
            semantics).
        ValueError: The posterior lacks full history, the callback is
            not callable, or its first batch violates the shape or
            precision contract.
    """
    if isinstance(posterior, LiuWestPosterior):
        raise ValueError(
            "smoothing_weights cannot ignore LiuWestPosterior.filtered_params"
        )
    if not isinstance(posterior, ParticleFilterResult):
        raise ValueError(
            "posterior must be a ParticleFilterResult: a record with "
            "marginal_loglik, filtered_particles, filtered_log_weights, "
            "ancestors, ess, and log_evidence_increments fields"
        )
    if not callable(log_transition):
        raise ValueError("log_transition must be callable")
    ntime, _ = _validate_particle_result_axes(
        posterior, "smoothing_weights", particles=True, full_history=True
    )
    log_weights = posterior.filtered_log_weights
    particle_history = posterior.filtered_particles
    _validate_minimum_float_precision(
        log_weights, name="posterior.filtered_log_weights"
    )
    inputs = None if inputs is None else _canonicalize_inputs(inputs, ntime)
    terminal = log_weights[-1]
    if ntime == 1:
        _raise_if_degenerate(jnp.zeros((), dtype=log_weights.dtype))
        return terminal[None]

    def _reweight(
        logits: Array, log_next: Array
    ) -> tuple[Array, Bool[Array, ""]]:
        invalid = jnp.any(jnp.isnan(logits) | jnp.isposinf(logits))
        denominators = logsumexp(logits, axis=-1, keepdims=True)
        combined = logsumexp(
            logits + (log_next - denominators[:, 0])[:, None], axis=0
        )
        row = combined - logsumexp(combined)
        row_valid = jnp.any(jnp.isfinite(row)) & ~invalid
        return row, row_valid

    peeled_cloud = tree.map(lambda leaf: leaf[-2], particle_history)
    next_cloud = tree.map(lambda leaf: leaf[-1], particle_history)
    peeled_logits = _backward_logits(
        next_cloud,
        peeled_cloud,
        log_weights[-2],
        log_transition,
        params,
        None if inputs is None else inputs[-1],
        validate=True,
    )
    peeled_row, valid = _reweight(peeled_logits, terminal)

    earlier_next = tree.map(lambda leaf: leaf[1:-1], particle_history)
    earlier_clouds = tree.map(lambda leaf: leaf[:-2], particle_history)
    earlier_inputs = None if inputs is None else inputs[1:-1]

    def _step(carry, step_inputs):
        log_next, all_valid = carry
        cloud_t, next_states, log_weights_t, input_t = step_inputs
        logits = _backward_logits(
            next_states,
            cloud_t,
            log_weights_t,
            log_transition,
            params,
            input_t,
        )
        row, row_valid = _reweight(logits, log_next)
        return (row, all_valid & row_valid), row

    (_, valid), earlier_rows = lax.scan(
        _step,
        (peeled_row, valid),
        (earlier_clouds, earlier_next, log_weights[:-2], earlier_inputs),
        reverse=True,
    )
    stacked = jnp.concatenate((
        earlier_rows,
        peeled_row[None],
        terminal[None],
    ))
    stacked = jnp.where(valid, stacked, jnp.full_like(stacked, jnp.nan))
    checked = jnp.where(valid, jnp.zeros((), dtype=log_weights.dtype), -jnp.inf)
    _raise_if_degenerate(checked)
    return stacked
