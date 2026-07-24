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

The implementation expresses the full time-loop as one `jax.lax.scan`.
"""

import math
from typing import NamedTuple, cast

import jax.numpy as jnp
import jax.random as jr
from jax import core, device_put, lax, tree, vmap
from jaxtyping import Array, Float, Int, Shaped

from smcx._numerics import _neumaier_add
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
    _validate_filter_inputs,
    _validate_log_density_batch,
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
    ParticleCloud,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
    TransitionSampler,
    TransitionSamplerWithInput,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize


def _checkpoint_tolerance(dtype: object, num_particles: int) -> float:
    """Return a reduction-depth-scaled checkpoint comparison budget."""
    reduction_depth = max(1, math.ceil(math.log2(num_particles)))
    # Normalization and ESS use up to two logaddexp reductions plus scalar
    # arithmetic. Thirty-two dtype eps per reduction level covers the
    # observed CPU/Metal f32 rounding without admitting an O(1) shift.
    return 32.0 * float(jnp.finfo(dtype).eps) * reduction_depth


def _validate_checkpoint(checkpoint: BootstrapCheckpoint) -> _TreeSignature:
    """Validate structural and concrete semantic checkpoint invariants."""
    state = checkpoint.state
    log_weights = jnp.asarray(state.log_weights)
    if log_weights.ndim != 1:
        raise ValueError("checkpoint log_weights must be rank 1")
    num_particles = log_weights.shape[0]
    if num_particles == 0:
        raise ValueError("checkpoint must contain at least one particle")
    if not jnp.issubdtype(log_weights.dtype, jnp.floating):
        raise ValueError("checkpoint log_weights must be floating")
    signature = _validate_particle_cloud(
        state.particles, num_particles, name="checkpoint particles"
    )
    scalar_values: dict[str, Array] = {}
    for name, value in (
        ("log_marginal_likelihood", state.log_marginal_likelihood),
        ("ess", checkpoint.ess),
        ("log_evidence_compensation", checkpoint.log_evidence_compensation),
    ):
        value = jnp.asarray(value)
        if value.ndim != 0:
            raise ValueError(f"checkpoint {name} must be scalar")
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(f"checkpoint {name} must be floating")
        scalar_values[name] = value

    # Value checks belong to concrete public shells. A transformed
    # bootstrap_step retains the same traceable behavior as the pure step;
    # malformed values then propagate instead of raising during tracing.
    if isinstance(log_weights, core.Tracer) or any(
        isinstance(value, core.Tracer) for value in scalar_values.values()
    ):
        return signature

    ess_value = float(scalar_values["ess"])
    if not math.isfinite(ess_value) or ess_value < 0:
        raise ValueError("checkpoint ess must be finite and nonnegative")
    tolerance = _checkpoint_tolerance(log_weights.dtype, num_particles)
    _, log_total = log_normalize(log_weights)
    log_total_value = float(log_total)
    if not math.isfinite(log_total_value) or abs(log_total_value) > tolerance:
        raise ValueError(
            "checkpoint log_weights must be normalized (logsumexp = 0)"
        )
    expected_ess = float(compute_ess(log_weights))
    if not math.isclose(
        ess_value,
        expected_ess,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError("checkpoint ess must match checkpoint log_weights")
    _raise_if_degenerate(
        scalar_values["log_marginal_likelihood"]
        + scalar_values["log_evidence_compensation"]
    )
    return signature


def bootstrap_init(
    init_key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    first_emission: Float[Array, " emission_dim"],
    num_particles: int,
    *,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[BootstrapCheckpoint, BootstrapStepInfo]:
    """Initialize a resumable bootstrap filter at observation zero.

    Args:
        init_key: Explicit key for sampling the initial particle cloud.
        initial_sampler: Initial-state callback, optionally input-aware.
        log_observation_fn: Observation log-density callback returning a
            scalar with at least float32 precision.
        first_emission: Observation ``y[0]``.
        num_particles: Number of particles.
        input_t: Optional ``inputs[0]`` passed to both callbacks.

    Returns:
        Normalized checkpoint plus identity, non-resampled time-zero details.

    Raises:
        ValueError: The particle count or sampled particle cloud is invalid.
        DegenerateWeightsError: Initial importance weights cannot normalize.
    """
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}")
    log_n = jnp.asarray(math.log(num_particles))
    initialized = _init_standard(
        init_key,
        initial_sampler,
        log_observation_fn,
        first_emission,
        num_particles,
        log_n,
        input_t,
    )
    _, _, log_ev_0, ess_0, identity, state, _ = initialized
    _raise_if_degenerate(log_ev_0)
    ess_arr = jnp.asarray(ess_0)
    checkpoint = BootstrapCheckpoint(state, ess_arr, jnp.zeros_like(log_ev_0))
    info = BootstrapStepInfo(identity, ess_arr, jnp.asarray(False), log_ev_0)
    return checkpoint, info


class _BootstrapParticleStepResult(NamedTuple):
    """Normalized particle update before shell-owned evidence accumulation."""

    particles: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    info: BootstrapStepInfo


def _bootstrap_particle_step(
    step_key: PRNGKeyT,
    state: ParticleState,
    current_ess: Float[Array, ""],
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emission_t: Float[Array, " emission_dim"],
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion,
    input_t: Float[Array, " input_dim"] | None,
    state_signature: _TreeSignature,
    time_index: Int[Array, ""] | None = None,
) -> _BootstrapParticleStepResult:
    """Propagate, weight, and normalize one bootstrap particle cloud."""
    num_particles = state.log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resample_key, transition_key = jr.split(step_key)
    do_resample, ancestors = _conditional_resample(
        resample_key,
        state.log_weights,
        current_ess,
        resampling_fn,
        resampling_threshold,
        num_particles,
        identity,
        time_index,
    )
    resampled_particles = _gather_particles(state.particles, ancestors)
    particle_keys = jr.split(transition_key, num_particles)

    if input_t is None:
        transition_fn = cast(TransitionSampler, transition_sampler)
        observation_fn = cast(LogObservationFn, log_observation_fn)
        propagated = vmap(transition_fn)(particle_keys, resampled_particles)
        log_obs = vmap(observation_fn, (None, 0))(emission_t, propagated)
    else:
        transition_fn_u = cast(TransitionSamplerWithInput, transition_sampler)
        observation_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
        propagated = vmap(transition_fn_u, in_axes=(0, 0, None))(
            particle_keys, resampled_particles, input_t
        )
        log_obs = vmap(observation_fn_u, in_axes=(None, 0, None))(
            emission_t, propagated, input_t
        )
    sample = tree.map(lambda leaf: leaf[0], propagated)
    _validate_state_tree(
        sample, state_signature, name="transition_sampler output"
    )
    _validate_log_density_batch(
        cast(Array, log_obs),
        num_particles,
        name="log_observation_fn",
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
    ess_t = jnp.asarray(compute_ess(log_w_norm))
    info = BootstrapStepInfo(ancestors, ess_t, do_resample, log_ev_inc)
    return _BootstrapParticleStepResult(propagated, log_w_norm, info)


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
    result = _bootstrap_particle_step(
        step_key,
        state,
        checkpoint.ess,
        transition_sampler,
        log_observation_fn,
        emission_t,
        resampling_fn,
        resampling_threshold,
        input_t,
        state_signature,
    )
    log_ev_sum, correction = _neumaier_add(
        jnp.asarray(state.log_marginal_likelihood),
        checkpoint.log_evidence_compensation,
        result.info.log_evidence_increment,
    )
    new_state = ParticleState(
        result.particles,
        result.log_weights,
        log_ev_sum,
    )
    new_checkpoint = BootstrapCheckpoint(
        new_state,
        result.info.ess,
        correction,
    )
    return new_checkpoint, result.info


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
    """Advance a resumable bootstrap filter by one observation.

    Args:
        step_key: Explicit key for resampling and propagation.
        checkpoint: State returned by initialization or a prior step. Its
            log weights must be normalized and its stored ESS must agree
            with them.
        transition_sampler: Transition callback, optionally input-aware.
        log_observation_fn: Observation log-density callback returning a
            scalar with at least float32 precision.
        emission_t: Current observation ``y[t]``.
        resampling_fn: Particle resampling algorithm.
        resampling_threshold: ESS resampling threshold as a particle fraction.
        input_t: Optional ``inputs[t]`` reaching the transition and density.

    Returns:
        Updated normalized checkpoint and current-step diagnostics.

    Raises:
        ValueError: The checkpoint or propagated state is malformed.
        DegenerateWeightsError: Updated importance weights cannot normalize.

    Note:
        Concrete calls validate the checkpoint's weight normalization, ESS,
        and evidence state. Under a JAX transform, those data-dependent
        checks are skipped and malformed values propagate into the result.
    """
    state_signature = _validate_checkpoint(checkpoint)

    def body(carry, args):
        return _bootstrap_step(
            args[0],
            carry,
            transition_sampler,
            log_observation_fn,
            args[1],
            resampling_fn,
            resampling_threshold,
            input_t,
            state_signature,
        )

    new_checkpoint, batched_info = lax.scan(
        body, checkpoint, (step_key[None], emission_t[None])
    )
    info = tree.map(lambda leaf: leaf[0], batched_info)
    total = new_checkpoint.state.log_marginal_likelihood
    _raise_if_degenerate(total + new_checkpoint.log_evidence_compensation)
    return new_checkpoint, info


def bootstrap_update(
    step_keys: Shaped[Array, "..."],
    checkpoint: BootstrapCheckpoint,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions_chunk: Float[Array, "ntime emission_dim"],
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> tuple[BootstrapCheckpoint, ParticleFilterPosterior]:
    """Advance a checkpoint over an explicitly keyed observation chunk.

    Inputs align by index. Chunk evidence is conditional. MPS uses an eager
    public-step loop; other platforms use a compiled scan.

    Args:
        step_keys: One explicit PRNG key per chunk observation.
        checkpoint: State returned by initialization or an earlier update.
            Its log weights must be normalized and its stored ESS must agree
            with them.
        transition_sampler: Transition callback, optionally input-aware.
        log_observation_fn: Observation log-density callback returning a
            scalar with at least float32 precision.
        emissions_chunk: Consecutive observations after the checkpoint.
        resampling_fn: Particle resampling algorithm.
        resampling_threshold: ESS resampling threshold as a particle fraction.
        inputs: Optional inputs aligned one-for-one with the chunk.
        store_history: Whether to retain every chunk particle cloud.

    Returns:
        Updated checkpoint and chunk posterior. With ``store_history=False``,
        particle, weight, and ancestor histories have time-axis length one.

    Raises:
        TypeError: The host shell is called with traced checkpoint arrays.
        ValueError: The checkpoint, chunk, keys, inputs, or transition output
            is malformed.
        DegenerateWeightsError: Cumulative importance weights cannot normalize.
    """
    num_steps = emissions_chunk.shape[0]
    if num_steps == 0:
        raise ValueError("emissions_chunk must contain at least one row")
    key_error = "step_keys must be a batched PRNG key array"
    try:
        key_data = jr.key_data(step_keys)
    except (TypeError, ValueError) as error:
        raise ValueError(key_error) from error
    if key_data.ndim != 2:
        raise ValueError(key_error)
    if key_data.shape[0] != num_steps:
        raise ValueError(
            "step_keys and emissions_chunk must have the same leading "
            f"dimension; got {key_data.shape[0]} and {num_steps}"
        )
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_steps)
    )
    log_weights_0 = jnp.asarray(checkpoint.state.log_weights)
    if isinstance(log_weights_0, core.Tracer):
        raise TypeError("bootstrap_update cannot run under a JAX transform")
    device = next(iter(log_weights_0.devices()))
    checkpoint = device_put(checkpoint, device)
    if isinstance(checkpoint.state.log_weights, core.Tracer):
        raise TypeError("bootstrap_update cannot run under a JAX transform")
    platform = device.platform
    state_signature = _validate_checkpoint(checkpoint)
    scan_inputs = (step_keys, emissions_chunk)
    if inputs_arr is not None:
        scan_inputs = (*scan_inputs, inputs_arr)
    zero = jnp.zeros_like(jnp.asarray(checkpoint.state.log_marginal_likelihood))
    retained = jnp.arange(log_weights_0.shape[0], dtype=jnp.int32)

    def _record(next_checkpoint, info):
        traces = info.ess, info.log_evidence_increment
        if store_history:
            return (
                next_checkpoint.state.particles,
                next_checkpoint.state.log_weights,
                info.ancestors,
                *traces,
            )
        return traces

    def _advance(current, args):
        current_checkpoint, chunk_sum, chunk_correction, _retained = current
        if inputs_arr is None:
            step_key, emission_t = args
            input_t = None
        else:
            step_key, emission_t, input_t = args
        next_checkpoint, info = _bootstrap_step(
            step_key,
            current_checkpoint,
            transition_sampler,
            log_observation_fn,
            emission_t,
            resampling_fn,
            resampling_threshold,
            input_t,
            state_signature,
        )
        chunk_sum, chunk_correction = _neumaier_add(
            chunk_sum, chunk_correction, info.log_evidence_increment
        )
        return (
            (next_checkpoint, chunk_sum, chunk_correction, info.ancestors),
            _record(next_checkpoint, info),
        )

    if platform == "mps":
        # Remove this containment under smcx#38 after a fixed jax-mps release.
        current, chunk_sum, chunk_correction = checkpoint, zero, zero
        records = []
        for index in range(num_steps):
            input_t = None if inputs_arr is None else inputs_arr[index]
            current, info = bootstrap_step(
                step_keys[index],
                current,
                transition_sampler,
                log_observation_fn,
                emissions_chunk[index],
                resampling_fn,
                resampling_threshold,
                input_t=input_t,
            )
            if isinstance(current.state.log_weights, core.Tracer):
                raise TypeError(
                    "bootstrap_update cannot run under a JAX transform"
                )
            chunk_sum, chunk_correction = _neumaier_add(
                chunk_sum, chunk_correction, info.log_evidence_increment
            )
            retained = info.ancestors
            records.append(_record(current, info))
        carry = current, chunk_sum, chunk_correction, retained
        outputs = tree.map(lambda *values: jnp.stack(values), *records)
    else:
        carry, outputs = lax.scan(
            _advance,
            (checkpoint, zero, zero, retained),
            scan_inputs,
        )
    final_checkpoint, chunk_sum, chunk_correction, retained = carry
    if store_history:
        particles, log_weights, ancestors, ess, increments = outputs
    else:
        ess, increments = outputs
        particles = _particle_time_axis(final_checkpoint.state.particles)
        log_weights = final_checkpoint.state.log_weights[None]
        ancestors = retained[None]

    _raise_if_degenerate(
        final_checkpoint.state.log_marginal_likelihood
        + final_checkpoint.log_evidence_compensation
    )
    return final_checkpoint, ParticleFilterPosterior(
        marginal_loglik=chunk_sum + chunk_correction,
        filtered_particles=particles,
        filtered_log_weights=log_weights,
        ancestors=ancestors,
        ess=ess,
        log_evidence_increments=increments,
    )


def bootstrap_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a bootstrap (SIR) particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws from $p(z_1)$. ``particles`` may
            be a dense array or a nonempty PyTree whose array leaves all
            have leading size ``num_particles``.
        transition_sampler: Function ``(key, state[, input_t]) -> state``
            drawing from $p(z_t \mid z_{t-1})$. It receives one
            particle PyTree and must preserve its structure, leaf shapes,
            and dtypes. smcx ``vmap``-s it internally.
        log_observation_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            observation log-density $\log p(y_t \mid z_t)$.
            It must return a scalar with at least float32 precision.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles $N$.
        resampling_fn: Resampling algorithm matching the BlackJAX
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to `smcx.resampling.systematic`.
        resampling_threshold: ESS fraction (e.g. 0.5 means resample when
            ``ESS < 0.5 * N``), or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            Time is the zero-based emission index from 1 through T - 1.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. The latter becomes ``(T, 1)``. ``inputs[0]``
            reaches the initial sampler and observation callback;
            ``inputs[t]`` then reaches the transition into t and the
            observation at t.
        store_history: When False, the scan stacks no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full —
            dropping memory from O(T*N) to O(N).

    Returns:
        `smcx.containers.ParticleFilterPosterior` containing
        filtered particles, log weights, ancestor indices, the marginal
        log-likelihood estimate, and ESS trace. Structured particle
        histories preserve the state PyTree and add ``(T, N)`` to every
        leaf.

    Raises:
        DegenerateWeightsError: A particle-weight stage cannot be normalized
            (eager execution only; under ``jax.jit`` its nonfinite signal
            propagates).
        ValueError: Inputs are malformed, a criterion result is not a scalar
            Boolean, the initial state tree is empty or has a wrong leading
            axis, a transition changes its state contract, or an observation
            callback output is malformed.
    """
    num_timesteps = _validate_filter_inputs(emissions, num_particles)
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
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
        carry: tuple[ParticleState, Array, Array, Array],
        args: tuple[Array, ...],
    ):
        state, current_ess, correction, _prev_ancestors = carry
        if inputs_arr is None:
            step_key, y_t, time_index = args
            input_t = None
        else:
            step_key, y_t, input_t, time_index = args
        result = _bootstrap_particle_step(
            step_key,
            state,
            current_ess,
            transition_sampler,
            log_observation_fn,
            y_t,
            resampling_fn,
            resampling_threshold,
            input_t,
            state_signature,
            time_index,
        )
        info = result.info
        log_evidence, correction = _neumaier_add(
            jnp.asarray(state.log_marginal_likelihood),
            correction,
            info.log_evidence_increment,
        )
        new_state = ParticleState(
            particles=result.particles,
            log_weights=result.log_weights,
            log_marginal_likelihood=log_evidence,
        )
        if store_history:
            return (new_state, info.ess, correction, info.ancestors), (
                result.particles,
                result.log_weights,
                info.ancestors,
                info.ess,
                info.log_evidence_increment,
            )
        # Final-only mode: ancestors ride the carry (O(N)), the scan
        # stacks just the scalar traces.
        return (new_state, info.ess, correction, info.ancestors), (
            info.ess,
            info.log_evidence_increment,
        )

    # Run the scan over t = 1 ... T-1
    step_keys = jr.split(key, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    scan_inputs = (
        (step_keys, emissions[1:], time_indices)
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:], time_indices)
    )
    init_carry = (
        init_state,
        ess_0,
        jnp.zeros_like(log_ev_0),
        identity_ancestors,
    )
    if store_history:
        (
            (final_state, _, final_correction, _),
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
            (final_state, _, final_correction, final_ancestors),
            (ess_rest, log_ev_inc_rest),
        ) = lax.scan(_step, init_carry, scan_inputs)
        all_particles = _particle_time_axis(final_state.particles)
        all_log_w = final_state.log_weights[None]
        all_ancestors = final_ancestors[None]
    ess_0_arr: Array = jnp.asarray(ess_0)
    all_ess = _prepend(ess_0_arr, ess_rest)
    all_log_ev_inc = _prepend(jnp.asarray(log_ev_0), log_ev_inc_rest)
    final_log_evidence = final_state.log_marginal_likelihood + final_correction

    _raise_if_degenerate(final_log_evidence)

    return ParticleFilterPosterior(
        marginal_loglik=final_log_evidence,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=all_ess,
        log_evidence_increments=all_log_ev_inc,
    )
