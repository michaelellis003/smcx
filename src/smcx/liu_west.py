# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling and validation, shrinkage
# guidance, typed callbacks, exogenous inputs, and optional history storage.

r"""Liu-West particle filter for joint state-parameter estimation.

The Liu-West filter (Liu & West, 2001) extends the auxiliary particle
filter to estimate static model parameters alongside latent states.
Parameters are propagated using kernel density smoothing:

.. math::

    \phi_t^i = a \phi_{t-1}^{a_i}
             + (1 - a) \bar{\phi}_{t-1}
             + h \, \varepsilon^i, \quad
    \varepsilon^i \sim \mathcal{N}(0, V_{t-1})

where :math:`a` is the shrinkage parameter, :math:`\bar{\phi}` is the
weighted parameter mean, :math:`V` is the weighted parameter covariance,
and :math:`h^2 = 1 - a^2`.

The implementation uses :func:`jax.lax.scan` so the full time-loop is
compiled into a single XLA program.

References:
    Liu, J. and West, M. (2001). Combined Parameter and State Estimation
    in Simulation-Based Filtering. *Sequential Monte Carlo Methods in
    Practice*, 197--223.
    https://doi.org/10.1007/978-1-4757-3437-9_10
"""

import math
from typing import NamedTuple, cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Float, Int

from smcx._utils import (
    _canonicalize_inputs,
    _conditional_resample,
    _prepend,
    _raise_if_degenerate,
    _TreeSignature,
    _validate_filter_inputs,
    _validate_log_density_batch,
    _validate_particle_cloud,
    _validate_state_tree,
)
from smcx.containers import LiuWestPosterior
from smcx.resampling import systematic
from smcx.types import (
    DenseInitialSampler,
    DenseInitialSamplerWithInput,
    InputSequence,
    ParamInitialSampler,
    ParamLogObservationFn,
    ParamLogObservationFnWithInput,
    ParamTransitionSampler,
    ParamTransitionSamplerWithInput,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize, normalize


class _LiuWestStepCarry(NamedTuple):
    particles: Float[Array, "num_particles state_dim"]
    params: Float[Array, "num_particles param_dim"]
    log_weights: Float[Array, " num_particles"]
    log_marginal_likelihood: Float[Array, ""]
    ancestors: Int[Array, " num_particles"]


class _LiuWestStepInput(NamedTuple):
    emission: Float[Array, " emission_dim"]
    input_t: Float[Array, " input_dim"] | None
    time_index: Int[Array, ""]


class _LiuWestStepOutput(NamedTuple):
    particles: Float[Array, "num_particles state_dim"]
    params: Float[Array, "num_particles param_dim"]
    log_weights: Float[Array, " num_particles"]
    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    log_evidence_increment: Float[Array, ""]


def _validate_dense_initial_cloud(
    values: object,
    num_particles: int,
    *,
    name: str,
) -> _TreeSignature:
    """Require one nonempty floating matrix with a particle axis."""
    signature = _validate_particle_cloud(
        values,
        num_particles,
        name=name,
    )
    if signature.paths != ("<root>",):
        raise ValueError(
            f"{name} must be a JAX array with shape (num_particles, dimension)"
        )
    array = cast(Array, values)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape (num_particles, dimension) with "
            f"dimension >= 1; got {array.shape}"
        )
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise ValueError(
            f"{name} must have a floating dtype; got {array.dtype}"
        )
    return signature


def _init_liu_west(
    init_key: PRNGKeyT,
    initial_sampler: DenseInitialSampler | DenseInitialSamplerWithInput,
    param_initial_sampler: ParamInitialSampler,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    first_emission: Array,
    num_particles: int,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, _TreeSignature]:
    """Initialise Liu-West filter at t=0.

    Args:
        init_key: PRNG key for initialisation.
        initial_sampler: State prior sampler.
        param_initial_sampler: Parameter prior sampler.
        log_observation_fn: Observation log-density.
        first_emission: First observation y_0.
        num_particles: Number of particles N.
        input_t: Optional t=0 input passed to the state initializer
            and after parameters to the observation callback.

    Returns:
        Tuple of (particles_0, params_0, log_w_0, log_ev_0, ess_0,
        identity_ancestors, state_signature).
    """
    log_n = jnp.asarray(math.log(num_particles))
    k_z, k_p = jr.split(init_key)
    if input_t is None:
        state_init = cast(DenseInitialSampler, initial_sampler)
        particles_0 = state_init(k_z, num_particles)
    else:
        state_init_u = cast(DenseInitialSamplerWithInput, initial_sampler)
        particles_0 = state_init_u(k_z, num_particles, input_t)
    params_0 = param_initial_sampler(k_p, num_particles)
    state_signature = _validate_dense_initial_cloud(
        particles_0,
        num_particles,
        name="initial_sampler output",
    )
    _validate_dense_initial_cloud(
        params_0,
        num_particles,
        name="param_initial_sampler output",
    )

    if input_t is None:
        observation_fn = cast(ParamLogObservationFn, log_observation_fn)
        log_obs_0 = cast(
            Array,
            vmap(lambda z, p: observation_fn(first_emission, z, p))(
                particles_0, params_0
            ),
        )
    else:
        observation_fn_u = cast(
            ParamLogObservationFnWithInput, log_observation_fn
        )
        log_obs_0 = cast(
            Array,
            vmap(lambda z, p: observation_fn_u(first_emission, z, p, input_t))(
                particles_0, params_0
            ),
        )

    _validate_log_density_batch(
        log_obs_0,
        num_particles,
        name="log_observation_fn",
    )
    log_w_0, log_sum_0 = log_normalize(log_obs_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0 = jnp.asarray(compute_ess(log_w_0))
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    return (
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity,
        state_signature,
    )


def _liu_west_step(
    carry: _LiuWestStepCarry,
    inputs_t: _LiuWestStepInput,
    key_t: PRNGKeyT,
    *,
    transition_sampler: ParamTransitionSampler
    | ParamTransitionSamplerWithInput,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    log_auxiliary_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    resampling_fn: ResamplingFn,
    resampling_threshold: float | ResamplingCriterion,
    log_num_particles: Float[Array, ""],
    shrinkage: Float[Array, ""],
    kernel_variance: Float[Array, ""],
    state_signature: _TreeSignature,
) -> tuple[_LiuWestStepCarry, _LiuWestStepOutput]:
    particles, params, log_weights, log_ml, _ = carry
    emission_t, input_t, time_index = inputs_t
    num_particles = log_weights.shape[0]
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    resample_key, parameter_key, transition_key = jr.split(key_t, 3)

    def _evaluate(
        callback: ParamLogObservationFn | ParamLogObservationFnWithInput,
        values: Float[Array, "num_particles state_dim"],
        parameter_values: Float[Array, "num_particles param_dim"],
    ) -> Array:
        if input_t is None:
            callback_fn = cast(ParamLogObservationFn, callback)
            result = vmap(lambda z, p: callback_fn(emission_t, z, p))(
                values, parameter_values
            )
        else:
            callback_fn_u = cast(ParamLogObservationFnWithInput, callback)
            result = vmap(
                lambda z, p: callback_fn_u(emission_t, z, p, input_t)
            )(values, parameter_values)
        return cast(Array, result)

    weights = normalize(log_weights)
    param_mean = jnp.sum(weights[:, None] * params, axis=0)
    param_dev = params - param_mean[None, :]
    param_cov = jnp.einsum("n,nd,ne->de", weights, param_dev, param_dev)
    shrunk = shrinkage * params + (1.0 - shrinkage) * param_mean[None, :]
    log_aux = _evaluate(log_auxiliary_fn, particles, shrunk)
    _validate_log_density_batch(log_aux, num_particles, name="log_auxiliary_fn")
    log_first_norm, log_first_sum = log_normalize(log_weights + log_aux)
    first_ess = jnp.asarray(compute_ess(log_first_norm))
    do_resample, ancestors = _conditional_resample(
        resample_key,
        log_first_norm,
        first_ess,
        resampling_fn,
        resampling_threshold,
        num_particles,
        identity,
        time_index,
    )
    param_dim = params.shape[1]
    jitter = 1e-8 * jnp.eye(param_dim)
    chol = jnp.linalg.cholesky(kernel_variance * param_cov + jitter)
    eps = jr.normal(parameter_key, (num_particles, param_dim))
    new_params = shrunk[ancestors] + eps @ chol.T
    particle_keys = jr.split(transition_key, num_particles)
    if input_t is None:
        transition_fn = cast(ParamTransitionSampler, transition_sampler)
        propagated = vmap(transition_fn)(
            particle_keys, particles[ancestors], new_params
        )
    else:
        transition_fn_u = cast(
            ParamTransitionSamplerWithInput, transition_sampler
        )
        propagated = vmap(transition_fn_u, in_axes=(0, 0, 0, None))(
            particle_keys, particles[ancestors], new_params, input_t
        )
    batched_shapes = ((num_particles, *state_signature.shapes[0]),)
    batched_signature = state_signature._replace(shapes=batched_shapes)
    _validate_state_tree(
        propagated, batched_signature, name="transition_sampler output"
    )
    log_obs = _evaluate(log_observation_fn, propagated, new_params)
    _validate_log_density_batch(
        log_obs, num_particles, name="log_observation_fn"
    )
    log_w_unnorm = jnp.where(
        do_resample,
        log_obs - log_aux[ancestors],
        log_weights + log_obs,
    )
    log_w_norm, log_sum = log_normalize(log_w_unnorm)
    log_ev_inc = jnp.where(
        do_resample,
        log_first_sum + log_sum - log_num_particles,
        log_sum,
    )
    new_carry = _LiuWestStepCarry(
        propagated,
        new_params,
        log_w_norm,
        log_ml + log_ev_inc,
        ancestors,
    )
    ess = jnp.asarray(compute_ess(log_w_norm))
    output = _LiuWestStepOutput(
        propagated, new_params, log_w_norm, ancestors, ess, log_ev_inc
    )
    return new_carry, output


def liu_west_filter(
    key: PRNGKeyT,
    initial_sampler: DenseInitialSampler | DenseInitialSamplerWithInput,
    transition_sampler: ParamTransitionSampler
    | ParamTransitionSamplerWithInput,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    log_auxiliary_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    param_initial_sampler: ParamInitialSampler,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    shrinkage: float = 0.95,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    *,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> LiuWestPosterior:
    r"""Run a Liu-West particle filter (Liu & West, 2001).

    Jointly estimates latent states and static parameters using
    auxiliary particle filtering with kernel density smoothing for
    parameter propagation.

    Args:
        key: JAX PRNG key.
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws a nonempty floating array of shape
            ``(num_particles, state_dim)``.
        transition_sampler: Function
            ``(key, state, params[, input_t]) -> state`` that draws from
            the transition distribution while preserving the initial state
            shape and dtype.
        log_observation_fn: Function
            ``(emission, state, params[, input_t]) -> log_prob`` that
            evaluates the observation log-density.
        log_auxiliary_fn: Function
            ``(emission, state, params[, input_t]) -> log_prob`` that
            evaluates the look-ahead log-density.
        param_initial_sampler: Function
            ``(key, num_particles) -> params`` that draws from the
            prior parameter distribution. Returns a nonempty floating array
            of shape ``(num_particles, param_dim)``.
        emissions: Observed emissions, shape ``(T, D)``.
        num_particles: Number of particles :math:`N`.
        shrinkage: Shrinkage parameter :math:`a \in (0, 1)`.
            Controls the balance between the kernel smoothing
            exploration and prior concentration.  Higher values
            give tighter parameter posteriors.

            .. warning::

                The shrinkage parameter has no generative
                interpretation: it introduces artificial dynamics
                into the parameter evolution that do not correspond
                to any probabilistic model.  Results can be
                sensitive to this choice.  We recommend running the
                filter under several values (e.g. 0.95, 0.975,
                0.99) and reporting the range of posterior and
                evidence estimates.
        resampling_fn: Resampling algorithm.  Defaults to systematic.
        resampling_threshold: ESS fraction, or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            The callback receives the first-stage weights and ESS at the
            zero-based emission indices 1 through T - 1.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Inputs follow ``params`` in every callback;
            the parameter initializer remains input-independent.
        store_history: When False, only the final step's
            particle/param/weight/ancestor arrays are returned (time
            axis length 1); ``ess``/``log_evidence_increments`` stay
            full.

    Returns:
        :class:`~smcx.containers.LiuWestPosterior` containing
        filtered particles, parameters, log weights, ancestor indices,
        the marginal log-likelihood estimate, and ESS trace.

    Raises:
        ValueError: Inputs, particle count, shrinkage, callback output, or a
            criterion result is structurally invalid.
    """
    num_timesteps = _validate_filter_inputs(emissions, num_particles)
    if not 0.0 < shrinkage < 1.0:
        raise ValueError(
            f"shrinkage must be in the open interval (0, 1); got {shrinkage}"
        )
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    key, init_key = jr.split(key)
    log_n = jnp.asarray(math.log(num_particles))
    a = jnp.asarray(shrinkage)
    h_sq = 1.0 - a**2

    (
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        ess_0,
        identity_ancestors,
        state_signature,
    ) = (
        _init_liu_west(
            init_key,
            initial_sampler,
            param_initial_sampler,
            log_observation_fn,
            emissions[0],
            num_particles,
        )
        if inputs_arr is None
        else _init_liu_west(
            init_key,
            initial_sampler,
            param_initial_sampler,
            log_observation_fn,
            emissions[0],
            num_particles,
            inputs_arr[0],
        )
    )

    # --- Scan body for t = 1, ..., T-1 -------------------------------------
    def _step(carry: _LiuWestStepCarry, args: tuple[Array, ...]):
        if inputs_arr is None:
            step_key, y_t, time_index = args
            input_t = None
        else:
            step_key, y_t, input_t, time_index = args
        next_carry, output = _liu_west_step(
            carry,
            _LiuWestStepInput(y_t, input_t, time_index),
            step_key,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation_fn,
            log_auxiliary_fn=log_auxiliary_fn,
            resampling_fn=resampling_fn,
            resampling_threshold=resampling_threshold,
            log_num_particles=log_n,
            shrinkage=a,
            kernel_variance=h_sq,
            state_signature=state_signature,
        )
        if store_history:
            return next_carry, output
        # In final-only mode, the scan stacks only the scalar traces;
        # final arrays come from the carry.
        return next_carry, (output.ess, output.log_evidence_increment)

    init_carry = _LiuWestStepCarry(
        particles_0,
        params_0,
        log_w_0,
        log_ev_0,
        identity_ancestors,
    )
    step_keys = jr.split(key, num_timesteps - 1)
    time_indices = jnp.arange(1, num_timesteps, dtype=jnp.int32)
    scan_inputs = (
        (step_keys, emissions[1:], time_indices)
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:], time_indices)
    )

    if store_history:
        final_carry, outputs = lax.scan(_step, init_carry, scan_inputs)
        all_particles = _prepend(particles_0, outputs.particles)
        all_params = _prepend(params_0, outputs.params)
        all_log_w = _prepend(log_w_0, outputs.log_weights)
        all_ancestors = _prepend(identity_ancestors, outputs.ancestors)
        ess_rest = outputs.ess
        log_ev_inc_rest = outputs.log_evidence_increment
    else:
        final_carry, (ess_rest, log_ev_inc_rest) = lax.scan(
            _step, init_carry, scan_inputs
        )
        all_particles = final_carry.particles[None]
        all_params = final_carry.params[None]
        all_log_w = final_carry.log_weights[None]
        all_ancestors = final_carry.ancestors[None]

    final_log_ml = final_carry.log_marginal_likelihood

    _raise_if_degenerate(final_log_ml)

    return LiuWestPosterior(
        marginal_loglik=final_log_ml,
        filtered_particles=all_particles,
        filtered_log_weights=all_log_w,
        ancestors=all_ancestors,
        ess=_prepend(jnp.asarray(ess_0), ess_rest),
        log_evidence_increments=_prepend(
            jnp.asarray(log_ev_0), log_ev_inc_rest
        ),
        filtered_params=all_params,
    )
