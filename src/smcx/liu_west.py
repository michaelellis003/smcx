# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

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
"""

import math
from typing import cast

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Float

from smcx._utils import (
    _canonicalize_inputs,
    _conditional_resample,
    _prepend,
    _raise_if_degenerate,
)
from smcx.containers import LiuWestPosterior
from smcx.resampling import systematic
from smcx.types import (
    InitialSampler,
    InitialSamplerWithInput,
    InputSequence,
    ParamInitialSampler,
    ParamLogObservationFn,
    ParamLogObservationFnWithInput,
    ParamTransitionSampler,
    ParamTransitionSamplerWithInput,
    PRNGKeyT,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize, normalize

# Carry type: (particles, params, log_weights, log_marginal_likelihood)
_Carry = tuple[Array, Array, Array, Array, Array]


def _init_liu_west(
    init_key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    param_initial_sampler: ParamInitialSampler,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    first_emission: Array,
    num_particles: int,
    input_t: Float[Array, " input_dim"] | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
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
        identity_ancestors).
    """
    log_n = jnp.asarray(math.log(num_particles))
    k_z, k_p = jr.split(init_key)
    if input_t is None:
        state_init = cast(InitialSampler, initial_sampler)
        particles_0 = state_init(k_z, num_particles)
    else:
        state_init_u = cast(InitialSamplerWithInput, initial_sampler)
        particles_0 = state_init_u(k_z, num_particles, input_t)
    params_0 = param_initial_sampler(k_p, num_particles)

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

    log_w_0, log_sum_0 = log_normalize(log_obs_0)
    log_ev_0 = log_sum_0 - log_n
    ess_0 = jnp.asarray(compute_ess(log_w_0))
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    return particles_0, params_0, log_w_0, log_ev_0, ess_0, identity


def liu_west_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: ParamTransitionSampler
    | ParamTransitionSamplerWithInput,
    log_observation_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    log_auxiliary_fn: ParamLogObservationFn | ParamLogObservationFnWithInput,
    param_initial_sampler: ParamInitialSampler,
    emissions: Float[Array, "ntime emission_dim"],
    num_particles: int,
    shrinkage: float = 0.95,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
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
            particles`` that draws from the initial state distribution.
        transition_sampler: Function
            ``(key, state, params[, input_t]) -> state`` that draws from
            the transition distribution.
        log_observation_fn: Function
            ``(emission, state, params[, input_t]) -> log_prob`` that
            evaluates the observation log-density.
        log_auxiliary_fn: Function
            ``(emission, state, params[, input_t]) -> log_prob`` that
            evaluates the look-ahead log-density.
        param_initial_sampler: Function
            ``(key, num_particles) -> params`` that draws from the
            prior parameter distribution.  Returns array of shape
            ``(num_particles, param_dim)``.
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
        resampling_threshold: ESS fraction triggering resampling.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)``. Inputs follow ``params`` in every callback;
            the parameter initializer remains input-independent.
        store_history: When False (ADR-0011), only the final step's
            particle/param/weight/ancestor arrays are returned (time
            axis length 1); ``ess``/``log_evidence_increments`` stay
            full.

    Returns:
        :class:`~smcx.containers.LiuWestPosterior` containing
        filtered particles, parameters, log weights, ancestor indices,
        the marginal log-likelihood estimate, and ESS trace.

    Raises:
        ValueError: ``inputs`` is not rank one or two, or its leading
            dimension does not match ``emissions``.
    """
    inputs_arr = (
        None
        if inputs is None
        else _canonicalize_inputs(inputs, emissions.shape[0])
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
    def _step(
        carry: _Carry,
        args: tuple[Array, ...],
    ):
        particles, params, log_weights, log_ml, _prev_anc = carry
        if inputs_arr is None:
            step_key, y_t = args
            input_t = None
        else:
            step_key, y_t, input_t = args
        k1, k2, k3 = jr.split(step_key, 3)

        # Weighted parameter moments for kernel smoothing
        w = normalize(log_weights)
        param_mean = jnp.sum(w[:, None] * params, axis=0)
        param_dev = params - param_mean[None, :]
        param_cov = jnp.einsum("n,nd,ne->de", w, param_dev, param_dev)

        # Shrunk means: m_i = a * phi_i + (1-a) * phi_bar
        shrunk = a * params + (1.0 - a) * param_mean[None, :]

        # 1. First-stage weights using shrunk params
        if input_t is None:
            auxiliary_fn = cast(ParamLogObservationFn, log_auxiliary_fn)
            log_aux = cast(
                Array,
                vmap(lambda z, p: auxiliary_fn(y_t, z, p))(particles, shrunk),
            )
        else:
            auxiliary_fn_u = cast(
                ParamLogObservationFnWithInput, log_auxiliary_fn
            )
            log_aux = cast(
                Array,
                vmap(lambda z, p: auxiliary_fn_u(y_t, z, p, input_t))(
                    particles, shrunk
                ),
            )
        log_first_norm, log_first_sum = log_normalize(log_weights + log_aux)

        # 2. Conditionally resample
        threshold = resampling_threshold * num_particles
        do_resample, ancestors = _conditional_resample(
            k1,
            log_first_norm,
            resampling_fn,
            threshold,
            num_particles,
            identity_ancestors,
        )

        # 3. Propagate params via kernel smoothing + propagate states
        param_dim = params.shape[1]
        # Jitter prevents NaN from cholesky on singular covariance
        # (e.g. when all particles share the same parameter value).
        jitter = 1e-8 * jnp.eye(param_dim)
        chol = jnp.linalg.cholesky(h_sq * param_cov + jitter)
        eps = jr.normal(k2, (num_particles, param_dim))
        new_params = shrunk[ancestors] + eps @ chol.T

        keys = jr.split(k3, num_particles)
        if input_t is None:
            transition_fn = cast(ParamTransitionSampler, transition_sampler)
            propagated = vmap(transition_fn)(
                keys,
                particles[ancestors],
                new_params,
            )
        else:
            transition_fn_u = cast(
                ParamTransitionSamplerWithInput, transition_sampler
            )
            propagated = vmap(transition_fn_u, in_axes=(0, 0, 0, None))(
                keys,
                particles[ancestors],
                new_params,
                input_t,
            )

        # 4. Second-stage weights
        if input_t is None:
            observation_fn = cast(ParamLogObservationFn, log_observation_fn)
            log_obs = cast(
                Array,
                vmap(lambda z, p: observation_fn(y_t, z, p))(
                    propagated, new_params
                ),
            )
        else:
            observation_fn_u = cast(
                ParamLogObservationFnWithInput, log_observation_fn
            )
            log_obs = cast(
                Array,
                vmap(lambda z, p: observation_fn_u(y_t, z, p, input_t))(
                    propagated, new_params
                ),
            )
        log_w_unnorm = jnp.where(
            do_resample,
            log_obs - log_aux[ancestors],
            log_weights + log_obs,
        )
        log_w_norm, log_sum = log_normalize(log_w_unnorm)

        log_ev_inc = jnp.where(
            do_resample,
            log_first_sum + log_sum - log_n,
            log_sum,
        )

        new_carry = (
            propagated,
            new_params,
            log_w_norm,
            log_ml + log_ev_inc,
            ancestors,
        )
        ess_t = jnp.asarray(compute_ess(log_w_norm))
        if store_history:
            return new_carry, (
                propagated,
                new_params,
                log_w_norm,
                ancestors,
                ess_t,
                log_ev_inc,
            )
        # Final-only mode (ADR-0011): the lean scan stacks only the
        # scalar traces; final arrays come from the carry.
        return new_carry, (ess_t, log_ev_inc)

    init_carry = (particles_0, params_0, log_w_0, log_ev_0, identity_ancestors)
    step_keys = jr.split(key, emissions.shape[0] - 1)
    scan_inputs = (
        (step_keys, emissions[1:])
        if inputs_arr is None
        else (step_keys, emissions[1:], inputs_arr[1:])
    )

    if store_history:
        (
            final_carry,
            (
                particles_rest,
                params_rest,
                log_w_rest,
                ancestors_rest,
                ess_rest,
                log_ev_inc_rest,
            ),
        ) = lax.scan(_step, init_carry, scan_inputs)
        all_particles = _prepend(particles_0, particles_rest)
        all_params = _prepend(params_0, params_rest)
        all_log_w = _prepend(log_w_0, log_w_rest)
        all_ancestors = _prepend(identity_ancestors, ancestors_rest)
    else:
        (
            final_carry,
            (ess_rest, log_ev_inc_rest),
        ) = lax.scan(_step, init_carry, scan_inputs)
        fp, fpar, flw, _, fanc = final_carry
        all_particles = fp[None]
        all_params = fpar[None]
        all_log_w = flw[None]
        all_ancestors = fanc[None]

    final_log_ml = final_carry[3]

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
