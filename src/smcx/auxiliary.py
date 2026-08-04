# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS/resampling and validation, typed callback
# protocols, exogenous inputs, structured state, and optional history storage.

r"""Auxiliary particle filter (Pitt & Shephard, 1999).

The auxiliary particle filter (APF) improves on the bootstrap filter by
using a *look-ahead* step that biases resampling towards particles
likely to match the next observation **before** propagation.

At each time step the APF:

1. **First-stage weights** — combines the current normalised weights
   with the look-ahead log-density
   $\log g(y_{t+1} \mid x_t^i)$ to form first-stage weights.
2. **Resamples** (conditionally on ESS) using the first-stage weights.
3. **Propagates** resampled particles through the transition prior.
4. **Second-stage weights** — corrects for the look-ahead bias:
   $w_t^{(2)} = p(y_{t+1} \mid x_{t+1}^i) /
   g(y_{t+1} \mid x_t^{a_i})$.

When ``log_auxiliary_fn`` returns zero for all inputs, the APF
reduces to the bootstrap filter.

MPS uses a sequence of one-step scans; other platforms use one full scan.
"""

from typing import cast

from smcx._utils import (
    _canonicalize_inputs,
    _validate_filter_inputs,
    _validate_resampling_threshold,
)
from smcx.containers import ParticleFilterPosterior
from smcx.fk import CallbackNames, FeynmanKac, run_smc
from smcx.resampling import systematic
from smcx.types import (
    EmissionSequence,
    InitialSampler,
    InitialSamplerWithInput,
    InputSequence,
    LogObservationFn,
    LogObservationFnWithInput,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
    TransitionSampler,
    TransitionSamplerWithInput,
)


def _auxiliary_fk(
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    log_auxiliary_fn: LogObservationFn | LogObservationFnWithInput,
    # Canonicalized (T, D) arrays with model-owned dtypes (integer
    # and Boolean supported): runtime-lax sequence aliases, never Float.
    emissions: EmissionSequence,
    inputs_arr: InputSequence | None,
) -> FeynmanKac:
    """Derive the auxiliary Feynman-Kac model from its callbacks.

    The mutation kernel is the transition prior and the potential is
    the observation density; the look-ahead enters as the twist, whose
    first-stage selection, ancestor correction, and two-factor
    evidence increment are owned by the generic loop.
    """
    names = CallbackNames(
        m0="initial_sampler output",
        m="transition_sampler output",
        log_g="log_observation_fn",
        log_eta="log_auxiliary_fn",
    )
    if inputs_arr is None:
        init_fn = cast(InitialSampler, initial_sampler)
        transition_fn = cast(TransitionSampler, transition_sampler)
        observation_fn = cast(LogObservationFn, log_observation_fn)
        auxiliary_fn = cast(LogObservationFn, log_auxiliary_fn)

        def m0(key, num_particles, context_t):
            del context_t
            return init_fn(key, num_particles)

        def m(key, parent, context_t):
            del context_t
            return transition_fn(key, parent)

        def log_g(parent, state, context_t):
            del parent
            return observation_fn(context_t[0], state)

        def log_eta(state, context_t):
            return auxiliary_fn(context_t[0], state)

        return FeynmanKac(
            m0=m0,
            m=m,
            log_g=log_g,
            contexts=(emissions,),
            names=names,
            log_eta=log_eta,
        )

    init_fn_u = cast(InitialSamplerWithInput, initial_sampler)
    transition_fn_u = cast(TransitionSamplerWithInput, transition_sampler)
    observation_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
    auxiliary_fn_u = cast(LogObservationFnWithInput, log_auxiliary_fn)

    def m0_u(key, num_particles, context_t):
        return init_fn_u(key, num_particles, context_t[1])

    def m_u(key, parent, context_t):
        return transition_fn_u(key, parent, context_t[1])

    def log_g_u(parent, state, context_t):
        del parent
        return observation_fn_u(context_t[0], state, context_t[1])

    def log_eta_u(state, context_t):
        return auxiliary_fn_u(context_t[0], state, context_t[1])

    return FeynmanKac(
        m0=m0_u,
        m=m_u,
        log_g=log_g_u,
        contexts=(emissions, inputs_arr),
        names=names,
        log_eta=log_eta_u,
    )


def auxiliary_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    log_auxiliary_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: EmissionSequence,
    num_particles: int,
    *,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run an auxiliary particle filter (Pitt & Shephard, 1999).

    Args:
        key: JAX PRNG key. Split once for the initial cloud, then
            into one key per subsequent step; each step key splits
            into a resampling key and per-particle transition keys
            (the frozen ``run_smc`` schedule).
        initial_sampler: Function ``(key, num_particles[, input_0]) ->
            particles`` that draws from $p(z_1)$. ``particles`` may
            be a dense array or a nonempty PyTree whose array leaves all
            have leading size ``num_particles``.
        transition_sampler: Function ``(key, state[, input_t]) -> state`` that
            draws from $p(z_t \mid z_{t-1})$. It receives one
            particle PyTree and must preserve its structure, leaf shapes,
            and dtypes. smcx ``vmap``-s it internally.
        log_observation_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            observation log-density $\log p(y_t \mid z_t)$.
            It must return a scalar with at least float32 precision.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.
        log_auxiliary_fn: Function
            ``(emission, state[, input_t]) -> log_prob`` that evaluates the
            look-ahead log-density
            $\log g(y_{t+1} \mid x_t)$.
            It must return a scalar with at least float32 precision.
            Will be ``vmap``-ped over the particle dimension (second
            argument) internally.  When this returns zero for all
            inputs the APF reduces to the bootstrap filter.
        emissions: Scalar ``(T,)`` or vector ``(T, D)`` observations.
            Rank-one data become ``(T, 1)``; dtype is preserved.
        num_particles: Number of particles $N$.
        resampling_fn: Resampling algorithm matching the BlackJAX
            signature ``(key, weights, num_samples) -> indices``.
            Defaults to `smcx.resampling.systematic`.
        resampling_threshold: ESS fraction (e.g. 0.5 means resample when
            ``ESS < 0.5 * N``), or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            Numeric values must be finite and nonnegative; zero disables
            resampling and values above one force it at every update.
            The callback receives the first-stage weights and ESS at the
            zero-based emission indices 1 through T - 1.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)`` and a nonempty event. Input zero reaches
            initialization; each later
            input reaches the transition, observation, and auxiliary
            callbacks aligned at the same time step.
        store_history: When False, the filter retains no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

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
        ValueError: Observations, inputs, or the threshold are malformed, a
            criterion result is not a scalar Boolean, the initial state tree
            is empty or has a wrong leading axis, a transition changes its
            state contract, or a log-density callback output is malformed.
    """
    _validate_resampling_threshold(resampling_threshold)
    emissions, num_timesteps = _validate_filter_inputs(
        emissions,
        num_particles,
    )
    inputs_arr = (
        None if inputs is None else _canonicalize_inputs(inputs, num_timesteps)
    )
    fk = _auxiliary_fk(
        initial_sampler,
        transition_sampler,
        log_observation_fn,
        log_auxiliary_fn,
        emissions,
        inputs_arr,
    )
    return run_smc(
        key,
        fk,
        num_particles,
        resampling_fn=resampling_fn,
        resampling_threshold=resampling_threshold,
        store_history=store_history,
        gate_stage_normalizers=True,
    )
