# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Guided (proposal-based) particle filter.

The guided filter propagates through a user proposal
$q(z_t \mid z_{t-1}, y_t)$ — which, unlike the bootstrap
transition prior, can see the current observation — and corrects with
the general importance weight
$w \propto g(y_t \mid z_t)\, f(z_t \mid z_{t-1}) /
q(z_t \mid z_{t-1}, y_t)$ [Doucet, Godsill & Andrieu, 2000].
Approximate proposals (EKF/UKF/Laplace) MUST use this general
formula — the predictive-likelihood shortcut is exact only for the
locally optimal proposal. With ``q = f`` the filter reduces to
bootstrap (same key stream, agreement to floating-point tolerance —
the ``f/q`` cancellation is mathematical, not bitwise; tested).
"""

from typing import cast

from jax import vmap
from jaxtyping import Array

from smcx._utils import (
    _canonicalize_inputs,
    _validate_filter_inputs,
    _validate_log_density_batch,
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
    LogProposalFn,
    LogProposalFnWithInput,
    LogTransitionFn,
    LogTransitionFnWithInput,
    PRNGKeyT,
    ProposalSampler,
    ProposalSamplerWithInput,
    ResamplingCriterion,
    ResamplingFn,
)


def _guided_fk(
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    proposal_sampler: ProposalSampler | ProposalSamplerWithInput,
    log_proposal_fn: LogProposalFn | LogProposalFnWithInput,
    log_transition_fn: LogTransitionFn | LogTransitionFnWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    # Canonicalized (T, D) arrays with model-owned dtypes (integer
    # and Boolean supported): runtime-lax sequence aliases, never Float.
    emissions: EmissionSequence,
    inputs_arr: InputSequence | None,
) -> FeynmanKac:
    """Derive the guided Feynman-Kac model from its callbacks.

    The mutation kernel is the proposal (which sees the emission) and
    the step potential is the composite ``g * f / q``, expressed as a
    cloud-level potential so each factor keeps its own error name.
    Initialization weights by the observation density alone, exactly
    as before the loop extraction.
    """
    names = CallbackNames(
        m0="initial_sampler output",
        m="proposal_sampler output",
        log_g="log_observation_fn",
    )
    if inputs_arr is None:
        proposal_fn = cast(ProposalSampler, proposal_sampler)
        observation_fn = cast(LogObservationFn, log_observation_fn)
        transition_fn = cast(LogTransitionFn, log_transition_fn)
        proposal_density = cast(LogProposalFn, log_proposal_fn)

        def m0(key, num_particles, context_t):
            del context_t
            return cast(InitialSampler, initial_sampler)(key, num_particles)

        def m(key, parent, context_t):
            return proposal_fn(key, parent, context_t[0])

        def log_g(parent, state, context_t):
            del parent
            return observation_fn(context_t[0], state)

        def log_g_batch(parents, propagated, context_t):
            y_t = context_t[0]
            log_obs = cast(
                Array, vmap(lambda z: observation_fn(y_t, z))(propagated)
            )
            log_f = vmap(transition_fn)(propagated, parents)
            log_q = vmap(
                lambda z_new, z_old: proposal_density(y_t, z_new, z_old)
            )(propagated, parents)
            num_particles = log_obs.shape[0]
            for name, values in (
                ("log_observation_fn", log_obs),
                ("log_transition_fn", log_f),
                ("log_proposal_fn", log_q),
            ):
                _validate_log_density_batch(
                    cast(Array, values), num_particles, name=name
                )
            return log_obs + log_f - log_q

        return FeynmanKac(
            m0=m0,
            m=m,
            log_g=log_g,
            contexts=(emissions,),
            names=names,
            log_g_batch=log_g_batch,
        )

    proposal_fn_u = cast(ProposalSamplerWithInput, proposal_sampler)
    observation_fn_u = cast(LogObservationFnWithInput, log_observation_fn)
    transition_fn_u = cast(LogTransitionFnWithInput, log_transition_fn)
    proposal_density_u = cast(LogProposalFnWithInput, log_proposal_fn)

    def m0_u(key, num_particles, context_t):
        return cast(InitialSamplerWithInput, initial_sampler)(
            key, num_particles, context_t[1]
        )

    def m_u(key, parent, context_t):
        return proposal_fn_u(key, parent, context_t[0], context_t[1])

    def log_g_u(parent, state, context_t):
        del parent
        return observation_fn_u(context_t[0], state, context_t[1])

    def log_g_batch_u(parents, propagated, context_t):
        y_t, input_t = context_t
        log_obs = cast(
            Array,
            vmap(lambda z: observation_fn_u(y_t, z, input_t))(propagated),
        )
        log_f = vmap(transition_fn_u, in_axes=(0, 0, None))(
            propagated, parents, input_t
        )
        log_q = vmap(
            lambda z_new, z_old: proposal_density_u(y_t, z_new, z_old, input_t)
        )(propagated, parents)
        num_particles = log_obs.shape[0]
        for name, values in (
            ("log_observation_fn", log_obs),
            ("log_transition_fn", log_f),
            ("log_proposal_fn", log_q),
        ):
            _validate_log_density_batch(
                cast(Array, values), num_particles, name=name
            )
        return log_obs + log_f - log_q

    return FeynmanKac(
        m0=m0_u,
        m=m_u,
        log_g=log_g_u,
        contexts=(emissions, inputs_arr),
        names=names,
        log_g_batch=log_g_batch_u,
    )


def guided_filter(
    key: PRNGKeyT,
    initial_sampler: InitialSampler | InitialSamplerWithInput,
    proposal_sampler: ProposalSampler | ProposalSamplerWithInput,
    log_proposal_fn: LogProposalFn | LogProposalFnWithInput,
    log_transition_fn: LogTransitionFn | LogTransitionFnWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: EmissionSequence,
    num_particles: int,
    *,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    inputs: InputSequence | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a guided particle filter.

    Args:
        key: JAX PRNG key.
        initial_sampler: ``(key, num_particles[, input_0]) -> particles``
            drawing from $p(z_1)$. ``particles`` may be a dense
            array or a nonempty PyTree whose array leaves all have leading
            size ``num_particles``.
        proposal_sampler: Per-particle
            ``(key, z_prev, y_t[, input_t]) -> z_t``
            drawing from $q(z_t \mid z_{t-1}, y_t)$. It receives
            one particle PyTree and must preserve its structure, leaf
            shapes, and dtypes. smcx ``vmap``-s it internally.
        log_proposal_fn: Per-particle
            ``(y_t, z_t, z_prev[, input_t]) -> scalar`` log proposal
            density with at least float32 precision.
        log_transition_fn: Per-particle
            ``(z_t, z_prev[, input_t]) -> scalar`` log transition
            density $\log f$ with at least float32 precision.
        log_observation_fn: Per-particle
            ``(y_t, z_t[, input_t]) -> scalar`` log observation density
            $\log g$ with at least float32 precision.
        emissions: Scalar ``(T,)`` or vector ``(T, D)`` observations.
            Rank-one data become ``(T, 1)``; dtype is preserved.
        num_particles: Number of particles $N$.
        resampling_fn: Resampler with signature
            ``(key, weights, num_samples) -> indices``.
        resampling_threshold: ESS fraction, or a JAX-traceable criterion
            ``(normalized_log_weights, absolute_ess, time_index) -> bool``.
            Numeric values must be finite and nonnegative; zero disables
            resampling and values above one force it at every update.
            The callback receives carried weights and ESS at the zero-based
            emission indices 1 through T - 1.
        inputs: Optional exogenous inputs with shape ``(T, input_dim)``
            or ``(T,)`` and a nonempty event. Input zero reaches
            initialization; each later
            input reaches every guided callback at that time step.
        store_history: When False, the filter retains no
            per-step particle/weight/ancestor histories — the returned
            arrays cover only the final step (time axis length 1)
            while ``ess``/``log_evidence_increments`` stay full.

    Returns:
        `smcx.containers.ParticleFilterPosterior`. Structured
        particle histories preserve the state PyTree and add ``(T, N)``
        to every leaf.

    Raises:
        DegenerateWeightsError: A particle-weight stage cannot be normalized
            (eager execution only; under ``jax.jit`` its nonfinite signal
            propagates).
        ValueError: Observations, inputs, or the threshold are malformed, a
            criterion result is not a scalar Boolean, the initial state tree
            is empty or has a wrong leading axis, a proposal changes its
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
    fk = _guided_fk(
        initial_sampler,
        proposal_sampler,
        log_proposal_fn,
        log_transition_fn,
        log_observation_fn,
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
    )
