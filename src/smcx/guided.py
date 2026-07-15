# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

r"""Guided (proposal-based) particle filter.

The guided filter propagates through a user proposal
:math:`q(x_t \mid x_{t-1}, y_t)` — which, unlike the bootstrap
transition prior, can see the current observation — and corrects with
the general importance weight
:math:`w \propto g(y_t \mid x_t)\, f(x_t \mid x_{t-1}) /
q(x_t \mid x_{t-1}, y_t)` [Doucet, Godsill & Andrieu, 2000].
Approximate proposals (EKF/UKF/Laplace) MUST use this general
formula — the predictive-likelihood shortcut is exact only for the
locally optimal proposal. With ``q = f`` the filter reduces to
bootstrap (matches at f32 rounding tolerance at the same key — the
``f/q`` cancellation is mathematical, not bitwise; tested).

New relative to the frozen smcjax surface (ADR-0008 item 2); the
proposal is the accuracy axis the ecosystem treats as first-class
(docs/research/library-survey.md).
"""

import mlx.core as mx
from jaxtyping import Float

from smcx import _utils
from smcx._fk import FKModel, run_filter
from smcx.containers import ParticleFilterPosterior
from smcx.resampling import systematic
from smcx.types import (
    InitialSampler,
    KeyT,
    LogObservationFn,
    LogObservationFnWithInput,
    LogProposalFn,
    LogProposalFnWithInput,
    LogTransitionFn,
    LogTransitionFnWithInput,
    ProposalSampler,
    ProposalSamplerWithInput,
)


def guided_filter(
    key: KeyT,
    initial_sampler: InitialSampler,
    proposal_sampler: ProposalSampler | ProposalSamplerWithInput,
    log_proposal_fn: LogProposalFn | LogProposalFnWithInput,
    log_transition_fn: LogTransitionFn | LogTransitionFnWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[mx.array, "ntime emission_dim"]
    | Float[mx.array, " ntime"],
    num_particles: int,
    resampling_fn=systematic,
    resampling_threshold: float = 0.5,
    *,
    inputs: mx.array | None = None,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    r"""Run a guided particle filter.

    Args:
        key: PRNG key.
        initial_sampler: ``(key, num_particles) -> particles`` from
            :math:`p(z_1)` (t=0 is weighted by the observation only).
        proposal_sampler: ``(key, state, emission[, input_t]) ->
            state`` drawing from :math:`q(x_t \mid x_{t-1}, y_t)`;
            vmapped internally.
        log_proposal_fn: ``(emission, new_state, old_state
            [, input_t])`` evaluating :math:`\log q`; vmapped
            internally.
        log_transition_fn: ``(new_state, old_state[, input_t])``
            evaluating :math:`\log f`; vmapped internally.
        log_observation_fn: ``(emission, state[, input_t])``
            evaluating :math:`\log g`; vmapped internally.
        emissions: Observations ``(T, D)`` (or ``(T,)``,
            canonicalized).
        num_particles: Number of particles N.
        resampling_fn: ADR-0004 contract resampler.
        resampling_threshold: Resample when ESS < threshold * N.
        inputs: Optional per-step inputs (ADR-0008 alignment).
        store_history: ADR-0011 memory option.

    Returns:
        :class:`~smcx.containers.ParticleFilterPosterior`.

    Raises:
        DegenerateWeightsError: All weights collapsed at some step.
        TypeError: Callback arity inconsistent with ``inputs``.
        ValueError: Malformed shapes or ``num_particles < 1``.
    """
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}")
    emissions = _utils.canonicalize_emissions(emissions)
    num_timesteps = emissions.shape[0]

    has_inputs = inputs is not None
    for fn, name, base in (
        (proposal_sampler, "proposal_sampler", 3),
        (log_proposal_fn, "log_proposal_fn", 3),
        (log_transition_fn, "log_transition_fn", 2),
        (log_observation_fn, "log_observation_fn", 2),
    ):
        _utils.check_callback_arity(fn, name, base, has_inputs)

    if has_inputs:
        inputs_arr = _utils.canonicalize_inputs(inputs, num_timesteps)

        def mutate(key, particles, data):
            y_t, input_t = data
            keys = mx.random.split(key, particles.shape[0])
            return mx.vmap(proposal_sampler, in_axes=(0, 0, None, None))(
                keys, particles, y_t, input_t
            )

        def log_g(prev, particles, data):
            y_t, input_t = data
            obs = mx.vmap(log_observation_fn, in_axes=(None, 0, None))(
                y_t, particles, input_t
            )
            trans = mx.vmap(log_transition_fn, in_axes=(0, 0, None))(
                particles, prev, input_t
            )
            prop = mx.vmap(log_proposal_fn, in_axes=(None, 0, 0, None))(
                y_t, particles, prev, input_t
            )
            return obs + trans - prop

        def log_g0(particles, data):
            y_0, input_0 = data
            return mx.vmap(log_observation_fn, in_axes=(None, 0, None))(
                y_0, particles, input_0
            )

        data = (emissions, inputs_arr)
    else:

        def mutate(key, particles, data):
            (y_t,) = data
            keys = mx.random.split(key, particles.shape[0])
            return mx.vmap(proposal_sampler, in_axes=(0, 0, None))(
                keys, particles, y_t
            )

        def log_g(prev, particles, data):
            (y_t,) = data
            obs = mx.vmap(log_observation_fn, in_axes=(None, 0))(y_t, particles)
            trans = mx.vmap(log_transition_fn)(particles, prev)
            prop = mx.vmap(log_proposal_fn, in_axes=(None, 0, 0))(
                y_t, particles, prev
            )
            return obs + trans - prop

        def log_g0(particles, data):
            (y_0,) = data
            return mx.vmap(log_observation_fn, in_axes=(None, 0))(
                y_0, particles
            )

        data = (emissions,)

    fk = FKModel(m0=initial_sampler, m=mutate, log_g=log_g, log_g0=log_g0)
    return run_filter(
        key,
        fk,
        data,
        num_particles,
        resampling_fn,
        resampling_threshold,
        store_history=store_history,
    )
