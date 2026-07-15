# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

#
# Ported to MLX from smcjax (https://github.com/michaelellis003/smcjax,
# frozen @ e93d527), Apache-2.0. Modified: FK-core twist formulation,
# MLX arrays, inputs channel, store_history.

r"""Auxiliary particle filter (APF).

The APF [Pitt & Shephard, 1999] looks one observation ahead: a
first-stage weight :math:`W_i \cdot \eta(y_t \mid x_{t-1}^i)` steers
resampling toward particles likely to explain :math:`y_t`, and the
look-ahead is divided back out after propagation (second-stage
correction; general weights per Johansen & Doucet, 2008). With a flat
``log_auxiliary_fn`` (zero), the APF reduces exactly to the bootstrap
filter — bit-identical at the same key (tested). APF is *not*
uniformly better than bootstrap (Johansen & Doucet 2008); it pays off
when the look-ahead is informative.
"""

from typing import Any

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
    TransitionSampler,
    TransitionSamplerWithInput,
)


def auxiliary_filter(
    key: KeyT,
    initial_sampler: InitialSampler,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    log_auxiliary_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[mx.array, "ntime emission_dim"]
    | Float[mx.array, " ntime"],
    num_particles: int,
    resampling_fn=systematic,
    resampling_threshold: float = 0.5,
    *,
    inputs: mx.array | None = None,
    store_history: bool = True,
    batched: bool = False,
) -> ParticleFilterPosterior:
    r"""Run an auxiliary particle filter.

    Args:
        key: PRNG key.
        initial_sampler: ``(key, num_particles) -> particles``.
        transition_sampler: ``(key, state[, input_t]) -> state``;
            vmapped internally.
        log_observation_fn: ``(emission, state[, input_t])``;
            vmapped internally.
        log_auxiliary_fn: Look-ahead ``(emission, state[, input_t])``
            approximating :math:`\log p(y_t \mid x_{t-1})`, evaluated
            on the *pre-propagation* particles; vmapped internally.
            Zero everywhere reduces the APF to the bootstrap filter.
        emissions: Observations ``(T, D)`` (or ``(T,)``,
            canonicalized).
        num_particles: Number of particles N.
        resampling_fn: ADR-0004 contract resampler.
        resampling_threshold: Resample when the ESS of the
            *first-stage* weights drops below ``threshold * N``.
        inputs: Optional per-step inputs (ADR-0008 alignment).
        store_history: ADR-0011 memory option.
        batched: ADR-0013 fast path — callbacks receive the whole
            cloud with one key; same arities.

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
    for fn, name in (
        (transition_sampler, "transition_sampler"),
        (log_observation_fn, "log_observation_fn"),
        (log_auxiliary_fn, "log_auxiliary_fn"),
    ):
        _utils.check_callback_arity(fn, name, 2, has_inputs)

    trans_any: Any = transition_sampler  # ty: union not narrowable by flags
    obs_any: Any = log_observation_fn  # ty: union not narrowable by flags
    aux_any: Any = log_auxiliary_fn  # ty: union not narrowable by flags

    if has_inputs:
        inputs_arr = _utils.canonicalize_inputs(inputs, num_timesteps)
        if batched:

            def mutate(key, particles, data):
                _, input_t = data
                return trans_any(key, particles, input_t)

            def log_g(prev, particles, data):
                y_t, input_t = data
                del prev
                return obs_any(y_t, particles, input_t)

            def log_eta(particles, data):
                y_t, input_t = data
                return aux_any(y_t, particles, input_t)

        else:

            def mutate(key, particles, data):
                _, input_t = data
                keys = mx.random.split(key, particles.shape[0])
                return mx.vmap(transition_sampler, in_axes=(0, 0, None))(
                    keys, particles, input_t
                )

            def log_g(prev, particles, data):
                y_t, input_t = data
                del prev
                return mx.vmap(log_observation_fn, in_axes=(None, 0, None))(
                    y_t, particles, input_t
                )

            def log_eta(particles, data):
                y_t, input_t = data
                return mx.vmap(log_auxiliary_fn, in_axes=(None, 0, None))(
                    y_t, particles, input_t
                )

        data = (emissions, inputs_arr)
    else:
        if batched:

            def mutate(key, particles, data):
                del data
                return trans_any(key, particles)

            def log_g(prev, particles, data):
                (y_t,) = data
                del prev
                return obs_any(y_t, particles)

            def log_eta(particles, data):
                (y_t,) = data
                return aux_any(y_t, particles)

        else:

            def mutate(key, particles, data):
                del data
                keys = mx.random.split(key, particles.shape[0])
                return mx.vmap(transition_sampler)(keys, particles)

            def log_g(prev, particles, data):
                (y_t,) = data
                del prev
                return mx.vmap(log_observation_fn, in_axes=(None, 0))(
                    y_t, particles
                )

            def log_eta(particles, data):
                (y_t,) = data
                return mx.vmap(log_auxiliary_fn, in_axes=(None, 0))(
                    y_t, particles
                )

        data = (emissions,)

    fk = FKModel(m0=initial_sampler, m=mutate, log_g=log_g, log_eta=log_eta)
    return run_filter(
        key,
        fk,
        data,
        num_particles,
        resampling_fn,
        resampling_threshold,
        store_history=store_history,
    )
