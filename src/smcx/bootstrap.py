# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

r"""Bootstrap (SIR) particle filter.

The bootstrap filter [Gordon *et al.*, 1993] propagates particles
through the transition prior and weights them by the observation
likelihood, resampling conditionally on ESS. Built as a thin
constructor over the internal Feynman-Kac core (ADR-0002): the time
loop is Python over one ``mx.compile``d step (MLX has no scan;
async + lagged-eval cadence per mlx-performance.md).
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
    ResamplingFn,
    TransitionSampler,
    TransitionSamplerWithInput,
)


def bootstrap_filter(
    key: KeyT,
    initial_sampler: InitialSampler,
    transition_sampler: TransitionSampler | TransitionSamplerWithInput,
    log_observation_fn: LogObservationFn | LogObservationFnWithInput,
    emissions: Float[mx.array, "ntime emission_dim"]
    | Float[mx.array, " ntime"],
    num_particles: int,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
    *,
    inputs: mx.array | None = None,
    store_history: bool = True,
    batched: bool = False,
) -> ParticleFilterPosterior:
    r"""Run a bootstrap (SIR) particle filter.

    Args:
        key: PRNG key.
        initial_sampler: ``(key, num_particles) -> particles`` drawing
            the whole initial cloud from :math:`p(z_1)`.
        transition_sampler: ``(key, state) -> state`` drawing from
            :math:`p(z_t \mid z_{t-1})`; vmapped internally over
            particles. With ``inputs``: ``(key, state, input_t)``.
        log_observation_fn: ``(emission, state) -> log_prob``
            evaluating :math:`\log p(y_t \mid z_t)`; vmapped
            internally (second argument). With ``inputs``:
            ``(emission, state, input_t)``. NaN emissions are passed
            through untouched — mask them here (design §4):
            ``mx.where(mx.isnan(y), 0.0, logpdf)``.
        emissions: Observations, shape ``(T, D)`` (a ``(T,)`` series
            is canonicalized to ``(T, 1)``).
        num_particles: Number of particles N.
        resampling_fn: ADR-0004 contract resampler
            (``(key, weights, num_samples) -> indices``); defaults to
            :func:`smcx.systematic`.
        resampling_threshold: Resample when
            ``ESS < resampling_threshold * num_particles``.
        inputs: Optional per-step exogenous inputs, leading dim T
            aligned with emissions; ``inputs[t]`` feeds the
            transition *into* t and the observation *at* t
            (ADR-0008).
        batched: When True (ADR-0013), callbacks receive the whole
            cloud with ONE key — ``trans_any(key,
            particles[, input_t])`` and ``log_observation_fn(
            emission, particles[, input_t]) -> (N,)`` — skipping the
            internal vmap/per-particle key split. Use for
            matrix-valued models (MLX's vmap does not fuse matvecs
            to GEMM; measured 5.1x). Same arities as the default
            convention.
        store_history: When False (ADR-0011), particle/weight/
            ancestor arrays cover only the final step (time axis
            length 1) and memory drops from O(T*N) to O(N);
            ``ess`` and ``log_evidence_increments`` stay full and
            ``marginal_loglik`` is bit-identical at the same key.
            Use for log-ML-only workloads (model comparison, PMMH
            inner loops); genealogy/smoothing needs the default.

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
    _utils.check_callback_arity(
        transition_sampler, "transition_sampler", 2, has_inputs
    )
    _utils.check_callback_arity(
        log_observation_fn, "log_observation_fn", 2, has_inputs
    )

    trans_any: Any = transition_sampler  # ty: union not narrowable by flags
    obs_any: Any = log_observation_fn  # ty: union not narrowable by flags

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

        data = (emissions,)

    fk = FKModel(m0=initial_sampler, m=mutate, log_g=log_g)
    return run_filter(
        key,
        fk,
        data,
        num_particles,
        resampling_fn,
        resampling_threshold,
        store_history=store_history,
    )
