# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Iterated batch importance sampling for static parameters."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from smcx._ibis import _IBISPopulation, _run_ibis_stages
from smcx._static_mutation import _resolve_static_mutation
from smcx._utils import (
    _canonicalize_emissions,
    _canonicalize_inputs,
    _positive_integer,
    _validate_log_density_batch,
    _validate_resampling_threshold,
)
from smcx.containers import IBISPosterior
from smcx.resampling import systematic
from smcx.types import (
    EmissionSequence,
    IBISLogLikelihoodFn,
    InputSequence,
    ParamInitialSampler,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
    StaticLogDensity,
    StaticMutation,
    StaticMutationInitFn,
    StaticMutationStepFn,
)


def ibis(
    key: PRNGKeyT,
    param_initial_sampler: ParamInitialSampler,
    log_prior_fn: StaticLogDensity,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
    emissions: EmissionSequence,
    num_particles: int,
    *,
    inputs: InputSequence | None = None,
    num_mcmc_steps: int = 5,
    resampling_threshold: float | ResamplingCriterion = 0.5,
    resampling_fn: ResamplingFn = systematic,
    mutation: StaticMutation | None = None,
    mutation_init_fn: StaticMutationInitFn | None = None,
    mutation_step_fn: StaticMutationStepFn | None = None,
    store_history: bool = True,
) -> IBISPosterior:
    r"""Run IBIS for exact sequential static-parameter inference.

    ``log_likelihood_increment_fn(emission_t, params, input_t)`` returns the
    deterministic conditional log-likelihood contribution at one datum;
    ``input_t`` is always present and is ``None`` when ``inputs`` is omitted.
    Rank-one sequences are canonicalized to ``(T, 1)``. Resampled stages use
    a fixed-count kernel invariant for the posterior through the current
    datum. The default random-walk proposal covariance is ``2.38**2 / d``
    times the corrected weighted parameter covariance.

    The outer function is host-driven and cannot itself be jitted or vmapped.
    It implements Chopin (2002), https://doi.org/10.1093/biomet/89.3.539,
    with the sequential parameter-learning presentation of Chopin, Jacob, and
    Papaspiliopoulos (2013),
    https://doi.org/10.1111/j.1467-9868.2012.01046.x.

    Args:
        key: JAX PRNG key.
        param_initial_sampler: ``(key, N) -> (N, d)`` normalized-prior draw.
        log_prior_fn: Per-parameter scalar log-prior.
        log_likelihood_increment_fn: Three-argument conditional increment
            described above; vmapped internally.
        emissions: Nonempty rank-one or rank-two observation sequence.
        num_particles: Number of parameter particles N.
        inputs: Optional aligned rank-one or rank-two input sequence.
        num_mcmc_steps: Fixed mutation sweeps per resampled stage.
        resampling_threshold: Finite nonnegative ESS fraction or
            ``(log_weights, selection_ess, time_index) -> bool`` criterion.
            Zero disables resampling and values above one force it.
        resampling_fn: Resampler used when the criterion fires.
        mutation: Optional `smcx.StaticMutation` record carrying the
            initializer and step together; the primary form of the
            legacy pair below. Supply the record or the pair, not both.
        mutation_init_fn: Optional caller mutation initializer.
        mutation_step_fn: Optional paired caller mutation step.
        store_history: If false, retain only the final cloud and weights;
            scalar diagnostic traces always retain length T.

    Returns:
        An `smcx.containers.IBISPosterior`. ``selection_ess`` is pre-resample;
        ``ess`` is implied by stored post-move weights and equals N after a
        resampled stage. Acceptance is zero where no move ran.

    Raises:
        ValueError: A count, sequence, prior cloud, callback output,
            resampling rule, or mutation contract is structurally invalid.
        DegenerateWeightsError: A weight stage cannot be normalized.
    """
    num_particles = _positive_integer(num_particles, name="num_particles")
    num_mcmc_steps = _positive_integer(num_mcmc_steps, name="num_mcmc_steps")
    if not isinstance(store_history, bool):
        raise ValueError("store_history must be a boolean")
    _validate_resampling_threshold(resampling_threshold)
    mutation_init_fn, mutation_step_fn = _resolve_static_mutation(
        mutation,
        mutation_init_fn,
        mutation_step_fn,
    )
    if (mutation_init_fn is None) != (mutation_step_fn is None):
        raise ValueError(
            "mutation_init_fn and mutation_step_fn must be supplied together"
        )

    emissions = _canonicalize_emissions(emissions)
    num_time = emissions.shape[0]
    inputs = None if inputs is None else _canonicalize_inputs(inputs, num_time)
    prior_key, stage_root_key = jr.split(key)
    params = param_initial_sampler(prior_key, num_particles)
    if not isinstance(params, jax.Array):
        raise ValueError(
            "param_initial_sampler output must be a JAX array with shape "
            "(num_particles, param_dim)"
        )
    if params.ndim != 2 or params.shape != (num_particles, params.shape[1]):
        raise ValueError(
            "param_initial_sampler output must have shape "
            f"({num_particles}, param_dim); got {params.shape}"
        )
    if params.shape[1] == 0:
        raise ValueError("param_initial_sampler output has empty param_dim")
    if not jnp.issubdtype(params.dtype, jnp.floating):
        raise ValueError(
            "param_initial_sampler output must have a floating dtype; "
            f"got {params.dtype}"
        )

    log_prior = jnp.asarray(vmap(log_prior_fn)(params))
    _validate_log_density_batch(log_prior, num_particles, name="log_prior_fn")
    input_0 = None if inputs is None else inputs[0]
    first_increment = jnp.asarray(
        vmap(
            lambda position: log_likelihood_increment_fn(
                emissions[0],
                position,
                input_0,
            )
        )(params)
    )
    _validate_log_density_batch(
        first_increment,
        num_particles,
        name="log_likelihood_increment_fn",
    )
    target_dtype = jnp.result_type(log_prior, first_increment)
    prior_population = _IBISPopulation(
        params=params,
        log_target=log_prior.astype(target_dtype),
        log_target_correction=jnp.zeros((num_particles,), dtype=target_dtype),
    )
    run = _run_ibis_stages(
        stage_root_key,
        prior_population,
        first_increment.astype(target_dtype),
        emissions,
        inputs=inputs,
        num_mcmc_steps=num_mcmc_steps,
        resampling_threshold=resampling_threshold,
        resampling_fn=resampling_fn,
        log_prior_fn=log_prior_fn,
        log_likelihood_increment_fn=log_likelihood_increment_fn,
        mutation_init_fn=mutation_init_fn,
        mutation_step_fn=mutation_step_fn,
        store_history=store_history,
    )
    return IBISPosterior(
        marginal_loglik=(
            run.state.log_evidence + run.state.log_evidence_correction
        ),
        filtered_params=run.filtered_params,
        filtered_log_weights=run.filtered_log_weights,
        ess=run.ess,
        log_evidence_increments=run.log_evidence_increments,
        acceptance_rates=run.acceptance_rates,
        selection_ess=run.selection_ess,
        resampled=run.resampled,
    )
