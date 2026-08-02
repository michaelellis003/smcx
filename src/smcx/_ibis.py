# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Private numerical kernels for iterated batch importance sampling."""

import math
from functools import partial
from typing import NamedTuple, Protocol, cast, runtime_checkable

import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap
from jaxtyping import Array, Bool, Float, Int, Shaped

from smcx._covariance import _weighted_covariance_factor
from smcx._numerics import _neumaier_add
from smcx._static_mutation import _run_custom_mutation_sweep
from smcx._static_smc import (
    _static_smc_stage,
    _StaticMoveResult,
    _StaticSMCState,
)
from smcx._utils import _validate_log_density_batch
from smcx.types import (
    Emission,
    IBISLogLikelihoodFn,
    ModelInput,
    ParticleCloud,
    PRNGKeyT,
    ResamplingCriterion,
    ResamplingFn,
    StaticLogDensity,
    StaticMutationInitFn,
    StaticMutationStepFn,
)


class _IBISPopulation(NamedTuple):
    """Parameters with a compensated expansion of their current target."""

    params: Float[Array, "num_particles param_dim"]
    log_target: Float[Array, " num_particles"]
    log_target_correction: Float[Array, " num_particles"]


class _IBISRun(NamedTuple):
    """Final private state and public-aligned IBIS histories."""

    state: _StaticSMCState
    filtered_params: Float[Array, "stored_time num_particles param_dim"]
    filtered_log_weights: Float[Array, "stored_time num_particles"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]
    acceptance_rates: Float[Array, " ntime"]
    selection_ess: Float[Array, " ntime"]
    resampled: Bool[Array, " ntime"]


@runtime_checkable
class _IBISRWMStageKernel(Protocol):
    """Apply one compiled default mutation stage."""

    def __call__(
        self,
        key: PRNGKeyT,
        population: _IBISPopulation,
        time_index: Int[Array, ""],
        proposal_factor: Float[Array, "param_dim param_dim"],
        /,
    ) -> tuple[_IBISPopulation, Float[Array, ""]]: ...  # pragma: no cover


@runtime_checkable
class _IBISCustomStageKernel(Protocol):
    """Apply one compiled caller-owned mutation stage."""

    def __call__(
        self,
        key: PRNGKeyT,
        population: _IBISPopulation,
        time_index: Int[Array, ""],
        /,
    ) -> tuple[
        _IBISPopulation,
        Float[Array, ""],
        Bool[Array, ""],
    ]: ...  # pragma: no cover


def _ibis_expansion_log_ratio(
    proposed_total: Float[Array, " num_particles"],
    proposed_correction: Float[Array, " num_particles"],
    current_total: Float[Array, " num_particles"],
    current_correction: Float[Array, " num_particles"],
) -> Float[Array, " num_particles"]:
    """Subtract two target expansions without resolving either one first."""
    total, correction = _neumaier_add(
        proposed_total,
        proposed_correction,
        -current_total,
    )
    total, correction = _neumaier_add(
        total,
        correction,
        -current_correction,
    )
    return total + correction


def _ibis_prefix_expansion(
    params: Float[Array, " param_dim"],
    time_index: Int[Array, ""],
    target_template: Float[Array, ""],
    *,
    emissions: Shaped[Array, "ntime emission_dim"],
    inputs: Shaped[Array, "ntime input_dim"] | None,
    log_prior_fn: StaticLogDensity,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Return a compensated expansion of the target through ``time_index``."""
    target_zero = jnp.zeros_like(target_template)
    log_prior = jnp.asarray(log_prior_fn(params), dtype=target_template.dtype)
    indices = jnp.arange(emissions.shape[0], dtype=time_index.dtype)

    def evaluate(index):
        emission = lax.dynamic_index_in_dim(
            emissions,
            index,
            axis=0,
            keepdims=False,
        )
        if inputs is None:
            return log_likelihood_increment_fn(emission, params, None)
        input_t = lax.dynamic_index_in_dim(
            inputs,
            index,
            axis=0,
            keepdims=False,
        )
        return log_likelihood_increment_fn(emission, params, input_t)

    def add_factor(carry, index):

        def active(_):
            return jnp.asarray(
                evaluate(index),
                dtype=target_template.dtype,
            )

        factor = lax.cond(
            index <= time_index,
            active,
            lambda _: target_zero,
            operand=None,
        )
        total, correction = carry
        return _neumaier_add(total, correction, factor), None

    # Scan one index leaf and fetch rows only inside the active branch. This
    # both excludes future callbacks and avoids the jax-mps multi-leaf scan
    # corruption tracked by smcx#38.
    expansion, _ = lax.scan(
        add_factor,
        (log_prior, target_zero),
        indices,
    )
    return expansion


def _ibis_prefix_logdensity(
    params: Float[Array, " param_dim"],
    time_index: Int[Array, ""],
    target_template: Float[Array, ""],
    *,
    emissions: Shaped[Array, "ntime emission_dim"],
    inputs: Shaped[Array, "ntime input_dim"] | None,
    log_prior_fn: StaticLogDensity,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
) -> Float[Array, ""]:
    """Return the resolved target log-density through ``time_index``."""
    total, correction = _ibis_prefix_expansion(
        params,
        time_index,
        target_template,
        emissions=emissions,
        inputs=inputs,
        log_prior_fn=log_prior_fn,
        log_likelihood_increment_fn=log_likelihood_increment_fn,
    )
    return total + correction


def _run_ibis_rwm_sweep(
    key: PRNGKeyT,
    population: _IBISPopulation,
    time_index: Int[Array, ""],
    proposal_factor: Float[Array, "param_dim param_dim"],
    *,
    num_steps: int,
    emissions: Shaped[Array, "ntime emission_dim"],
    inputs: Shaped[Array, "ntime input_dim"] | None,
    log_prior_fn: StaticLogDensity,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
) -> tuple[_IBISPopulation, Float[Array, ""]]:
    """Run fixed-count RWM sweeps while retaining aligned target caches."""
    num_particles = population.params.shape[0]
    target_template = jnp.zeros((), dtype=population.log_target.dtype)

    def evaluate(params):
        return _ibis_prefix_expansion(
            params,
            time_index,
            target_template,
            emissions=emissions,
            inputs=inputs,
            log_prior_fn=log_prior_fn,
            log_likelihood_increment_fn=log_likelihood_increment_fn,
        )

    sweep_keys = jr.split(key, num_steps)

    def apply_sweep(current, sweep_key):
        proposal_key, acceptance_key = jr.split(sweep_key)
        noise = jr.normal(
            proposal_key,
            current.params.shape,
            dtype=current.params.dtype,
        )
        proposed_params = current.params + noise @ proposal_factor.T
        proposed_total, proposed_correction = vmap(evaluate)(proposed_params)
        log_ratio = _ibis_expansion_log_ratio(
            proposed_total,
            proposed_correction,
            current.log_target,
            current.log_target_correction,
        )
        uniforms = jr.uniform(
            acceptance_key,
            (num_particles,),
            dtype=log_ratio.dtype,
        )
        log_uniforms = jnp.log(
            jnp.maximum(uniforms, jnp.finfo(uniforms.dtype).tiny)
        )
        accepted = log_uniforms < log_ratio
        next_population = _IBISPopulation(
            params=jnp.where(
                accepted[:, None],
                proposed_params,
                current.params,
            ),
            log_target=jnp.where(
                accepted,
                proposed_total,
                current.log_target,
            ),
            log_target_correction=jnp.where(
                accepted,
                proposed_correction,
                current.log_target_correction,
            ),
        )
        acceptance_rate = jnp.mean(accepted.astype(log_ratio.dtype))
        return next_population, acceptance_rate

    population, acceptance_rates = lax.scan(
        apply_sweep,
        population,
        sweep_keys,
    )
    return population, jnp.mean(acceptance_rates)


def _run_ibis_custom_mutation_sweep(
    key: PRNGKeyT,
    population: _IBISPopulation,
    time_index: Int[Array, ""],
    *,
    num_steps: int,
    emissions: Shaped[Array, "ntime emission_dim"],
    inputs: Shaped[Array, "ntime input_dim"] | None,
    log_prior_fn: StaticLogDensity,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
    initialize: StaticMutationInitFn,
    mutate: StaticMutationStepFn,
) -> tuple[
    _IBISPopulation,
    Float[Array, ""],
    Bool[Array, ""],
]:
    """Run caller mutation and rebuild its current-prefix target caches."""
    target_template = jnp.zeros((), dtype=population.log_target.dtype)

    def logdensity(params):
        return _ibis_prefix_logdensity(
            params,
            time_index,
            target_template,
            emissions=emissions,
            inputs=inputs,
            log_prior_fn=log_prior_fn,
            log_likelihood_increment_fn=log_likelihood_increment_fn,
        )

    params, acceptance_rate, acceptance_valid = _run_custom_mutation_sweep(
        key,
        population.params,
        logdensity,
        num_steps=num_steps,
        initialize=initialize,
        mutate=mutate,
    )

    def expansion(position):
        return _ibis_prefix_expansion(
            position,
            time_index,
            target_template,
            emissions=emissions,
            inputs=inputs,
            log_prior_fn=log_prior_fn,
            log_likelihood_increment_fn=log_likelihood_increment_fn,
        )

    log_target, correction = vmap(expansion)(params)
    return (
        _IBISPopulation(
            params=params,
            log_target=log_target,
            log_target_correction=correction,
        ),
        acceptance_rate,
        acceptance_valid,
    )


def _ibis_move_population(
    key: PRNGKeyT,
    corrected_population: _IBISPopulation,
    corrected_log_weights: Float[Array, " num_particles"],
    selected_population: _IBISPopulation,
    time_index: Int[Array, ""],
    *,
    rwm_kernel: _IBISRWMStageKernel,
    custom_kernel: _IBISCustomStageKernel | None,
) -> _StaticMoveResult:
    """Move selected seeds using a proposal fitted to the corrected cloud."""
    if custom_kernel is None:
        param_dim = corrected_population.params.shape[1]
        proposal_factor = _weighted_covariance_factor(
            corrected_population.params,
            jnp.exp(corrected_log_weights),
            scale=2.38**2 / param_dim,
        )
        population, acceptance_rate = rwm_kernel(
            key,
            selected_population,
            time_index,
            proposal_factor,
        )
        acceptance_valid = jnp.asarray(True)
    else:
        population, acceptance_rate, acceptance_valid = custom_kernel(
            key,
            selected_population,
            time_index,
        )
    return _StaticMoveResult(
        population=population,
        acceptance_rate=acceptance_rate,
        acceptance_valid=acceptance_valid,
    )


def _add_ibis_increment(
    population: _IBISPopulation,
    increments: Float[Array, " num_particles"],
) -> _IBISPopulation:
    """Advance the resident target cache by one validated increment."""
    log_target, correction = _neumaier_add(
        population.log_target,
        population.log_target_correction,
        increments,
    )
    return _IBISPopulation(
        params=population.params,
        log_target=log_target,
        log_target_correction=correction,
    )


def _advance_ibis_target(
    population: _IBISPopulation,
    emission_t: Emission,
    input_t: ModelInput | None,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
) -> tuple[_IBISPopulation, Float[Array, " num_particles"]]:
    """Evaluate one datum and advance every resident target expansion."""
    num_particles = population.params.shape[0]
    increments = jnp.asarray(
        vmap(
            lambda params: log_likelihood_increment_fn(
                emission_t,
                params,
                input_t,
            )
        )(population.params)
    )
    _validate_log_density_batch(
        increments,
        num_particles,
        name="log_likelihood_increment_fn",
    )
    increments = increments.astype(population.log_target.dtype)
    return _add_ibis_increment(population, increments), increments


def _run_ibis_stages(
    stage_root_key: PRNGKeyT,
    prior_population: _IBISPopulation,
    initial_log_increment: Float[Array, " num_particles"],
    emissions: Shaped[Array, "ntime emission_dim"],
    *,
    inputs: Shaped[Array, "ntime input_dim"] | None,
    num_mcmc_steps: int,
    resampling_threshold: float | ResamplingCriterion,
    resampling_fn: ResamplingFn,
    log_prior_fn: StaticLogDensity,
    log_likelihood_increment_fn: IBISLogLikelihoodFn,
    mutation_init_fn: StaticMutationInitFn | None,
    mutation_step_fn: StaticMutationStepFn | None,
    store_history: bool,
) -> _IBISRun:
    """Run the fixed IBIS data schedule through the shared host stage."""
    num_particles = prior_population.params.shape[0]
    num_time = emissions.shape[0]
    target_dtype = prior_population.log_target.dtype
    uniform_log_weights = jnp.full(
        (num_particles,),
        -math.log(num_particles),
        dtype=target_dtype,
    )
    target_zero = jnp.zeros((), dtype=target_dtype)
    state = _StaticSMCState(
        population=prior_population,
        log_weights=uniform_log_weights,
        log_evidence=target_zero,
        log_evidence_correction=target_zero,
    )
    stage_keys = jr.split(stage_root_key, num_time)

    compiled_advance_target = jit(
        partial(
            _advance_ibis_target,
            log_likelihood_increment_fn=log_likelihood_increment_fn,
        )
    )
    rwm_kernel = jit(
        partial(
            _run_ibis_rwm_sweep,
            num_steps=num_mcmc_steps,
            emissions=emissions,
            inputs=inputs,
            log_prior_fn=log_prior_fn,
            log_likelihood_increment_fn=log_likelihood_increment_fn,
        )
    )
    custom_kernel: _IBISCustomStageKernel | None = None
    if mutation_init_fn is not None:
        initialize = mutation_init_fn
        mutate = cast(StaticMutationStepFn, mutation_step_fn)

        custom_kernel = jit(
            partial(
                _run_ibis_custom_mutation_sweep,
                num_steps=num_mcmc_steps,
                emissions=emissions,
                inputs=inputs,
                log_prior_fn=log_prior_fn,
                log_likelihood_increment_fn=log_likelihood_increment_fn,
                initialize=initialize,
                mutate=mutate,
            )
        )

    params_history: list[Array] = []
    log_weights_history: list[Array] = []
    ess_history: list[Array] = []
    increment_history: list[Array] = []
    acceptance_history: list[Array] = []
    selection_ess_history: list[Array] = []
    resampled_history: list[Array] = []

    for time in range(num_time):
        resampling_key, mutation_key = jr.split(stage_keys[time])
        time_index = jnp.asarray(time, dtype=jnp.int32)
        population = cast(_IBISPopulation, state.population)
        if time == 0:
            increments = initial_log_increment
            population = _add_ibis_increment(population, increments)
        else:
            input_t = None if inputs is None else inputs[time]
            population, increments = compiled_advance_target(
                population,
                emissions[time],
                input_t,
            )
        state = state._replace(population=population)

        def move_population(
            key: PRNGKeyT,
            corrected_population: ParticleCloud,
            corrected_log_weights: Float[Array, " num_particles"],
            selected_population: ParticleCloud,
            stage_time_index: Int[Array, ""] = time_index,
        ) -> _StaticMoveResult:
            return _ibis_move_population(
                key,
                cast(_IBISPopulation, corrected_population),
                corrected_log_weights,
                cast(_IBISPopulation, selected_population),
                stage_time_index,
                rwm_kernel=rwm_kernel,
                custom_kernel=custom_kernel,
            )

        state, diagnostics = _static_smc_stage(
            state,
            increments,
            resampling_key,
            mutation_key,
            resampling_fn=resampling_fn,
            resampling_threshold=resampling_threshold,
            time_index=time_index,
            move_fn=move_population,
            uniform_log_weights=uniform_log_weights,
        )
        population = cast(_IBISPopulation, state.population)
        if store_history:
            params_history.append(population.params)
            log_weights_history.append(state.log_weights)
        ess_history.append(diagnostics.ess)
        increment_history.append(diagnostics.log_evidence_increment)
        acceptance_history.append(diagnostics.acceptance_rate)
        selection_ess_history.append(diagnostics.selection_ess)
        resampled_history.append(diagnostics.resampled)

    if not store_history:
        population = cast(_IBISPopulation, state.population)
        params_history.append(population.params)
        log_weights_history.append(state.log_weights)

    return _IBISRun(
        state=state,
        filtered_params=jnp.stack(params_history),
        filtered_log_weights=jnp.stack(log_weights_history),
        ess=jnp.stack(ess_history),
        log_evidence_increments=jnp.stack(increment_history),
        acceptance_rates=jnp.stack(acceptance_history),
        selection_ess=jnp.stack(selection_ess_history),
        resampled=jnp.stack(resampled_history),
    )
