# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Private numerical kernels for iterated batch importance sampling."""

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from jaxtyping import Array, Bool, Float, Int, Shaped

from smcx._numerics import _neumaier_add
from smcx._static_mutation import _run_custom_mutation_sweep
from smcx._utils import _validate_log_density_batch
from smcx.types import (
    Emission,
    IBISLogLikelihoodFn,
    ModelInput,
    PRNGKeyT,
    StaticLogDensity,
    StaticMutationInitFn,
    StaticMutationStepFn,
)


class _IBISPopulation(NamedTuple):
    """Parameters with a compensated expansion of their current target."""

    params: Float[Array, "num_particles param_dim"]
    log_target: Float[Array, " num_particles"]
    log_target_correction: Float[Array, " num_particles"]


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
    log_target, correction = _neumaier_add(
        population.log_target,
        population.log_target_correction,
        increments,
    )
    return (
        _IBISPopulation(
            params=population.params,
            log_target=log_target,
            log_target_correction=correction,
        ),
        increments,
    )
