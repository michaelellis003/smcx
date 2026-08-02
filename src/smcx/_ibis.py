# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Private numerical kernels for iterated batch importance sampling."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax, vmap
from jaxtyping import Array, Float, Int, Shaped

from smcx._numerics import _neumaier_add
from smcx._utils import _validate_log_density_batch
from smcx.types import (
    Emission,
    IBISLogLikelihoodFn,
    ModelInput,
    StaticLogDensity,
)


class _IBISPopulation(NamedTuple):
    """Parameters with a compensated expansion of their current target."""

    params: Float[Array, "num_particles param_dim"]
    log_target: Float[Array, " num_particles"]
    log_target_correction: Float[Array, " num_particles"]


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

    if inputs is None:
        scan_inputs = jnp.zeros(
            (emissions.shape[0], 0),
            dtype=target_template.dtype,
        )

        def evaluate(emission, _input_t):
            return log_likelihood_increment_fn(emission, params, None)

    else:
        scan_inputs = inputs

        def evaluate(emission, input_t):
            return log_likelihood_increment_fn(emission, params, input_t)

    def add_factor(carry, row):
        index, emission, input_t = row

        def active(_):
            return jnp.asarray(
                evaluate(emission, input_t),
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

    expansion, _ = lax.scan(
        add_factor,
        (log_prior, target_zero),
        (indices, emissions, scan_inputs),
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
