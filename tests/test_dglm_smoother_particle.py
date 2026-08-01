# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Particle corroboration for DGLM retrospective moments."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import numpy as np
from jax.scipy.special import gammaln

import smcx


def _working_dtype() -> jnp.dtype:
    """Return the configured float dtype on CPU and Metal."""
    if jax.config.read("jax_enable_x64"):
        return jnp.dtype(jnp.float64)
    return jnp.dtype(jnp.float32)


def test_dglm_smoother_matches_particle_ffbs_within_five_se() -> None:
    """A fixed Poisson fixture passes a tier-2 particle smoothing gate."""
    dtype = _working_dtype()
    num_replicates, num_particles, num_draws = 32, 2_048, 128
    emissions = jnp.asarray([1, 1, 1, 2])

    def sample_initial(key: jax.Array, count: int) -> jax.Array:
        return jnp.asarray(0.2, dtype=dtype) * jr.normal(
            key, (count, 1), dtype=dtype
        )

    def sample_transition(key: jax.Array, state: jax.Array) -> jax.Array:
        return state + jnp.asarray(0.1, dtype=state.dtype) * jr.normal(
            key, state.shape, dtype=state.dtype
        )

    def log_observation(
        emission: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        count = emission[0]
        log_rate = state[0]
        return count * log_rate - jnp.exp(log_rate) - gammaln(count + 1)

    def log_transition(
        state: jax.Array,
        previous: jax.Array,
        params: object,
        input_t: object,
    ) -> jax.Array:
        del params, input_t
        return jstats.norm.logpdf(
            state[0],
            loc=previous[0],
            scale=jnp.asarray(0.1, dtype=state.dtype),
        )

    def replicate(root: jax.Array) -> jax.Array:
        smoother_key, filter_key = jr.split(root)
        filtered = smcx.bootstrap_filter(
            filter_key,
            sample_initial,
            sample_transition,
            log_observation,
            emissions,
            num_particles,
            resampling_threshold=1.1,
            store_history=True,
        )
        paths = smcx.backward_simulation(
            smoother_key,
            filtered,
            log_transition,
            None,
            num_draws=num_draws,
        ).smoothed_trajectories[:, 1, 0]
        return jnp.stack((jnp.mean(paths), jnp.mean(jnp.square(paths))))

    roots = jr.split(jr.key(354), num_replicates)
    rows = np.asarray(
        jax.jit(jax.vmap(replicate))(roots),
        dtype=np.float64,
    )
    assert np.all(np.isfinite(rows))
    aggregate = rows.mean(axis=0)
    # Each row is one independent complete PF-plus-FFBS estimator; paths
    # within a row share its filter. Thus SE = sample_sd(ddof=1) / sqrt(R).
    estimator_se = rows.std(axis=0, ddof=1) / math.sqrt(num_replicates)
    five_se = 5.0 * estimator_se
    np.testing.assert_array_less(five_se, [0.025, 0.008])

    transition = jnp.ones((1, 1), dtype=dtype)
    evolution = jnp.asarray([[0.01]], dtype=dtype)
    filtered = smcx.dglm_filter(
        jnp.zeros((1,), dtype=dtype),
        jnp.asarray([[0.04]], dtype=dtype),
        transition,
        jnp.ones((1,), dtype=dtype),
        emissions,
        family=smcx.poisson(),
        transition_covariance=evolution,
        dispersion_discount=1.0,
    )
    smoothed = smcx.dglm_smoother(
        filtered,
        transition,
        transition_covariance=evolution,
    )
    mean = smoothed.smoothed_means[1, 0]
    dglm_target = np.asarray(
        jnp.stack((
            mean,
            smoothed.smoothed_covariances[1, 0, 0] + jnp.square(mean),
        )),
        dtype=np.float64,
    )
    # Frozen normalized-grid forward/backward quadrature from ADR-0032.
    grid_target = np.asarray([0.037082929332, 0.04336664793])
    for name, target in (("DGLM", dglm_target), ("grid", grid_target)):
        np.testing.assert_array_less(
            np.abs(aggregate - target),
            five_se,
            err_msg=f"particle aggregate versus {name}",
        )
