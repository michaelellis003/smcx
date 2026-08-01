# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Reference-oracle tests for particle backward simulation."""

import math
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import numpy as np

import smcx
from smcx.containers import ParticleFilterPosterior

_A = 0.9
_TRANSITION_SD = 0.5
_OBSERVATION_SD = 1.0
_REFERENCE_TIMES = jnp.asarray([0, 4, 9, 14])
_NUM_REPLICATES = 16
_NUM_PARTICLES = 512
_NUM_DRAWS = 512


def _initial_sampler(key: jax.Array, count: int) -> jax.Array:
    return jr.normal(key, (count, 1))


def _transition_sampler(key: jax.Array, state: jax.Array) -> jax.Array:
    return _A * state + _TRANSITION_SD * jr.normal(key, state.shape)


def _log_observation(emission: jax.Array, state: jax.Array) -> jax.Array:
    return jstats.norm.logpdf(emission[0], state[0], scale=_OBSERVATION_SD)


def _log_transition(
    state: jax.Array,
    previous: jax.Array,
    params: Any,
    input_t: Any,
) -> jax.Array:
    del params, input_t
    return jstats.norm.logpdf(state[0], _A * previous[0], scale=_TRANSITION_SD)


def _lgssm_replicate(
    key: jax.Array,
    emissions: jax.Array,
    exact_means: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    smoother_key, filter_key = jr.split(key)
    posterior = smcx.bootstrap_filter(
        filter_key,
        _initial_sampler,
        _transition_sampler,
        _log_observation,
        emissions,
        _NUM_PARTICLES,
        resampling_threshold=1.1,
    )
    result = smcx.backward_simulation(
        smoother_key,
        posterior,
        _log_transition,
        None,
        num_draws=_NUM_DRAWS,
    )
    selected = result.smoothed_trajectories[:, _REFERENCE_TIMES, 0]
    return (
        jnp.mean(selected, axis=0),
        jnp.mean(jnp.square(selected - exact_means), axis=0),
    )


def _assert_five_se(
    estimates: np.ndarray,
    expected: np.ndarray,
    *,
    maximum_band: float,
) -> None:
    # Each row is an independent complete filter-plus-smoother estimate.
    # Hence SE(mean across R rows) = sample_sd(rows) / sqrt(R).
    estimator_se = estimates.std(axis=0, ddof=1) / math.sqrt(estimates.shape[0])
    five_se = 5.0 * estimator_se
    np.testing.assert_array_less(five_se, maximum_band)
    np.testing.assert_array_less(
        np.abs(estimates.mean(axis=0) - expected),
        five_se + 2e-5,
    )


def _discrete_record(
    particles: jax.Array, ancestors: jax.Array
) -> ParticleFilterPosterior:
    ntime, num_particles = particles.shape[:2]
    log_weight = -jnp.log(jnp.asarray(num_particles, dtype=jnp.float32))
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(0.0, dtype=jnp.float32),
        filtered_particles=particles,
        filtered_log_weights=jnp.full(
            (ntime, num_particles), log_weight, dtype=jnp.float32
        ),
        ancestors=ancestors,
        ess=jnp.full((ntime,), num_particles, dtype=jnp.float32),
        log_evidence_increments=jnp.zeros(ntime, dtype=jnp.float32),
    )


def test_lgssm_moments_match_rts_at_five_estimator_se(
    lgssm_params: dict[str, jax.Array],
    lgssm_data: tuple[jax.Array, jax.Array],
) -> None:
    """Complete PF-plus-FFBS estimates match exact smoothing moments."""
    emissions = lgssm_data[1][:15]
    filtered = smcx.kalman_filter(
        lgssm_params["initial_mean"],
        lgssm_params["initial_cov"],
        lgssm_params["dynamics_weights"],
        lgssm_params["dynamics_cov"],
        lgssm_params["emissions_weights"],
        lgssm_params["emissions_cov"],
        emissions,
    )
    exact = smcx.rts_smoother(filtered, lgssm_params["dynamics_weights"])
    exact_means = exact.smoothed_means[_REFERENCE_TIMES, 0]
    exact_variances = exact.smoothed_covariances[_REFERENCE_TIMES, 0, 0]
    keys = jax.vmap(jr.key)(jnp.arange(900, 900 + _NUM_REPLICATES))

    replicate = jax.jit(jax.vmap(_lgssm_replicate, in_axes=(0, None, None)))
    mean_estimates, centered_seconds = replicate(keys, emissions, exact_means)

    _assert_five_se(
        np.asarray(mean_estimates, dtype=np.float64),
        np.asarray(exact_means, dtype=np.float64),
        maximum_band=0.25,
    )
    _assert_five_se(
        np.asarray(centered_seconds, dtype=np.float64),
        np.asarray(exact_variances, dtype=np.float64),
        maximum_band=0.10,
    )


def test_unique_parent_support_matches_filter_genealogy() -> None:
    """Unique-parent support makes FFBS and genealogy exactly equal."""
    ntime, num_particles, num_draws = 4, 5, 7

    def initial(key: jax.Array, count: int) -> jax.Array:
        del key
        return jnp.arange(count, dtype=jnp.int32)[:, None]

    def transition(key: jax.Array, state: jax.Array) -> jax.Array:
        del key
        return state + num_particles

    def log_observation(emission: jax.Array, state: jax.Array) -> jax.Array:
        del emission, state
        return jnp.asarray(0.0, dtype=jnp.float32)

    def log_transition(
        state: jax.Array,
        previous: jax.Array,
        params: Any,
        input_t: Any,
    ) -> jax.Array:
        del params, input_t
        return jnp.where(
            state[0] == previous[0] + num_particles,
            0.0,
            -jnp.inf,
        )

    posterior = smcx.bootstrap_filter(
        jr.key(1000),
        initial,
        transition,
        log_observation,
        jnp.zeros((ntime, 1), dtype=jnp.float32),
        num_particles,
        resampling_threshold=0.0,
    )
    identity = jnp.broadcast_to(
        jnp.arange(num_particles, dtype=jnp.int32),
        (ntime, num_particles),
    )
    np.testing.assert_array_equal(posterior.ancestors, identity)

    result = smcx.backward_simulation(
        jr.key(1001),
        posterior,
        log_transition,
        None,
        num_draws=num_draws,
    )
    terminal = result.backward_indices[:, -1]
    np.testing.assert_array_equal(
        result.backward_indices,
        jnp.broadcast_to(terminal[:, None], (num_draws, ntime)),
    )
    genealogy = smcx.reconstruct_trajectories(posterior)
    expected = jnp.swapaxes(genealogy[:, terminal], 0, 1)
    np.testing.assert_array_equal(result.smoothed_trajectories, expected)


def test_backward_simulation_is_independent_of_ancestor_field() -> None:
    """Valid genealogy metadata cannot alter the FFBS target or draws."""
    ntime, num_particles = 4, 5
    particles = jnp.arange(ntime * num_particles, dtype=jnp.int32).reshape(
        ntime, num_particles, 1
    )
    identity = jnp.broadcast_to(
        jnp.arange(num_particles, dtype=jnp.int32),
        (ntime, num_particles),
    )
    posterior = _discrete_record(particles, identity)
    shifted = identity.at[1:].set((identity[1:] + 1) % num_particles)
    altered = posterior._replace(ancestors=shifted)

    original_genealogy = smcx.reconstruct_trajectories(posterior)
    altered_genealogy = smcx.reconstruct_trajectories(altered)
    assert not np.array_equal(original_genealogy, altered_genealogy)

    def log_transition(
        state: jax.Array,
        previous: jax.Array,
        params: Any,
        input_t: Any,
    ) -> jax.Array:
        del state, previous, params, input_t
        return jnp.asarray(0.0, dtype=jnp.float32)

    original = smcx.backward_simulation(
        jr.key(1002), posterior, log_transition, None, num_draws=9
    )
    changed = smcx.backward_simulation(
        jr.key(1002), altered, log_transition, None, num_draws=9
    )
    np.testing.assert_array_equal(
        original.smoothed_trajectories, changed.smoothed_trajectories
    )
    np.testing.assert_array_equal(
        original.backward_indices, changed.backward_indices
    )
