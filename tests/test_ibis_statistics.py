# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Analytical and Monte Carlo acceptance test for public IBIS."""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

import smcx
from smcx.types import ModelInput, PRNGKeyT

_PRIOR_MEAN = -0.25
_PRIOR_SD = 1.10
_OBSERVATION_SD = 0.65
_OBSERVATIONS = np.asarray([1.2, -0.4, 0.8, 1.6, -0.2])
_NUM_RUNS = 32
_NUM_PARTICLES = 1_024


def _normal_prefix_oracle() -> tuple[np.ndarray, np.ndarray]:
    """Return exact posterior moments and cumulative predictive evidence."""
    prior_variance = _PRIOR_SD**2
    observation_variance = _OBSERVATION_SD**2
    counts = np.arange(1, _OBSERVATIONS.size + 1, dtype=np.float64)
    posterior_variances = 1.0 / (
        1.0 / prior_variance + counts / observation_variance
    )
    posterior_means = posterior_variances * (
        _PRIOR_MEAN / prior_variance
        + np.cumsum(_OBSERVATIONS) / observation_variance
    )
    raw_second_moments = posterior_variances + posterior_means**2

    previous_means = np.concatenate(([_PRIOR_MEAN], posterior_means[:-1]))
    previous_variances = np.concatenate((
        [prior_variance],
        posterior_variances[:-1],
    ))
    predictive_variances = observation_variance + previous_variances
    predictive_log_densities = -0.5 * (
        np.log(2.0 * math.pi * predictive_variances)
        + (_OBSERVATIONS - previous_means) ** 2 / predictive_variances
    )
    expected = np.stack((
        posterior_means,
        raw_second_moments,
        np.ones(_OBSERVATIONS.size),
    ))
    return expected, np.cumsum(predictive_log_densities)


def _run_normal_ibis(key: PRNGKeyT) -> np.ndarray:
    """Return prefix mean, raw-second-moment, and evidence estimates."""
    prior_mean = jnp.float32(_PRIOR_MEAN)
    prior_sd = jnp.float32(_PRIOR_SD)
    observation_sd = jnp.float32(_OBSERVATION_SD)
    log_two_pi = jnp.float32(math.log(2.0 * math.pi))

    def sample_prior(
        sample_key: PRNGKeyT,
        count: int,
    ) -> Float[Array, "num_particles 1"]:
        draws = jr.normal(sample_key, (count, 1), dtype=jnp.float32)
        return prior_mean + prior_sd * draws

    def log_prior(position: Float[Array, " param_dim"]) -> Float[Array, ""]:
        standardized = (position[0] - prior_mean) / prior_sd
        return -0.5 * standardized**2 - jnp.log(prior_sd) - 0.5 * log_two_pi

    def log_increment(
        emission: Float[Array, " emission_dim"],
        position: Float[Array, " param_dim"],
        input_t: ModelInput | None,
    ) -> Float[Array, ""]:
        assert input_t is None
        standardized = (emission[0] - position[0]) / observation_sd
        return (
            -0.5 * standardized**2 - jnp.log(observation_sd) - 0.5 * log_two_pi
        )

    posterior = smcx.ibis(
        key,
        sample_prior,
        log_prior,
        log_increment,
        jnp.asarray(_OBSERVATIONS, dtype=jnp.float32),
        _NUM_PARTICLES,
        num_mcmc_steps=3,
        resampling_threshold=0.5,
    )
    positions = np.asarray(
        posterior.filtered_params[..., 0],
        dtype=np.float64,
    )
    log_weights = np.asarray(
        posterior.filtered_log_weights,
        dtype=np.float64,
    )
    log_weights -= np.logaddexp.reduce(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    means = np.sum(weights * positions, axis=1)
    raw_second_moments = np.sum(weights * positions**2, axis=1)

    _, exact_log_evidence = _normal_prefix_oracle()
    estimated_log_evidence = np.cumsum(
        np.asarray(posterior.log_evidence_increments, dtype=np.float64)
    )
    estimated_log_evidence[-1] = float(posterior.marginal_loglik)
    evidence_ratios = np.exp(estimated_log_evidence - exact_log_evidence)
    return np.stack((means, raw_second_moments, evidence_ratios))


def test_ibis_matches_every_normal_partial_posterior_and_evidence():
    """Check all partial targets with MC-error-honest repeated runs."""
    expected, _ = _normal_prefix_oracle()
    keys = jr.split(jr.key(278_001), _NUM_RUNS)
    estimates = np.stack([_run_normal_ibis(key) for key in keys])
    observed = estimates.mean(axis=0)

    # Complete IBIS runs are the independent units, so the estimator SE is
    # their sample SD / sqrt(R). Five SE is the committed Monte Carlo bound.
    estimator_se = estimates.std(axis=0, ddof=1) / math.sqrt(_NUM_RUNS)
    five_se = 5.0 * estimator_se
    # 2e-5 is under 168 f32 eps at unit scale and covers the five-prefix
    # log/exp/reduction path; it is arithmetic allowance, not MC slack.
    np.testing.assert_array_less(
        np.abs(observed - expected),
        five_se + 2e-5,
    )
    # These ceilings prevent a high-variance fixture from making 5 SE vacuous.
    np.testing.assert_array_less(
        five_se.max(axis=1),
        np.asarray([0.02, 0.03, 0.05]),
    )
