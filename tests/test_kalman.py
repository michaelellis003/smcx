# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact linear-Gaussian filtering and smoothing."""

import jax.numpy as jnp
import numpy as np

import smcx
from tests._lgssm_reference import EXACT_LOG_LIKELIHOOD, REFERENCE_TIMES
from tests._lgssm_reference import FILTERED_MEANS as EXACT_FILTERED_MEANS
from tests._lgssm_reference import FILTERED_VARIANCES as EXACT_FILTERED_VARS


def test_kalman_filter_matches_frozen_dynamax_reference(
    lgssm_params, lgssm_data
):
    """The exact filter reproduces independently generated moments."""
    _, emissions = lgssm_data
    posterior = smcx.kalman_filter(
        lgssm_params["initial_mean"],
        lgssm_params["initial_cov"],
        lgssm_params["dynamics_weights"],
        lgssm_params["dynamics_cov"],
        lgssm_params["emissions_weights"],
        lgssm_params["emissions_cov"],
        emissions,
    )

    is_f64 = posterior.filtered_means.dtype == jnp.float64
    # Dynamax's PSD solve adds 1e-9 jitter. Against the unjittered
    # covariance-form recurrence this shifts the 50-step f64 log evidence
    # by 2.3e-9 and selected variances by at most 5.1e-10. The 5e-9
    # absolute gate admits that known oracle-policy difference; 2e-5 is
    # the explicit f32/Metal arithmetic budget.
    atol = 5e-9 if is_f64 else 2e-5
    np.testing.assert_allclose(
        posterior.marginal_loglik,
        EXACT_LOG_LIKELIHOOD,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.filtered_means[REFERENCE_TIMES, 0],
        EXACT_FILTERED_MEANS,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.filtered_covariances[REFERENCE_TIMES, 0, 0],
        EXACT_FILTERED_VARS,
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.predicted_means[0],
        lgssm_params["initial_mean"],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.predicted_means[1:, 0],
        0.9 * posterior.filtered_means[:-1, 0],
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        posterior.log_evidence_increments.sum(),
        posterior.marginal_loglik,
        rtol=0.0,
        atol=atol,
    )
