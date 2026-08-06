# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Partially missing rows in kalman_filter and the sqrt pair (#433)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx

MU0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y_FULL = jnp.asarray([
    [0.3, -0.1],
    [0.6, 0.2],
    [-0.4, 0.9],
    [0.1, 0.05],
    [0.7, -0.3],
])
# Component 0 missing at t = 1; component 1 missing at t = 3.
Y_PARTIAL = Y_FULL.at[1, 0].set(jnp.nan).at[3, 1].set(jnp.nan)

MODEL = smcx.LinearGaussianModel(
    initial_mean=MU0,
    initial_covariance=P0,
    transition_matrix=A,
    transition_covariance=Q,
    observation_matrix=H,
    observation_covariance=R,
)


def _rtol(array):
    return 1e-10 if array.dtype == jnp.float64 else 2e-4


def _subset_oracle(emissions):
    """Textbook recursion using only each row's observed components."""
    mean = np.asarray(MU0, dtype=np.float64)
    cov = np.asarray(P0, dtype=np.float64)
    a = np.asarray(A, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    h = np.asarray(H, dtype=np.float64)
    r = np.asarray(R, dtype=np.float64)
    means, covs, increments = [], [], []
    for t, row in enumerate(np.asarray(emissions, dtype=np.float64)):
        if t > 0:
            mean = a @ mean
            cov = a @ cov @ a.T + q
            cov = 0.5 * (cov + cov.T)
        observed = ~np.isnan(row)
        if observed.any():
            h_o = h[observed]
            r_o = r[np.ix_(observed, observed)]
            y_o = row[observed]
            innovation = y_o - h_o @ mean
            s = h_o @ cov @ h_o.T + r_o
            gain = np.linalg.solve(s, h_o @ cov).T
            mean = mean + gain @ innovation
            joseph = np.eye(2) - gain @ h_o
            cov = joseph @ cov @ joseph.T + gain @ r_o @ gain.T
            cov = 0.5 * (cov + cov.T)
            _, logdet = np.linalg.slogdet(s)
            increments.append(
                -0.5
                * (
                    observed.sum() * math.log(2.0 * math.pi)
                    + logdet
                    + innovation @ np.linalg.solve(s, innovation)
                )
            )
        else:
            increments.append(0.0)
        means.append(mean.copy())
        covs.append(cov.copy())
    return np.asarray(means), np.asarray(covs), np.asarray(increments)


def test_partial_rows_match_the_subset_oracle():
    """The masked update equals the observed-subvector recursion."""
    posterior = smcx.kalman_filter(MODEL, Y_PARTIAL)
    means, covs, increments = _subset_oracle(Y_PARTIAL)
    rtol = _rtol(posterior.filtered_means)
    np.testing.assert_allclose(
        np.asarray(posterior.filtered_means), means, rtol=rtol, atol=rtol
    )
    np.testing.assert_allclose(
        np.asarray(posterior.filtered_covariances), covs, rtol=rtol, atol=rtol
    )
    np.testing.assert_allclose(
        np.asarray(posterior.log_evidence_increments),
        increments,
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(posterior.marginal_loglik),
        increments.sum(),
        rtol=1e-6,
    )


def test_fully_observed_rows_are_bitwise_unchanged():
    """A no-NaN series takes the existing path exactly."""
    with_mask_path = smcx.kalman_filter(MODEL, Y_FULL)
    means, _covs, _increments = _subset_oracle(Y_FULL)
    rtol = _rtol(with_mask_path.filtered_means)
    np.testing.assert_allclose(
        np.asarray(with_mask_path.filtered_means), means, rtol=rtol, atol=rtol
    )


def test_all_nan_rows_keep_the_identity_contract():
    """An entirely NaN row still stores the prediction, increment 0."""
    emissions = Y_FULL.at[2].set(jnp.nan)
    posterior = smcx.kalman_filter(MODEL, emissions)
    np.testing.assert_array_equal(
        np.asarray(posterior.filtered_means[2]),
        np.asarray(posterior.predicted_means[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.filtered_covariances[2]),
        np.asarray(posterior.predicted_covariances[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.log_evidence_increments[2]), 0.0
    )


def test_sqrt_filter_agrees_on_partial_rows():
    """The square-root form matches the covariance form with masks."""
    sqrt = smcx.sqrt_kalman_filter(MODEL, Y_PARTIAL)
    exact = smcx.kalman_filter(MODEL, Y_PARTIAL)
    rtol = _rtol(sqrt.filtered_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(sqrt.log_evidence_increments),
        np.asarray(exact.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )
    gram = np.einsum(
        "tij,tkj->tik",
        np.asarray(sqrt.filtered_factors),
        np.asarray(sqrt.filtered_factors),
    )
    np.testing.assert_allclose(
        gram, np.asarray(exact.filtered_covariances), rtol=rtol, atol=rtol
    )


def test_partial_records_feed_the_smoothers():
    """Both smoothers consume partial-row records without change."""
    smoothed = smcx.rts_smoother(smcx.kalman_filter(MODEL, Y_PARTIAL), MODEL)
    assert np.all(np.isfinite(np.asarray(smoothed.smoothed_means)))
    sqrt_smoothed = smcx.sqrt_rts_smoother(
        smcx.sqrt_kalman_filter(MODEL, Y_PARTIAL), MODEL
    )
    rtol = _rtol(smoothed.smoothed_means)
    np.testing.assert_allclose(
        np.asarray(sqrt_smoothed.smoothed_means),
        np.asarray(smoothed.smoothed_means),
        rtol=rtol,
        atol=rtol,
    )


def test_gradients_stay_finite_through_partial_rows():
    """The double-where keeps gradients NaN-free at masked components."""

    def loss(transition):
        model = MODEL._replace(transition_matrix=transition)
        return smcx.kalman_filter(model, Y_PARTIAL).marginal_loglik

    gradient = jax.grad(loss)(A)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_infinite_entries_are_still_rejected():
    """Infinities remain never-meaningful and fail eagerly."""
    for filter_fn in (smcx.kalman_filter, smcx.sqrt_kalman_filter):
        with pytest.raises(ValueError, match="finite"):
            filter_fn(MODEL, Y_FULL.at[1, 0].set(jnp.inf))


# The nonlinear family's partial-row behavior is gated in
# tests/test_partial_missing_nonlinear.py (the second #433 slice).
