# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""as_covariance gates: the explicit adapter back to covariance form."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

MU0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y = jnp.asarray([
    [0.3, -0.1],
    [0.6, 0.2],
    [-0.4, 0.9],
    [0.1, 0.05],
])

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


def test_filter_conversion_agrees_with_the_covariance_run():
    """The converted record matches kalman_filter field by field."""
    converted = smcx.as_covariance(smcx.sqrt_kalman_filter(MODEL, Y))
    exact = smcx.kalman_filter(MODEL, Y)
    assert type(converted) is smcx.GaussianFilterPosterior
    rtol = _rtol(converted.filtered_means)
    np.testing.assert_allclose(
        np.asarray(converted.filtered_covariances),
        np.asarray(exact.filtered_covariances),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(converted.predicted_covariances),
        np.asarray(exact.predicted_covariances),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_array_equal(
        np.asarray(converted.filtered_means),
        np.asarray(smcx.sqrt_kalman_filter(MODEL, Y).filtered_means),
    )


def test_smoother_conversion_agrees_with_the_covariance_run():
    """The converted smoother record matches rts_smoother."""
    sqrt = smcx.sqrt_rts_smoother(smcx.sqrt_kalman_filter(MODEL, Y), MODEL)
    converted = smcx.as_covariance(sqrt)
    exact = smcx.rts_smoother(smcx.kalman_filter(MODEL, Y), MODEL)
    assert type(converted) is smcx.GaussianSmootherPosterior
    rtol = _rtol(converted.smoothed_means)
    np.testing.assert_allclose(
        np.asarray(converted.smoothed_covariances),
        np.asarray(exact.smoothed_covariances),
        rtol=rtol,
        atol=rtol,
    )


def test_converted_records_feed_the_existing_consumers():
    """posterior_sample, forecasts, and cross-covariances consume it."""
    converted = smcx.as_covariance(smcx.sqrt_kalman_filter(MODEL, Y))
    draws = smcx.posterior_sample(jr.key(3), converted, MODEL, num_draws=8)
    assert draws.shape == (8, Y.shape[0], 2)
    forecast = smcx.kalman_forecast(converted, MODEL, num_steps=2)
    assert forecast.state_means.shape == (2, 2)
    smoothed = smcx.rts_smoother(converted, MODEL)
    cross = smcx.smoothed_cross_covariances(smoothed, MODEL)
    assert cross.shape == (Y.shape[0] - 1, 2, 2)


def test_conversion_is_psd_at_every_step():
    """Reconstructed covariances stay PSD (the Gram guarantee)."""
    converted = smcx.as_covariance(smcx.sqrt_kalman_filter(MODEL, Y))
    for stack in (
        converted.filtered_covariances,
        converted.predicted_covariances,
    ):
        for step in np.asarray(stack):
            assert np.linalg.eigvalsh(step).min() > -1e-10


def test_unsupported_input_is_rejected():
    """Records outside the square-root family get a named error."""
    plain: Any = smcx.kalman_filter(MODEL, Y)
    with pytest.raises(ValueError, match="square-root"):
        smcx.as_covariance(plain)
    arbitrary: Any = object()
    with pytest.raises(ValueError, match="square-root"):
        smcx.as_covariance(arbitrary)
