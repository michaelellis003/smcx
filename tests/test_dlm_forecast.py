# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""dlm_forecast gates: W&H section 6.3.3 conventions and boundaries (#381)."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import smcx

G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
F = jnp.asarray([1.0, 0.0])
W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
M0 = jnp.asarray([0.2, -0.1])
C0 = jnp.asarray([[1.0, 0.1], [0.1, 0.5]])
Y = jnp.asarray([0.4, 0.9, 1.1, 1.6, 2.2])
DISCOUNT = 0.9

FILTERED_W = smcx.dlm_filter(
    M0, C0, G, F, Y, scale_free_transition_covariance=W
)
FILTERED_D = smcx.dlm_filter(M0, C0, G, F, Y, discount=DISCOUNT)


def test_explicit_w_forecast_matches_the_nan_padded_filter_bitwise():
    """With explicit W the forecast equals filtering through a NaN gap."""
    num_steps = 3
    padded = smcx.dlm_filter(
        M0,
        C0,
        G,
        F,
        jnp.concatenate((Y, jnp.full((num_steps,), jnp.nan))),
        scale_free_transition_covariance=W,
    )
    forecast = smcx.dlm_forecast(
        FILTERED_W,
        G,
        F,
        num_steps=num_steps,
        scale_free_transition_covariance=W,
    )
    ntime = Y.shape[0]
    np.testing.assert_array_equal(
        np.asarray(forecast.state_means),
        np.asarray(padded.filtered_means[ntime:]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_scale_free_covariances),
        np.asarray(padded.filtered_scale_free_covariances[ntime:]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.scale_shapes),
        np.asarray(padded.scale_shapes[ntime:]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.scale_estimates),
        np.asarray(padded.scale_estimates[ntime:]),
    )


def test_discount_forecast_horizon_one_matches_the_padded_filter():
    """At k=1 the frozen-W and rediscounting conventions coincide."""
    padded = smcx.dlm_filter(
        M0,
        C0,
        G,
        F,
        jnp.concatenate((Y, jnp.asarray([jnp.nan]))),
        discount=DISCOUNT,
    )
    forecast = smcx.dlm_forecast(
        FILTERED_D, G, F, num_steps=1, discount=DISCOUNT
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_means[0]),
        np.asarray(padded.filtered_means[-1]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_scale_free_covariances[0]),
        np.asarray(padded.filtered_scale_free_covariances[-1]),
    )


def test_discount_forecast_freezes_the_frontier_evolution_variance():
    """Beyond k=1 the W&H frozen-W convention rules, not rediscounting."""
    num_steps = 3
    forecast = smcx.dlm_forecast(
        FILTERED_D, G, F, num_steps=num_steps, discount=DISCOUNT
    )
    g_np = np.asarray(G, dtype=np.float64)
    covariance = np.asarray(
        FILTERED_D.filtered_scale_free_covariances[-1], dtype=np.float64
    )
    propagated = g_np @ covariance @ g_np.T
    frontier_w = propagated * (1.0 / DISCOUNT - 1.0)
    rediscounted = propagated / DISCOUNT
    expected = [rediscounted]
    for _ in range(num_steps - 1):
        expected.append(g_np @ expected[-1] @ g_np.T + frontier_w)
    dtype = np.asarray(forecast.state_scale_free_covariances).dtype
    rtol = 1e-10 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(
        np.asarray(forecast.state_scale_free_covariances),
        np.stack(expected),
        rtol=rtol,
    )
    padded = smcx.dlm_filter(
        M0,
        C0,
        G,
        F,
        jnp.concatenate((Y, jnp.full((num_steps,), jnp.nan))),
        discount=DISCOUNT,
    )
    assert not np.allclose(
        np.asarray(forecast.state_scale_free_covariances[-1]),
        np.asarray(padded.filtered_scale_free_covariances[-1]),
        rtol=1e-3,
    )


def test_one_step_forecast_density_matches_the_evidence_increment():
    """The k=1 Student-t density at the held-out row is the increment."""
    held_out = smcx.dlm_filter(
        M0, C0, G, F, Y[:-1], scale_free_transition_covariance=W
    )
    forecast = smcx.dlm_forecast(
        held_out,
        G,
        F,
        num_steps=1,
        scale_free_transition_covariance=W,
    )
    dof = float(forecast.scale_shapes[0])
    location = float(forecast.observation_means[0])
    scale = np.sqrt(float(forecast.observation_scales[0]))
    log_density = stats.t.logpdf(
        float(Y[-1]), df=dof, loc=location, scale=scale
    )
    increment = np.asarray(FILTERED_W.log_evidence_increments[-1])
    rtol = 1e-9 if increment.dtype == np.float64 else 1e-4
    np.testing.assert_allclose(log_density, increment, rtol=rtol)


def test_variance_discount_decays_the_forecast_dof():
    """Each horizon applies one variance-discount evolution to the dof."""
    variance_discount = 0.95
    filtered = smcx.dlm_filter(
        M0,
        C0,
        G,
        F,
        Y,
        scale_free_transition_covariance=W,
        variance_discount=variance_discount,
    )
    forecast = smcx.dlm_forecast(
        filtered,
        G,
        F,
        num_steps=3,
        scale_free_transition_covariance=W,
        variance_discount=variance_discount,
    )
    terminal = np.asarray(filtered.scale_shapes[-1], dtype=np.float64)
    expected = terminal * variance_discount ** np.arange(1, 4)
    dtype = np.asarray(forecast.scale_shapes).dtype
    rtol = 1e-12 if dtype == np.float64 else 1e-6
    np.testing.assert_allclose(
        np.asarray(forecast.scale_shapes), expected, rtol=rtol
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.scale_estimates),
        np.full(3, np.asarray(filtered.scale_estimates[-1])),
    )


def test_per_horizon_evolution_covariances():
    """A (num_steps, d, d) future-W history is accepted and applied."""
    w_future = jnp.stack([W, W * 2.0])
    forecast = smcx.dlm_forecast(
        FILTERED_W,
        G,
        F,
        num_steps=2,
        scale_free_transition_covariance=w_future,
    )
    static = smcx.dlm_forecast(
        FILTERED_W,
        G,
        F,
        num_steps=2,
        scale_free_transition_covariance=W,
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_scale_free_covariances[0]),
        np.asarray(static.state_scale_free_covariances[0]),
    )
    assert not np.array_equal(
        np.asarray(forecast.state_scale_free_covariances[1]),
        np.asarray(static.state_scale_free_covariances[1]),
    )


def test_supply_exactly_one_evolution_specification():
    """W and discount together, or neither, get the documented error."""
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dlm_forecast(
            FILTERED_W,
            G,
            F,
            num_steps=1,
            scale_free_transition_covariance=W,
            discount=DISCOUNT,
        )
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dlm_forecast(FILTERED_W, G, F, num_steps=1)


def _forecast_with_count(num_steps):
    return smcx.dlm_forecast(
        FILTERED_W,
        G,
        F,
        num_steps=num_steps,
        scale_free_transition_covariance=W,
    )


def test_num_steps_boundary_matrix():
    """Count validation matches the shared positive-integer contract."""
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _forecast_with_count(bad)


def test_discount_above_one_is_rejected():
    """The forecast validates the discount like the filter."""
    with pytest.raises(ValueError, match="discount"):
        smcx.dlm_forecast(FILTERED_D, G, F, num_steps=1, discount=1.5)
