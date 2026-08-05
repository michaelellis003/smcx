# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""dglm_forecast gates: conjugate composition and boundaries (#381)."""

import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import pytest

import smcx

G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
F = jnp.asarray([1.0, 0.0])
W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
M0 = jnp.asarray([0.3, 0.1])
C0 = jnp.asarray([[0.5, 0.1], [0.1, 0.2]])
Y_COUNTS = jnp.asarray([2.0, 1.0, 4.0, 3.0, 5.0])
DISCOUNT = 0.9

FILTERED_W = smcx.dglm_filter(
    M0,
    C0,
    G,
    F,
    Y_COUNTS,
    family=smcx.poisson(),
    transition_covariance=W,
)
FILTERED_D = smcx.dglm_filter(
    M0, C0, G, F, Y_COUNTS, family=smcx.poisson(), discount=DISCOUNT
)


def _normal_family(variance: float) -> smcx.DGLMFamily:
    """Known-variance normal family (the tests/test_dglm.py reduction)."""

    def match_moments(forecast_mean, forecast_variance):
        return forecast_mean, forecast_variance

    def log_forecast(emission, mean, var):
        return jstats.norm.logpdf(
            emission, loc=mean, scale=jnp.sqrt(var + variance)
        )

    def update(emission, mean, var):
        gain = var / (var + variance)
        return mean + gain * (emission - mean), var * variance / (
            var + variance
        )

    def posterior_moments(mean, var):
        return mean, var

    return smcx.DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )


def test_explicit_w_forecast_matches_the_nan_padded_filter_bitwise():
    """With explicit W the forecast equals filtering through a NaN gap."""
    num_steps = 3
    padded = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        jnp.concatenate((Y_COUNTS, jnp.full((num_steps,), jnp.nan))),
        family=smcx.poisson(),
        transition_covariance=W,
    )
    forecast = smcx.dglm_forecast(
        FILTERED_W,
        G,
        F,
        family=smcx.poisson(),
        num_steps=num_steps,
        transition_covariance=W,
    )
    ntime = Y_COUNTS.shape[0]
    np.testing.assert_array_equal(
        np.asarray(forecast.state_means),
        np.asarray(padded.filtered_means[ntime:]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_covariances),
        np.asarray(padded.filtered_covariances[ntime:]),
    )
    # Conjugate rows compare at the eager-vs-compiled contract: the
    # filter matches moments inside its compiled scan, the forecast's
    # first horizon matches eagerly, and special functions may round
    # one ulp apart between the two lowerings.
    eps = float(np.finfo(np.asarray(forecast.conjugate_alphas).dtype).eps)
    np.testing.assert_allclose(
        np.asarray(forecast.conjugate_alphas),
        np.asarray(padded.conjugate_alphas[ntime:]),
        rtol=64 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(forecast.conjugate_betas),
        np.asarray(padded.conjugate_betas[ntime:]),
        rtol=64 * eps,
    )


def test_discount_forecast_horizon_one_matches_the_padded_filter():
    """At k=1 the frozen-W and rediscounting conventions coincide."""
    padded = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        jnp.concatenate((Y_COUNTS, jnp.asarray([jnp.nan]))),
        family=smcx.poisson(),
        discount=DISCOUNT,
    )
    forecast = smcx.dglm_forecast(
        FILTERED_D,
        G,
        F,
        family=smcx.poisson(),
        num_steps=1,
        discount=DISCOUNT,
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_means[0]),
        np.asarray(padded.filtered_means[-1]),
    )
    np.testing.assert_array_equal(
        np.asarray(forecast.state_covariances[0]),
        np.asarray(padded.filtered_covariances[-1]),
    )
    eps = float(np.finfo(np.asarray(forecast.conjugate_alphas).dtype).eps)
    np.testing.assert_allclose(
        np.asarray(forecast.conjugate_alphas[0]),
        np.asarray(padded.conjugate_alphas[-1]),
        rtol=64 * eps,
    )


def test_discount_forecast_freezes_the_frontier_evolution_variance():
    """Beyond k=1 the W&H frozen-W convention rules, not rediscounting."""
    num_steps = 3
    forecast = smcx.dglm_forecast(
        FILTERED_D,
        G,
        F,
        family=smcx.poisson(),
        num_steps=num_steps,
        discount=DISCOUNT,
    )
    g_np = np.asarray(G, dtype=np.float64)
    covariance = np.asarray(
        FILTERED_D.filtered_covariances[-1], dtype=np.float64
    )
    propagated = g_np @ covariance @ g_np.T
    frontier_w = propagated * (1.0 / DISCOUNT - 1.0)
    expected = [propagated / DISCOUNT]
    for _ in range(num_steps - 1):
        expected.append(g_np @ expected[-1] @ g_np.T + frontier_w)
    dtype = np.asarray(forecast.state_covariances).dtype
    rtol = 1e-10 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(
        np.asarray(forecast.state_covariances), np.stack(expected), rtol=rtol
    )


def test_one_step_forecast_density_matches_the_evidence_increment():
    """The k=1 conjugate forecast density at the held-out count matches."""
    held_out = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        Y_COUNTS[:-1],
        family=smcx.poisson(),
        transition_covariance=W,
    )
    forecast = smcx.dglm_forecast(
        held_out,
        G,
        F,
        family=smcx.poisson(),
        num_steps=1,
        transition_covariance=W,
    )
    log_density = smcx.poisson().log_forecast(
        Y_COUNTS[-1],
        forecast.conjugate_alphas[0],
        forecast.conjugate_betas[0],
    )
    eps = float(np.finfo(np.asarray(log_density).dtype).eps)
    np.testing.assert_allclose(
        np.asarray(log_density),
        np.asarray(FILTERED_W.log_evidence_increments[-1]),
        rtol=1e3 * eps,
    )


def test_normal_family_reduction_to_kalman_forecast():
    """The normal family reproduces kalman_forecast at every horizon."""
    variance = 0.3
    emissions = jnp.asarray([0.2, -0.4, 0.9, 0.1, 0.5])
    num_steps = 3
    dglm = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        emissions,
        family=_normal_family(variance),
        transition_covariance=W,
    )
    kalman = smcx.kalman_filter(
        M0,
        C0,
        G,
        W,
        F[None, :],
        jnp.asarray([[variance]]),
        emissions[:, None],
    )
    dglm_fc = smcx.dglm_forecast(
        dglm,
        G,
        F,
        family=_normal_family(variance),
        num_steps=num_steps,
        transition_covariance=W,
    )
    kalman_fc = smcx.kalman_forecast(
        kalman,
        G,
        W,
        F[None, :],
        jnp.asarray([[variance]]),
        num_steps=num_steps,
    )
    eps = float(jnp.finfo(dglm_fc.state_means.dtype).eps)
    np.testing.assert_allclose(
        np.asarray(dglm_fc.state_means),
        np.asarray(kalman_fc.state_means),
        rtol=1e3 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(dglm_fc.state_covariances),
        np.asarray(kalman_fc.state_covariances),
        rtol=1e3 * eps,
        atol=1e2 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(dglm_fc.linear_predictor_means),
        np.asarray(kalman_fc.observation_means[:, 0]),
        rtol=1e3 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(dglm_fc.linear_predictor_variances) + variance,
        np.asarray(kalman_fc.observation_covariances[:, 0, 0]),
        rtol=1e3 * eps,
    )


def test_supply_exactly_one_evolution_specification():
    """W and discount together, or neither, get the documented error."""
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dglm_forecast(
            FILTERED_W,
            G,
            F,
            family=smcx.poisson(),
            num_steps=1,
            transition_covariance=W,
            discount=DISCOUNT,
        )
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dglm_forecast(FILTERED_W, G, F, family=smcx.poisson(), num_steps=1)


def _forecast_with_count(num_steps):
    return smcx.dglm_forecast(
        FILTERED_W,
        G,
        F,
        family=smcx.poisson(),
        num_steps=num_steps,
        transition_covariance=W,
    )


def test_num_steps_boundary_matrix():
    """Count validation matches the shared positive-integer contract."""
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _forecast_with_count(bad)


def test_discount_above_one_is_rejected():
    """The forecast validates the discount like the filter."""
    with pytest.raises(ValueError, match="discount"):
        smcx.dglm_forecast(
            FILTERED_D,
            G,
            F,
            family=smcx.poisson(),
            num_steps=1,
            discount=1.5,
        )


def test_dispersion_discount_above_one_is_rejected():
    """The dispersion discount is validated like the filter's."""
    with pytest.raises(ValueError, match="dispersion_discount"):
        smcx.dglm_forecast(
            FILTERED_W,
            G,
            F,
            family=smcx.poisson(),
            num_steps=1,
            transition_covariance=W,
            dispersion_discount=1.5,
        )
