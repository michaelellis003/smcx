# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""gamma DGLM family gates: conjugate algebra and goldens (#383)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

_CPU_X64 = jax.default_backend() == "cpu" and jax.config.read("jax_enable_x64")

# mpmath goldens at 50 digits: (f, q, shape, y) ->
# (alpha, beta, log_forecast). alpha solves trigamma(alpha) = q; beta
# = exp(digamma(alpha) + f - log(shape)); the forecast density is the
# compound gamma. The q = 1e-6 row probes the small-variance boundary
# where alpha reaches 1e6.
_GOLDENS = [
    (
        0.5,
        0.3,
        2.0,
        1.7,
        3.8087190120511670098,
        2.7378194201852255205,
        -1.3815908441271525013,
    ),
    (
        -1.0,
        0.05,
        1.0,
        0.4,
        20.495835239154604799,
        7.3568229735027506319,
        -0.11349619000183561853,
    ),
    (
        2.0,
        1.5,
        3.0,
        9.0,
        1.0656560393745264854,
        1.5330794083627854118,
        -3.5067375572617987914,
    ),
    (
        0.0,
        1e-6,
        2.5,
        1.0,
        1000000.4999999167119,
        399999.99999998335143,
        -0.49395729078586483292,
    ),
    (
        3.0,
        0.01,
        0.7,
        25.0,
        100.49916668194318926,
        2869.350462275155781,
        -4.451664575300284101,
    ),
]

G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
F = jnp.asarray([1.0, 0.0])
W = jnp.asarray([[0.02, 0.005], [0.005, 0.04]])
M0 = jnp.asarray([0.3, 0.05])
C0 = jnp.asarray([[0.3, 0.05], [0.05, 0.1]])
Y_POSITIVE = jnp.asarray([1.2, 0.8, 2.5, 1.9, 3.1])


@pytest.mark.skipif(not _CPU_X64, reason="frozen CPU/x64 arithmetic contract")
@pytest.mark.parametrize(
    "f, q, shape, y, alpha_ref, beta_ref, log_forecast_ref", _GOLDENS
)
def test_match_and_forecast_reproduce_the_mpmath_goldens(
    f, q, shape, y, alpha_ref, beta_ref, log_forecast_ref
):
    """The conjugate match and forecast density hit 50-digit goldens."""
    family = smcx.gamma(shape=shape)
    alpha, beta = family.match_moments(jnp.float64(f), jnp.float64(q))
    np.testing.assert_allclose(float(alpha), alpha_ref, rtol=1e-9)
    np.testing.assert_allclose(float(beta), beta_ref, rtol=1e-8)
    log_density = family.log_forecast(
        jnp.float64(y), jnp.float64(alpha_ref), jnp.float64(beta_ref)
    )
    np.testing.assert_allclose(float(log_density), log_forecast_ref, rtol=1e-8)


def test_moment_match_roundtrip():
    """posterior_moments inverts match_moments (the sign contract)."""
    family = smcx.gamma(shape=2.0)
    for f, q in ((0.5, 0.3), (-1.0, 0.05), (2.0, 1.5)):
        alpha, beta = family.match_moments(jnp.asarray(f), jnp.asarray(q))
        f_back, q_back = family.posterior_moments(alpha, beta)
        f_back = jnp.asarray(f_back)
        rtol = 1e-8 if f_back.dtype == jnp.float64 else 1e-4
        np.testing.assert_allclose(float(f_back), f, rtol=rtol, atol=1e-6)
        np.testing.assert_allclose(float(q_back), q, rtol=rtol)


def test_conjugate_update_is_exact_algebra():
    """Observing y adds the shape to alpha and y to beta."""
    family = smcx.gamma(shape=1.7)
    alpha, beta = family.update(
        jnp.asarray(2.5), jnp.asarray(3.0), jnp.asarray(4.0)
    )
    rtol = 1e-12 if jnp.asarray(alpha).dtype == jnp.float64 else 1e-6
    np.testing.assert_allclose(float(alpha), 3.0 + 1.7, rtol=rtol)
    np.testing.assert_allclose(float(beta), 4.0 + 2.5, rtol=rtol)


def test_filter_runs_on_positive_data_with_missing_rows():
    """The family composes with dglm_filter and ADR-0034 gaps."""
    emissions = Y_POSITIVE.at[2].set(jnp.nan)
    posterior = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        emissions,
        family=smcx.gamma(shape=2.0),
        transition_covariance=W,
    )
    increments = np.asarray(posterior.log_evidence_increments)
    assert np.all(np.isfinite(np.asarray(posterior.filtered_means)))
    np.testing.assert_array_equal(increments[2], 0.0)
    assert np.all(np.isfinite(increments))


def test_forecast_sample_mixture_mean():
    """The log-link recipe reproduces the lognormal mixture mean."""
    shape = 2.0
    posterior = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        Y_POSITIVE,
        family=smcx.gamma(shape=shape),
        transition_covariance=W,
    )
    closed = smcx.dglm_forecast(
        posterior,
        G,
        F,
        family=smcx.gamma(shape=shape),
        num_steps=1,
        transition_covariance=W,
    )
    paths = smcx.dglm_forecast_sample(
        jr.key(31),
        posterior,
        G,
        F,
        sample_emission=lambda key, lam: (
            jr.gamma(key, shape) * jnp.exp(lam) / shape
        ),
        num_steps=1,
        num_draws=15_000,
        transition_covariance=W,
    )
    predictor_mean = float(closed.linear_predictor_means[0])
    predictor_var = float(closed.linear_predictor_variances[0])
    mixture_mean = np.exp(predictor_mean + predictor_var / 2.0)
    draws = np.asarray(paths.emission_paths, dtype=np.float64)[:, 0]
    se = draws.std(ddof=1) / np.sqrt(draws.shape[0])
    assert abs(draws.mean() - mixture_mean) < 8.0 * se


def test_emission_validation_rejects_nonpositive_values():
    """Nonpositive values get the family error, infinities the row error."""

    def run(bad):
        return smcx.dglm_filter(
            M0,
            C0,
            G,
            F,
            Y_POSITIVE.at[1].set(bad),
            family=smcx.gamma(shape=2.0),
            transition_covariance=W,
        )

    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="gamma emissions"):
            run(bad)
    with pytest.raises(ValueError, match="fully observed finite"):
        run(jnp.inf)


def _build_with(shape):
    return smcx.gamma(shape=shape)


def test_shape_boundary_matrix():
    """The observation shape must be a finite positive scalar."""
    for bad in (0.0, -1.0, jnp.nan):
        with pytest.raises(ValueError, match="shape"):
            _build_with(bad)
