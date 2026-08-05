# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Timed observation vectors in the DLM and DGLM filters (#380)."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import pytest

import smcx

G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
F_STATIC = jnp.asarray([1.0, 0.0])
F_TIMED = jnp.asarray([
    [1.0, 0.0],
    [0.5, 1.0],
    [1.5, -0.5],
    [1.0, 0.25],
    [0.75, 0.5],
])
W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
M0 = jnp.asarray([0.2, -0.1])
C0 = jnp.asarray([[1.0, 0.1], [0.1, 0.5]])
Y = jnp.asarray([0.4, 0.9, 1.1, 1.6, 2.2])
Y_COUNTS = jnp.asarray([2.0, 1.0, 4.0, 3.0, 5.0])

_CPU_X64 = jax.default_backend() == "cpu" and jax.config.read("jax_enable_x64")


def _assert_posteriors_bitwise_equal(left, right):
    for leaf_l, leaf_r in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(leaf_l), np.asarray(leaf_r))


def test_dlm_static_equals_broadcast_timed_bitwise():
    """A broadcast static row reproduces the static call exactly."""
    static = smcx.dlm_filter(
        M0, C0, G, F_STATIC, Y, scale_free_transition_covariance=W
    )
    timed = smcx.dlm_filter(
        M0,
        C0,
        G,
        jnp.broadcast_to(F_STATIC, (Y.shape[0], 2)),
        Y,
        scale_free_transition_covariance=W,
    )
    _assert_posteriors_bitwise_equal(timed, static)


def test_dglm_static_equals_broadcast_timed_bitwise():
    """The DGLM filter honors the same broadcast identity."""
    static = smcx.dglm_filter(
        M0,
        C0,
        G,
        F_STATIC,
        Y_COUNTS,
        family=smcx.poisson(),
        transition_covariance=W,
    )
    timed = smcx.dglm_filter(
        M0,
        C0,
        G,
        jnp.broadcast_to(F_STATIC, (Y_COUNTS.shape[0], 2)),
        Y_COUNTS,
        family=smcx.poisson(),
        transition_covariance=W,
    )
    _assert_posteriors_bitwise_equal(timed, static)


@pytest.mark.skipif(not _CPU_X64, reason="frozen CPU/x64 arithmetic contract")
def test_dlm_fraction_oracle_with_distinct_rows():
    """Exact rational recursion with per-step rows matches the filter."""
    half = Fraction(1, 2)
    g = [[Fraction(1), Fraction(1)], [Fraction(0), Fraction(1)]]
    w = [[Fraction(1, 20), Fraction(1, 100)], [Fraction(1, 100), half / 5]]
    rows = [
        [Fraction(1), Fraction(0)],
        [half, Fraction(1)],
        [Fraction(3, 2), -half],
    ]
    emissions = [Fraction(2, 5), Fraction(9, 10), Fraction(11, 10)]
    mean = [Fraction(1, 5), -Fraction(1, 10)]
    cov = [[Fraction(1), Fraction(1, 10)], [Fraction(1, 10), half]]
    dof, scale = Fraction(1), Fraction(1)

    def matvec(matrix, vector):
        return [
            sum(m_ij * v_j for m_ij, v_j in zip(row, vector, strict=True))
            for row in matrix
        ]

    def quadratic(vector, matrix):
        inner = matvec(matrix, vector)
        return sum(v * i for v, i in zip(vector, inner, strict=True))

    means, covs, dofs, scales = [], [], [], []
    for t, (row, emission) in enumerate(zip(rows, emissions, strict=True)):
        if t > 0:
            mean = matvec(g, mean)
            cov = [
                [
                    sum(
                        g[i][k] * cov[k][m] * g[j][m]
                        for k in range(2)
                        for m in range(2)
                    )
                    + w[i][j]
                    for j in range(2)
                ]
                for i in range(2)
            ]
        forecast = sum(r * m for r, m in zip(row, mean, strict=True))
        scale_free = quadratic(row, cov) + 1
        residual = emission - forecast
        gain = [
            sum(cov[i][j] * row[j] for j in range(2)) / scale_free
            for i in range(2)
        ]
        mean = [m + gi * residual for m, gi in zip(mean, gain, strict=True)]
        cov = [
            [cov[i][j] - gain[i] * gain[j] * scale_free for j in range(2)]
            for i in range(2)
        ]
        new_dof = dof + 1
        ratio = dof / new_dof
        half_width_sq = residual * residual / (scale_free * dof)
        scale = ratio * scale + ratio * half_width_sq
        dof = new_dof
        means.append(list(mean))
        covs.append([list(r) for r in cov])
        dofs.append(dof)
        scales.append(scale)

    filtered = smcx.dlm_filter(
        M0,
        C0,
        G,
        jnp.asarray(np.asarray(rows, dtype=np.float64)),
        jnp.asarray(np.asarray(emissions, dtype=np.float64)),
        scale_free_transition_covariance=W,
    )
    np.testing.assert_allclose(
        np.asarray(filtered.filtered_means),
        np.asarray([[float(m) for m in row] for row in means]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(filtered.scale_estimates),
        np.asarray([float(s) for s in scales]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(filtered.scale_shapes),
        np.asarray([float(d) for d in dofs]),
        rtol=1e-15,
    )


def test_dglm_normal_reduction_with_timed_rows():
    """The normal family with timed rows reduces to kalman_filter."""
    variance = 0.3

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

    family = smcx.DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )
    emissions = jnp.asarray([0.2, -0.4, 0.9, 0.1, 0.5])
    dglm = smcx.dglm_filter(
        M0,
        C0,
        G,
        F_TIMED,
        emissions,
        family=family,
        transition_covariance=W,
    )
    kalman = smcx.kalman_filter(
        M0,
        C0,
        G,
        W,
        F_TIMED[:, None, :],
        jnp.asarray([[variance]]),
        emissions[:, None],
    )
    eps = float(jnp.finfo(dglm.filtered_means.dtype).eps)
    np.testing.assert_allclose(
        dglm.filtered_means, kalman.filtered_means, rtol=1e3 * eps
    )
    np.testing.assert_allclose(
        dglm.filtered_covariances,
        kalman.filtered_covariances,
        rtol=1e3 * eps,
        atol=1e2 * eps,
    )
    np.testing.assert_allclose(
        dglm.marginal_loglik, kalman.marginal_loglik, rtol=1e3 * eps
    )


def test_missing_row_ignores_its_observation_row():
    """The F_t row at an all-NaN datum cannot influence the run."""
    emissions = Y.at[2].set(jnp.nan)
    other_rows = F_TIMED.at[2].set(jnp.asarray([9.0, -9.0]))
    base = smcx.dlm_filter(
        M0, C0, G, F_TIMED, emissions, scale_free_transition_covariance=W
    )
    perturbed = smcx.dlm_filter(
        M0, C0, G, other_rows, emissions, scale_free_transition_covariance=W
    )
    _assert_posteriors_bitwise_equal(perturbed, base)


def test_bad_timed_shape_is_rejected():
    """A timed history with the wrong leading length is rejected."""
    with pytest.raises(ValueError, match="observation_vector"):
        smcx.dlm_filter(
            M0,
            C0,
            G,
            F_TIMED[:3],
            Y,
            scale_free_transition_covariance=W,
        )
    with pytest.raises(ValueError, match="observation_vector"):
        smcx.dglm_filter(
            M0,
            C0,
            G,
            F_TIMED[:3],
            Y_COUNTS,
            family=smcx.poisson(),
            transition_covariance=W,
        )
