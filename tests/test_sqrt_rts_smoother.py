# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""sqrt_rts_smoother gates: agreement, regimes, the type wall."""

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
    return 1e-9 if array.dtype == jnp.float64 else 2e-4


def _gram(factors):
    return np.einsum("tij,tkj->tik", np.asarray(factors), np.asarray(factors))


def test_well_conditioned_agrees_with_the_covariance_smoother():
    """Smoothed means and reconstructed covariances match rts_smoother."""
    sqrt = smcx.sqrt_rts_smoother(smcx.sqrt_kalman_filter(MODEL, Y), MODEL)
    exact = smcx.rts_smoother(smcx.kalman_filter(MODEL, Y), MODEL)
    rtol = _rtol(sqrt.smoothed_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.smoothed_means),
        np.asarray(exact.smoothed_means),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        _gram(sqrt.smoothed_factors),
        np.asarray(exact.smoothed_covariances),
        rtol=rtol,
        atol=rtol,
    )


def test_record_path_matches_array_path_bitwise():
    """A model record and loose arrays produce identical output."""
    filtered = smcx.sqrt_kalman_filter(MODEL, Y)
    from_record = smcx.sqrt_rts_smoother(filtered, MODEL)
    from_arrays = smcx.sqrt_rts_smoother(filtered, A, Q)
    for left, right in zip(
        jax.tree.leaves(from_record), jax.tree.leaves(from_arrays), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_covariance_form_posterior_hits_the_type_wall():
    """The covariance record is rejected with the pairing error."""
    from typing import Any

    filtered: Any = smcx.kalman_filter(MODEL, Y)
    with pytest.raises(ValueError, match="sqrt_kalman_filter"):
        smcx.sqrt_rts_smoother(filtered, MODEL)


def test_non_record_posterior_is_rejected():
    """An arbitrary object gets the documented record error."""
    from typing import Any

    non_record: Any = object()
    with pytest.raises(ValueError, match="SqrtGaussianFilterPosterior"):
        smcx.sqrt_rts_smoother(non_record, MODEL)


def test_malformed_record_shapes_are_rejected():
    """Shape and value violations raise at the eager boundary."""
    filtered = smcx.sqrt_kalman_filter(MODEL, Y)
    with pytest.raises(ValueError, match="filtered_means"):
        smcx.sqrt_rts_smoother(
            filtered._replace(filtered_means=filtered.filtered_means[:, :0]),
            MODEL,
        )
    with pytest.raises(ValueError, match="predicted_factors shape"):
        smcx.sqrt_rts_smoother(
            filtered._replace(
                predicted_factors=filtered.predicted_factors[:-1]
            ),
            MODEL,
        )
    with pytest.raises(ValueError, match="marginal_loglik"):
        smcx.sqrt_rts_smoother(
            filtered._replace(marginal_loglik=jnp.zeros((2,))), MODEL
        )
    with pytest.raises(ValueError, match="finite"):
        smcx.sqrt_rts_smoother(
            filtered._replace(
                filtered_factors=filtered.filtered_factors.at[0, 0, 0].set(
                    jnp.nan
                )
            ),
            MODEL,
        )


def test_missing_rows_ride_through_the_backward_pass():
    """A gap record smooths and agrees with the covariance form."""
    emissions = Y.at[2].set(jnp.nan)
    sqrt = smcx.sqrt_rts_smoother(
        smcx.sqrt_kalman_filter(MODEL, emissions), MODEL
    )
    exact = smcx.rts_smoother(smcx.kalman_filter(MODEL, emissions), MODEL)
    rtol = _rtol(sqrt.smoothed_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.smoothed_means),
        np.asarray(exact.smoothed_means),
        rtol=rtol,
        atol=rtol,
    )


def test_one_time_record_smooths_to_itself():
    """T = 1: the smoothed row equals the filtered row bitwise."""
    filtered = smcx.sqrt_kalman_filter(MODEL, Y[:1])
    smoothed = smcx.sqrt_rts_smoother(filtered, MODEL)
    np.testing.assert_array_equal(
        np.asarray(smoothed.smoothed_means),
        np.asarray(filtered.filtered_means),
    )
    np.testing.assert_array_equal(
        np.asarray(smoothed.smoothed_factors),
        np.asarray(filtered.filtered_factors),
    )


def test_smoothed_factor_diagonals_are_nonnegative():
    """The sign convention holds through the backward pass."""
    sqrt = smcx.sqrt_rts_smoother(smcx.sqrt_kalman_filter(MODEL, Y), MODEL)
    diagonals = np.diagonal(
        np.asarray(sqrt.smoothed_factors), axis1=-2, axis2=-1
    )
    assert np.all(diagonals >= 0.0)


def _ill_conditioned_fixture(dtype):
    rng = np.random.default_rng(3)
    d = 4
    u, _ = np.linalg.qr(rng.normal(size=(d, d)))
    v, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = (u * np.array([1.0, 0.5, 1e-2, 1e-5])) @ v
    q = 1e-10 * np.eye(d)
    h = np.eye(d)[:2]
    r = 0.1 * np.eye(2)
    xs = [rng.normal(size=d)]
    ys = []
    for t in range(120):
        if t > 0:
            xs.append(a @ xs[-1])
        ys.append(h @ xs[-1] + rng.multivariate_normal(np.zeros(2), r))

    def to(value):
        return jnp.asarray(np.asarray(value), dtype=dtype)

    return (
        to(np.zeros(d)),
        to(np.eye(d)),
        to(a),
        to(q),
        to(h),
        to(r),
        to(np.asarray(ys)),
    )


def test_rejected_regime_smooths_where_the_covariance_form_cannot():
    """The f32 regime the covariance smoother rejects runs here."""
    args32 = _ill_conditioned_fixture(jnp.float32)
    a, q = args32[2], args32[3]
    covariance_form = smcx.kalman_filter(*args32)
    with pytest.raises(ValueError):
        smcx.rts_smoother(covariance_form, a)

    sqrt = smcx.sqrt_rts_smoother(smcx.sqrt_kalman_filter(*args32), a, q)
    assert np.all(np.isfinite(np.asarray(sqrt.smoothed_means)))
    gram = _gram(sqrt.smoothed_factors)
    min_eig = min(np.linalg.eigvalsh(step).min() for step in gram)
    assert min_eig > -1e-8

    if jax.config.read("jax_enable_x64"):
        args64 = _ill_conditioned_fixture(jnp.float64)
        reference = smcx.rts_smoother(smcx.kalman_filter(*args64), args64[2])
        np.testing.assert_allclose(
            np.asarray(sqrt.smoothed_means[-1]),
            np.asarray(reference.smoothed_means[-1]),
            rtol=1e-3,
            atol=1e-3,
        )


def test_non_triangular_factors_are_rejected():
    """A caller-built record with dense factors gets a named error."""
    filtered = smcx.sqrt_kalman_filter(MODEL, Y)
    dense = filtered._replace(
        filtered_factors=filtered.filtered_factors
        + 0.1 * jnp.ones_like(filtered.filtered_factors)
    )
    with pytest.raises(ValueError, match="lower-triangular"):
        smcx.sqrt_rts_smoother(dense, MODEL)


def test_loose_arrays_require_the_transition_covariance():
    """Omitting Q without a record raises the documented error."""
    filtered = smcx.sqrt_kalman_filter(MODEL, Y)
    with pytest.raises(ValueError, match="transition_covariance"):
        smcx.sqrt_rts_smoother(filtered, A)
