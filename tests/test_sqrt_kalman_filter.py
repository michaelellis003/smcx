# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""sqrt_kalman_filter gates: agreement, regimes, boundaries (ADR-0037)."""

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
B_TRANS = jnp.asarray([0.1, -0.2])
B_OBS = jnp.asarray([0.05, 0.15])
G_TRANS = jnp.asarray([[0.4], [0.7]])
G_OBS = jnp.asarray([[0.2], [-0.3]])
U = jnp.asarray([[0.5], [-1.0], [0.25], [0.75]])

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


def _gram(factors):
    return np.einsum("tij,tkj->tik", np.asarray(factors), np.asarray(factors))


def test_well_conditioned_agrees_with_kalman_filter():
    """Both forms agree at roundoff on a benign fixture."""
    sqrt = smcx.sqrt_kalman_filter(MODEL, Y)
    exact = smcx.kalman_filter(MODEL, Y)
    rtol = _rtol(sqrt.filtered_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(sqrt.marginal_loglik),
        np.asarray(exact.marginal_loglik),
        rtol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(sqrt.log_evidence_increments),
        np.asarray(exact.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        _gram(sqrt.filtered_factors),
        np.asarray(exact.filtered_covariances),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        _gram(sqrt.predicted_factors),
        np.asarray(exact.predicted_covariances),
        rtol=rtol,
        atol=rtol,
    )


def test_factor_diagonals_are_nonnegative():
    """The sign convention pins every triangular diagonal at >= 0."""
    sqrt = smcx.sqrt_kalman_filter(MODEL, Y)
    for factors in (sqrt.filtered_factors, sqrt.predicted_factors):
        diagonals = np.diagonal(np.asarray(factors), axis1=-2, axis2=-1)
        assert np.all(diagonals >= 0.0)


def test_record_path_matches_array_path_bitwise():
    """Record and loose-array runs are identical."""
    from_record = smcx.sqrt_kalman_filter(MODEL, Y)
    from_arrays = smcx.sqrt_kalman_filter(MU0, P0, A, Q, H, R, Y)
    for left, right in zip(
        jax.tree.leaves(from_record), jax.tree.leaves(from_arrays), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_missing_row_is_the_identity_update():
    """An all-NaN row stores the prediction and a zero increment."""
    emissions = Y.at[2].set(jnp.nan)
    sqrt = smcx.sqrt_kalman_filter(MODEL, emissions)
    np.testing.assert_array_equal(
        np.asarray(sqrt.filtered_factors[2]),
        np.asarray(sqrt.predicted_factors[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(sqrt.filtered_means[2]),
        np.asarray(sqrt.predicted_means[2]),
    )
    np.testing.assert_array_equal(
        np.asarray(sqrt.log_evidence_increments[2]), 0.0
    )
    exact = smcx.kalman_filter(MODEL, emissions)
    rtol = _rtol(sqrt.filtered_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.marginal_loglik),
        np.asarray(exact.marginal_loglik),
        rtol=rtol,
    )


def test_partial_nan_rows_share_the_uniform_rejection():
    """Partially observed rows get the family-wide row error."""
    with pytest.raises(ValueError, match="fully observed finite"):
        smcx.sqrt_kalman_filter(MODEL, Y.at[1, 0].set(jnp.nan))


def test_biases_and_inputs_agree_with_kalman_filter():
    """The offset and input folding matches the covariance form."""
    kwargs = dict(
        transition_bias=B_TRANS,
        observation_bias=B_OBS,
        transition_input_matrix=G_TRANS,
        observation_input_matrix=G_OBS,
        inputs=U,
    )
    sqrt = smcx.sqrt_kalman_filter(MU0, P0, A, Q, H, R, Y, **kwargs)
    exact = smcx.kalman_filter(MU0, P0, A, Q, H, R, Y, **kwargs)
    rtol = _rtol(sqrt.filtered_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=rtol,
        atol=rtol,
    )
    np.testing.assert_allclose(
        np.asarray(sqrt.marginal_loglik),
        np.asarray(exact.marginal_loglik),
        rtol=rtol,
    )


def test_timed_operators_agree_with_kalman_filter():
    """Per-step operator histories ride through both forms alike."""
    a_t = jnp.stack([A, A * 0.95, A * 1.05])
    r_t = jnp.stack([R, R * 1.5, R * 0.8, R])
    sqrt = smcx.sqrt_kalman_filter(MU0, P0, a_t, Q, H, r_t, Y)
    exact = smcx.kalman_filter(MU0, P0, a_t, Q, H, r_t, Y)
    rtol = _rtol(sqrt.filtered_means)
    np.testing.assert_allclose(
        np.asarray(sqrt.filtered_means),
        np.asarray(exact.filtered_means),
        rtol=rtol,
        atol=rtol,
    )


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


def test_rejected_regime_runs_and_tracks_the_f64_reference():
    """The regime that breaks the f32 covariance form works here."""
    args32 = _ill_conditioned_fixture(jnp.float32)
    sqrt = smcx.sqrt_kalman_filter(*args32)
    assert np.all(np.isfinite(np.asarray(sqrt.filtered_means)))
    gram = _gram(sqrt.filtered_factors)
    min_eig_sqrt = min(np.linalg.eigvalsh(step).min() for step in gram)
    assert min_eig_sqrt > -1e-8

    covariance_form = smcx.kalman_filter(*args32)
    min_eig_cov = min(
        np.linalg.eigvalsh(step).min()
        for step in np.asarray(covariance_form.filtered_covariances)
    )
    assert min_eig_cov < min_eig_sqrt

    if jax.config.read("jax_enable_x64"):
        args64 = _ill_conditioned_fixture(jnp.float64)
        reference = smcx.kalman_filter(*args64)
        np.testing.assert_allclose(
            np.asarray(sqrt.filtered_means[-1]),
            np.asarray(reference.filtered_means[-1]),
            rtol=1e-4,
            atol=1e-4,
        )


def test_record_with_loose_model_array_is_rejected():
    """The shared resolver names this function in its error."""
    with pytest.raises(ValueError, match="sqrt_kalman_filter"):
        smcx.sqrt_kalman_filter(MODEL, Y, transition_matrix=A)


def test_observation_covariance_domain_matches_kalman_filter():
    """A non-PD observation covariance gets the shared boundary error."""
    singular = jnp.asarray([[0.5, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="observation_covariance"):
        smcx.sqrt_kalman_filter(MU0, P0, A, Q, H, singular, Y)
