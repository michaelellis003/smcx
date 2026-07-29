# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for representable Kalman covariance boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
import smcx.kalman as kalman_module


def _scalar_linear_model(dtype=None) -> dict[str, jax.Array]:
    """Return a deterministic scalar model in the test-platform dtype."""
    dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
    zero = jnp.zeros((1, 1), dtype=dtype)
    return {
        "initial_mean": jnp.zeros(1, dtype=dtype),
        "initial_covariance": zero,
        "transition_matrix": jnp.ones((1, 1), dtype=dtype),
        "transition_covariance": zero,
        "observation_matrix": zero,
        "observation_covariance": jnp.ones((1, 1), dtype=dtype),
        "emissions": zero,
    }


def _identity(state):
    """Return a state unchanged."""
    return state


def _zero_mean(state):
    """Return a scalar zero mean."""
    return jnp.zeros(1, dtype=state.dtype)


def _observation_model(covariance, *, timed=False):
    """Return a zero-state model exposing one observation factor path."""
    observation_dim = covariance.shape[-1]
    dtype = covariance.dtype
    model = _scalar_linear_model(dtype)
    model["observation_matrix"] = jnp.zeros((observation_dim, 1), dtype=dtype)
    model["observation_covariance"] = (
        jnp.stack((jnp.eye(observation_dim, dtype=dtype), covariance))
        if timed
        else covariance
    )
    model["emissions"] = jnp.zeros((1 + timed, observation_dim), dtype=dtype)
    return model


def _backend_factorable(covariance):
    """Report the active backend's represented Cholesky result."""
    factor = np.asarray(
        jnp.linalg.cholesky(
            covariance,
            symmetrize_input=False,
        )
    )
    diagonal = np.diagonal(factor, axis1=-2, axis2=-1)
    return bool(np.all(np.isfinite(factor)) and np.all(diagonal > 0.0))


def _hilbert_correlation(dtype, dimension=None):
    """Return a represented SPD matrix near the backend factor boundary."""
    dimension = dimension or (13 if dtype == jnp.dtype(jnp.float64) else 8)
    index = np.arange(dimension, dtype=np.float64)
    hilbert = 1.0 / (index[:, None] + index[None, :] + 1.0)
    scale = np.sqrt(np.diag(hilbert))
    return jnp.asarray(hilbert / scale[:, None] / scale[None, :], dtype=dtype)


def test_concrete_covariance_rejects_nonzero_subnormal():
    """A representable value that arithmetic flushes cannot enter the loop."""
    dtype = jnp.asarray(0.0).dtype
    host_dtype = np.dtype(dtype)
    subnormal = np.nextafter(
        np.asarray(0.0, dtype=host_dtype),
        np.asarray(1.0, dtype=host_dtype),
    )
    model = _scalar_linear_model()
    model["initial_covariance"] = jnp.asarray([[subnormal]], dtype=dtype)

    with pytest.raises(
        ValueError,
        match="initial_covariance must not contain nonzero subnormal values",
    ):
        smcx.kalman_filter(**model)


@pytest.mark.parametrize("kind", ["hilbert", "subnormal_pivot"])
@pytest.mark.parametrize("timed", [False, True])
def test_factorized_covariance_matches_active_backend(kind, timed):
    """Concrete validation agrees with the represented factor operation."""
    dtype = jnp.asarray(0.0).dtype
    if kind == "hilbert":
        covariance = _hilbert_correlation(dtype)
    else:
        tiny = jnp.finfo(dtype).tiny
        covariance = jnp.asarray(
            [[2.0 * tiny, tiny], [tiny, tiny]],
            dtype=dtype,
        )
    model = _observation_model(covariance, timed=timed)

    if _backend_factorable(model["observation_covariance"]):
        posterior = smcx.kalman_filter(**model)
        assert all(jnp.all(jnp.isfinite(field)) for field in posterior)
    else:
        with pytest.raises(
            ValueError,
            match="observation_covariance must be positive definite",
        ):
            smcx.kalman_filter(**model)


def test_factorized_covariance_accepts_backend_factorable_roundoff():
    """A roundoff-negative host eigenvalue cannot mask a finite factor."""
    dtype = jnp.asarray(0.0).dtype
    host_dtype = np.dtype(dtype)
    one = host_dtype.type(1.0)
    rho = np.nextafter(one, host_dtype.type(0.0))
    covariance = np.full((4, 4), rho, dtype=host_dtype)
    np.fill_diagonal(covariance, one)

    posterior = smcx.kalman_filter(
        **_observation_model(jnp.asarray(covariance))
    )

    assert all(jnp.all(jnp.isfinite(field)) for field in posterior)


@pytest.mark.parametrize("timed", [False, True])
def test_factorized_covariance_rejects_negative_backend_diagonal(timed):
    """A finite but nonpositive backend factor is not positive definite."""
    covariance = _hilbert_correlation(jnp.float32, dimension=10)
    factor = np.asarray(jnp.linalg.cholesky(covariance, symmetrize_input=False))
    diagonal = np.diagonal(factor)
    assert not _backend_factorable(covariance)
    if jax.default_backend() == "mps":
        assert np.all(np.isfinite(factor)) and np.any(diagonal <= 0.0)
    with pytest.raises(ValueError, match="must be positive definite"):
        smcx.kalman_filter(**_observation_model(covariance, timed=timed))


def test_factor_probe_normalizes_debug_nan_failure():
    """Expected factor failure remains the public ValueError under debugging."""
    covariance = _hilbert_correlation(jnp.asarray(0.0).dtype)
    if _backend_factorable(covariance):
        return

    with (
        jax.debug_nans(True),
        pytest.raises(
            ValueError,
            match="observation_covariance must be positive definite",
        ),
    ):
        smcx.kalman_filter(**_observation_model(covariance))


def test_scalar_normal_minimum_covariance_remains_supported():
    """The active backend retains the smallest normal scalar factor."""
    dtype = jnp.asarray(0.0).dtype
    model = _scalar_linear_model()
    model["observation_covariance"] = jnp.asarray(
        [[jnp.finfo(dtype).tiny]],
        dtype=dtype,
    )

    posterior = smcx.kalman_filter(**model)

    for field in posterior:
        assert jnp.all(jnp.isfinite(field))


def test_maximum_observation_covariance_evidence_is_finite():
    """Linear evidence supports a factorable maximum scalar."""
    dtype = jnp.asarray(0.0).dtype
    maximum = jnp.asarray([[jnp.finfo(dtype).max]], dtype=dtype)
    model = _scalar_linear_model()
    model["observation_covariance"] = maximum

    posterior = smcx.kalman_filter(**model)

    assert jnp.isfinite(posterior.marginal_loglik)
    assert jnp.all(jnp.isfinite(posterior.log_evidence_increments))


def test_unscented_maximum_covariance_path_is_finite():
    """The UKF keeps a factorable maximum scalar through paired moments."""
    dtype = jnp.asarray(0.0).dtype
    maximum = jnp.asarray([[jnp.finfo(dtype).max]], dtype=dtype)
    one = jnp.ones((1, 1), dtype=dtype)
    zero = jnp.zeros((1, 1), dtype=dtype)

    def run(initial_covariance):
        return smcx.unscented_kalman_filter(
            jnp.zeros(1, dtype=dtype),
            initial_covariance,
            _identity,
            one,
            _zero_mean,
            one,
            zero,
        )

    for posterior in (run(maximum), jax.jit(run)(maximum)):
        for field in posterior:
            assert jnp.all(jnp.isfinite(field))


def test_rts_maximum_factor_path_is_finite():
    """The smoother factors maximum finite predicted covariance."""
    dtype = jnp.asarray(0.0).dtype
    maximum = jnp.asarray([[jnp.finfo(dtype).max]], dtype=dtype)
    covariances = jnp.broadcast_to(maximum, (2, 1, 1))
    means = jnp.zeros((2, 1), dtype=dtype)
    posterior = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0, dtype=dtype),
        means,
        covariances,
        means,
        covariances,
        jnp.zeros(2, dtype=dtype),
    )

    def run(predicted_covariances):
        return smcx.rts_smoother(
            posterior._replace(
                predicted_covariances=predicted_covariances,
            ),
            jnp.ones((1, 1), dtype=dtype),
        )

    smoothed = run(covariances)

    assert jnp.all(jnp.isfinite(smoothed.smoothed_means))
    # At the exact dtype-max corner the true smoothed covariance is the
    # maximum itself, and the Joseph-form product (#286) may round the
    # last ulp upward to inf; NaN is the only forbidden outcome.
    assert not jnp.any(jnp.isnan(smoothed.smoothed_covariances))


def test_ordinary_unscented_moments_retain_legacy_bits():
    """Overflow guards leave ordinary paired moments bitwise unchanged."""
    dtype = jnp.asarray(0.0).dtype
    values = jnp.asarray([[0.25], [0.75], [-0.35]], dtype=dtype)
    rule = kalman_module._scaled_unscented_rule(1, dtype, 1.0, 2.0, 0.0)
    _, actual = kalman_module._unscented_moments(values, rule)
    deltas = values[1:] - values[0]
    delta_sum = deltas.sum(axis=0)
    expected = 0.5 * jnp.einsum(
        "ij,ik->jk",
        deltas,
        deltas,
    ) + 0.25 * jnp.outer(delta_sum, delta_sum)
    expected = 0.5 * (expected + expected.T)

    np.testing.assert_array_equal(actual, expected)


def _huge_residual_linear_model(dtype):
    """Return identity LGSSM arrays whose t=1 residual is near 2e19."""
    eye = jnp.eye(1, dtype=dtype)
    return {
        "initial_mean": jnp.zeros(1, dtype=dtype),
        "initial_covariance": eye,
        "transition_matrix": eye,
        "transition_covariance": eye,
        "observation_matrix": eye,
        "observation_covariance": eye,
        "emissions": jnp.asarray([[0.0], [3e19], [0.0]], dtype=dtype),
    }


def test_float32_kalman_evidence_survives_representable_residual():
    """A representable log density never becomes -inf or NaN (#281)."""

    def run(dtype):
        arrays = _huge_residual_linear_model(dtype)
        return smcx.kalman_filter(
            arrays["initial_mean"],
            arrays["initial_covariance"],
            arrays["transition_matrix"],
            arrays["transition_covariance"],
            arrays["observation_matrix"],
            arrays["observation_covariance"],
            arrays["emissions"],
        )

    reference = run(jnp.float64)
    posterior = run(jnp.float32)
    np.testing.assert_allclose(
        np.asarray(posterior.log_evidence_increments),
        np.asarray(reference.log_evidence_increments),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(posterior.marginal_loglik),
        float(reference.marginal_loglik),
        rtol=1e-5,
    )


def test_float32_kalman_true_underflow_is_minus_inf_not_nan():
    """Below the float32 range the marginal is -inf, never NaN (#281)."""
    arrays = _huge_residual_linear_model(jnp.float32)
    posterior = smcx.kalman_filter(
        arrays["initial_mean"],
        arrays["initial_covariance"],
        arrays["transition_matrix"],
        arrays["transition_covariance"],
        arrays["observation_matrix"],
        arrays["observation_covariance"],
        jnp.asarray([[0.0], [1e20], [0.0]], dtype=jnp.float32),
    )
    assert np.isneginf(float(posterior.marginal_loglik))


@pytest.mark.parametrize("filter_name", ["extended", "unscented"])
def test_float32_gaussian_evidence_survives_representable_residual(
    filter_name,
):
    """EKF and UKF share the overflow-safe evidence kernel (#281)."""

    def run(dtype):
        arrays = _huge_residual_linear_model(dtype)
        # 3e19 pushes the unscented marginal just past the float32
        # range, where -inf is exact; 2.2e19 keeps it representable.
        arrays["emissions"] = jnp.asarray([[0.0], [2.2e19], [0.0]], dtype=dtype)
        args = (
            arrays["initial_mean"],
            arrays["initial_covariance"],
            _identity,
            arrays["transition_covariance"],
            _identity,
            arrays["observation_covariance"],
            arrays["emissions"],
        )
        if filter_name == "extended":
            return smcx.extended_kalman_filter(
                args[0],
                args[1],
                args[2],
                lambda state: jnp.eye(1, dtype=state.dtype),
                args[3],
                args[4],
                lambda state: jnp.eye(1, dtype=state.dtype),
                args[5],
                args[6],
            )
        return smcx.unscented_kalman_filter(*args)

    reference = run(jnp.float64)
    posterior = run(jnp.float32)
    np.testing.assert_allclose(
        float(posterior.marginal_loglik),
        float(reference.marginal_loglik),
        rtol=1e-5,
    )


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="the float32-vs-float64 comparison needs a float64 backend",
)
def test_float32_rts_smoothed_variances_stay_nonnegative():
    """The Joseph-form backward update cannot cancel negative (#286)."""

    def run(dtype):
        emissions = jnp.full((4, 1), 0.1, dtype)
        observation_covariances = jnp.asarray(
            [1.0, 1.0, 1.0, 1e-15], dtype
        ).reshape(4, 1, 1)
        posterior = smcx.kalman_filter(
            jnp.zeros(1, dtype),
            jnp.eye(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            jnp.zeros((1, 1), dtype),
            jnp.eye(1, dtype=dtype),
            observation_covariances,
            emissions,
        )
        return smcx.rts_smoother(posterior, jnp.eye(1, dtype=dtype))

    single = run(jnp.float32)
    double = run(jnp.float64)
    variances = np.asarray(single.smoothed_covariances).ravel()
    reference = np.asarray(double.smoothed_covariances).ravel()

    # The subtractive form returned about -1e-7 here: negative, and
    # seven to eight orders above the true magnitude. The Joseph form
    # carries only the squared gain-rounding residue, so the float32
    # result must be nonnegative and within two orders of float64.
    assert np.all(variances >= 0.0)
    assert np.all(variances <= 100.0 * reference)
    # With zero process noise the state is constant, so the float64
    # smoothed variances all equal the final filtered variance.
    np.testing.assert_allclose(
        reference,
        float(double.filtered_covariances[-1, 0, 0]),
        rtol=1e-9,
    )
