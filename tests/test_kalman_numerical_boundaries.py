# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for representable Kalman covariance boundaries."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import solve_triangular

import smcx


def _scalar_linear_model() -> dict[str, jax.Array]:
    """Return a deterministic scalar model in the test-platform dtype."""
    dtype = jnp.asarray(0.0).dtype
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


def _identity_jacobian(state):
    """Return the Jacobian of the identity map."""
    return jnp.eye(state.shape[0], dtype=state.dtype)


def _zero_mean(state):
    """Return a scalar zero mean."""
    return jnp.zeros(1, dtype=state.dtype)


def _zero_jacobian(state):
    """Return the Jacobian of a constant scalar map."""
    return jnp.zeros((1, state.shape[0]), dtype=state.dtype)


def test_concrete_covariance_rejects_nonzero_subnormal():
    """A representable value that arithmetic flushes cannot enter the loop."""
    dtype = jnp.asarray(0.0).dtype
    host_dtype = np.dtype(dtype)
    subnormal = np.nextafter(
        np.asarray(0.0, dtype=host_dtype),
        np.asarray(1.0, dtype=host_dtype),
    )
    model = _scalar_linear_model()
    model["initial_covariance"] = jnp.asarray(
        [[subnormal]],
        dtype=dtype,
    )

    with pytest.raises(
        ValueError,
        match="initial_covariance must not contain nonzero subnormal values",
    ):
        smcx.kalman_filter(**model)


def test_factorized_covariance_rejects_subnormal_pivot_scale():
    """An all-normal SPD input must retain a normal Cholesky scale."""
    dtype = jnp.asarray(0.0).dtype
    tiny = jnp.finfo(dtype).tiny
    model = {
        "initial_mean": jnp.zeros(2, dtype=dtype),
        "initial_covariance": jnp.eye(2, dtype=dtype),
        "transition_matrix": jnp.eye(2, dtype=dtype),
        "transition_covariance": jnp.eye(2, dtype=dtype),
        "observation_matrix": jnp.zeros((2, 2), dtype=dtype),
        "observation_covariance": jnp.asarray(
            [[2.0 * tiny, tiny], [tiny, tiny]],
            dtype=dtype,
        ),
        "emissions": jnp.zeros((1, 2), dtype=dtype),
    }

    with pytest.raises(
        ValueError,
        match="observation_covariance must be positive definite",
    ):
        smcx.kalman_filter(**model)


def test_scalar_normal_minimum_covariance_remains_supported():
    """The inclusive factorability floor retains the smallest normal scalar."""
    dtype = jnp.asarray(0.0).dtype
    model = _scalar_linear_model()
    model["observation_covariance"] = jnp.asarray(
        [[jnp.finfo(dtype).tiny]],
        dtype=dtype,
    )

    posterior = smcx.kalman_filter(**model)

    for field in posterior:
        assert jnp.all(jnp.isfinite(field))


@pytest.mark.parametrize("method", ["linear", "extended"])
def test_maximum_covariance_evidence_is_finite_under_transformations(method):
    """Linear and extended evidence support a factorable maximum scalar."""
    dtype = jnp.asarray(0.0).dtype
    maximum = jnp.asarray([[jnp.finfo(dtype).max]], dtype=dtype)
    zero = jnp.zeros((1, 1), dtype=dtype)

    def run(observation_covariance):
        if method == "linear":
            model = _scalar_linear_model()
            model["observation_covariance"] = observation_covariance
            return smcx.kalman_filter(**model)
        return smcx.extended_kalman_filter(
            jnp.zeros(1, dtype=dtype),
            zero,
            _identity,
            _identity_jacobian,
            zero,
            _zero_mean,
            _zero_jacobian,
            observation_covariance,
            zero,
        )

    posterior = run(maximum)

    assert jnp.isfinite(posterior.marginal_loglik)
    assert jnp.all(jnp.isfinite(posterior.log_evidence_increments))


def test_unscented_maximum_covariance_path_is_finite():
    """The UKF keeps a factorable maximum scalar through paired moments."""
    dtype = jnp.asarray(0.0).dtype
    maximum = jnp.asarray([[jnp.finfo(dtype).max]], dtype=dtype)
    tiny = jnp.asarray([[jnp.finfo(dtype).tiny]], dtype=dtype)
    zero = jnp.zeros((1, 1), dtype=dtype)

    def run(initial_covariance):
        return smcx.unscented_kalman_filter(
            jnp.zeros(1, dtype=dtype),
            initial_covariance,
            _identity,
            tiny,
            _zero_mean,
            tiny,
            zero,
        )

    posterior = run(maximum)

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
    assert jnp.all(jnp.isfinite(smoothed.smoothed_covariances))


def test_ordinary_unscented_outputs_retain_legacy_bits():
    """Overflow guards leave every ordinary public result bitwise unchanged."""
    dtype = jnp.asarray(0.0).dtype
    mean = jnp.asarray([0.25], dtype=dtype)
    covariance = jnp.asarray([[0.7]], dtype=dtype)
    transition_covariance = jnp.asarray([[0.3]], dtype=dtype)
    observation_covariance = jnp.asarray([[0.4]], dtype=dtype)
    emissions = jnp.asarray([[-0.2]], dtype=dtype)

    posterior = smcx.unscented_kalman_filter(
        mean,
        covariance,
        _identity,
        transition_covariance,
        _zero_mean,
        observation_covariance,
        emissions,
    )
    symmetric = 0.5 * (covariance + covariance.T)
    factor = jnp.linalg.cholesky(symmetric)
    points = jnp.concatenate((
        mean[None],
        mean[None] + factor.T,
        mean[None] - factor.T,
    ))
    positive = points[1:2] - points[0]
    negative = points[2:] - points[0]
    delta_sum = (positive + negative).sum(axis=0)
    residual_covariance = 0.5 * (
        jnp.einsum("ij,ik->jk", positive, positive)
        + jnp.einsum("ij,ik->jk", negative, negative)
    ) + 0.25 * jnp.outer(delta_sum, delta_sum)
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    innovation = 0.5 * (observation_covariance + observation_covariance.T)
    innovation_cholesky = jnp.linalg.cholesky(innovation)
    whitened = solve_triangular(
        innovation_cholesky,
        emissions[0],
        lower=True,
    )
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=dtype)
    increment = -0.5 * (
        log_two_pi
        + 2.0 * jnp.log(jnp.diag(innovation_cholesky)).sum()
        + whitened @ whitened
    )
    expected = (
        increment,
        mean[None],
        covariance[None],
        mean[None],
        residual_covariance[None],
        increment[None],
    )
    for actual_field, expected_field in zip(
        posterior,
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(actual_field, expected_field)
