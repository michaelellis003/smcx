# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic evolution oracles for DGLM retrospective moments."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

import smcx


def _working_dtype() -> jnp.dtype:
    """Return the configured float dtype on CPU and Metal."""
    if jax.config.read("jax_enable_x64"):
        return jnp.dtype(jnp.float64)
    return jnp.dtype(jnp.float32)


def _dglm_record(
    filtered_means: jax.Array,
    filtered_covariances: jax.Array,
) -> smcx.DGLMFilterPosterior:
    """Build a valid public record around moments targeted by an oracle."""
    dtype = filtered_means.dtype
    num_timesteps = filtered_means.shape[0]
    return smcx.DGLMFilterPosterior(
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        conjugate_alphas=jnp.arange(num_timesteps, dtype=dtype) - 1.0,
        conjugate_betas=jnp.arange(1, num_timesteps + 1, dtype=dtype),
        marginal_loglik=jnp.asarray(-1.0, dtype=dtype),
        log_evidence_increments=-jnp.ones(num_timesteps, dtype=dtype),
    )


def _assert_roundoff_close(actual: jax.Array, expected: object) -> None:
    """Compare one deterministic result within 32 working-dtype eps."""
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected, dtype=actual_array.dtype)
    scale = max(1.0, float(np.max(np.abs(expected_array))))
    tolerance = 32 * np.finfo(actual_array.dtype).eps * scale
    np.testing.assert_allclose(
        actual_array,
        expected_array,
        rtol=0.0,
        atol=tolerance,
    )


def test_dglm_smoother_matches_timed_evolution_fraction_oracle() -> None:
    """Distinct W entries have exact scalar alignment in the backward pass."""
    dtype = _working_dtype()
    filtered = _dglm_record(
        jnp.asarray([[0.5], [-0.25], [0.75]], dtype=dtype),
        jnp.asarray(
            [[[0.5]], [[float(Fraction(1, 3))]], [[0.25]]],
            dtype=dtype,
        ),
    )
    transition = jnp.ones((1, 1), dtype=dtype)
    timed_evolution = jnp.asarray(
        [[[0.25]], [[float(Fraction(2, 3))]]],
        dtype=dtype,
    )

    actual = smcx.dglm_smoother(
        filtered,
        transition,
        transition_covariance=timed_evolution,
    )
    # Direct Fraction substitution gives gains 2/3 and 1/3. This classic
    # recursion shares neither the implementation's solve nor its Joseph form.
    expected_means = jnp.asarray(
        [[float(Fraction(2, 9))], [float(Fraction(1, 12))], [0.75]],
        dtype=dtype,
    )
    expected_covariances = jnp.asarray(
        [
            [[float(Fraction(5, 18))]],
            [[0.25]],
            [[0.25]],
        ],
        dtype=dtype,
    )
    _assert_roundoff_close(actual.smoothed_means, expected_means)
    _assert_roundoff_close(
        actual.smoothed_covariances,
        expected_covariances,
    )

    static_evolution = jnp.asarray([[0.25]], dtype=dtype)
    static = smcx.dglm_smoother(
        filtered,
        transition,
        transition_covariance=static_evolution,
    )
    repeated = smcx.dglm_smoother(
        filtered,
        transition,
        transition_covariance=jnp.broadcast_to(
            static_evolution,
            (2, 1, 1),
        ),
    )
    for static_field, repeated_field in zip(static, repeated, strict=True):
        np.testing.assert_array_equal(static_field, repeated_field)


def test_dglm_smoother_discount_matches_closed_form_backward_identity() -> None:
    """A nonsymmetric model matches B = delta G^-1 and its implied W."""
    dtype = _working_dtype()
    transition = jnp.asarray(
        [[1.0, 0.5], [-0.5, 0.75]],
        dtype=dtype,
    )
    observation = jnp.asarray([1.0, -0.5], dtype=dtype)
    initial_mean = jnp.asarray([0.25, -0.5], dtype=dtype)
    initial_covariance = jnp.asarray(
        [[1.0, 0.25], [0.25, 0.75]],
        dtype=dtype,
    )
    discount = jnp.asarray(0.75, dtype=dtype)
    filtered = smcx.dglm_filter(
        initial_mean,
        initial_covariance,
        transition,
        observation,
        jnp.asarray([1, 0, 3, 2]),
        family=smcx.poisson(),
        discount=discount,
    )
    actual = smcx.dglm_smoother(
        filtered,
        transition,
        discount=discount,
    )

    means = np.asarray(filtered.filtered_means)
    covariances = np.asarray(filtered.filtered_covariances)
    covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
    expected_means = means.copy()
    expected_covariances = covariances.copy()
    transition_array = np.asarray(transition)
    # The determinant of G is one, so this exact dyadic matrix is 0.75 G^-1.
    gain = np.asarray(
        [[0.5625, -0.375], [0.375, 0.75]],
        dtype=means.dtype,
    )
    for time in range(means.shape[0] - 2, -1, -1):
        expected_means[time] = means[time] + gain @ (
            expected_means[time + 1] - transition_array @ means[time]
        )
        expected_covariances[time] = (
            0.25 * covariances[time]
            + gain @ expected_covariances[time + 1] @ gain.T
        )
        expected_covariances[time] = 0.5 * (
            expected_covariances[time] + expected_covariances[time].T
        )

    _assert_roundoff_close(actual.smoothed_means, expected_means)
    _assert_roundoff_close(
        actual.smoothed_covariances,
        expected_covariances,
    )

    def smooth(value: jax.Array) -> smcx.DGLMSmootherPosterior:
        return smcx.dglm_smoother(
            filtered,
            transition,
            discount=value,
        )

    compiled = jax.jit(smooth)(discount)
    for eager_field, compiled_field in zip(actual, compiled, strict=True):
        _assert_roundoff_close(compiled_field, eager_field)

    canonical_covariances = 0.5 * (
        filtered.filtered_covariances
        + filtered.filtered_covariances.swapaxes(-1, -2)
    )
    propagated = (transition @ canonical_covariances[:-1]) @ transition.T
    implied_evolution = propagated * ((1.0 - discount) / discount)
    implied_evolution = 0.5 * (
        implied_evolution + implied_evolution.swapaxes(-1, -2)
    )
    explicit = smcx.dglm_smoother(
        filtered,
        transition,
        transition_covariance=implied_evolution,
    )
    for discount_field, explicit_field in zip(actual, explicit, strict=True):
        _assert_roundoff_close(explicit_field, discount_field)
