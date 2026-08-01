# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Conjugate DLM retrospective-analysis tests (W&H 1997, ch. 4)."""

from fractions import Fraction
from inspect import unwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="frozen CPU/x64 arithmetic contract",
)
def test_dlm_smoother_matches_timed_evolution_fraction_oracle():
    """The scale-free backward recursion matches exact rational arithmetic."""
    observations = jnp.asarray([1.0, -0.5, 0.75])
    evolution = jnp.asarray([[[0.25]], [[float(Fraction(2, 3))]]])

    filtered = smcx.dlm_filter(
        jnp.asarray([0.5]),
        jnp.asarray([[3.0]]),
        jnp.eye(1),
        jnp.asarray([1.0]),
        observations,
        scale_free_transition_covariance=evolution,
        prior_shape=5.0,
        prior_scale=2.0,
    )
    posterior = smcx.dlm_smoother(
        filtered,
        jnp.eye(1),
        scale_free_transition_covariance=evolution,
    )

    # Exact substitution in the scalar recurrences gives these Fractions.
    # Sixty-four eps covers their conversion and a few CPU/x64 operations.
    expected_means = [Fraction(95, 208), Fraction(33, 104), Fraction(51, 104)]
    expected_covariances = [Fraction(21, 52), Fraction(5, 13), Fraction(7, 13)]
    eps = np.finfo(np.asarray(posterior.smoothed_means).dtype).eps
    np.testing.assert_allclose(
        np.asarray(posterior.smoothed_means[:, 0]),
        [float(value) for value in expected_means],
        rtol=0.0,
        atol=64 * eps,
    )
    np.testing.assert_allclose(
        np.asarray(posterior.smoothed_scale_free_covariances[:, 0, 0]),
        [float(value) for value in expected_covariances],
        rtol=0.0,
        atol=64 * eps,
    )
    np.testing.assert_allclose(posterior.scale_shapes[-1], float(Fraction(8)))
    np.testing.assert_allclose(
        posterior.scale_estimates[-1], float(Fraction(145, 104)), rtol=1e-13
    )
    assert type(posterior)._fields == (
        *type(filtered)._fields,
        "smoothed_means",
        "smoothed_scale_free_covariances",
    )
    for retained, original in zip(posterior[:6], filtered, strict=True):
        np.testing.assert_array_equal(retained, original)
    np.testing.assert_array_equal(
        posterior.smoothed_means[-1], posterior.filtered_means[-1]
    )
    np.testing.assert_array_equal(
        posterior.smoothed_scale_free_covariances[-1],
        posterior.filtered_scale_free_covariances[-1],
    )


def test_dlm_smoother_canonicalizes_released_f32_roundoff_under_jit():
    """A released producer crosses only the obsolete symmetry boundary."""
    dtype = jnp.float32
    initial_covariance = jnp.asarray(
        [
            [10.720845, -0.54652244, 3.1010354],
            [-0.54652244, 0.9465956, -0.46825573],
            [3.1010354, -0.46825573, 1.4951488],
        ],
        dtype=dtype,
    )
    transition = jnp.asarray(
        [
            [-0.84977686, -0.20510903, -0.20211807],
            [-0.15791872, 0.11738589, 0.00980542],
            [0.5523522, 0.53633, -0.14769909],
        ],
        dtype=dtype,
    )
    evolution = 0.1 * jnp.eye(3, dtype=dtype)
    filtered = smcx.dlm_filter(
        jnp.zeros(3, dtype=dtype),
        initial_covariance,
        transition,
        jnp.asarray([0.63770854, -0.22345555, -1.3328147], dtype=dtype),
        jnp.asarray(
            [0.25107875, 1.1324171, 0.37735105, 1.4852368, 2.473733],
            dtype=dtype,
        ),
        scale_free_transition_covariance=evolution,
    )
    raw = np.asarray(filtered.filtered_scale_free_covariances).copy()
    transpose = raw.swapaxes(-1, -2)
    eps = np.finfo(raw.dtype).eps
    assert not np.array_equal(raw, transpose)

    def run(record, matrix, covariance):
        return smcx.dlm_smoother(
            record, matrix, scale_free_transition_covariance=covariance
        )

    eager = run(filtered, transition, evolution)
    compiled = jax.jit(run)(filtered, transition, evolution)
    np.testing.assert_array_equal(
        eager.filtered_scale_free_covariances, 0.5 * (raw + transpose)
    )
    np.testing.assert_array_equal(raw, filtered.filtered_scale_free_covariances)
    np.testing.assert_array_equal(
        compiled.filtered_scale_free_covariances,
        compiled.filtered_scale_free_covariances.swapaxes(-1, -2),
    )
    np.testing.assert_array_equal(
        eager.smoothed_scale_free_covariances[-1],
        eager.filtered_scale_free_covariances[-1],
    )
    for eager_field, compiled_field in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(
            eager_field, compiled_field, rtol=32 * eps, atol=32 * eps
        )
    with pytest.raises(ValueError, match="scalar"):
        jax.jit(
            lambda value: unwrap(smcx.dlm_smoother)(
                filtered, transition, discount=value
            )
        )(jnp.asarray([0.9, 0.8, 0.7], dtype=dtype))


_F32_EPS = np.finfo(np.float32).eps
_F32_SUBNORMAL = np.nextafter(
    np.float32(0.0), np.float32(1.0), dtype=np.float32
)


def _one_step_dlm_record(covariance: jax.Array) -> smcx.DLMFilterPosterior:
    """Build the smallest structurally valid smoother input record."""
    dtype = covariance.dtype
    return smcx.DLMFilterPosterior(
        filtered_means=jnp.asarray([[0.25, -0.5]], dtype=dtype),
        filtered_scale_free_covariances=covariance[None],
        scale_shapes=jnp.asarray([2.0], dtype=dtype),
        scale_estimates=jnp.asarray([1.5], dtype=dtype),
        marginal_loglik=jnp.asarray(-1.0, dtype=dtype),
        log_evidence_increments=jnp.asarray([-1.0], dtype=dtype),
    )


def _smooth_one_step(record: smcx.DLMFilterPosterior):
    """Exercise installed-package validation, not the test-only type hook."""
    dtype = record.filtered_means.dtype
    return unwrap(smcx.dlm_smoother)(
        record,
        jnp.zeros((2, 2), dtype=dtype),
        discount=jnp.asarray(1.0, dtype=dtype),
    )


def _dlm_record(
    filtered_means: jax.Array,
    filtered_covariances: jax.Array,
    *,
    scale: float = 1.0,
) -> smcx.DLMFilterPosterior:
    """Build a valid record around moments targeted by a numerical gate."""
    dtype = filtered_means.dtype
    num_timesteps = filtered_means.shape[0]
    return smcx.DLMFilterPosterior(
        filtered_means,
        filtered_covariances,
        jnp.arange(2, num_timesteps + 2, dtype=dtype),
        jnp.full((num_timesteps,), scale, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.zeros(num_timesteps, dtype=dtype),
    )


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        pytest.param(
            [[np.nan, 0.0], [0.0, 1.0]],
            "must contain only finite values",
            id="nonfinite",
        ),
        pytest.param(
            [[1.0, _F32_SUBNORMAL], [-_F32_SUBNORMAL, 1.0]],
            "contains nonzero subnormal values",
            id="subnormal",
        ),
        pytest.param(
            [[-1.0, 0.0], [0.0, 1.0]],
            "must be positive semidefinite",
            id="negative-diagonal",
        ),
        pytest.param(
            [[0.0, _F32_EPS], [-_F32_EPS, 1.0]],
            "has skew at a zero diagonal",
            id="zero-diagonal-skew",
        ),
        pytest.param(
            [[2.0**30, 4.0625], [-4.0625, 2.0**10]],
            "must be symmetric within roundoff",
            id="one-epsilon-above-normalized-limit",
        ),
    ],
)
def test_dlm_smoother_rejects_raw_covariance_defects(
    covariance: list[list[float]], message: str
):
    """Raw defects cannot disappear through symmetric projection."""
    record = _one_step_dlm_record(jnp.asarray(covariance, dtype=jnp.float32))

    with pytest.raises(ValueError, match=message):
        _smooth_one_step(record)


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="the normalization-overflow fixture requires float64",
)
def test_dlm_smoother_rejects_nonfinite_normalized_covariance():
    """Finite entries whose normalization overflows are not admitted."""
    finfo = np.finfo(np.float64)
    covariance = jnp.asarray(
        [[finfo.tiny, finfo.max], [finfo.max, finfo.tiny]], dtype=jnp.float64
    )

    with pytest.raises(ValueError, match="must be positive semidefinite"):
        _smooth_one_step(_one_step_dlm_record(covariance))


@pytest.mark.parametrize(
    ("covariance", "canonical"),
    [
        pytest.param(
            [[2.0**30, 4.0], [-4.0, 2.0**10]],
            [[2.0**30, 0.0], [0.0, 2.0**10]],
            id="normalized-skew-limit",
        ),
        pytest.param(
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            id="semidefinite-terminal",
        ),
    ],
)
def test_dlm_smoother_accepts_one_step_covariance_boundaries_under_jit(
    covariance: list[list[float]], canonical: list[list[float]]
):
    """The inclusive skew edge and factor-free singular terminal are valid."""
    value = jnp.asarray(covariance, dtype=jnp.float32)
    record = _one_step_dlm_record(value)

    for posterior in (
        _smooth_one_step(record),
        jax.jit(_smooth_one_step)(record),
    ):
        assert all(
            np.all(np.isfinite(np.asarray(field))) for field in posterior
        )
        np.testing.assert_array_equal(
            posterior.filtered_scale_free_covariances[0], canonical
        )
        np.testing.assert_array_equal(
            posterior.smoothed_means, posterior.filtered_means
        )
        np.testing.assert_array_equal(
            posterior.smoothed_scale_free_covariances,
            posterior.filtered_scale_free_covariances,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("filtered_means", "filtered_means must have shape"),
        (
            "filtered_scale_free_covariances",
            "filtered_scale_free_covariances shape",
        ),
        ("scale_shapes", "scale_shapes must contain finite positive values"),
        (
            "scale_estimates",
            "scale_estimates must contain finite positive values",
        ),
        ("marginal_loglik", "marginal_loglik must be scalar"),
    ],
)
def test_dlm_smoother_rejects_malformed_filter_record(field: str, message: str):
    """Retained record shapes and concrete scale domains fail eagerly."""
    dtype = jnp.float32
    invalid = {
        "filtered_means": jnp.empty((0, 2), dtype=dtype),
        "filtered_scale_free_covariances": jnp.empty((1, 2, 1), dtype=dtype),
        "scale_shapes": jnp.asarray([0.0], dtype=dtype),
        "scale_estimates": jnp.asarray([jnp.inf], dtype=dtype),
        "marginal_loglik": jnp.asarray([0.0], dtype=dtype),
    }
    record = _one_step_dlm_record(jnp.eye(2, dtype=dtype))._replace(**{
        field: invalid[field]
    })

    with pytest.raises(ValueError, match=message):
        _smooth_one_step(record)


@pytest.mark.parametrize(
    ("transition", "options", "message"),
    [
        pytest.param(jnp.eye(2), {}, "supply exactly one", id="neither"),
        pytest.param(
            jnp.eye(2),
            {
                "discount": 1.0,
                "scale_free_transition_covariance": jnp.zeros((2, 2)),
            },
            "supply exactly one",
            id="both",
        ),
        pytest.param(
            jnp.ones((1, 1)), {"discount": 1.0}, "shape", id="transition"
        ),
        pytest.param(jnp.eye(2), {"discount": 1.1}, "outside", id="discount"),
        pytest.param(jnp.eye(2), {"discount": 0.0}, "positive", id="zero"),
        pytest.param(jnp.eye(2), {"discount": np.nan}, "positive", id="nan"),
    ],
)
def test_dlm_smoother_rejects_malformed_evolution_specification(
    transition: jax.Array, options: dict[str, object], message: str
):
    """Evolution structure and concrete discount domains fail eagerly."""
    record = _one_step_dlm_record(jnp.eye(2, dtype=jnp.float32))

    with pytest.raises(ValueError, match=message):
        unwrap(smcx.dlm_smoother)(
            record, transition.astype(jnp.float32), **options
        )


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="the concentrated scale trace requires CPU/x64",
)
def test_dlm_smoother_reduces_to_rts_at_concentrated_scale():
    """The finite-n DLM result obeys its exact known-variance reduction."""
    dtype = jnp.float64
    variance = 0.49
    prior_shape = 1e8
    transition = jnp.asarray([[0.9, 0.1], [0.0, 0.8]], dtype=dtype)
    observation = jnp.asarray([1.0, 0.5], dtype=dtype)
    evolution = jnp.asarray([[0.3, 0.05], [0.05, 0.2]], dtype=dtype)
    initial_mean = jnp.asarray([0.2, -0.1], dtype=dtype)
    initial_covariance = jnp.asarray([[1.0, 0.1], [0.1, 0.8]], dtype=dtype)
    emissions = jnp.asarray([0.3, -0.2, 0.5, 0.1], dtype=dtype)

    filtered = smcx.dlm_filter(
        initial_mean,
        initial_covariance,
        transition,
        observation,
        emissions,
        scale_free_transition_covariance=evolution,
        prior_shape=prior_shape,
        prior_scale=variance,
    )
    smoothed = smcx.dlm_smoother(
        filtered,
        transition,
        scale_free_transition_covariance=evolution,
    )
    gaussian_filtered = smcx.kalman_filter(
        initial_mean,
        variance * initial_covariance,
        transition,
        variance * evolution,
        observation[None],
        jnp.asarray([[variance]], dtype=dtype),
        emissions[:, None],
    )
    gaussian_smoothed = smcx.rts_smoother(gaussian_filtered, transition)

    # Derive S_T independently from Gaussian innovations; n0 needs float64.
    predicted_means = np.asarray(gaussian_filtered.predicted_means)
    predicted_covariances = np.asarray(gaussian_filtered.predicted_covariances)
    observation_np = np.asarray(observation)
    residuals = np.asarray(emissions) - predicted_means @ observation_np
    forecast_variances = (
        np.einsum(
            "i,tij,j->t",
            observation_np,
            predicted_covariances,
            observation_np,
        )
        + variance
    )
    expected_scale = (
        prior_shape * variance
        + variance * np.sum(residuals**2 / forecast_variances)
    ) / (prior_shape + emissions.shape[0])

    rts_means = np.asarray(gaussian_smoothed.smoothed_means)
    rts_covariances = np.asarray(gaussian_smoothed.smoothed_covariances)
    scale_free = np.asarray(smoothed.smoothed_scale_free_covariances)
    eps = np.finfo(np.float64).eps
    # All covariance conds are <2.53 and CPU error is <=1 eps; 128 eps covers
    # both filter/smoother pipelines and the final scale product.
    tolerance = float(
        128
        * eps
        * max(1.0, np.max(np.abs(rts_means)), np.max(np.abs(rts_covariances)))
    )
    for actual, expected in (
        (smoothed.smoothed_means, rts_means),
        (variance * scale_free, rts_covariances),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)

    student_scale = float(filtered.scale_estimates[-1]) * scale_free
    finite_expected = (expected_scale / variance) * rts_covariances
    np.testing.assert_allclose(
        student_scale,
        finite_expected,
        rtol=0.0,
        atol=tolerance,
    )
    limiting_gap = np.max(np.abs(student_scale - rts_covariances))
    concentration = abs(expected_scale / variance - 1.0) * np.max(
        np.abs(rts_covariances)
    )
    assert limiting_gap > tolerance
    assert limiting_gap <= concentration + tolerance


def test_dlm_smoother_discount_matches_closed_form_backward_identity():
    """A nonsymmetric two-state model matches B = delta G^-1 exactly."""
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    transition = jnp.asarray([[1.0, 0.5], [-0.5, 0.75]], dtype=dtype)
    observation = jnp.asarray([1.0, -0.5], dtype=dtype)
    initial_mean = jnp.asarray([0.25, -0.5], dtype=dtype)
    initial_covariance = jnp.asarray([[1.0, 0.25], [0.25, 0.75]], dtype=dtype)
    emissions = jnp.asarray([0.75, -0.25, 1.0, 0.125], dtype=dtype)
    delta = 0.75
    filtered = smcx.dlm_filter(
        initial_mean,
        initial_covariance,
        transition,
        observation,
        emissions,
        discount=delta,
        prior_shape=4.0,
        prior_scale=1.2,
    )
    actual = smcx.dlm_smoother(filtered, transition, discount=delta)

    means = np.asarray(filtered.filtered_means)
    covariances = np.asarray(filtered.filtered_scale_free_covariances)
    covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
    expected_means = means.copy()
    expected_covariances = covariances.copy()
    transition_np = np.asarray(transition)
    # Exact dyadic B=delta G^-1 shares no solve or orientation with smcx.
    gain = np.asarray([[0.5625, -0.375], [0.375, 0.75]], dtype=means.dtype)
    for time in range(means.shape[0] - 2, -1, -1):
        expected_means[time] = means[time] + gain @ (
            expected_means[time + 1] - transition_np @ means[time]
        )
        expected_covariances[time] = (1.0 - delta) * covariances[
            time
        ] + gain @ expected_covariances[time + 1] @ gain.T
        expected_covariances[time] = 0.5 * (
            expected_covariances[time] + expected_covariances[time].T
        )

    # CPU/MPS error is <=1 eps; 32 eps covers the three-step solve/scan.
    eps = np.finfo(means.dtype).eps
    scale = max(
        1.0,
        np.max(np.abs(expected_means)),
        np.max(np.abs(expected_covariances)),
    )
    tolerance = 32 * eps * scale
    np.testing.assert_allclose(
        actual.smoothed_means,
        expected_means,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual.smoothed_scale_free_covariances,
        expected_covariances,
        rtol=0.0,
        atol=tolerance,
    )


def test_dlm_smoother_shares_backward_kernel_bitwise():
    """Equivalent DLM and Gaussian records take one float32 scan path."""
    dtype = jnp.float32
    transition = jnp.asarray([[0.9, 0.2], [-0.3, 0.8]], dtype=dtype)
    covariance_0 = jnp.asarray([[8.0, 2.0], [2.0, 10.0]], dtype=dtype)

    raw_1 = (transition @ covariance_0) @ transition.T
    predicted_1 = 0.5 * (raw_1 + raw_1.T)
    covariance_1 = 0.5 * predicted_1
    raw_2 = (transition @ covariance_1) @ transition.T
    predicted_2 = 0.5 * (raw_2 + raw_2.T)
    covariance_2 = 0.5 * predicted_2
    means = jnp.asarray([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]], dtype=dtype)
    covariances = jnp.stack((covariance_0, covariance_1, covariance_2))
    dlm_record = _dlm_record(means, covariances)
    gaussian_record = smcx.GaussianFilterPosterior(
        marginal_loglik=jnp.asarray(0.0, dtype=dtype),
        predicted_means=jnp.concatenate((
            means[:1],
            means[:-1] @ transition.T,
        )),
        predicted_covariances=jnp.stack((
            covariance_0,
            predicted_1,
            predicted_2,
        )),
        filtered_means=means,
        filtered_covariances=covariances,
        log_evidence_increments=jnp.zeros(3, dtype=dtype),
    )

    dlm = smcx.dlm_smoother(
        dlm_record,
        transition,
        scale_free_transition_covariance=jnp.zeros((2, 2), dtype=dtype),
    )
    gaussian = smcx.rts_smoother(gaussian_record, transition)
    np.testing.assert_array_equal(dlm.smoothed_means, gaussian.smoothed_means)
    np.testing.assert_array_equal(
        dlm.smoothed_scale_free_covariances,
        gaussian.smoothed_covariances,
    )


def test_dlm_smoother_discount_vmap_and_jit_are_lane_independent():
    """Batching preserves both state directions and every retained field."""
    dtype = jnp.float32
    transition = jnp.asarray([[1.0, 0.25], [-0.5, 1.0]], dtype=dtype)
    covariances = jnp.stack((
        jnp.eye(2, dtype=dtype),
        0.5 * jnp.eye(2, dtype=dtype),
    ))
    records = (
        _dlm_record(
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=dtype),
            covariances,
        ),
        _dlm_record(
            jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=dtype),
            covariances,
        ),
    )
    batched = jax.tree.map(lambda *values: jnp.stack(values), *records)

    def smooth(record: smcx.DLMFilterPosterior):
        return smcx.dlm_smoother(record, transition, discount=0.75)

    independent = jax.tree.map(
        lambda *values: jnp.stack(values),
        *(smooth(record) for record in records),
    )
    vectorized = jax.vmap(smooth)(batched)
    compiled = jax.jit(jax.vmap(smooth))(batched)
    # CPU/MPS error is <=1 eps; 32 eps covers batching and compiled transforms.
    eps = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        vectorized.smoothed_means[:, 0],
        [[2.0 / 3.0, 1.0 / 3.0], [-1.0 / 6.0, 2.0 / 3.0]],
        rtol=0.0,
        atol=32 * eps,
    )
    for actual in (vectorized, compiled):
        for actual_field, expected_field in zip(
            actual, independent, strict=True
        ):
            np.testing.assert_allclose(
                actual_field,
                expected_field,
                rtol=0.0,
                atol=32 * eps,
            )


def test_dlm_smoother_gradient_matches_scalar_derivative():
    """Autodiff crosses reconstruction, solve, mean, and covariance updates."""
    dtype = jnp.float32
    record = _dlm_record(
        jnp.asarray([[0.0], [0.7]], dtype=dtype),
        jnp.asarray([[[0.8]], [[0.4]]], dtype=dtype),
    )

    def objective(coefficient: jax.Array) -> jax.Array:
        smoothed = smcx.dlm_smoother(
            record,
            coefficient.reshape((1, 1)),
            scale_free_transition_covariance=jnp.asarray([[0.3]], dtype=dtype),
        )
        return (
            smoothed.smoothed_means[0, 0]
            + 0.25 * smoothed.smoothed_scale_free_covariances[0, 0, 0]
        )

    coefficient = jnp.asarray(0.6, dtype=dtype)
    # Direct derivative; CPU/MPS error <=1.42 eps, so 8 eps covers autodiff.
    expected = -0.14262764664383037
    gradient = jax.grad(objective)
    for actual in (gradient(coefficient), jax.jit(gradient)(coefficient)):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=8 * float(np.finfo(np.float32).eps),
        )


def test_dlm_smoother_never_materializes_huge_student_scale():
    """A representable scale trace stays finite through a real backward step."""
    dtype = jnp.float32
    maximum = float(np.finfo(np.float32).max)
    record = _dlm_record(
        jnp.zeros((2, 1), dtype=dtype),
        jnp.asarray([[[2.0]], [[1.0]]], dtype=dtype),
        scale=maximum,
    )

    def smooth(value: smcx.DLMFilterPosterior):
        return smcx.dlm_smoother(
            value,
            jnp.ones((1, 1), dtype=dtype),
            scale_free_transition_covariance=jnp.ones((1, 1), dtype=dtype),
        )

    with np.errstate(over="ignore"):
        materialized = np.asarray(record.scale_estimates[-1]) * np.asarray(
            record.filtered_scale_free_covariances[0]
        )
    assert np.all(np.isinf(materialized))
    for posterior in (smooth(record), jax.jit(smooth)(record)):
        assert all(
            np.all(np.isfinite(np.asarray(field))) for field in posterior
        )
        np.testing.assert_array_equal(
            posterior.scale_estimates, record.scale_estimates
        )


@pytest.mark.parametrize(
    ("evolution", "message"),
    [
        pytest.param(
            jnp.zeros((2, 2, 2), dtype=jnp.float32),
            "must have shape",
            id="length",
        ),
        pytest.param(
            jnp.zeros((2, 2), dtype=jnp.int32),
            "must have a floating dtype",
            id="integer",
        ),
        pytest.param(
            -jnp.eye(2, dtype=jnp.float32),
            "positive semidefinite",
            id="negative",
        ),
    ],
)
def test_dlm_smoother_rejects_invalid_evolution_covariance(
    evolution: jax.Array, message: str
):
    """Timed alignment, dtype, and covariance domain are public boundaries."""
    dtype = jnp.float32
    record = _dlm_record(
        jnp.zeros((2, 2), dtype=dtype),
        jnp.tile(jnp.eye(2, dtype=dtype), (2, 1, 1)),
    )

    with pytest.raises(ValueError, match=message):
        unwrap(smcx.dlm_smoother)(
            record,
            jnp.eye(2, dtype=dtype),
            scale_free_transition_covariance=evolution,
        )


def test_dlm_smoother_rejects_singular_reconstructed_positive_time_prior():
    """Only a terminal filtered covariance may be singular."""
    dtype = jnp.float32
    record = _dlm_record(
        jnp.zeros((2, 2), dtype=dtype),
        jnp.stack((jnp.zeros((2, 2), dtype=dtype), jnp.eye(2, dtype=dtype))),
    )

    with pytest.raises(ValueError, match="positive definite"):
        smcx.dlm_smoother(
            record,
            jnp.eye(2, dtype=dtype),
            scale_free_transition_covariance=jnp.zeros((2, 2), dtype=dtype),
        )


def test_dlm_smoother_accepts_zero_length_timed_evolution_at_one_step():
    """A T=1 timed evolution contains exactly zero transitions."""
    dtype = jnp.float32
    record = _one_step_dlm_record(jnp.eye(2, dtype=dtype))
    posterior = smcx.dlm_smoother(
        record,
        jnp.eye(2, dtype=dtype),
        scale_free_transition_covariance=jnp.empty((0, 2, 2), dtype=dtype),
    )
    assert posterior.smoothed_means.shape == (1, 2)
