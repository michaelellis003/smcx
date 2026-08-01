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
