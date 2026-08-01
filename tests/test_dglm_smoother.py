# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic evolution oracles for DGLM retrospective moments."""

from fractions import Fraction
from inspect import unwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
        equal_nan=False,
    )


def _one_step_record(covariance: jax.Array) -> smcx.DGLMFilterPosterior:
    """Build the smallest structurally valid smoother input record."""
    dtype = covariance.dtype
    return _dglm_record(
        jnp.asarray([[0.25, -0.5]], dtype=dtype),
        covariance[None],
    )


_F32_EYE_2 = jnp.eye(2, dtype=jnp.float32)
_F32_ZERO_2 = jnp.zeros((2, 2), dtype=jnp.float32)


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


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("filtered_means", jnp.empty((0, 2), dtype=jnp.float32)),
        ("filtered_means", jnp.empty((2, 0), dtype=jnp.float32)),
        ("filtered_covariances", jnp.empty((2, 2, 1), dtype=jnp.float32)),
        ("conjugate_alphas", jnp.empty((1,), dtype=jnp.float32)),
        ("conjugate_betas", jnp.empty((2, 1), dtype=jnp.float32)),
        ("marginal_loglik", jnp.empty((1,), dtype=jnp.float32)),
        ("log_evidence_increments", jnp.empty((1,), dtype=jnp.float32)),
        *((field, None) for field in smcx.DGLMFilterPosterior._fields),
    ],
)
def test_dglm_smoother_validates_every_record_field(
    field: str,
    invalid: jax.Array | None,
) -> None:
    """Every retained field has a public shape and common-float contract."""
    dtype = jnp.float32
    record = _dglm_record(
        jnp.zeros((2, 2), dtype=dtype),
        jnp.tile(jnp.eye(2, dtype=dtype), (2, 1, 1)),
    )
    if invalid is None:
        invalid = jnp.zeros_like(getattr(record, field), dtype=jnp.int32)
    malformed = record._replace(**{field: invalid})

    def smooth(value: smcx.DGLMFilterPosterior):
        return unwrap(smcx.dglm_smoother)(
            value,
            jnp.eye(2, dtype=dtype),
            discount=jnp.asarray(1.0, dtype=dtype),
        )

    for call in (smooth, jax.jit(smooth)):
        with pytest.raises(ValueError, match=field):
            call(malformed)
    if invalid.dtype == jnp.int32 and jax.config.read("jax_enable_x64"):
        mixed = record._replace(**{
            field: getattr(record, field).astype(jnp.float64)
        })
        for call in (smooth, jax.jit(smooth)):
            with pytest.raises(ValueError, match="all arrays must have dtype"):
                call(mixed)


@pytest.mark.parametrize(
    ("transition", "options", "message"),
    [
        (_F32_EYE_2, {}, "supply exactly one"),
        (
            _F32_EYE_2,
            {"transition_covariance": _F32_ZERO_2, "discount": 1.0},
            "supply exactly one",
        ),
        (jnp.ones((1, 1), dtype=jnp.float32), {"discount": 1.0}, "shape"),
        (
            jnp.eye(2, dtype=jnp.int32),
            {"discount": 1.0},
            "floating dtype",
        ),
        (
            _F32_EYE_2,
            {"transition_covariance": jnp.zeros((2, 2, 2), dtype=jnp.float32)},
            "must have shape",
        ),
        (
            _F32_EYE_2,
            {"transition_covariance": jnp.zeros((2, 2), dtype=jnp.int32)},
            "floating dtype",
        ),
        (
            _F32_EYE_2,
            {"transition_covariance": -_F32_EYE_2},
            "positive semidefinite",
        ),
        (
            _F32_EYE_2,
            {"discount": jnp.asarray([0.9, 0.8], dtype=jnp.float32)},
            "scalar",
        ),
        (_F32_EYE_2, {"discount": 0.0}, "positive"),
        (_F32_EYE_2, {"discount": 1.1}, "outside"),
        (_F32_EYE_2, {"discount": np.nan}, "positive"),
    ],
)
def test_dglm_smoother_rejects_invalid_evolution(
    transition: jax.Array,
    options: dict[str, object],
    message: str,
) -> None:
    """Evolution shapes, dtypes, covariance, and scalar domains are eager."""
    dtype = jnp.float32
    record = _dglm_record(
        jnp.zeros((2, 2), dtype=dtype),
        jnp.tile(jnp.eye(2, dtype=dtype), (2, 1, 1)),
    )

    with pytest.raises(ValueError, match=message):
        unwrap(smcx.dglm_smoother)(
            record,
            transition,
            **options,
        )
    if message == "scalar":
        with pytest.raises(ValueError, match=message):
            jax.jit(
                lambda value: unwrap(smcx.dglm_smoother)(
                    record, transition, discount=value
                )
            )(jnp.asarray([0.9, 0.8], dtype=dtype))
    if message == "floating dtype" and jax.config.read("jax_enable_x64"):
        is_transition = transition.dtype == jnp.int32
        if is_transition:
            transition = _F32_EYE_2.astype(jnp.float64)
        else:
            options = {"transition_covariance": _F32_ZERO_2.astype(jnp.float64)}
        with pytest.raises(ValueError, match="all arrays must have dtype"):
            unwrap(smcx.dglm_smoother)(record, transition, **options)


@pytest.mark.parametrize("evolution", ["static", "timed", "discount"])
@pytest.mark.parametrize(
    ("covariance", "canonical"),
    [
        (
            [[2.0**30, 4.0], [-4.0, 2.0**10]],
            [[2.0**30, 0.0], [0.0, 2.0**10]],
        ),
        (
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ),
    ],
)
def test_dglm_smoother_one_step_is_factor_free(
    evolution: str,
    covariance: list[list[float]],
    canonical: list[list[float]],
) -> None:
    """Inclusive skew and singular PSD terminals need no factorization."""
    dtype = jnp.float32
    record = _one_step_record(jnp.asarray(covariance, dtype=dtype))
    transition = jnp.zeros((2, 2), dtype=dtype)

    def smooth(value: smcx.DGLMFilterPosterior):
        if evolution == "discount":
            return smcx.dglm_smoother(value, transition, discount=1.0)
        covariance = (
            _F32_ZERO_2
            if evolution == "static"
            else jnp.empty((0, 2, 2), dtype=dtype)
        )
        return smcx.dglm_smoother(
            value, transition, transition_covariance=covariance
        )

    for posterior in (smooth(record), jax.jit(smooth)(record)):
        assert all(
            np.all(np.isfinite(np.asarray(field))) for field in posterior
        )
        np.testing.assert_array_equal(
            posterior.filtered_covariances[0], canonical
        )
        np.testing.assert_array_equal(
            posterior.smoothed_means, posterior.filtered_means
        )
        np.testing.assert_array_equal(
            posterior.smoothed_covariances,
            posterior.filtered_covariances,
        )


def test_dglm_smoother_rejects_material_filter_covariance_skew() -> None:
    """One normalized epsilon beyond the inclusive edge is material."""
    covariance = jnp.asarray(
        [[2.0**30, 4.0625], [-4.0625, 2.0**10]], dtype=jnp.float32
    )
    record = _one_step_record(covariance)

    with pytest.raises(ValueError, match="symmetric within roundoff"):
        unwrap(smcx.dglm_smoother)(
            record,
            jnp.eye(2, dtype=jnp.float32),
            discount=1.0,
        )


def test_dglm_smoother_factors_only_reconstructed_priors() -> None:
    """A singular terminal is valid; a singular positive-time prior is not."""
    dtype = jnp.float32
    identity = jnp.eye(2, dtype=dtype)
    singular = jnp.diag(jnp.asarray([1.0, 0.0], dtype=dtype))
    accepted = _dglm_record(
        jnp.zeros((2, 2), dtype=dtype),
        jnp.stack((identity, singular)),
    )

    def smooth(value: smcx.DGLMFilterPosterior):
        return smcx.dglm_smoother(
            value,
            identity,
            transition_covariance=jnp.zeros((2, 2), dtype=dtype),
        )

    for posterior in (smooth(accepted), jax.jit(smooth)(accepted)):
        assert all(
            np.all(np.isfinite(np.asarray(field))) for field in posterior
        )
        np.testing.assert_array_equal(
            posterior.smoothed_means, jnp.zeros((2, 2), dtype=dtype)
        )
        np.testing.assert_array_equal(
            posterior.smoothed_covariances,
            jnp.stack((singular, singular)),
        )

    rejected = accepted._replace(
        filtered_covariances=jnp.stack((jnp.zeros_like(identity), identity))
    )
    with pytest.raises(ValueError, match="positive definite"):
        smooth(rejected)


def test_dglm_smoother_canonicalizes_released_producer_under_jit() -> None:
    """Real filter roundoff needs the dimension-scaled skew allowance."""
    dtype = jnp.float32

    def array(values: object) -> jax.Array:
        return jnp.asarray(values, dtype=dtype)

    initial_mean = array(
        [-0.05465821921825409, -0.06837236881256104, -0.6396454572677612],
    )
    initial_covariance = array([
        [1.1717324256896973, 0.33089298009872437, 3.908212900161743],
        [0.33089298009872437, 0.12562869489192963, 1.4273052215576172],
        [3.908212900161743, 1.4273052215576172, 24.911808013916016],
    ])
    transition = array([
        [0.42578643560409546, -0.43568071722984314, 0.2502269446849823],
        [-0.054558929055929184, 1.1346344947814941, -0.3339521288871765],
        [0.06975531578063965, -0.4907507300376892, 1.2281440496444702],
    ])
    observation = array([
        -3.0298609733581543,
        1.9809789657592773,
        0.4838603734970093,
    ])
    evolution = jnp.diag(
        array([
            0.0033386414870619774,
            0.004689060617238283,
            0.5420311093330383,
        ])
    )
    filtered = smcx.dglm_filter(
        initial_mean,
        initial_covariance,
        transition,
        observation,
        jnp.asarray([1, 0, 3, 2, 1]),
        family=smcx.poisson(),
        transition_covariance=evolution,
    )
    raw = np.asarray(filtered.filtered_covariances).copy()
    transpose = raw.swapaxes(-1, -2)
    raw64 = raw.astype(np.float64)
    diagonal_scale = np.sqrt(np.diagonal(raw64, axis1=-2, axis2=-1))
    normalized = (raw64 / diagonal_scale[..., :, None]) / (
        diagonal_scale[..., None, :]
    )
    skew = float(np.max(np.abs(normalized - normalized.swapaxes(-1, -2))))
    eps = float(np.finfo(np.float32).eps)
    assert 32 * eps < skew <= 32 * 3 * eps
    assert not np.array_equal(raw, transpose)

    def smooth(record, matrix, covariance):
        return smcx.dglm_smoother(
            record, matrix, transition_covariance=covariance
        )

    eager = smooth(filtered, transition, evolution)
    compiled = jax.jit(smooth)(filtered, transition, evolution)
    canonical = np.float32(0.5) * (raw + transpose)
    np.testing.assert_array_equal(eager.filtered_covariances, canonical)
    np.testing.assert_array_equal(raw, filtered.filtered_covariances)
    np.testing.assert_array_equal(
        eager.smoothed_means[-1], eager.filtered_means[-1]
    )
    np.testing.assert_array_equal(eager.smoothed_covariances[-1], canonical[-1])
    for eager_field, compiled_field in zip(eager, compiled, strict=True):
        _assert_roundoff_close(compiled_field, eager_field)


def test_dglm_smoother_vmap_and_jit_are_lane_independent() -> None:
    """Batching maps complete records and dynamic discounts by lane."""
    dtype = jnp.float32
    transition = jnp.asarray([[1.0, 0.25], [-0.5, 1.0]], dtype=dtype)
    covariances = jnp.stack((
        jnp.eye(2, dtype=dtype),
        0.5 * jnp.eye(2, dtype=dtype),
    ))
    records = (
        _dglm_record(
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=dtype), covariances
        ),
        _dglm_record(
            jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=dtype), covariances
        ),
    )
    discounts = jnp.asarray([0.75, 0.8], dtype=dtype)
    batched = jax.tree.map(lambda *values: jnp.stack(values), *records)

    def smooth(record: smcx.DGLMFilterPosterior, discount: jax.Array):
        return smcx.dglm_smoother(record, transition, discount=discount)

    independent = jax.tree.map(
        lambda *values: jnp.stack(values),
        *(
            smooth(record, discounts[lane])
            for lane, record in enumerate(records)
        ),
    )
    vectorized = jax.vmap(smooth)(batched, discounts)
    compiled = jax.jit(jax.vmap(smooth))(batched, discounts)
    assert not np.array_equal(
        vectorized.smoothed_means[0], vectorized.smoothed_means[1]
    )
    for actual in (vectorized, compiled):
        for actual_field, expected_field in zip(
            actual, independent, strict=True
        ):
            _assert_roundoff_close(actual_field, expected_field)


def test_dglm_smoother_gradient_matches_scalar_derivative() -> None:
    """Autodiff crosses reconstruction, solve, mean, and covariance updates."""
    dtype = _working_dtype()
    record = _dglm_record(
        jnp.asarray([[0.0], [0.7]], dtype=dtype),
        jnp.asarray([[[0.8]], [[0.4]]], dtype=dtype),
    )

    def objective(coefficient: jax.Array) -> jax.Array:
        smoothed = smcx.dglm_smoother(
            record,
            coefficient.reshape((1, 1)),
            transition_covariance=jnp.asarray([[0.3]], dtype=dtype),
        )
        return (
            smoothed.smoothed_means[0, 0]
            + 0.25 * smoothed.smoothed_covariances[0, 0, 0]
        )

    coefficient = jnp.asarray(0.6, dtype=dtype)
    # Direct differentiation gives -16780/117649. Measured CPU/MPS error is
    # at most 1.42 eps, so 8 eps covers eager and compiled autodiff.
    expected = -0.14262764664383037
    gradient = jax.grad(objective)
    for actual in (gradient(coefficient), jax.jit(gradient)(coefficient)):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=8 * float(np.finfo(np.asarray(actual).dtype).eps),
            equal_nan=False,
        )
