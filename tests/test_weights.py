# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for smcx.weights — validated against hand-computed values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx
from smcx.weights import log_normalize, normalize

_WEIGHT_UTILITIES = (
    pytest.param(smcx.log_normalize, id="log-normalize"),
    pytest.param(smcx.normalize, id="normalize"),
    pytest.param(smcx.log_ess, id="log-ess"),
    pytest.param(smcx.ess, id="ess"),
)


@pytest.mark.parametrize("weight_utility", _WEIGHT_UTILITIES)
@pytest.mark.parametrize(
    ("log_weights", "message"),
    [
        ([0.0], "log_weights must be a JAX array"),
        (np.asarray([0.0]), "log_weights must be a JAX array"),
        (jnp.asarray(0.0), "log_weights must be rank 1"),
        (jnp.ones((2, 1)), "log_weights must be rank 1"),
        (jnp.empty((0,)), "log_weights must contain at least one value"),
        (
            jnp.asarray([0], dtype=jnp.int32),
            "log_weights must have a floating dtype",
        ),
    ],
)
def test_weight_utilities_reject_invalid_inputs(
    weight_utility, log_weights, message
):
    """Every public weight utility owns the same structural contract."""
    with pytest.raises(ValueError, match=message):
        weight_utility(log_weights)


@pytest.mark.parametrize("weight_utility", _WEIGHT_UTILITIES)
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.zeros(512), id="uniform"),
        pytest.param(np.linspace(-8.0, 0.0, 512), id="skewed"),
    ],
)
def test_weight_utilities_reject_precision_below_float32(
    weight_utility, dtype, values
):
    """Low-precision inputs cannot satisfy normalization invariants."""
    log_weights = jnp.asarray(values, dtype=dtype)

    with pytest.raises(ValueError, match="at least float32 precision"):
        weight_utility(log_weights)


@pytest.mark.parametrize("weight_utility", _WEIGHT_UTILITIES)
def test_weight_utilities_remain_jit_compatible(weight_utility):
    """Structural validation admits tracers for valid JAX inputs."""
    log_weights = jnp.asarray([-2.0, -1.0, 0.0])

    eager = weight_utility(log_weights)
    compiled = jax.jit(weight_utility)(log_weights)

    eager_leaves = jax.tree.leaves(eager)
    compiled_leaves = jax.tree.leaves(compiled)
    assert len(compiled_leaves) == len(eager_leaves)
    for actual, expected in zip(compiled_leaves, eager_leaves, strict=True):
        assert jnp.allclose(actual, expected)


class TestLogNormalize:
    """Tests for log_normalize."""

    def test_uniform_weights(self):
        """Uniform log-weights [0, 0, 0] -> log(1/3) each."""
        lw = jnp.array([0.0, 0.0, 0.0])
        log_norm, log_z = log_normalize(lw)
        expected = jnp.full(3, jnp.log(1.0 / 3.0))
        assert jnp.allclose(log_norm, expected, atol=1e-7)
        # logsumexp([0, 0, 0]) = log(3)
        assert jnp.allclose(log_z, jnp.log(3.0), atol=1e-7)

    def test_degenerate_weights(self):
        """One particle has all weight, rest are -inf."""
        lw = jnp.array([0.0, -jnp.inf, -jnp.inf])
        log_norm, log_z = log_normalize(lw)
        assert jnp.allclose(log_norm[0], 0.0, atol=1e-7)
        assert log_norm[1] == -jnp.inf
        assert log_norm[2] == -jnp.inf
        # logsumexp([0, -inf, -inf]) = 0
        assert jnp.allclose(log_z, 0.0, atol=1e-7)

    def test_normalized_sum_to_zero(self):
        """Normalized log-weights should logsumexp to 0."""
        lw = jnp.array([1.0, 2.0, 3.0])
        log_norm, _ = log_normalize(lw)
        log_total = jnp.logaddexp.reduce(log_norm)
        assert jnp.allclose(log_total, 0.0, atol=1e-7)

    def test_numerical_stability_extreme(self):
        """Large magnitude log-weights should not overflow/underflow."""
        lw = jnp.array([1000.0, 1000.0, 999.0])
        log_norm, log_ev = log_normalize(lw)
        assert jnp.all(jnp.isfinite(log_norm))
        assert jnp.isfinite(log_ev)

    def test_large_negative_weights(self):
        """Very negative log-weights should still normalize correctly."""
        lw = jnp.array([-1000.0, -1000.0, -1001.0])
        log_norm, log_ev = log_normalize(lw)
        assert jnp.all(jnp.isfinite(log_norm))
        assert jnp.isfinite(log_ev)


class TestNormalize:
    """Tests for normalize (exp + normalize)."""

    def test_sums_to_one(self):
        """Normalized weights should sum to 1."""
        lw = jnp.array([1.0, 2.0, 3.0, 4.0])
        w = normalize(lw)
        assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-7)

    def test_uniform(self):
        """Equal log-weights should give equal normalized weights."""
        lw = jnp.zeros(5)
        w = normalize(lw)
        assert jnp.allclose(w, 0.2, atol=1e-7)

    def test_all_positive(self):
        """All normalized weights should be non-negative."""
        lw = jnp.array([-10.0, 0.0, 10.0])
        w = normalize(lw)
        assert jnp.all(w >= 0.0)
