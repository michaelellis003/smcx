# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared numerical primitives."""

import jax.numpy as jnp
import numpy as np

from smcx._numerics import _neumaier_add, _neumaier_prefix_sum


def test_neumaier_add_matches_plain_sum_on_ordinary_values():
    """Compensated addition reproduces the exact small-case sum."""
    total = jnp.asarray(0.0)
    correction = jnp.asarray(0.0)
    for value in (1.0, 1e-8, -1.0):
        total, correction = _neumaier_add(total, correction, value)
    # The recovered value is exact up to one rounding of 1e-8 in the
    # working dtype (float32 under SMCX_TEST_X64=0).
    np.testing.assert_allclose(
        float(total + correction),
        1e-8,
        rtol=8 * float(jnp.finfo(total.dtype).eps),
    )


def test_neumaier_add_propagates_minus_inf_without_nan():
    """An infinite increment yields an infinite total, never NaN (#281)."""
    total = jnp.asarray(0.0)
    correction = jnp.asarray(0.0)
    for value in (1.0, -jnp.inf, 2.0):
        total, correction = _neumaier_add(total, correction, value)
    assert np.isneginf(float(total))
    assert np.isfinite(float(correction))
    assert np.isneginf(float(total + correction))


def test_neumaier_prefix_sum_propagates_minus_inf_without_nan():
    """Prefix sums after an infinite value stay -inf, never NaN (#281)."""
    prefixes = _neumaier_prefix_sum(jnp.asarray([1.0, -jnp.inf, 2.0]))
    assert np.isfinite(float(prefixes[0]))
    assert np.isneginf(float(prefixes[1]))
    assert np.isneginf(float(prefixes[2]))
