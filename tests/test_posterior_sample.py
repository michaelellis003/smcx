# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for joint linear-Gaussian posterior sampling."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
import smcx.kalman as kalman_module

_SCALAR_MEANS = jnp.zeros((2, 1))
_SCALAR_POSTERIOR = smcx.GaussianFilterPosterior(
    jnp.asarray(0.0),
    _SCALAR_MEANS,
    jnp.asarray([[[1.0]], [[2.0]]]),
    _SCALAR_MEANS,
    jnp.ones((2, 1, 1)),
    jnp.zeros(2),
)


def test_posterior_sample_matches_fixed_key_scalar_conditional():
    """A two-time oracle binds the gain, variance, keys, and draw axes."""
    key = jr.key(330)
    keys = jr.split(key, 2)
    count = np.int64(3)

    actual = smcx.posterior_sample(
        key, _SCALAR_POSTERIOR, jnp.ones((1, 1)), num_draws=count
    )
    compiled = jax.jit(smcx.posterior_sample, static_argnames="num_draws")(
        key, _SCALAR_POSTERIOR, jnp.ones((1, 1)), num_draws=count
    )
    terminal = jr.normal(keys[1], (3, 1), dtype=_SCALAR_MEANS.dtype)
    earlier = 0.5 * terminal + jnp.sqrt(
        jnp.asarray(0.5, _SCALAR_MEANS.dtype)
    ) * jr.normal(keys[0], (3, 1), dtype=_SCALAR_MEANS.dtype)
    expected = jnp.stack((earlier, terminal), axis=1)

    # 16 eps covers scalar solve/multiply; prohibited mutations differ by O(1).
    tolerance = 16.0 * np.finfo(np.asarray(actual).dtype).eps
    for result in (actual, compiled):
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize("count", [True, 0, np.asarray(1.0)])
def test_posterior_sample_rejects_invalid_draw_counts(count):
    """Every nonpositive or non-index count is rejected at the public shell."""
    with pytest.raises(ValueError, match="positive integer"):
        smcx.posterior_sample(
            jr.key(0), _SCALAR_POSTERIOR, jnp.ones((1, 1)), num_draws=count
        )


def test_posterior_sample_one_time_preserves_zero_variance_coordinate():
    """The terminal fallback handles an empty backward pass without jitter."""
    mean = jnp.asarray([[2.0, -1.0]])
    zero = jnp.zeros((1, 2, 2))
    posterior = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0),
        mean,
        zero,
        mean,
        jnp.asarray([[[4.0, 0.0], [0.0, 0.0]]]),
        jnp.zeros(1),
    )

    draws = smcx.posterior_sample(
        jr.key(331), posterior, jnp.eye(2), num_draws=1
    )

    assert draws.shape == (1, 1, 2)
    np.testing.assert_array_equal(draws[:, 0, 1], -1.0)
    assert np.all(np.isfinite(draws))


def test_posterior_sample_preserves_joseph_forward_product_order():
    """A conditioned f32 fixture binds the shared Joseph reconstruction."""
    dtype = np.float32
    covariance = jnp.asarray(
        [[8.0, np.nextafter(dtype(2.0), dtype(np.inf))], [2.0, 10.0]],
        dtype=jnp.float32,
    )
    transition = jnp.asarray([[-1.2, -1.1], [-2.6, -2.2]], dtype=jnp.float32)
    predicted = kalman_module._symmetrize(
        (transition @ covariance) @ transition.T
    )

    _, _, process_noise = kalman_module._backward_gain_terms(
        covariance, predicted, transition, None
    )
    _, conditional = kalman_module._posterior_sample_setup((
        covariance,
        predicted,
        transition,
    ))
    tolerance = 32.0 * np.finfo(dtype).eps * float(jnp.max(covariance))

    np.testing.assert_array_equal(process_noise, jnp.zeros_like(covariance))
    assert float(jnp.max(jnp.abs(conditional))) <= tolerance
    assert np.linalg.eigvalsh(np.asarray(conditional, dtype=np.float64))[0] >= (
        -tolerance
    )

    means = jnp.zeros((2, 2), dtype=jnp.float32)
    posterior = smcx.GaussianFilterPosterior(
        jnp.asarray(0.0, dtype=jnp.float32),
        means,
        jnp.stack((covariance, predicted)),
        means,
        jnp.stack((covariance, jnp.zeros_like(covariance))),
        jnp.zeros(2, dtype=jnp.float32),
    )
    draws = smcx.posterior_sample(
        jr.key(332), posterior, transition, num_draws=2
    )

    assert np.all(np.isfinite(draws))
