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


def _one_time_posterior(covariance):
    """Build a minimal posterior whose terminal covariance is under test."""
    state_dim = covariance.shape[0]
    means = jnp.zeros((1, state_dim), dtype=covariance.dtype)
    identity = jnp.eye(state_dim, dtype=covariance.dtype)
    return smcx.GaussianFilterPosterior(
        jnp.asarray(0.0, dtype=covariance.dtype),
        means,
        identity[None],
        means,
        covariance[None],
        jnp.zeros(1, dtype=covariance.dtype),
    )


def _two_time_terminal_posterior(covariance):
    """Build a valid two-time record with the supplied terminal covariance."""
    state_dim = covariance.shape[0]
    means = jnp.zeros((2, state_dim), dtype=covariance.dtype)
    identity = jnp.eye(state_dim, dtype=covariance.dtype)
    return smcx.GaussianFilterPosterior(
        jnp.asarray(0.0, dtype=covariance.dtype),
        means,
        jnp.stack((identity, identity)),
        means,
        jnp.stack((identity, covariance)),
        jnp.zeros(2, dtype=covariance.dtype),
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
    # This is a fixture-separation bound, not a general forward-error claim;
    # both direct Schur orders exceed it by more than an order of magnitude.
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


def test_spectral_factor_clips_covariance_inside_normalized_band():
    """The fallback clips only the small negative normalized mode."""
    epsilon = np.finfo(np.float32).eps
    delta = np.float32(4.0 * epsilon)
    covariance = jnp.asarray(
        [[1.0, 1.0 + delta], [1.0 + delta, 1.0]], dtype=jnp.float32
    )

    factor = kalman_module._sampling_covariance_factor(covariance)
    reconstructed = np.asarray(factor, dtype=np.float64) @ np.asarray(
        factor.T, dtype=np.float64
    )
    expected = np.full((2, 2), 1.0 + float(delta) / 2.0)

    assert np.all(np.isfinite(factor))
    # Four eps covers the f32 eigensolver reconstruction. Adding the nominal
    # 16-eps admission band as jitter exceeds this bound.
    np.testing.assert_allclose(
        reconstructed, expected, rtol=0.0, atol=4.0 * epsilon
    )
    with jax.disable_jit(True), jax.debug_nans(True):
        draws = smcx.posterior_sample(
            jr.key(333),
            _two_time_terminal_posterior(covariance),
            jnp.zeros((2, 2), dtype=jnp.float32),
            num_draws=2,
        )
    assert np.all(np.isfinite(draws))


def test_spectral_factor_rescales_state_coordinates_on_left():
    """A scale-separated singular covariance binds root orientation."""
    epsilon = np.finfo(np.float32).eps
    scales = np.asarray([2.0**20, 2.0**-20], dtype=np.float32)
    covariance = jnp.asarray(scales[:, None] * scales[None, :])

    factor = kalman_module._sampling_covariance_factor(covariance)
    reconstructed = np.asarray(factor, dtype=np.float64) @ np.asarray(
        factor.T, dtype=np.float64
    )
    normalized = (reconstructed / scales[:, None]) / scales[None, :]

    # Eight eps covers factor reconstruction plus the two rescaling divisions;
    # multiplying the eigenvectors on the right differs here by order one.
    np.testing.assert_allclose(
        normalized, np.ones((2, 2)), rtol=0.0, atol=8.0 * epsilon
    )


def test_posterior_sample_normalizes_finite_raw_row_sum_overflow():
    """A valid normalized covariance survives raw f32 sum overflow."""
    covariance = jnp.full((2, 2), 2.0e38, dtype=jnp.float32)

    with jax.debug_nans(True), jax.debug_infs(True):
        draws = smcx.posterior_sample(
            jr.key(334),
            _one_time_posterior(covariance),
            jnp.eye(2, dtype=jnp.float32),
            num_draws=2,
        )

    assert np.all(np.isfinite(draws))


@pytest.mark.parametrize(
    "covariance",
    [
        np.asarray([[1.0, 2.0**-10], [2.0**-10, 0.0]], dtype=np.float32),
        np.diag(np.asarray([1.0, -4.0 * np.finfo(np.float32).eps])),
        np.asarray(
            [
                [1.0, 1.0 + 64.0 * np.finfo(np.float32).eps],
                [1.0 + 64.0 * np.finfo(np.float32).eps, 1.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [[1.0e20, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 1.0]],
            dtype=np.float32,
        ),
        np.asarray([[2.0e38, 2.5e38], [2.5e38, 2.0e38]], dtype=np.float32),
    ],
    ids=[
        "zero-row",
        "negative-diagonal",
        "outside-band",
        "scale-separated",
        "raw-norm-overflow",
    ],
)
def test_posterior_sample_rejects_invalid_covariance_eager_and_traced(
    covariance,
):
    """Concrete invalid covariances raise; traced ones produce only NaNs."""
    covariance = jnp.asarray(covariance, dtype=jnp.float32)

    def sample(value):
        state_dim = value.shape[0]
        return smcx.posterior_sample(
            jr.key(335),
            _one_time_posterior(value),
            jnp.eye(state_dim, dtype=value.dtype),
            num_draws=2,
        )

    with pytest.raises(ValueError, match="positive semidefinite"):
        sample(covariance)
    with jax.debug_nans(False), jax.debug_infs(False):
        traced = jax.jit(sample)(covariance)

    assert np.all(np.isnan(traced))


def test_posterior_sample_rejects_active_backend_factor(monkeypatch):
    """The public shell rejects a factor that the active backend cannot form."""

    def rejected_factor(covariance):
        return jnp.full_like(covariance, jnp.nan)

    monkeypatch.setattr(
        kalman_module, "_sampling_covariance_factor", rejected_factor
    )
    covariance = jnp.eye(2, dtype=jnp.float32)

    with pytest.raises(ValueError, match="factorable on the active backend"):
        smcx.posterior_sample(
            jr.key(336),
            _one_time_posterior(covariance),
            covariance,
            num_draws=2,
        )
