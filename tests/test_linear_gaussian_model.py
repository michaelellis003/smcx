# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""LinearGaussianModel record-or-arrays gates (ADR-0035, issue #408)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx

MU0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y = jnp.asarray([
    [0.3, -0.1],
    [0.6, 0.2],
    [jnp.nan, jnp.nan],
    [-0.4, 0.9],
    [0.1, 0.05],
])
B_TRANS = jnp.asarray([0.1, -0.2])
B_OBS = jnp.asarray([0.05, 0.15])
G_TRANS = jnp.asarray([[0.4], [0.7]])
G_OBS = jnp.asarray([[0.2], [-0.3]])
U = jnp.asarray([[0.5], [-1.0], [0.25], [0.75], [-0.5]])

MODEL = smcx.LinearGaussianModel(
    initial_mean=MU0,
    initial_covariance=P0,
    transition_matrix=A,
    transition_covariance=Q,
    observation_matrix=H,
    observation_covariance=R,
)


def _assert_trees_bitwise_equal(left, right):
    for leaf_l, leaf_r in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(leaf_l), np.asarray(leaf_r))


def test_record_path_matches_array_path_bitwise():
    """Static model: the record form reproduces the array form exactly."""
    from_arrays = smcx.kalman_filter(MU0, P0, A, Q, H, R, Y)
    from_record = smcx.kalman_filter(MODEL, Y)
    _assert_trees_bitwise_equal(from_record, from_arrays)


def test_record_path_matches_array_path_time_varying():
    """Time-varying operators ride through the record unchanged."""
    a_t = jnp.stack([A, A * 0.95, A * 1.05, A * 0.9])
    h_t = jnp.stack([H, H * 1.1, H * 0.9, H, H * 1.2])
    model = MODEL._replace(transition_matrix=a_t, observation_matrix=h_t)
    from_arrays = smcx.kalman_filter(MU0, P0, a_t, Q, h_t, R, Y)
    from_record = smcx.kalman_filter(model, Y)
    _assert_trees_bitwise_equal(from_record, from_arrays)


def test_record_path_matches_array_path_with_biases_and_inputs():
    """Biases and input matrices in the record match the keyword form."""
    model = MODEL._replace(
        transition_bias=B_TRANS,
        observation_bias=B_OBS,
        transition_input_matrix=G_TRANS,
        observation_input_matrix=G_OBS,
    )
    from_arrays = smcx.kalman_filter(
        MU0,
        P0,
        A,
        Q,
        H,
        R,
        Y,
        transition_bias=B_TRANS,
        observation_bias=B_OBS,
        transition_input_matrix=G_TRANS,
        observation_input_matrix=G_OBS,
        inputs=U,
    )
    from_record = smcx.kalman_filter(model, Y, inputs=U)
    _assert_trees_bitwise_equal(from_record, from_arrays)


def test_keyword_emissions_with_record():
    """The emissions may arrive by keyword beside the record."""
    from_positional = smcx.kalman_filter(MODEL, Y)
    from_keyword = smcx.kalman_filter(MODEL, emissions=Y)
    _assert_trees_bitwise_equal(from_keyword, from_positional)


def test_rts_smoother_accepts_the_record():
    """A record in the transition slot means its transition_matrix."""
    filtered = smcx.kalman_filter(MODEL, Y)
    from_array = smcx.rts_smoother(filtered, A)
    from_record = smcx.rts_smoother(filtered, MODEL)
    _assert_trees_bitwise_equal(from_record, from_array)


def test_gradient_through_a_model_leaf_matches_the_array_path():
    """Differentiating a record leaf reproduces the array-path gradient."""

    def loss_record(a):
        return smcx.kalman_filter(
            MODEL._replace(transition_matrix=a), Y
        ).marginal_loglik

    def loss_arrays(a):
        return smcx.kalman_filter(MU0, P0, a, Q, H, R, Y).marginal_loglik

    np.testing.assert_array_equal(
        np.asarray(jax.grad(loss_record)(A)),
        np.asarray(jax.grad(loss_arrays)(A)),
    )


def test_vmap_over_model_batches():
    """Vmap over a batched model leaf equals stacked individual runs."""
    scales = jnp.asarray([0.5, 1.0, 2.0])

    def run(scale):
        model = MODEL._replace(transition_covariance=Q * scale)
        return smcx.kalman_filter(model, Y).marginal_loglik

    batched = jax.vmap(run)(scales)
    stacked = jnp.stack([run(scale) for scale in scales])
    rtol = 1e-10 if batched.dtype == jnp.float64 else 2e-5
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(stacked), rtol=rtol
    )


def test_record_with_loose_model_array_is_rejected():
    """The record and loose model arrays cannot be mixed."""
    with pytest.raises(ValueError, match="record only"):
        smcx.kalman_filter(MODEL, Y, transition_matrix=A)


def test_record_with_loose_keyword_bias_is_rejected():
    """Keyword model pieces beside the record are rejected too."""
    with pytest.raises(ValueError, match="record only"):
        smcx.kalman_filter(MODEL, Y, observation_bias=B_OBS)


def test_record_without_emissions_is_rejected():
    """A record with no emissions gets the documented error."""
    with pytest.raises(ValueError, match="emissions"):
        smcx.kalman_filter(MODEL)


def test_record_with_emissions_twice_is_rejected():
    """Positional plus keyword emissions is ambiguous and rejected."""
    with pytest.raises(ValueError, match="emissions"):
        smcx.kalman_filter(MODEL, Y, emissions=Y)


def test_array_path_with_missing_model_array_is_rejected():
    """Omitting a model array without a record raises at the boundary."""
    with pytest.raises(ValueError, match="LinearGaussianModel"):
        smcx.kalman_filter(MU0, P0, A, Q, H, emissions=Y)


def test_posterior_sample_accepts_the_record():
    """A record in the sampler's transition slot means its matrix."""
    filtered = smcx.kalman_filter(MODEL, Y)
    from_array = smcx.posterior_sample(
        jax.random.key(11), filtered, A, num_draws=8
    )
    from_record = smcx.posterior_sample(
        jax.random.key(11), filtered, MODEL, num_draws=8
    )
    np.testing.assert_array_equal(
        np.asarray(from_record), np.asarray(from_array)
    )


def test_smoothed_cross_covariances_accepts_the_record():
    """A record in the cross-covariance slot means its matrix."""
    smoothed = smcx.rts_smoother(smcx.kalman_filter(MODEL, Y), MODEL)
    from_array = smcx.smoothed_cross_covariances(smoothed, A)
    from_record = smcx.smoothed_cross_covariances(smoothed, MODEL)
    np.testing.assert_array_equal(
        np.asarray(from_record), np.asarray(from_array)
    )
