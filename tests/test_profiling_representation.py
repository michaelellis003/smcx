# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Dense-versus-PyTree profiling fixture contracts."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from benchmarks.profiling.representation import (
    TrackingLGSSM,
    TrackingState,
    flatten_tracking_state,
    make_dense_tracking_callbacks,
    make_tracking_data,
    make_tree_tracking_callbacks,
    tracking_kalman_oracle,
)
from smcx import bootstrap_filter


def test_tracking_data_and_kalman_oracle_are_frozen() -> None:
    model = TrackingLGSSM(timesteps=6)
    data = make_tracking_data(model, seed=20260719)
    np.testing.assert_allclose(
        data.emissions,
        [
            [-0.59187156, -1.60547185],
            [-1.06440167, -1.63049857],
            [-1.52810632, -3.20364817],
            [-1.70075401, -3.74806765],
            [-1.55940252, -2.27701689],
            [-1.77668401, -3.82175040],
        ],
        rtol=0.0,
        atol=5e-8,
    )
    oracle = tracking_kalman_oracle(model, data.emissions, data.inputs)
    assert oracle.log_evidence == pytest.approx(-17.0280693886779)
    np.testing.assert_allclose(
        oracle.filtered_means[-1],
        [-1.84542347, -3.57373302, -0.15254001, -0.30722461],
        rtol=0.0,
        atol=5e-8,
    )
    assert oracle.filtered_covariances.shape == (6, 4, 4)


def test_diagonal_covariance_regime_removes_cross_terms() -> None:
    model = TrackingLGSSM(timesteps=6, covariance_regime="diagonal")
    transition = model.transition_covariance()
    observation = model.observation_covariance()
    np.testing.assert_array_equal(transition, np.diag(np.diag(transition)))
    np.testing.assert_array_equal(observation, np.diag(np.diag(observation)))


def test_dense_and_tree_callbacks_are_mathematically_identical() -> None:
    model = TrackingLGSSM(timesteps=6)
    data = make_tracking_data(model, seed=20260719)
    dense = make_dense_tracking_callbacks(model)
    tree = make_tree_tracking_callbacks(model)

    initial_key = jr.key(4)
    dense_initial = dense.initial_sampler(initial_key, 16, data.inputs[0])
    tree_initial = tree.initial_sampler(initial_key, 16, data.inputs[0])
    np.testing.assert_array_equal(
        np.asarray(dense_initial),
        np.asarray(flatten_tracking_state(tree_initial)),
    )

    state = jnp.asarray(data.states[2])
    tree_state = TrackingState(position=state[:2], velocity=state[2:])
    transition_key = jr.key(5)
    dense_next = dense.transition_sampler(transition_key, state, data.inputs[3])
    tree_next = tree.transition_sampler(
        transition_key, tree_state, data.inputs[3]
    )
    np.testing.assert_array_equal(
        np.asarray(dense_next),
        np.asarray(flatten_tracking_state(tree_next)),
    )
    dense_log_prob = dense.log_observation_fn(
        data.emissions[3], dense_next, data.inputs[3]
    )
    tree_log_prob = tree.log_observation_fn(
        data.emissions[3], tree_next, data.inputs[3]
    )
    assert float(dense_log_prob) == pytest.approx(float(tree_log_prob))
    assert np.asarray(dense_initial).dtype == np.float32
    assert all(
        np.asarray(leaf).dtype == np.float32
        for leaf in jax.tree.leaves(tree_initial)
    )
    assert np.asarray(dense_next).dtype == np.float32
    assert all(
        np.asarray(leaf).dtype == np.float32
        for leaf in jax.tree.leaves(tree_next)
    )
    assert np.asarray(dense_log_prob).dtype == np.float32
    assert np.asarray(tree_log_prob).dtype == np.float32


@pytest.mark.parametrize("covariance_regime", ["correlated", "diagonal"])
@pytest.mark.parametrize("store_history", [False, True])
def test_dense_and_tree_bootstrap_match_when_jitted(
    covariance_regime: str,
    store_history: bool,
) -> None:
    model = TrackingLGSSM(
        timesteps=12,
        covariance_regime=covariance_regime,
    )
    data = make_tracking_data(model, seed=20260719)
    dense = make_dense_tracking_callbacks(model)
    tree = make_tree_tracking_callbacks(model)
    emissions = jnp.asarray(data.emissions)
    inputs = jnp.asarray(data.inputs)
    num_particles = 256

    def run_dense(key):
        return bootstrap_filter(
            key,
            dense.initial_sampler,
            dense.transition_sampler,
            dense.log_observation_fn,
            emissions,
            num_particles,
            inputs=inputs,
            store_history=store_history,
        )

    def run_tree(key):
        return bootstrap_filter(
            key,
            tree.initial_sampler,
            tree.transition_sampler,
            tree.log_observation_fn,
            emissions,
            num_particles,
            inputs=inputs,
            store_history=store_history,
        )

    key = jr.key(11)
    dense_result = jax.jit(run_dense)(key)
    tree_result = jax.jit(run_tree)(key)
    tree_particles = flatten_tracking_state(tree_result.filtered_particles)

    np.testing.assert_allclose(
        np.asarray(tree_particles),
        np.asarray(dense_result.filtered_particles),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(tree_result.filtered_log_weights),
        np.asarray(dense_result.filtered_log_weights),
        rtol=1e-6,
        atol=1e-6,
    )
    assert float(tree_result.marginal_loglik) == pytest.approx(
        float(dense_result.marginal_loglik), rel=1e-6
    )
