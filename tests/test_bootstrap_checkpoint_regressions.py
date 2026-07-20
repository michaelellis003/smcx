# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Regression contracts preserved while checkpoint APIs are added."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx


def _initial(key, num_particles):
    return jr.normal(key, (num_particles, 1))


def _transition(key, state):
    return 0.8 * state + 0.3 * jr.normal(key, state.shape)


def _log_observation(emission, state):
    return -0.5 * (emission[0] - state[0]) ** 2


def _assert_tree_equal(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_particle_state_documents_normalized_weights():
    """The public checkpoint state names its actual weight invariant."""
    docstring = " ".join((smcx.ParticleState.__doc__ or "").split())
    assert "Normalized log importance weights" in docstring


@pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason="frozen CPU/x64 arithmetic contract",
)
def test_bootstrap_filter_preserves_frozen_fixed_key_output():
    """Checkpoint additions must not change the legacy one-shot output."""
    posterior = smcx.bootstrap_filter(
        jr.key(314159),
        _initial,
        _transition,
        _log_observation,
        jnp.array([[0.2], [-0.4]]),
        2,
    )
    expected = smcx.ParticleFilterPosterior(
        marginal_loglik=jnp.array(-0.3544712190193707),
        filtered_particles=jnp.array([
            [[0.2972758051680527], [-0.9316924856656459]],
            [[-0.26226881164133187], [-1.0838353146267612]],
        ]),
        filtered_log_weights=jnp.array([
            [-0.4250064795171949, -1.0606391294376925],
            [-0.35289219279946654, -1.2128552713619594],
        ]),
        ancestors=jnp.array([[0, 1], [0, 1]], dtype=jnp.int32),
        ess=jnp.array([1.8271925747186781, 1.7178103810647196]),
        log_evidence_increments=jnp.array([
            -0.2728719921782969,
            -0.08159922684107379,
        ]),
    )
    _assert_tree_equal(posterior, expected)
