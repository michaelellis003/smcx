# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional ArviZ reporting bridge."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from smcx.containers import ParticleFilterPosterior


def _filter() -> ParticleFilterPosterior:
    particles = jnp.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ],
        dtype=jnp.float32,
    )
    weights = jnp.array(
        [[0.05, 0.15, 0.3, 0.5], [0.5, 0.3, 0.15, 0.05]],
        dtype=jnp.float32,
    )
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(1.25),
        filtered_particles=particles,
        filtered_log_weights=jnp.log(weights),
        ancestors=jnp.tile(jnp.arange(4), (2, 1)),
        ess=jnp.array([2.74, 2.74]),
        log_evidence_increments=jnp.array([0.5, 0.75]),
    )


def _group(result, name):
    group = getattr(result, name)
    return group.ds if hasattr(group, "ds") else group


def test_fixed_key_gives_frozen_filter_draws():
    from smcx.reporting import to_arviz

    result = to_arviz(_filter(), key=jr.key(0), num_draws=3)

    np.testing.assert_array_equal(
        _group(result, "posterior")["theta"].values[0, :, :, 0],
        np.array([[2.0, 10.0], [3.0, 10.0], [3.0, 11.0]]),
    )
