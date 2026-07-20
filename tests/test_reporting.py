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


def test_independent_runs_map_to_chain_and_draw_dimensions():
    from smcx.reporting import to_arviz

    post = _filter()
    other = post._replace(filtered_particles=post.filtered_particles + 100.0)
    one = _group(to_arviz(post, key=jr.key(1), num_draws=5), "posterior")
    two = _group(
        to_arviz([post, other], key=jr.key(1), num_draws=5), "posterior"
    )

    assert one["theta"].shape == (1, 5, 2, 1)
    assert two["theta"].shape == (2, 5, 2, 1)


def test_weighted_cloud_keeps_raw_source_weights_in_sample_stats():
    from smcx.reporting import to_arviz

    result = to_arviz(_filter(), key=jr.key(0), num_draws=3)
    posterior = _group(result, "posterior")
    stats = _group(result, "sample_stats")

    assert posterior.sizes["draw"] == 3
    assert stats["log_weights"].dims == (
        "chain",
        "draw",
        "particle",
        "time",
    )
    np.testing.assert_allclose(
        stats["log_weights"].values[0, 0],
        np.asarray(_filter().filtered_log_weights).T,
    )


def test_dense_and_structured_states_have_stable_names_and_dims():
    from smcx.reporting import to_arviz

    post = _filter()
    dense = _group(to_arviz(post, key=jr.key(2)), "posterior")
    structured_post = post._replace(
        filtered_particles={
            "position": post.filtered_particles[..., 0],
            "vector": jnp.concatenate(
                [post.filtered_particles, post.filtered_particles + 1.0],
                axis=-1,
            ),
        }
    )
    structured = _group(
        to_arviz(
            structured_post,
            key=jr.key(2),
            var_names={"position": "x"},
            dims={"vector": ("axis",)},
        ),
        "posterior",
    )

    assert dense["theta"].dims == ("chain", "draw", "time", "theta_dim_0")
    assert set(structured.data_vars) == {"x", "vector"}
    assert structured["x"].dims == ("chain", "draw", "time")
    assert structured["vector"].dims[-1] == "axis"
