# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional ArviZ reporting bridge."""

import importlib
import subprocess
import sys
from unittest.mock import patch

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from smcx.containers import ParticleFilterPosterior
from smcx.reporting import to_arviz


def _filter() -> ParticleFilterPosterior:
    particles = jnp.arange(4, dtype=jnp.float32)[None, :, None]
    particles = particles + 10 * jnp.arange(2)[:, None, None]
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
    result = to_arviz(_filter(), key=jr.key(0), num_draws=3)

    np.testing.assert_array_equal(
        _group(result, "posterior")["theta"].values[0, :, :, 0],
        np.array([[2.0, 10.0], [3.0, 10.0], [3.0, 11.0]]),
    )


def test_independent_runs_map_to_chain_and_draw_dimensions():
    post = _filter()
    other = post._replace(filtered_particles=post.filtered_particles + 100.0)
    one = _group(to_arviz(post, key=jr.key(1), num_draws=5), "posterior")
    two = _group(
        to_arviz([post, other], key=jr.key(1), num_draws=5), "posterior"
    )

    assert one["theta"].shape == (1, 5, 2, 1)
    assert two["theta"].shape == (2, 5, 2, 1)


def test_weighted_cloud_keeps_raw_source_weights_in_sample_stats():
    result = to_arviz(_filter(), key=jr.key(0), num_draws=3)
    posterior = _group(result, "posterior")
    stats = _group(result, "sample_stats")

    assert posterior.sizes["draw"] == 3
    assert stats["log_weights"].dims[-2:] == ("particle", "time")
    np.testing.assert_allclose(
        stats["log_weights"].values[0, 0],
        np.asarray(_filter().filtered_log_weights).T,
    )


def test_dense_and_structured_states_have_stable_names_and_dims():
    post = _filter()
    dense = _group(to_arviz(post, key=jr.key(2)), "posterior")
    structured_post = post._replace(
        filtered_particles={
            "position": post.filtered_particles[..., 0],
            "vector": jnp.stack(
                [
                    post.filtered_particles[..., 0],
                    post.filtered_particles[..., 0] + 1,
                ],
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


def test_filter_metadata_and_observations_land_in_standard_groups():
    result = to_arviz(
        _filter(), key=jr.key(3), emissions=jnp.array([[1.0], [2.0]])
    )
    stats = _group(result, "sample_stats")

    assert stats["ess"].dims == ("chain", "draw", "time")
    assert {"pareto_k", "log_evidence_increments"} <= set(stats.data_vars)
    assert _group(result, "posterior").attrs["marginal_loglik"] == [1.25]
    np.testing.assert_array_equal(
        _group(result, "observed_data")["emissions"], [[1.0], [2.0]]
    )


def test_optional_import_is_lazy_and_missing_extra_is_actionable(monkeypatch):
    code = (
        "import sys, smcx; assert 'arviz' not in sys.modules; "
        "assert callable(smcx.to_arviz)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    from smcx import reporting

    def missing_arviz(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(reporting.importlib, "import_module", missing_arviz)
    with pytest.raises(ImportError, match=r"smcx\[arviz\]"):
        reporting.to_arviz(_filter(), key=jr.key(4))


def test_generation_dispatch_uses_resolved_constructor():
    import arviz

    module = importlib.import_module(
        "arviz_base" if int(arviz.__version__.split(".")[0]) >= 1 else "arviz"
    )
    with patch.object(
        module, "from_dict", wraps=module.from_dict
    ) as constructor:
        to_arviz(_filter(), key=jr.key(5))
    constructor.assert_called_once()


def test_unconstrained_draws_follow_the_posterior_resampling_indices():
    result = to_arviz(
        _filter(),
        key=jr.key(0),
        num_draws=3,
        unconstrained=-_filter().filtered_particles,
    )
    constrained = _group(result, "posterior")["theta"].values
    unconstrained = _group(result, "unconstrained_posterior")["theta"].values
    np.testing.assert_array_equal(unconstrained, -constrained)
