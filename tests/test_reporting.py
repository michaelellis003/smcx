# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional ArviZ reporting bridge."""

import subprocess
import sys
import warnings

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx.reporting as reporting
from smcx.containers import ParticleFilterPosterior, TemperedPosterior
from smcx.reporting import to_arviz
from smcx.types import ParticleCloud


def _filter() -> ParticleFilterPosterior:
    particles = jnp.arange(4, dtype=jnp.float32)[None, :, None]
    particles = particles + 10 * jnp.arange(2)[:, None, None]
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(1.25),
        filtered_particles=particles,
        filtered_log_weights=jnp.log(
            jnp.array([[0.05, 0.15, 0.3, 0.5], [0.5, 0.3, 0.15, 0.05]])
        ),
        ancestors=jnp.tile(jnp.arange(4), (2, 1)),
        ess=jnp.array([2.74, 2.74]),
        log_evidence_increments=jnp.array([0.5, 0.75]),
    )


def _tempered(particles: ParticleCloud) -> TemperedPosterior:
    return TemperedPosterior(
        particles=particles,
        log_weights=jnp.full(4, -jnp.log(4.0)),
        marginal_loglik=jnp.asarray(1.0),
        temperatures=jnp.array([0.0, 1.0]),
        ess=jnp.array([4.0, 4.0]),
        acceptance_rates=jnp.array([0.0, 0.8]),
        log_evidence_increments=jnp.zeros_like(jnp.array([0.0, 0.8])),
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = to_arviz([post, other], key=jr.key(1), num_draws=5)
    two = _group(result, "posterior")
    diagnostics = _group(result, "particle_diagnostics")
    assert one["theta"].shape == (1, 5, 2, 1)
    assert two["theta"].shape == (2, 5, 2, 1)
    assert np.all(two["theta"].values[1] >= 100)
    assert diagnostics["ess"].dims == ("run", "time")
    assert diagnostics["log_weights"].dims == ("run", "time", "particle")
    assert not [
        warning
        for warning in caught
        if "chain dimension" in str(warning.message)
    ]


def test_weighted_cloud_keeps_raw_source_weights_in_diagnostics():
    result = to_arviz(_filter(), key=jr.key(0), num_draws=3)
    diagnostics = _group(result, "particle_diagnostics")
    assert _group(result, "posterior").sizes["draw"] == 3
    assert diagnostics["log_weights"].dims == (
        "run",
        "time",
        "particle",
    )
    np.testing.assert_allclose(
        diagnostics["log_weights"].values[0],
        np.asarray(_filter().filtered_log_weights),
    )


def test_final_only_filter_requires_full_history():
    full = _filter()
    final_only = full._replace(
        filtered_particles=full.filtered_particles[-1:],
        filtered_log_weights=full.filtered_log_weights[-1:],
        ancestors=full.ancestors[-1:],
    )

    with pytest.raises(ValueError, match="store_history=True"):
        to_arviz(final_only, key=jr.key(0))


def test_dense_and_structured_states_have_stable_names_and_dims():
    post = _filter()
    tempered = _tempered(post.filtered_particles[0])
    dense_result = to_arviz(tempered, key=jr.key(2), num_draws=3)
    dense = _group(dense_result, "posterior")
    dense_diagnostics = _group(dense_result, "particle_diagnostics")
    tree = {"position": jnp.repeat(post.filtered_particles, 2, axis=-1)}
    structured_post = post._replace(filtered_particles=tree)
    structured = _group(
        to_arviz(
            structured_post,
            key=jr.key(2),
            var_names={"position": "x"},
            dims={"x": ("axis",)},
        ),
        "posterior",
    )
    assert dense["theta"].shape == (1, 3, 1)
    assert dense_diagnostics["log_weights"].dims == ("run", "particle")
    assert dense_diagnostics["temperatures"].dims == ("run", "stage")
    assert structured["x"].dims == ("chain", "draw", "time", "axis")


@pytest.mark.parametrize("collision_group", ["posterior", "unconstrained"])
def test_dotted_tree_paths_are_rejected(collision_group: str) -> None:
    collision = {
        "a": {"b": jnp.ones((4, 1))},
        "a.b": 2 * jnp.ones((4, 1)),
    }
    posterior = _tempered(
        collision if collision_group == "posterior" else jnp.ones((4, 1))
    )
    unconstrained = collision if collision_group == "unconstrained" else None

    with pytest.raises(
        ValueError,
        match=r"ambiguous ArviZ tree path 'a\.b'.*rename a tree key",
    ):
        to_arviz(
            posterior,
            key=jr.key(10),
            num_draws=3,
            var_names={"a.b": "renamed"},
            unconstrained=unconstrained,
        )


def test_var_names_must_resolve_to_unique_aliases() -> None:
    posterior = _tempered({
        "location": jnp.ones((4, 1)),
        "scale": 2 * jnp.ones((4, 1)),
    })
    with pytest.raises(
        ValueError,
        match=r"duplicate ArviZ variable name 'parameter'.*unique var_names",
    ):
        to_arviz(
            posterior,
            key=jr.key(11),
            var_names={"location": "parameter", "scale": "parameter"},
        )


@pytest.mark.parametrize(
    ("dimension", "timed"),
    [("chain", False), ("draw", False), ("time", True)],
)
@pytest.mark.parametrize("source", ["variable", "event"])
def test_sample_dimensions_cannot_share_particle_namespace(
    dimension: str,
    timed: bool,
    source: str,
) -> None:
    posterior = _filter() if timed else _tempered({"value": jnp.ones((4, 1))})
    name = "theta" if timed else "value"
    aliases = {name: dimension} if source == "variable" else None
    event_dims = {name: (dimension,)} if source == "event" else None
    with pytest.raises(ValueError, match="dimension"):
        to_arviz(
            posterior,
            key=jr.key(14),
            var_names=aliases,
            dims=event_dims,
        )


@pytest.mark.parametrize(
    ("dimension", "explicit"),
    [("a_dim_0", False), ("axis", True)],
)
def test_variable_names_cannot_shadow_event_dimensions(
    dimension: str,
    explicit: bool,
) -> None:
    posterior = _tempered({
        "a": jnp.ones((4, 2)),
        dimension: jnp.arange(4.0),
    })
    dims = {"a": (dimension,)} if explicit else None
    with pytest.raises(
        ValueError,
        match=rf"variable name '{dimension}'.*dimension",
    ):
        to_arviz(posterior, key=jr.key(15), dims=dims)


def test_event_dimensions_must_be_unique_per_variable() -> None:
    posterior = _tempered(jnp.ones((4, 2, 2)))
    with pytest.raises(ValueError, match=r"dims\['theta'\].*unique"):
        to_arviz(
            posterior,
            key=jr.key(16),
            dims={"theta": ("axis", "axis")},
        )


@pytest.mark.parametrize("second_size", [2, 3])
def test_shared_dimensions_require_one_size_within_group(
    second_size: int,
) -> None:
    posterior = _tempered({
        "a": jnp.ones((4, 2)),
        "state": {"b": jnp.ones((4, second_size))},
    })
    options = {"a": ("axis",), "state.b": ("axis",)}
    if second_size == 3:
        with pytest.raises(
            ValueError, match=r"dimension 'axis'.*sizes 2 and 3"
        ):
            to_arviz(posterior, key=jr.key(17), dims=options)
    else:
        group = _group(
            to_arviz(posterior, key=jr.key(18), dims=options),
            "posterior",
        )
        assert group["a"].dims == group["state.b"].dims


def test_constrained_and_unconstrained_schemas_are_group_scoped() -> None:
    posterior = _tempered(jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2))
    unconstrained = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    result = to_arviz(
        posterior,
        key=jr.key(13),
        unconstrained=unconstrained,
    )

    constrained = _group(result, "posterior")["theta"]
    u_space = _group(result, "unconstrained_posterior")["theta"]
    assert constrained.dims[-2:] == ("theta_dim_0", "theta_dim_1")
    assert u_space.dims[-1] == "theta_dim_0"


def test_particle_dimensions_do_not_leak_into_observed_emissions() -> None:
    posterior = _tempered({"emissions": jnp.ones((4, 2))})

    result = to_arviz(
        posterior,
        key=jr.key(19),
        dims={"emissions": ("state_axis",)},
        emissions=jnp.ones((3,)),
    )

    observed = _group(result, "observed_data")["emissions"]
    assert "state_axis" not in observed.dims


def test_adaptive_tempered_runs_pad_stage_diagnostics_with_validity_mask():
    particles = jnp.arange(4, dtype=jnp.float32)[:, None]
    short = TemperedPosterior(
        particles=particles,
        log_weights=jnp.full(4, -jnp.log(4.0)),
        marginal_loglik=jnp.asarray(1.0),
        temperatures=jnp.array([0.0, 1.0]),
        ess=jnp.array([4.0, 3.0]),
        acceptance_rates=jnp.array([0.0, 0.8]),
        log_evidence_increments=jnp.zeros_like(jnp.array([0.0, 0.8])),
    )
    long = TemperedPosterior(
        particles=particles + 10.0,
        log_weights=jnp.full(4, -jnp.log(4.0)),
        marginal_loglik=jnp.asarray(2.0),
        temperatures=jnp.array([0.0, 0.4, 1.0]),
        ess=jnp.array([4.0, 3.5, 3.0]),
        acceptance_rates=jnp.array([0.0, 0.7, 0.8]),
        log_evidence_increments=jnp.zeros_like(jnp.array([0.0, 0.7, 0.8])),
    )

    result = to_arviz([short, long], key=jr.key(8))
    posterior = _group(result, "posterior")
    diagnostics = _group(result, "particle_diagnostics")

    assert posterior["theta"].shape == (2, 4, 1)
    assert np.all(posterior["theta"].values[1] >= 10.0)
    assert diagnostics["stage_valid"].dims == ("run", "stage")
    np.testing.assert_array_equal(
        diagnostics["stage_valid"].values,
        np.array([[True, True, False], [True, True, True]]),
    )
    np.testing.assert_allclose(
        diagnostics["temperatures"].values,
        np.array([[0.0, 1.0, np.nan], [0.0, 0.4, 1.0]]),
    )
    assert np.isnan(diagnostics["ess"].values[0, -1])
    assert np.isnan(diagnostics["acceptance_rates"].values[0, -1])


def test_tempered_stage_diagnostics_require_aligned_lengths():
    particles = jnp.arange(4, dtype=jnp.float32)[:, None]
    posterior = TemperedPosterior(
        particles=particles,
        log_weights=jnp.full(4, -jnp.log(4.0)),
        marginal_loglik=jnp.asarray(1.0),
        temperatures=jnp.array([0.0, 1.0]),
        ess=jnp.array([4.0]),
        acceptance_rates=jnp.array([0.0, 0.8]),
        log_evidence_increments=jnp.zeros_like(jnp.array([0.0, 0.8])),
    )

    with pytest.raises(ValueError, match="matching stage lengths"):
        to_arviz(posterior, key=jr.key(9))


def test_filter_metadata_and_observations_land_in_standard_groups():
    result = to_arviz(
        _filter(), key=jr.key(3), emissions=jnp.array([[1.0], [2.0]])
    )
    diagnostics = _group(result, "particle_diagnostics")
    assert diagnostics["ess"].dims == ("run", "time")
    assert {"pareto_k", "log_evidence_increments"} <= set(diagnostics.data_vars)
    assert _group(result, "posterior").attrs["marginal_loglik"] == [1.25]
    assert _group(result, "observed_data")["emissions"].shape == (2, 1)


def test_optional_import_is_lazy_and_missing_extra_is_actionable(monkeypatch):
    code = (
        "import sys, smcx; assert 'arviz' not in sys.modules; "
        "assert callable(smcx.to_arviz)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    def missing_arviz(name):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(reporting.importlib, "import_module", missing_arviz)
    with pytest.raises(ImportError, match=r"smcx\[arviz\]"):
        reporting.to_arviz(_filter(), key=jr.key(4))


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
    bad_u = -_filter().filtered_particles[:, :-1]
    with pytest.raises(ValueError, match="particle axes"):
        to_arviz(_filter(), key=jr.key(0), unconstrained=bad_u)
