# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Optional ArviZ reporting bridge (ADR-0027).

This module uses ArviZ's public ``from_dict`` APIs; no ArviZ code is
ported. ArviZ is distributed under Apache-2.0.
"""

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.tree_util import keystr
from jaxtyping import Array

from smcx.containers import ParticleFilterPosterior, TemperedPosterior
from smcx.diagnostics import pareto_k_diagnostic
from smcx.resampling import systematic
from smcx.types import PRNGKeyT

_Posterior = ParticleFilterPosterior | TemperedPosterior


def _stack_particle_leaves(
    runs: Sequence[ParticleFilterPosterior],
) -> tuple[tuple[str, ...], tuple[Array, ...]]:
    """Stack corresponding particle-history leaves across runs."""
    path_leaves, structure = jax.tree.flatten_with_path(
        runs[0].filtered_particles
    )
    paths = tuple(
        keystr(path, simple=True, separator=".") or "theta"
        for path, _ in path_leaves
    )
    leaves_by_run = []
    for run in runs:
        leaves, run_structure = jax.tree.flatten(run.filtered_particles)
        if run_structure != structure:
            raise ValueError(
                "posterior runs must have matching PyTree structures"
            )
        leaves_by_run.append(leaves)
    stacked = tuple(
        jnp.stack([cast(Array, leaves[index]) for leaves in leaves_by_run])
        for index in range(len(paths))
    )
    return paths, stacked


def _construct_arviz(
    groups: dict[str, dict[str, np.ndarray]],
    dimensions: dict[str, list[str]],
    attrs: dict[str, dict[str, Any]],
) -> Any:
    """Construct the installed ArviZ generation's native result."""
    try:
        arviz: Any = importlib.import_module("arviz")
    except ModuleNotFoundError as error:
        if error.name != "arviz":
            raise
        raise ImportError(
            'to_arviz requires ArviZ; install it with pip install "smcx[arviz]"'
        ) from error

    major = int(arviz.__version__.split(".", maxsplit=1)[0])
    if major >= 1:
        arviz_base: Any = importlib.import_module("arviz_base")
        return arviz_base.from_dict(
            groups,
            dims=dimensions,
            attrs=attrs,
        )

    legacy_attrs = {f"{group}_attrs": values for group, values in attrs.items()}
    return arviz.from_dict(
        **groups,
        dims=dimensions,
        **legacy_attrs,
    )


def to_arviz(
    posteriors: _Posterior | Sequence[_Posterior],
    *,
    key: PRNGKeyT,
    num_draws: int | None = None,
    var_names: Mapping[str, str] | None = None,
    dims: Mapping[str, Sequence[str]] | None = None,
    emissions: object | None = None,
    unconstrained: object | None = None,
) -> Any:
    """Convert an smcx posterior to the installed ArviZ generation.

    Args:
        posteriors: One supported posterior or independent runs.
        key: Explicit key for equal-weight resampling.
        num_draws: Output draws per chain. Defaults to the particle count.
        var_names: Optional tree-path to output-variable mapping.
        dims: Optional output-variable to event-dimension mapping.
        emissions: Optional shared observations.
        unconstrained: Optional aligned u-space particle values.

    Returns:
        ``InferenceData`` on ArviZ 0.x or ``DataTree`` on ArviZ 1.x.
    """
    if isinstance(posteriors, ParticleFilterPosterior):
        runs = (posteriors,)
    else:
        runs = tuple(posteriors)
    if not runs or not all(
        isinstance(run, ParticleFilterPosterior) for run in runs
    ):
        raise TypeError("posteriors must be a supported smcx posterior")

    filter_runs = cast(tuple[ParticleFilterPosterior, ...], runs)
    paths, particle_leaves = _stack_particle_leaves(filter_runs)
    log_weights = jnp.stack([run.filtered_log_weights for run in filter_runs])
    num_chains, ntime, num_particles = log_weights.shape
    draws = num_particles if num_draws is None else num_draws
    if draws <= 0:
        raise ValueError("num_draws must be positive")

    step_keys = jr.split(key, num_chains * ntime)
    indices = jax.vmap(systematic, in_axes=(0, 0, None))(
        step_keys,
        jnp.exp(log_weights.reshape(-1, num_particles)),
        draws,
    ).reshape(num_chains, ntime, draws)
    posterior_group = {}
    dimensions = {}
    for path, particles in zip(paths, particle_leaves, strict=True):
        selected = jax.vmap(jax.vmap(lambda cloud, index: cloud[index]))(
            particles,
            indices,
        )
        selected = jnp.swapaxes(selected, 1, 2)
        name = path if var_names is None else var_names.get(path, path)
        event_rank = particles.ndim - 3
        event_dims = (
            [f"{name}_dim_{axis}" for axis in range(event_rank)]
            if dims is None or name not in dims
            else list(dims[name])
        )
        if len(event_dims) != event_rank:
            raise ValueError(
                f"dims[{name!r}] must name {event_rank} event axes"
            )
        posterior_group[name] = np.asarray(jax.device_get(selected))
        dimensions[name] = ["time", *event_dims]

    stats = {
        "log_weights": jnp.swapaxes(log_weights, 1, 2)[:, None],
        "ess": jnp.stack([run.ess for run in filter_runs])[:, None],
        "pareto_k": jnp.stack([
            pareto_k_diagnostic(run) for run in filter_runs
        ])[:, None],
        "log_evidence_increments": jnp.stack([
            run.log_evidence_increments for run in filter_runs
        ])[:, None],
    }
    groups = {
        "posterior": posterior_group,
        "sample_stats": {
            name: np.asarray(jax.device_get(value))
            for name, value in stats.items()
        },
    }
    if emissions is not None:
        groups["observed_data"] = {
            "emissions": np.asarray(jax.device_get(emissions))
        }
    dimensions["log_weights"] = ["particle", "time"]
    dimensions.update({
        name: ["time"] for name in stats if name != "log_weights"
    })
    evidence = np.asarray(
        jax.device_get(jnp.stack([run.marginal_loglik for run in filter_runs]))
    ).tolist()
    return _construct_arviz(
        groups,
        dimensions,
        {"posterior": {"marginal_loglik": evidence}},
    )
