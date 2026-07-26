# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Optional ArviZ reporting bridge using public constructors."""

import importlib
from collections.abc import Mapping, Sequence
from operator import getitem
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


def _resolve_names(
    value: object,
    var_names: Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Resolve one tree's names without touching its leaf values."""
    path_leaves, _ = jax.tree.flatten_with_path(value)
    raw_sources: dict[str, str] = {}
    raw_names = []
    sources = []
    for path, _ in path_leaves:
        raw_name = keystr(path, simple=True, separator=".") or "theta"
        source = keystr(path) or "<root>"
        if raw_name in raw_sources:
            raise ValueError(
                f"ambiguous ArviZ tree path {raw_name!r} resolved from "
                f"{raw_sources[raw_name]} and {source}; rename a tree key"
            )
        raw_sources[raw_name] = source
        raw_names.append(raw_name)
        sources.append(source)

    resolved_sources: dict[str, str] = {}
    names = []
    for raw_name, source in zip(raw_names, sources, strict=True):
        name = (var_names or {}).get(raw_name, raw_name)
        if name in resolved_sources:
            raise ValueError(
                f"duplicate ArviZ variable name {name!r} resolved from "
                f"{resolved_sources[name]} and {source}; assign unique "
                "var_names aliases"
            )
        resolved_sources[name] = source
        names.append(name)
    return tuple(names)


def _host(value: object) -> np.ndarray:
    """Transfer one reporting value to NumPy."""
    return np.asarray(jax.device_get(value))


def _stack(values: Sequence[object], name: str) -> Array:
    """Stack one array-valued posterior field across runs."""
    return jnp.stack([getattr(value, name) for value in values])


def _tempering_stats(
    runs: Sequence[TemperedPosterior],
) -> dict[str, Array]:
    """Pad adaptive stage traces while preserving their valid lengths."""
    stage_counts = [run.temperatures.shape[0] for run in runs]
    for run, count in zip(runs, stage_counts, strict=True):
        if run.ess.shape != (count,) or run.acceptance_rates.shape != (count,):
            raise ValueError(
                "tempering diagnostics must have matching stage lengths"
            )
    max_stages = max(stage_counts)

    def padded(name: str) -> Array:
        return jnp.stack([
            jnp.pad(
                getattr(run, name),
                (0, max_stages - count),
                constant_values=jnp.nan,
            )
            for run, count in zip(runs, stage_counts, strict=True)
        ])

    return {
        "temperatures": padded("temperatures"),
        "ess": padded("ess"),
        "acceptance_rates": padded("acceptance_rates"),
        "stage_valid": jnp.arange(max_stages)[None, :]
        < jnp.asarray(stage_counts)[:, None],
    }


def _resampled_group(
    values: Sequence[object],
    indices: Array,
    num_particles: int,
    names: Sequence[str],
    dims: Mapping[str, Sequence[str]] | None,
    *,
    timed: bool,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Resample aligned PyTree leaves into one named ArviZ group."""
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)
    expected = (*indices.shape[:-1], num_particles)
    leaves = jax.tree.leaves(stacked)
    if any(leaf.shape[: len(expected)] != expected for leaf in leaves):
        raise ValueError("particle axes must match posterior weights")
    group = {}
    dimensions = {}
    for name, particles in zip(names, leaves, strict=True):
        particles = cast(Array, particles)
        gather = jax.vmap(jax.vmap(getitem)) if timed else jax.vmap(getitem)
        selected = gather(particles, indices)
        selected = jnp.swapaxes(selected, 1, 2) if timed else selected
        event_rank = particles.ndim - (3 if timed else 2)
        default = [f"{name}_dim_{axis}" for axis in range(event_rank)]
        event_dims = list((dims or {}).get(name, default))
        if len(event_dims) != event_rank:
            raise ValueError(f"dims[{name!r}] needs {event_rank} event axes")
        group[name] = _host(selected)
        dimensions[name] = ["time", *event_dims] if timed else event_dims
    return group, dimensions


def _construct_arviz(
    groups: dict[str, dict[str, np.ndarray]],
    dimensions: dict[str, list[str]],
    attrs: dict[str, dict[str, Any]],
    diagnostics: dict[str, np.ndarray],
    diagnostic_dimensions: dict[str, list[str]],
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

    xarray: Any = importlib.import_module("xarray")
    diagnostic_group = xarray.Dataset({
        name: (("run", *diagnostic_dimensions[name]), values)
        for name, values in diagnostics.items()
    })
    if not arviz.__version__.startswith("0."):
        arviz_base: Any = importlib.import_module("arviz_base")
        result = arviz_base.from_dict(groups, dims=dimensions, attrs=attrs)
        result["particle_diagnostics"] = diagnostic_group
        return result

    supported = dict(groups)
    u_group = supported.pop("unconstrained_posterior", None)
    result = arviz.from_dict(
        **supported,
        dims=dimensions,
        **{f"{name}_attrs": values for name, values in attrs.items()},
    )
    if u_group is not None:
        result.add_groups({"unconstrained_posterior": u_group}, dims=dimensions)
    result.add_groups({"particle_diagnostics": diagnostic_group})
    return result


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

    Filter draws are per-time filtering marginals, not trajectories.

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

        Adaptive tempered runs retain their independent stage counts.
        Shorter diagnostic traces are padded with NaN and identified by
        ``particle_diagnostics.stage_valid``.

    Raises:
        ImportError: ArviZ is not installed.
        TypeError: The input is empty, contains mixed posterior types, or
            contains an unsupported posterior type.
        ValueError: Draw count, particle axes, diagnostic histories,
            unconstrained values, or named event dimensions are not aligned.
    """
    if isinstance(posteriors, (ParticleFilterPosterior, TemperedPosterior)):
        runs = (posteriors,)
    else:
        runs = tuple(posteriors)
    kind = type(runs[0]) if runs else None
    if kind not in (ParticleFilterPosterior, TemperedPosterior) or any(
        type(run) is not kind for run in runs
    ):
        raise TypeError("posteriors must be a supported smcx posterior")

    num_chains = len(runs)
    if isinstance(runs[0], ParticleFilterPosterior):
        filter_runs = cast(tuple[ParticleFilterPosterior, ...], runs)
        if any(
            run.filtered_log_weights.shape[0]
            != run.log_evidence_increments.shape[0]
            for run in filter_runs
        ):
            raise ValueError(
                "to_arviz requires full particle history; rerun the filter "
                "with store_history=True"
            )
        ntime, num_particles = filter_runs[0].filtered_log_weights.shape
        values = [run.filtered_particles for run in filter_runs]
        timed = True
    else:
        tempered_runs = cast(tuple[TemperedPosterior, ...], runs)
        num_particles = tempered_runs[0].log_weights.shape[-1]
        values = [run.particles for run in tempered_runs]
        timed = False
    draws = num_particles if num_draws is None else num_draws
    if draws <= 0:
        raise ValueError("num_draws must be positive")

    u_values = None
    if unconstrained is not None:
        u_values = (
            (unconstrained,)
            if num_chains == 1
            else tuple(cast(Any, unconstrained))
        )
        if len(u_values) != num_chains:
            raise ValueError("unconstrained must provide one value per run")
    posterior_names = _resolve_names(values[0], var_names)
    u_names = (
        _resolve_names(u_values[0], var_names) if u_values is not None else None
    )

    if isinstance(runs[0], ParticleFilterPosterior):
        log_weights = _stack(filter_runs, "filtered_log_weights")
        indices = jax.vmap(systematic, in_axes=(0, 0, None))(
            jr.split(key, num_chains * ntime),
            jnp.exp(log_weights.reshape(-1, num_particles)),
            draws,
        ).reshape(num_chains, ntime, draws)
        values = [run.filtered_particles for run in filter_runs]
        stats = {
            "log_weights": log_weights,
            "ess": _stack(filter_runs, "ess"),
            "pareto_k": jnp.stack([
                pareto_k_diagnostic(run) for run in filter_runs
            ]),
            "log_evidence_increments": _stack(
                filter_runs, "log_evidence_increments"
            ),
        }
        stat_dims = {name: ["time"] for name in stats}
        stat_dims["log_weights"] = ["time", "particle"]
    else:
        log_weights = _stack(tempered_runs, "log_weights")
        if draws == num_particles:
            indices = jnp.broadcast_to(
                jnp.arange(num_particles), (num_chains, draws)
            )
        else:
            indices = jax.vmap(systematic, in_axes=(0, 0, None))(
                jr.split(key, num_chains), jnp.ones_like(log_weights), draws
            )
        values = [run.particles for run in tempered_runs]
        stats = {
            "log_weights": log_weights,
            **_tempering_stats(tempered_runs),
        }
        stat_dims = {name: ["stage"] for name in stats}
        stat_dims["log_weights"] = ["particle"]

    posterior_group, dimensions = _resampled_group(
        values, indices, num_particles, posterior_names, dims, timed=timed
    )
    groups = {
        "posterior": posterior_group,
    }
    if u_values is not None and u_names is not None:
        groups["unconstrained_posterior"], u_dims = _resampled_group(
            u_values, indices, num_particles, u_names, dims, timed=timed
        )
        dimensions.update(u_dims)
    if emissions is not None:
        groups["observed_data"] = {"emissions": _host(emissions)}
    evidence = _host(jnp.stack([run.marginal_loglik for run in runs])).tolist()
    return _construct_arviz(
        groups,
        dimensions,
        {"posterior": {"marginal_loglik": evidence}},
        {name: _host(value) for name, value in stats.items()},
        stat_dims,
    )
