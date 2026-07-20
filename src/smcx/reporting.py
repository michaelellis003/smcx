# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Optional ArviZ reporting bridge using public APIs (ADR-0027)."""

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


def _host(value: object) -> np.ndarray:
    """Transfer one reporting value to NumPy."""
    return np.asarray(jax.device_get(value))


def _stack(values: Sequence[object], name: str) -> Array:
    """Stack one array-valued posterior field across runs."""
    return jnp.stack([getattr(value, name) for value in values])


def _resampled_group(
    values: Sequence[object],
    indices: Array,
    var_names: Mapping[str, str] | None,
    dims: Mapping[str, Sequence[str]] | None,
    *,
    timed: bool,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Resample aligned PyTree leaves into one named ArviZ group."""
    try:
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)
    except ValueError as error:
        raise ValueError(
            "posterior runs must have matching PyTree leaves"
        ) from error
    path_leaves, _ = jax.tree.flatten_with_path(stacked)
    group = {}
    dimensions = {}
    for path, particles in path_leaves:
        particles = cast(Array, particles)
        gather = jax.vmap(jax.vmap(getitem)) if timed else jax.vmap(getitem)
        selected = gather(particles, indices)
        selected = jnp.swapaxes(selected, 1, 2) if timed else selected
        path_name = keystr(path, simple=True, separator=".") or "theta"
        name = (var_names or {}).get(path_name, path_name)
        event_rank = particles.ndim - (3 if timed else 2)
        event_dims = list(
            (dims or {}).get(
                name, [f"{name}_dim_{axis}" for axis in range(event_rank)]
            )
        )
        if len(event_dims) != event_rank:
            raise ValueError(
                f"dims[{name!r}] must name {event_rank} event axes"
            )
        group[name] = _host(selected)
        dimensions[name] = ["time", *event_dims] if timed else event_dims
    return group, dimensions


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

    if not arviz.__version__.startswith("0."):
        arviz_base: Any = importlib.import_module("arviz_base")
        return arviz_base.from_dict(groups, dims=dimensions, attrs=attrs)

    supported = dict(groups)
    extension = supported.pop("unconstrained_posterior", None)
    result = arviz.from_dict(
        **supported,
        dims=dimensions,
        **{f"{name}_attrs": values for name, values in attrs.items()},
    )
    if extension is not None:
        result.add_groups(
            {"unconstrained_posterior": extension}, dims=dimensions
        )
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
    if isinstance(posteriors, (ParticleFilterPosterior, TemperedPosterior)):
        runs = (posteriors,)
    else:
        runs = tuple(posteriors)
    allowed = (ParticleFilterPosterior, TemperedPosterior)
    if (
        not runs
        or not isinstance(runs[0], allowed)
        or any(type(run) is not type(runs[0]) for run in runs)
    ):
        raise TypeError("posteriors must be a supported smcx posterior")

    num_chains = len(runs)
    if isinstance(runs[0], ParticleFilterPosterior):
        filter_runs = cast(tuple[ParticleFilterPosterior, ...], runs)
        log_weights = _stack(filter_runs, "filtered_log_weights")
    else:
        tempered_runs = cast(tuple[TemperedPosterior, ...], runs)
        log_weights = _stack(tempered_runs, "log_weights")
    num_particles = log_weights.shape[-1]
    draws = num_particles if num_draws is None else num_draws
    if draws <= 0:
        raise ValueError("num_draws must be positive")

    if isinstance(runs[0], ParticleFilterPosterior):
        ntime = log_weights.shape[1]
        indices = jax.vmap(systematic, in_axes=(0, 0, None))(
            jr.split(key, num_chains * ntime),
            jnp.exp(log_weights.reshape(-1, num_particles)),
            draws,
        ).reshape(num_chains, ntime, draws)
        values = [run.filtered_particles for run in filter_runs]
        stats = {
            "log_weights": jnp.swapaxes(log_weights, 1, 2)[:, None],
            "ess": _stack(filter_runs, "ess")[:, None],
            "pareto_k": jnp.stack([
                pareto_k_diagnostic(run) for run in filter_runs
            ])[:, None],
            "log_evidence_increments": _stack(
                filter_runs, "log_evidence_increments"
            )[:, None],
        }
        stat_dims = {name: ["time"] for name in stats}
        stat_dims["log_weights"] = ["particle", "time"]
        timed = True
    else:
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
            "log_weights": log_weights[:, None],
            "temperatures": _stack(tempered_runs, "temperatures")[:, None],
            "ess": _stack(tempered_runs, "ess")[:, None],
            "acceptance_rates": _stack(tempered_runs, "acceptance_rates")[
                :, None
            ],
        }
        stat_dims = {name: ["stage"] for name in stats}
        stat_dims["log_weights"] = ["particle"]
        timed = False

    posterior_group, dimensions = _resampled_group(
        values, indices, var_names, dims, timed=timed
    )
    groups = {
        "posterior": posterior_group,
        "sample_stats": {name: _host(value) for name, value in stats.items()},
    }
    if unconstrained is not None:
        u_values = (
            (unconstrained,)
            if num_chains == 1
            else tuple(cast(Sequence[object], unconstrained))
        )
        if len(u_values) != num_chains:
            raise ValueError("unconstrained must provide one value per run")
        groups["unconstrained_posterior"], u_dims = _resampled_group(
            u_values, indices, var_names, dims, timed=timed
        )
        dimensions.update(u_dims)
    if emissions is not None:
        groups["observed_data"] = {"emissions": _host(emissions)}
    dimensions.update(stat_dims)
    evidence = _host(jnp.stack([run.marginal_loglik for run in runs])).tolist()
    return _construct_arviz(
        groups,
        dimensions,
        {"posterior": {"marginal_loglik": evidence}},
    )
