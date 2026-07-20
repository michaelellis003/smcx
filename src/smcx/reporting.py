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
from jaxtyping import Array

from smcx.containers import ParticleFilterPosterior, TemperedPosterior
from smcx.resampling import systematic
from smcx.types import PRNGKeyT

_Posterior = ParticleFilterPosterior | TemperedPosterior


def _construct_arviz(
    groups: dict[str, dict[str, np.ndarray]],
    dimensions: dict[str, list[str]],
    attrs: dict[str, dict[str, Any]],
) -> Any:
    """Construct the installed ArviZ generation's native result."""
    try:
        arviz = importlib.import_module("arviz")
    except ModuleNotFoundError as error:
        if error.name != "arviz":
            raise
        raise ImportError(
            'to_arviz requires ArviZ; install it with pip install "smcx[arviz]"'
        ) from error

    major = int(arviz.__version__.split(".", maxsplit=1)[0])
    if major >= 1:
        arviz_base = importlib.import_module("arviz_base")
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
    if not isinstance(posteriors, ParticleFilterPosterior):
        raise TypeError("posteriors must be a supported smcx posterior")

    particles = cast(Array, posteriors.filtered_particles)
    log_weights = posteriors.filtered_log_weights
    ntime, num_particles = log_weights.shape
    draws = num_particles if num_draws is None else num_draws
    if draws <= 0:
        raise ValueError("num_draws must be positive")

    step_keys = jr.split(key, ntime)
    indices = jax.vmap(systematic, in_axes=(0, 0, None))(
        step_keys,
        jnp.exp(log_weights),
        draws,
    )
    selected = jax.vmap(lambda cloud, index: cloud[index])(
        particles,
        indices,
    )
    selected = jnp.swapaxes(selected, 0, 1)[None]

    name = "theta" if var_names is None else var_names.get("theta", "theta")
    event_rank = particles.ndim - 2
    event_dims = (
        [f"{name}_dim_{axis}" for axis in range(event_rank)]
        if dims is None or name not in dims
        else list(dims[name])
    )
    groups = {"posterior": {name: np.asarray(jax.device_get(selected))}}
    dimensions = {name: ["time", *event_dims]}
    return _construct_arviz(groups, dimensions, {})
