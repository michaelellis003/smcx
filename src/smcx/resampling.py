# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Inverse-CDF resampling kernels (ADR-0004 contract, JAX port).

Every kernel takes ``(key, weights, num_samples)`` — probability-space
weights, any positive scale — and returns ``int32`` ancestor indices in
``[0, num_particles)``. Systematic, stratified, and multinomial outputs
are nondecreasing; residual returns its deterministic block followed by
iid remainder draws. Query grids are clamped strictly below 1 so a grid
point that rounds to 1.0 in float32 cannot select past the final
positive-weight slot (the ADR-0017 endpoint guard, inherited from the
MLX implementation).
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32

from smcx.types import PRNGKeyT

# Largest float32 below 1 (see module docstring).
_BELOW_ONE = 1.0 - 2.0**-24
# Avoids 0/0 on all-zero CDFs; outputs are masked upstream wherever
# that can engage.
_TINY = 1e-30


def _normalized_cdf(
    weights: Float[Array, " num_particles"],
) -> Float[Array, " num_particles"]:
    """Cumulative distribution normalized so the final entry is 1."""
    cdf = jnp.cumsum(weights)
    return cdf / jnp.maximum(cdf[-1], _TINY)


def _searchsorted_clipped(
    cdf: Float[Array, " num_particles"],
    queries: Float[Array, " num_samples"],
) -> Int32[Array, " num_samples"]:
    """Right-bisect with indices clipped into ``[0, n - 1]``."""
    idx = jnp.searchsorted(cdf, queries, side="right")
    return jnp.clip(idx, 0, cdf.shape[0] - 1).astype(jnp.int32)


def systematic(
    key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int32[Array, " num_samples"]:
    """Systematic resampling: one shared uniform, evenly spaced grid.

    Args:
        key: PRNG key.
        weights: Probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.
    """
    u0 = jax.random.uniform(key)
    grid = (u0 + jnp.arange(num_samples)) / num_samples
    queries = jnp.minimum(grid, _BELOW_ONE)
    return _searchsorted_clipped(_normalized_cdf(weights), queries)


def stratified(
    key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int32[Array, " num_samples"]:
    """Stratified resampling: one uniform per stratum.

    Args:
        key: PRNG key.
        weights: Probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.
    """
    v = jax.random.uniform(key, (num_samples,))
    grid = (jnp.arange(num_samples) + v) / num_samples
    queries = jnp.minimum(grid, _BELOW_ONE)
    return _searchsorted_clipped(_normalized_cdf(weights), queries)


def multinomial(
    key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int32[Array, " num_samples"]:
    """Multinomial (iid) resampling via sorted uniforms.

    Sorted order statistics come from normalized running sums of iid
    Exp(1) spacings (Devroye 1986, Ch. V.3.1) — O(N), no sort — using
    ``-log1p(-u)`` so a uniform that returns exactly 0 never reaches
    ``log(0)``. Sorted queries keep the ancestor gather monotone.

    Args:
        key: PRNG key.
        weights: Probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.
    """
    e = -jnp.log1p(-jax.random.uniform(key, (num_samples + 1,)))
    s = jnp.cumsum(e)
    queries = jnp.minimum(s[:-1] / jnp.maximum(s[-1], _TINY), _BELOW_ONE)
    return _searchsorted_clipped(_normalized_cdf(weights), queries)


def residual(
    key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int32[Array, " num_samples"]:
    """Residual resampling (deterministic floor + multinomial remainder).

    Static-shape formulation: the deterministic ``floor(m * w)`` copies
    and the stochastic remainder are expressed as one cumulative
    schedule so the output size stays ``num_samples`` under jit.

    Args:
        key: PRNG key.
        weights: Probability-space weights.
        num_samples: Number of ancestors to draw.

    Returns:
        Int32 ancestor indices (deterministic block first, remainder
        drawn multinomially from the residual weights).

    References:
        Douc, R., Cappe, O., and Moulines, E. (2005). Comparison of
        resampling schemes for particle filtering.
        https://doi.org/10.1109/ISPA.2005.195385
    """
    m = num_samples
    w = weights / jnp.maximum(jnp.sum(weights), _TINY)
    counts = jnp.floor(m * w)
    residual_w = m * w - counts
    # Deterministic block: positions [0, sum(counts)) filled by
    # repeating each index counts[i] times, via searchsorted on the
    # count schedule; positions >= sum(counts) get remainder draws.
    schedule = jnp.cumsum(counts)
    n_det = schedule[-1]
    positions = jnp.arange(m, dtype=weights.dtype)
    det_idx = jnp.clip(
        jnp.searchsorted(schedule, positions, side="right"),
        0,
        w.shape[0] - 1,
    ).astype(jnp.int32)
    # Draw iid candidates, then keep exactly the ``m - n_det`` entries
    # selected by the static-shape mask below. Using sorted order
    # statistics here would bias that selected suffix toward larger CDF
    # values: an arbitrary fixed subset is iid only before sorting.
    rem_queries = jnp.minimum(
        jax.random.uniform(key, (m,), dtype=weights.dtype), _BELOW_ONE
    )
    rem_idx = _searchsorted_clipped(_normalized_cdf(residual_w), rem_queries)
    return jnp.where(positions < n_det, det_idx, rem_idx)
