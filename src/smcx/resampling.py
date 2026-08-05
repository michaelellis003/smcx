# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Inverse-CDF resampling kernels.

Every kernel takes ``(key, weights, num_samples)`` — probability-space
weights with at least float32 precision, at any positive scale — and returns
``int32`` ancestor indices in ``[0, num_particles)``. Systematic, stratified,
and multinomial outputs are nondecreasing; residual returns its deterministic
block followed by iid remainder draws. Calls with concrete weights require
finite, nonnegative values with positive total mass. Data-dependent validation
is skipped for traced values, where Python exceptions cannot be staged. Query
grids are clamped strictly below 1 so a grid point that rounds to 1.0 in
float32 cannot select past the final positive-weight slot. This endpoint guard
is inherited from smcx's former MLX implementation. Normalized CDFs are capped
at one, repaired with a cumulative maximum, and given an exact unit endpoint
before search. This can change fixed-key ancestors relative to releases that
passed locally inverted float32 CDFs to ``searchsorted``; already-ordered
seeded fixtures retain their draws. Numerical correctness fixes are not a
promise of cross-version random-stream identity.
"""

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, Float, Int32

from smcx._numerics import _validate_minimum_float_precision
from smcx.types import PRNGKeyT

# Avoids a zero denominator in the exponential-spacing construction.
_TINY = 1e-30


def _validate_inputs(weights: Array, num_samples: int) -> None:
    """Validate the public resampling contract where values are concrete."""
    if weights.ndim != 1:
        raise ValueError(
            f"weights must have shape (N,); got shape {weights.shape}"
        )
    if weights.shape[0] == 0:
        raise ValueError("weights must contain at least one value")
    if not jnp.issubdtype(weights.dtype, jnp.floating):
        raise ValueError(
            f"weights must have a floating dtype; got {weights.dtype}"
        )
    _validate_minimum_float_precision(weights, name="weights")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1; got {num_samples}")
    if isinstance(weights, Tracer):
        return
    all_finite = jnp.all(jnp.isfinite(weights))
    # Closed-over JAX arrays remain concrete at function entry but their
    # first value operation is traced under jit/vmap.
    if isinstance(all_finite, Tracer):
        return
    if not bool(all_finite):
        raise ValueError("weights must contain only finite values")
    if bool(jnp.any(weights < 0)):
        raise ValueError("weights must be nonnegative")
    if not bool(jnp.any(weights > 0)):
        raise ValueError("weights must have positive total mass")


def _below_one(dtype: jnp.dtype) -> Array:
    """Return the largest representable value below one for ``dtype``."""
    one = jnp.ones((), dtype=dtype)
    half_epsilon = jnp.asarray(jnp.finfo(dtype).eps / 2, dtype=dtype)
    return one - half_epsilon


def _scale_by_max(
    weights: Float[Array, " num_particles"],
) -> Float[Array, " num_particles"]:
    """Scale finite nonnegative weights without overflowing their sum."""
    # Dividing by a near-f32-max value can underflow through a reciprocal
    # optimization. A power-of-two shift changes no relative weight and is
    # exact for every finite normal or subnormal value.
    _, exponent = jnp.frexp(jnp.max(weights))
    return jnp.ldexp(weights, -exponent)


def _normalized_cdf(
    weights: Float[Array, " num_particles"],
) -> Float[Array, " num_particles"]:
    """Cumulative distribution normalized so the final entry is 1."""
    cdf = jnp.cumsum(_scale_by_max(weights))
    # Preserve every positive finite scale; use one only for the invalid
    # all-zero fallback.
    total = cdf[-1]
    denominator = jnp.where(total > 0, total, jnp.ones_like(total))
    return _monotone_cdf(cdf / denominator)


def _monotone_cdf(
    cdf: Float[Array, "*batch num_particles"],
) -> Float[Array, "*batch num_particles"]:
    """Repair prefix-rounding inversions along the particle axis."""
    one = jnp.ones((), dtype=cdf.dtype)
    endpoint = cdf[..., -1:]
    has_positive_mass = jnp.isfinite(endpoint) & (endpoint > 0)
    bounded = jnp.where(
        has_positive_mass,
        jnp.minimum(cdf, one),
        cdf,
    )
    # Promote the whole terminal plateau, not just its final entry. Otherwise
    # a sub-unit rounded endpoint followed by zero weights would create
    # artificial mass at the last (zero-weight) particle.
    bounded = jnp.where(
        has_positive_mass & (cdf == endpoint),
        one,
        bounded,
    )
    # ``maximum.accumulate`` has a pathological jax-mps 0.10.9 lowering.
    # An explicit associative prefix has the same semantics and supports
    # arbitrary leading batch axes on both maintained backends.
    repaired = jax.lax.associative_scan(
        jnp.maximum,
        bounded,
        axis=-1,
    )
    final = jnp.where(has_positive_mass[..., 0], one, endpoint[..., 0])
    return repaired.at[..., -1].set(final)


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
        key: PRNG key, consumed whole in a single scalar uniform
            draw (no split).
        weights: Finite, nonnegative probability-space weights with positive
            total mass and at least float32 precision, on any positive scale.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.

    Raises:
        ValueError: The weights or sample count are invalid. Data-dependent
            weight checks run while their values remain concrete.
    """
    _validate_inputs(weights, num_samples)
    u0 = jax.random.uniform(key)
    grid = (u0 + jnp.arange(num_samples)) / num_samples
    queries = jnp.minimum(grid, _below_one(weights.dtype))
    return _searchsorted_clipped(_normalized_cdf(weights), queries)


def stratified(
    key: PRNGKeyT,
    weights: Float[Array, " num_particles"],
    num_samples: int,
) -> Int32[Array, " num_samples"]:
    """Stratified resampling: one uniform per stratum.

    Args:
        key: PRNG key, consumed whole in one uniform draw of
            ``num_samples`` stratum offsets (no split).
        weights: Finite, nonnegative probability-space weights with positive
            total mass and at least float32 precision, on any positive scale.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.

    Raises:
        ValueError: The weights or sample count are invalid. Data-dependent
            weight checks run while their values remain concrete.
    """
    _validate_inputs(weights, num_samples)
    v = jax.random.uniform(key, (num_samples,))
    grid = (jnp.arange(num_samples) + v) / num_samples
    queries = jnp.minimum(grid, _below_one(weights.dtype))
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
    ``log(0)``. A cumulative maximum restores the mathematical monotonicity
    that parallel float32 prefix rounding can otherwise violate locally.
    Sorted queries keep the ancestor gather monotone.

    Args:
        key: PRNG key, consumed whole in one uniform draw of
            ``num_samples + 1`` exponential spacings (no split).
        weights: Finite, nonnegative probability-space weights with positive
            total mass and at least float32 precision, on any positive scale.
        num_samples: Number of ancestors to draw.

    Returns:
        Nondecreasing int32 ancestor indices.

    Raises:
        ValueError: The weights or sample count are invalid. Data-dependent
            weight checks run while their values remain concrete.
    """
    _validate_inputs(weights, num_samples)
    e = -jnp.log1p(-jax.random.uniform(key, (num_samples + 1,)))
    # ``maximum.accumulate`` has a pathological jax-mps 0.10.9 lowering.
    # The explicit associative prefix has the same semantics and stays O(N)
    # on both supported backends.
    s = jax.lax.associative_scan(jnp.maximum, jnp.cumsum(e))
    queries = jnp.minimum(
        s[:-1] / jnp.maximum(s[-1], _TINY),
        _below_one(weights.dtype),
    )
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
        key: PRNG key, consumed whole in one uniform draw for the
            residual multinomial stage (no split).
        weights: Finite, nonnegative probability-space weights with positive
            total mass and at least float32 precision, on any positive scale.
        num_samples: Number of ancestors to draw.

    Returns:
        Int32 ancestor indices (deterministic block first, remainder
        drawn multinomially from the residual weights).

    Raises:
        ValueError: The weights or sample count are invalid. Data-dependent
            weight checks run while their values remain concrete.

    References:
        Douc, R., Cappe, O., and Moulines, E. (2005). Comparison of
        resampling schemes for particle filtering.
        https://doi.org/10.1109/ISPA.2005.195385
    """
    _validate_inputs(weights, num_samples)
    m = num_samples
    scaled_weights = _scale_by_max(weights)
    total = jnp.sum(scaled_weights)
    denominator = jnp.where(total > 0, total, jnp.ones_like(total))
    w = scaled_weights / denominator
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
        jax.random.uniform(key, (m,), dtype=weights.dtype),
        _below_one(weights.dtype),
    )
    rem_idx = _searchsorted_clipped(_normalized_cdf(residual_w), rem_queries)
    return jnp.where(positions < n_det, det_idx, rem_idx)
