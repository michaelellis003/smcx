# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local normalization, ESS, and log-ESS helpers.

"""Log-space weight normalization utilities.

Normalization uses the max-shifted formulation analysed by Blanchard,
Higham, and Higham (2021), https://doi.org/10.1093/imanum/draa038.
"""

from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, Float

from smcx._numerics import _validate_minimum_float_precision
from smcx.types import Scalar

if TYPE_CHECKING:
    _LogWeightVector: TypeAlias = Float[Array, " num_particles"]
else:
    # Runtime checking must admit malformed values so this module's public
    # validator owns the documented ValueError contract.
    _LogWeightVector: TypeAlias = Any


class _LogExpansion(NamedTuple):
    """A large shift and its small additive correction."""

    shift: Array
    correction: Array

    def resolve(self) -> Array:
        """Return the represented sum of both components."""
        return self.shift + self.correction


def _validate_log_weights(log_weights: _LogWeightVector) -> None:
    """Require a nonempty rank-one floating JAX array."""
    if not isinstance(log_weights, (jax.Array, Tracer)):
        raise ValueError(
            f"log_weights must be a JAX array; got {type(log_weights).__name__}"
        )
    if log_weights.ndim != 1:
        raise ValueError(
            f"log_weights must be rank 1; got ndim={log_weights.ndim}"
        )
    if log_weights.shape[0] == 0:
        raise ValueError("log_weights must contain at least one value")
    if not jnp.issubdtype(log_weights.dtype, jnp.floating):
        raise ValueError(
            f"log_weights must have a floating dtype; got {log_weights.dtype}"
        )
    _validate_minimum_float_precision(log_weights, name="log_weights")


def _log_normalize_axis_parts(
    log_weights: Array,
    *,
    axis: int,
) -> tuple[Array, _LogExpansion]:
    """Normalize one axis while retaining its shifted decomposition."""
    maximum = jnp.max(log_weights, axis=axis, keepdims=True)
    # Preserve the established nonfinite behavior: all -inf rows have an
    # absolute normalizer of -inf, while their normalized values are NaN.
    shift = jax.lax.stop_gradient(
        jnp.where(jnp.isfinite(maximum), maximum, jnp.zeros_like(maximum))
    )
    shifted = log_weights - shift
    shifted_log_normalizer = jax.nn.logsumexp(
        shifted,
        axis=axis,
        keepdims=True,
    )
    log_normalized = shifted - shifted_log_normalizer
    return log_normalized, _LogExpansion(
        shift=jnp.squeeze(shift, axis=axis),
        correction=jnp.squeeze(shifted_log_normalizer, axis=axis),
    )


def _log_normalize_axis(
    log_weights: Array,
    *,
    axis: int,
) -> tuple[Array, Array]:
    """Normalize one axis before restoring its absolute offset."""
    log_normalized, expansion = _log_normalize_axis_parts(
        log_weights,
        axis=axis,
    )
    return log_normalized, expansion.resolve()


def log_normalize(
    log_weights: _LogWeightVector,
) -> tuple[Float[Array, " num_particles"], Scalar]:
    """Normalize log weights and return the log normalizing constant.

    Args:
        log_weights: Unnormalized log importance weights with at least
            float32 precision.

    Returns:
        A tuple ``(log_normalized, log_normalizer)`` where
        *log_normalized* has ``logsumexp == 0`` and
        *log_normalizer* is ``logsumexp(log_weights)``.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one JAX array
            with at least float32 precision.
    """
    _validate_log_weights(log_weights)
    log_normalized, log_normalizer = _log_normalize_axis(log_weights, axis=0)
    return log_normalized, log_normalizer


def normalize(
    log_weights: _LogWeightVector,
) -> Float[Array, " num_particles"]:
    """Exponentiate and normalize log weights.

    Args:
        log_weights: Unnormalized log importance weights with at least
            float32 precision.

    Returns:
        Normalized weights that sum to one.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one JAX array
            with at least float32 precision.
    """
    log_norm, _ = log_normalize(log_weights)
    return jnp.exp(log_norm)


def log_ess(
    log_weights: _LogWeightVector,
) -> Scalar:
    """Log effective sample size from (possibly unnormalized) log weights.

    Shift-invariant: ``log_ess = 2*LSE(lw) - LSE(2*lw)``.

    Args:
        log_weights: Log importance weights (any normalization) with at
            least float32 precision.

    Returns:
        ``log(ESS)`` as a scalar array.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one JAX array
            with at least float32 precision.
    """
    _validate_log_weights(log_weights)
    log_normalized, _ = _log_normalize_axis(log_weights, axis=0)
    return -jax.nn.logsumexp(2.0 * log_normalized)


def ess(
    log_weights: _LogWeightVector,
) -> Scalar:
    """Effective sample size ``1 / sum(w_norm**2)`` from log weights.

    Args:
        log_weights: Log importance weights (any normalization) with at
            least float32 precision.

    Returns:
        The ESS as a scalar array in ``(0, num_particles]``.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one JAX array
            with at least float32 precision.
    """
    return jnp.exp(log_ess(log_weights))
