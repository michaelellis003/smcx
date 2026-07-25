# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS and log-ESS helpers.

"""Log-space weight normalization utilities."""

from typing import TYPE_CHECKING, Any, TypeAlias

import jax
import jax.numpy as jnp
from jax.core import Tracer
from jaxtyping import Array, Float

from smcx.types import Scalar

if TYPE_CHECKING:
    _LogWeightVector: TypeAlias = Float[Array, " num_particles"]
else:
    # Runtime checking must admit malformed values so this module's public
    # validator owns the documented ValueError contract.
    _LogWeightVector: TypeAlias = Any


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


def log_normalize(
    log_weights: _LogWeightVector,
) -> tuple[Float[Array, " num_particles"], Scalar]:
    """Normalize log weights and return the log normalizing constant.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        A tuple ``(log_normalized, log_normalizer)`` where
        *log_normalized* has ``logsumexp == 0`` and
        *log_normalizer* is ``logsumexp(log_weights)``.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one floating JAX
            array.
    """
    _validate_log_weights(log_weights)
    log_normalizer = jnp.logaddexp.reduce(log_weights)  # type: ignore[union-attr]
    log_normalized = log_weights - log_normalizer
    return log_normalized, log_normalizer


def normalize(
    log_weights: _LogWeightVector,
) -> Float[Array, " num_particles"]:
    """Exponentiate and normalize log weights.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        Normalized weights that sum to one.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one floating JAX
            array.
    """
    log_norm, _ = log_normalize(log_weights)
    return jnp.exp(log_norm)


def log_ess(
    log_weights: _LogWeightVector,
) -> Scalar:
    """Log effective sample size from (possibly unnormalized) log weights.

    Shift-invariant: ``log_ess = 2*LSE(lw) - LSE(2*lw)``.

    Args:
        log_weights: Log importance weights (any normalization).

    Returns:
        ``log(ESS)`` as a scalar array.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one floating JAX
            array.
    """
    _validate_log_weights(log_weights)
    two_lse = 2.0 * jnp.logaddexp.reduce(log_weights)  # type: ignore[union-attr]
    lse_two = jnp.logaddexp.reduce(2.0 * log_weights)  # type: ignore[union-attr]
    return two_lse - lse_two


def ess(
    log_weights: _LogWeightVector,
) -> Scalar:
    """Effective sample size ``1 / sum(w_norm**2)`` from log weights.

    Args:
        log_weights: Log importance weights (any normalization).

    Returns:
        The ESS as a scalar array in ``(0, num_particles]``.

    Raises:
        ValueError: ``log_weights`` is not a nonempty rank-one floating JAX
            array.
    """
    return jnp.exp(log_ess(log_weights))
