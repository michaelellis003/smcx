# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: local ESS and log-ESS helpers.

"""Log-space weight normalization utilities."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from smcx.types import Scalar


def log_normalize(
    log_weights: Float[Array, " num_particles"],
) -> tuple[Float[Array, " num_particles"], Scalar]:
    """Normalize log weights and return the log normalizing constant.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        A tuple ``(log_normalized, log_normalizer)`` where
        *log_normalized* has ``logsumexp == 0`` and
        *log_normalizer* is ``logsumexp(log_weights)``.
    """
    log_normalizer = jnp.logaddexp.reduce(log_weights)  # type: ignore[union-attr]
    log_normalized = log_weights - log_normalizer
    return log_normalized, log_normalizer


def normalize(
    log_weights: Float[Array, " num_particles"],
) -> Float[Array, " num_particles"]:
    """Exponentiate and normalize log weights.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        Normalized weights that sum to one.
    """
    log_norm, _ = log_normalize(log_weights)
    return jnp.exp(log_norm)


def log_ess(
    log_weights: Float[Array, " num_particles"],
) -> Scalar:
    """Log effective sample size from (possibly unnormalized) log weights.

    Shift-invariant: ``log_ess = 2*LSE(lw) - LSE(2*lw)``.

    Args:
        log_weights: Log importance weights (any normalization).

    Returns:
        ``log(ESS)`` as a scalar array.
    """
    two_lse = 2.0 * jnp.logaddexp.reduce(log_weights)  # type: ignore[union-attr]
    lse_two = jnp.logaddexp.reduce(2.0 * log_weights)  # type: ignore[union-attr]
    return two_lse - lse_two


def ess(
    log_weights: Float[Array, " num_particles"],
) -> Scalar:
    """Effective sample size ``1 / sum(w_norm**2)`` from log weights.

    Args:
        log_weights: Log importance weights (any normalization).

    Returns:
        The ESS as a scalar array in ``(0, num_particles]``.
    """
    return jnp.exp(log_ess(log_weights))
