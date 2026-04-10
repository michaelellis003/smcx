# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Log-space weight normalization utilities."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from smcjax.types import Scalar


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
