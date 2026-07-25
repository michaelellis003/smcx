# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Host-side covariance factors for adaptive particle proposals."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def _weighted_covariance_factor(
    particles: Float[Array, "num_particles dimension"],
    weights: Float[Array, " num_particles"],
    *,
    scale: float,
) -> Float[Array, "dimension dimension"]:
    """Return a target-dtype factor of a scaled weighted covariance.

    The weighted covariance is evaluated in float64 on the host to avoid
    cancellation at ordinary posterior offsets. Well-conditioned matrices
    are factored without regularization. Failed factors retain the existing
    trace-relative jitter schedule; only the final fallback imposes a
    dtype-epsilon variance floor. That floor has squared parameter units
    (relative to one parameter unit), remains positive for a zero-trace
    cloud, and is not used when a target-dtype factor already exists.
    """
    x = np.asarray(particles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mean = w @ x
    centered = x - mean
    covariance = scale * ((centered * w[:, None]).T @ centered)

    dimension = covariance.shape[0]
    identity = np.eye(dimension)
    relative_base = np.trace(covariance) / max(dimension, 1)

    def target_factor(matrix: np.ndarray) -> Array | None:
        """Factor one candidate and check its target-dtype diagonal."""
        try:
            lower = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            return None
        candidate = jnp.asarray(lower, dtype=particles.dtype)
        candidate_host = np.asarray(candidate)
        if not np.all(np.isfinite(candidate_host)):
            return None
        if not np.all(np.diag(candidate_host) > 0.0):
            return None
        return candidate

    for jitter_scale in (0.0, 1e-8, 1e-6, 1e-4):
        factor = target_factor(
            covariance + relative_base * jitter_scale * identity
        )
        if factor is not None:
            return factor

    # Covariance has squared parameter units. Machine epsilon is therefore
    # interpreted as eps * (one parameter unit)**2, an absolute last-resort
    # floor that does not disappear with a zero or underflowed trace.
    variance_floor = max(
        relative_base * 1e-6,
        float(jnp.finfo(particles.dtype).eps),
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, variance_floor, None)
    regularized = (eigenvectors * eigenvalues) @ eigenvectors.T
    factor = target_factor(regularized + variance_floor * identity)
    if factor is None:  # pragma: no cover - the positive floor is definitive
        raise np.linalg.LinAlgError(
            "regularized covariance is not positive definite"
        )
    return factor
