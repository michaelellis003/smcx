# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Log-space weight utilities.

All weight arithmetic in smcx is log-domain (ADR-0003): these are the
only sanctioned conversions between log-weights and probability-space
weights. On fully degenerate input (every log-weight ``-inf``) these
functions do not raise — ``log_normalize`` returns a ``-inf``
normalizer and ``ess`` returns NaN; raising ``DegenerateWeightsError``
on those signals is the filter loop shell's job (design §6).

The ESS identity ``2*LSE(logw) - LSE(2*logw)`` matches BlackJAX's
``blackjax.smc.ess`` semantics, so smcjax call sites port unchanged.
"""

import mlx.core as mx
from jaxtyping import Float


def log_normalize(
    log_weights: Float[mx.array, " num_particles"],
) -> tuple[Float[mx.array, " num_particles"], Float[mx.array, ""]]:
    """Normalize log weights and return the log normalizing constant.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        A tuple ``(log_normalized, log_normalizer)`` where
        *log_normalized* has ``logsumexp == 0`` and *log_normalizer*
        is ``logsumexp(log_weights)`` (``-inf`` when every input
        weight is ``-inf`` — the degeneracy signal).
    """
    log_normalizer = mx.logsumexp(log_weights)
    log_normalized = log_weights - log_normalizer
    return log_normalized, log_normalizer


def normalize(
    log_weights: Float[mx.array, " num_particles"],
) -> Float[mx.array, " num_particles"]:
    """Exponentiate and normalize log weights.

    Args:
        log_weights: Unnormalized log importance weights.

    Returns:
        Normalized probability-space weights that sum to one.
    """
    log_normalized, _ = log_normalize(log_weights)
    return mx.exp(log_normalized)


def log_ess(
    log_weights: Float[mx.array, " num_particles"],
) -> Float[mx.array, ""]:
    """Compute the logarithm of the effective sample size.

    Uses the max-shift-safe identity
    ``log ESS = 2*logsumexp(logw) - logsumexp(2*logw)``, which never
    materializes probability-space weights (identity error <= 1.6e-6
    at N = 1e6 in float32; see docs/research/numerical-methods.md).

    Args:
        log_weights: Log importance weights, normalized or not.

    Returns:
        ``log(ESS)``, a scalar in ``[0, log num_particles]``.
    """
    return 2.0 * mx.logsumexp(log_weights) - mx.logsumexp(2.0 * log_weights)


def ess(
    log_weights: Float[mx.array, " num_particles"],
) -> Float[mx.array, ""]:
    """Compute the effective sample size ``1 / sum(W**2)``.

    ESS is a resampling trigger, not a convergence certificate
    (Elvira, Martino & Robert 2022): it reads ≈N even when the
    proposal has missed the target entirely.

    Args:
        log_weights: Log importance weights, normalized or not.

    Returns:
        The effective sample size, a scalar in ``[1, num_particles]``
        (NaN when every weight is ``-inf`` — the degeneracy signal).
    """
    return mx.exp(log_ess(log_weights))
