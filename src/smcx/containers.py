# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Result containers (field-for-field parity with smcjax).

NamedTuples so ``mx.compile`` round-trips them class-intact
(engineering-practices.md); the ``ParticleFilterResult`` Protocol is
the structural type all diagnostics accept, mirroring smcjax's
Protocol + NamedTuple pattern (NamedTuples cannot subclass).
Weights are stored as normalized log-weights with a time axis; t=0 is
included so every array covers all ``ntime`` steps.
"""

from typing import NamedTuple, Protocol, runtime_checkable

import mlx.core as mx
from jaxtyping import Float, Int32


@runtime_checkable
class ParticleFilterResult(Protocol):
    """Structural type for filter outputs accepted by diagnostics."""

    marginal_loglik: Float[mx.array, ""]
    filtered_particles: Float[mx.array, "ntime num_particles state_dim"]
    filtered_log_weights: Float[mx.array, "ntime num_particles"]
    ancestors: Int32[mx.array, "ntime num_particles"]
    ess: Float[mx.array, " ntime"]
    log_evidence_increments: Float[mx.array, " ntime"]


class ParticleState(NamedTuple):
    """One-step filter carry.

    Invariant: ``log_weights`` are normalized (``logsumexp == 0``).
    """

    particles: Float[mx.array, "num_particles state_dim"]
    log_weights: Float[mx.array, " num_particles"]
    log_marginal_likelihood: Float[mx.array, ""]


class ParticleFilterPosterior(NamedTuple):
    """Filtered posterior (Dynamax ``PosteriorGSSMFiltered`` convention).

    ``log_evidence_increments`` sums to ``marginal_loglik`` (tested
    invariant; the total is Neumaier-compensated, ADR-0003).
    """

    marginal_loglik: Float[mx.array, ""]
    filtered_particles: Float[mx.array, "ntime num_particles state_dim"]
    filtered_log_weights: Float[mx.array, "ntime num_particles"]
    ancestors: Int32[mx.array, "ntime num_particles"]
    ess: Float[mx.array, " ntime"]
    log_evidence_increments: Float[mx.array, " ntime"]
