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
    """Structural type for filter outputs accepted by diagnostics.

    Members are read-only properties so NamedTuple posteriors (whose
    fields are immutable) satisfy the protocol structurally.
    """

    @property
    def marginal_loglik(self) -> Float[mx.array, ""]: ...

    @property
    def filtered_particles(
        self,
    ) -> Float[mx.array, "ntime num_particles state_dim"]: ...

    @property
    def filtered_log_weights(
        self,
    ) -> Float[mx.array, "ntime num_particles"]: ...

    @property
    def ancestors(self) -> Int32[mx.array, "ntime num_particles"]: ...

    @property
    def ess(self) -> Float[mx.array, " ntime"]: ...

    @property
    def log_evidence_increments(self) -> Float[mx.array, " ntime"]: ...


class ParticleState(NamedTuple):
    """One-step filter carry.

    Invariant: ``log_weights`` are normalized (``logsumexp == 0``).
    """

    particles: Float[mx.array, "num_particles state_dim"]
    log_weights: Float[mx.array, " num_particles"]
    log_marginal_likelihood: Float[mx.array, ""]


class LiuWestPosterior(NamedTuple):
    """Liu-West filter output: the Protocol fields + parameter cloud.

    ``filtered_params`` carries the per-step parameter particles;
    the state fields match :class:`ParticleFilterPosterior` exactly
    (NamedTuples cannot subclass — the Protocol is the shared type).
    """

    marginal_loglik: Float[mx.array, ""]
    filtered_particles: Float[mx.array, "ntime num_particles state_dim"]
    filtered_log_weights: Float[mx.array, "ntime num_particles"]
    ancestors: Int32[mx.array, "ntime num_particles"]
    ess: Float[mx.array, " ntime"]
    log_evidence_increments: Float[mx.array, " ntime"]
    filtered_params: Float[mx.array, "ntime num_particles param_dim"]


class TemperedPosterior(NamedTuple):
    """Tempered-SMC output (ADR-0008 item 6).

    ``particles`` are equal-weight draws from the target (final
    resample + pi-invariant moves), so ``log_weights`` is uniform —
    kept for interface symmetry and Rao-Blackwell reminders: compute
    summaries from weighted clouds when you have them.
    ``marginal_loglik`` is the Neumaier-compensated log-evidence;
    E[exp(marginal_loglik)] = Z (log Zhat itself is Jensen-biased).
    """

    particles: Float[mx.array, "num_particles dim"]
    log_weights: Float[mx.array, " num_particles"]
    marginal_loglik: Float[mx.array, ""]
    temperatures: Float[mx.array, " num_stages"]
    ess: Float[mx.array, " num_stages"]
    acceptance_rates: Float[mx.array, " num_stages"]


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
