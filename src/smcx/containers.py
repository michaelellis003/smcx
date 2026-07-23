# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: structured states and additional posterior,
# streaming-checkpoint, and reporting containers.

"""Containers for particle filter state and posteriors.

All containers are :class:`~typing.NamedTuple` subclasses so they are
registered as JAX PyTrees by default.
"""

from typing import NamedTuple, Protocol, runtime_checkable

from jaxtyping import Array, Bool, Float, Int

from smcx.types import ParticleCloud, ParticleHistory, Scalar


@runtime_checkable
class ParticleFilterResult(Protocol):
    r"""Structural type for any particle filter posterior.

    Both :class:`ParticleFilterPosterior` and
    :class:`LiuWestPosterior` satisfy this protocol, so diagnostic
    functions can accept either without type errors.

    Attributes:
        marginal_loglik: Scalar estimate of
            :math:`\log p(y_{1:T})`.
        filtered_particles: Latent-state PyTree with every leaf shaped
            ``(ntime, num_particles, ...)``.
        filtered_log_weights: Normalised log weights at each step,
            shape ``(ntime, num_particles)``.
        ancestors: Resampled ancestor indices at each time step,
            shape ``(ntime, num_particles)``.
        ess: Effective sample size at each time step,
            shape ``(ntime,)``.
        log_evidence_increments: Per-step log marginal likelihood
            increments, shape ``(ntime,)``.
    """

    # Read-only properties, not bare attributes: NamedTuple fields are
    # immutable, and a mutable protocol member would make the concrete
    # posteriors fail structural checks under strict variance rules.
    @property
    def marginal_loglik(self) -> Scalar: ...

    @property
    def filtered_particles(
        self,
    ) -> ParticleHistory: ...

    @property
    def filtered_log_weights(
        self,
    ) -> Float[Array, "ntime num_particles"]: ...

    @property
    def ancestors(self) -> Int[Array, "ntime num_particles"]: ...

    @property
    def ess(self) -> Float[Array, " ntime"]: ...

    @property
    def log_evidence_increments(self) -> Float[Array, " ntime"]: ...


class ParticleState(NamedTuple):
    r"""State of a particle cloud at a single time step.

    Attributes:
        particles: Latent-state PyTree with every leaf shaped
            ``(num_particles, ...)``.
        log_weights: Normalized log importance weights,
            shape ``(num_particles,)``.
        log_marginal_likelihood: Running log marginal likelihood estimate.
    """

    particles: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    log_marginal_likelihood: Scalar


class BootstrapCheckpoint(NamedTuple):
    """Resumable state and ESS; evidence is its sum plus its correction."""

    state: ParticleState
    ess: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]


class BootstrapStepInfo(NamedTuple):
    """Ancestors, ESS, resampling flag, and conditional evidence increment."""

    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    resampled: Bool[Array, ""]
    log_evidence_increment: Float[Array, ""]


class ParticleFilterPosterior(NamedTuple):
    r"""Full output of a particle filter run.

    Follows the Dynamax ``PosteriorGSSMFiltered`` convention of storing
    the marginal log-likelihood as a scalar summary alongside the
    time-indexed arrays.

    Attributes:
        marginal_loglik: Scalar estimate of
            :math:`\log p(y_{1:T})`.
        filtered_particles: Latent-state PyTree with every leaf shaped
            ``(ntime, num_particles, ...)``. A dense state remains one
            array of shape ``(ntime, num_particles, state_dim)``.
        filtered_log_weights: Normalized log weights at each time step,
            shape ``(ntime, num_particles)``.
        ancestors: Resampled ancestor indices at each time step,
            shape ``(ntime, num_particles)``.
        ess: Effective sample size at each time step,
            shape ``(ntime,)``.
        log_evidence_increments: Per-step log marginal likelihood
            increments, shape ``(ntime,)``.  These sum to
            ``marginal_loglik``.
    """

    marginal_loglik: Scalar
    filtered_particles: ParticleHistory
    filtered_log_weights: Float[Array, "ntime num_particles"]
    ancestors: Int[Array, "ntime num_particles"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]


class GaussianFilterPosterior(NamedTuple):
    r"""Exact Gaussian filtering output.

    Attributes:
        marginal_loglik: Exact :math:`\log p(y_{1:T})`.
        predicted_means: Means before conditioning at each step,
            shape ``(ntime, state_dim)``.
        predicted_covariances: Covariances before conditioning at each
            step, shape ``(ntime, state_dim, state_dim)``.
        filtered_means: Means after conditioning at each step,
            shape ``(ntime, state_dim)``.
        filtered_covariances: Covariances after conditioning at each
            step, shape ``(ntime, state_dim, state_dim)``.
        log_evidence_increments: Per-step log marginal likelihood
            increments, shape ``(ntime,)``.
    """

    marginal_loglik: Scalar
    predicted_means: Float[Array, "ntime state_dim"]
    predicted_covariances: Float[Array, "ntime state_dim state_dim"]
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    log_evidence_increments: Float[Array, " ntime"]


class GaussianSmootherPosterior(NamedTuple):
    r"""Exact Gaussian filtering and smoothing output.

    The filtering fields are retained so a downstream method can consume
    one self-contained posterior without rerunning the forward pass.

    Attributes:
        marginal_loglik: Exact :math:`\log p(y_{1:T})`.
        predicted_means: Means before conditioning at each step.
        predicted_covariances: Covariances before conditioning at each step.
        filtered_means: Means after conditioning at each step.
        filtered_covariances: Covariances after conditioning at each step.
        log_evidence_increments: Per-step log marginal likelihood increments.
        smoothed_means: Means conditional on all observations, shape
            ``(ntime, state_dim)``.
        smoothed_covariances: Covariances conditional on all observations,
            shape ``(ntime, state_dim, state_dim)``.
    """

    marginal_loglik: Scalar
    predicted_means: Float[Array, "ntime state_dim"]
    predicted_covariances: Float[Array, "ntime state_dim state_dim"]
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    log_evidence_increments: Float[Array, " ntime"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_covariances: Float[Array, "ntime state_dim state_dim"]


class LiuWestPosterior(NamedTuple):
    r"""Full output of a Liu-West particle filter run.

    Extends :class:`ParticleFilterPosterior` with parameter samples.
    The Liu-West filter (Liu & West, 2001) jointly estimates latent
    states and static parameters using kernel density smoothing.

    Attributes:
        marginal_loglik: Scalar estimate of
            :math:`\log p(y_{1:T})`.
        filtered_particles: Particle values at each time step,
            shape ``(ntime, num_particles, state_dim)``.
        filtered_log_weights: Normalized log weights at each step,
            shape ``(ntime, num_particles)``.
        ancestors: Resampled ancestor indices at each time step,
            shape ``(ntime, num_particles)``.
        ess: Effective sample size at each time step,
            shape ``(ntime,)``.
        log_evidence_increments: Per-step log marginal likelihood
            increments, shape ``(ntime,)``.  These sum to
            ``marginal_loglik``.
        filtered_params: Parameter samples at each time step,
            shape ``(ntime, num_particles, param_dim)``.
    """

    marginal_loglik: Scalar
    filtered_particles: Float[Array, "ntime num_particles state_dim"]
    filtered_log_weights: Float[Array, "ntime num_particles"]
    ancestors: Int[Array, "ntime num_particles"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]
    filtered_params: Float[Array, "ntime num_particles param_dim"]


class TemperedPosterior(NamedTuple):
    """Tempered-SMC output.

    ``particles`` are equal-weight draws from the target (final
    resample + pi-invariant moves), so ``log_weights`` is uniform —
    kept for interface symmetry and Rao-Blackwell reminders: compute
    summaries from weighted clouds when you have them.
    ``marginal_loglik`` is the Neumaier-compensated log-evidence
    estimate.
    """

    particles: Float[Array, "num_particles dim"]
    log_weights: Float[Array, " num_particles"]
    marginal_loglik: Float[Array, ""]
    temperatures: Float[Array, " num_stages"]
    ess: Float[Array, " num_stages"]
    acceptance_rates: Float[Array, " num_stages"]


class SMC2Posterior(NamedTuple):
    """SMC² posterior over static parameters.

    The outer layer is an SMC sampler over ``num_theta`` parameter
    particles; each carries an ``num_x``-particle inner filter whose
    unbiased likelihood estimate drives the outer weights.
    ``filtered_params`` is the parameter cloud and
    ``filtered_log_weights`` its normalized outer log-weights at each
    step (final step only when ``store_history=False``) —
    the field name matches ``LiuWestPosterior`` so
    ``param_weighted_mean`` and ``param_weighted_quantile`` apply
    directly. ``marginal_loglik`` is the Neumaier-compensated SMC²
    log-evidence; E[exp(marginal_loglik)] = Z (log Zhat itself is
    Jensen-biased). ``acceptance_rates`` is the PMMH move acceptance
    at each step (0 where the outer ESS stayed above threshold and no
    move fired).
    """

    marginal_loglik: Float[Array, ""]
    filtered_params: Float[Array, "ntime num_theta param_dim"]
    filtered_log_weights: Float[Array, "ntime num_theta"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]
    acceptance_rates: Float[Array, " ntime"]
