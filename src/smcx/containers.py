# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Containers for particle filter state and posteriors.

All containers are :class:`~typing.NamedTuple` subclasses so they are
registered as JAX PyTrees by default.
"""

from typing import NamedTuple, Protocol, runtime_checkable

from jaxtyping import Array, Float, Int

from smcx.types import Scalar


@runtime_checkable
class ParticleFilterResult(Protocol):
    r"""Structural type for any particle filter posterior.

    Both :class:`ParticleFilterPosterior` and
    :class:`LiuWestPosterior` satisfy this protocol, so diagnostic
    functions can accept either without type errors.

    Attributes:
        marginal_loglik: Scalar estimate of
            :math:`\log p(y_{1:T})`.
        filtered_particles: Particle values at each time step,
            shape ``(ntime, num_particles, state_dim)``.
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
    ) -> Float[Array, "ntime num_particles state_dim"]: ...

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
        particles: Particle values, shape ``(num_particles, state_dim)``.
        log_weights: Unnormalized log importance weights,
            shape ``(num_particles,)``.
        log_marginal_likelihood: Running log marginal likelihood estimate.
    """

    particles: Float[Array, "num_particles state_dim"]
    log_weights: Float[Array, " num_particles"]
    log_marginal_likelihood: Scalar


class ParticleFilterPosterior(NamedTuple):
    r"""Full output of a particle filter run.

    Follows the Dynamax ``PosteriorGSSMFiltered`` convention of storing
    the marginal log-likelihood as a scalar summary alongside the
    time-indexed arrays.

    Attributes:
        marginal_loglik: Scalar estimate of
            :math:`\log p(y_{1:T})`.
        filtered_particles: Particle values at each time step,
            shape ``(ntime, num_particles, state_dim)``.
        filtered_log_weights: Unnormalized log weights at each time step,
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
    filtered_particles: Float[Array, "ntime num_particles state_dim"]
    filtered_log_weights: Float[Array, "ntime num_particles"]
    ancestors: Int[Array, "ntime num_particles"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]


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
        filtered_log_weights: Unnormalized log weights at each step,
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
    """Tempered-SMC output (ADR-0008 item 6).

    ``particles`` are equal-weight draws from the target (final
    resample + pi-invariant moves), so ``log_weights`` is uniform —
    kept for interface symmetry and Rao-Blackwell reminders: compute
    summaries from weighted clouds when you have them.
    ``marginal_loglik`` is the Neumaier-compensated log-evidence;
    E[exp(marginal_loglik)] = Z (log Zhat itself is Jensen-biased).
    """

    particles: Float[Array, "num_particles dim"]
    log_weights: Float[Array, " num_particles"]
    marginal_loglik: Float[Array, ""]
    temperatures: Float[Array, " num_stages"]
    ess: Float[Array, " num_stages"]
    acceptance_rates: Float[Array, " num_stages"]


class SMC2Posterior(NamedTuple):
    """SMC² output (ADR-0014): a posterior over static parameters.

    The outer layer is an SMC sampler over ``num_theta`` parameter
    particles; each carries an ``num_x``-particle inner filter whose
    unbiased likelihood estimate drives the outer weights.
    ``filtered_params`` is the parameter cloud and
    ``filtered_log_weights`` its normalized outer log-weights at each
    step (final step only when ``store_history=False``, ADR-0011) —
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
