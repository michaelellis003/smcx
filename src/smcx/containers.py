# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: structured states and additional posterior,
# streaming-checkpoint, and reporting containers.

"""Containers for particle filter state and posteriors.

All containers are `typing.NamedTuple` subclasses so they are
registered as JAX PyTrees by default.
"""

from typing import NamedTuple, Protocol, runtime_checkable

from jaxtyping import Array, Bool, Float, Int, Int32, PyTree, Shaped

from smcx.types import ParticleCloud, ParticleHistory, Scalar


@runtime_checkable
class ParticleFilterResult(Protocol):
    r"""Structural type for any particle filter posterior.

    Both `smcx.containers.ParticleFilterPosterior` and
    `smcx.containers.LiuWestPosterior` satisfy this protocol, so diagnostic
    functions can accept either without type errors.

    With full history, every time axis below has length T. For a final-only
    result, particle, weight, and ancestor histories have length one while
    ESS and evidence increments retain length T.

    Attributes:
        marginal_loglik: Scalar estimate of $\log p(y_{1:T})$.
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


@runtime_checkable
class ParameterFilterResult(Protocol):
    """Structural type for results with weighted parameter histories.

    `smcx.containers.LiuWestPosterior`, `smcx.containers.SMC2Posterior`,
    `smcx.containers.IBISPosterior`, and caller-owned records satisfy this
    protocol when they provide its two fields. Parameter diagnostics validate
    array dtypes, shapes, and aligned axes at their public boundary.

    Attributes:
        filtered_params: Parameter particles, shape
            ``(ntime, num_particles, param_dim)``.
        filtered_log_weights: Normalized log weights, shape
            ``(ntime, num_particles)``.
    """

    @property
    def filtered_params(
        self,
    ) -> Float[Array, "ntime num_particles param_dim"]: ...

    @property
    def filtered_log_weights(
        self,
    ) -> Float[Array, "ntime num_particles"]: ...


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
    """Resumable normalized bootstrap-filter state.

    Attributes:
        state: Particle state with normalized log weights and finite running
            log evidence.
        ess: Effective sample size implied by ``state.log_weights``.
        log_evidence_compensation: Finite correction summed with the running
            log evidence.
    """

    state: ParticleState
    ess: Float[Array, ""]
    log_evidence_compensation: Float[Array, ""]


class BootstrapStepInfo(NamedTuple):
    """Ancestors, ESS, resampling flag, and conditional evidence increment."""

    ancestors: Int[Array, " num_particles"]
    ess: Float[Array, ""]
    resampled: Bool[Array, ""]
    log_evidence_increment: Float[Array, ""]


class ParticleFilterRecord(NamedTuple):
    """Standard per-time record returned by caller-owned filter kernels.

    Attributes:
        particles: Current particle PyTree. Every leaf has leading size
            ``num_particles``.
        log_weights: Normalized log importance weights.
        ancestors: Indices into the preceding particle cloud. Identity
            indices are conventional at time zero.
        log_evidence_increment: Current evidence increment, computed by the
            caller's algorithm.
    """

    particles: ParticleCloud
    log_weights: Float[Array, " num_particles"]
    ancestors: Int[Array, " num_particles"]
    log_evidence_increment: Float[Array, ""]


class ParticleFilterPosterior(NamedTuple):
    r"""Full output of a particle filter run.

    Follows the Dynamax ``PosteriorGSSMFiltered`` convention of storing
    the marginal log-likelihood as a scalar summary alongside the
    time-indexed arrays. ``marginal_loglik`` uses Neumaier-compensated
    accumulation; an ordinary floating-point reduction of the retained
    increments can therefore differ on long series.

    With ``store_history=False``, particle, weight, and ancestor histories
    contain only the final row; ESS and evidence increments still cover all
    T observations.

    Each retained log-weight row is the normalized filtering weight of its
    stored particle cloud, including the corrective weight after an auxiliary
    look-ahead resample.

    Attributes:
        marginal_loglik: Scalar estimate of $\log p(y_{1:T})$.
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
            increments, shape ``(ntime,)``. Their compensated sum is
            ``marginal_loglik``.
    """

    marginal_loglik: Scalar
    filtered_particles: ParticleHistory
    filtered_log_weights: Float[Array, "ntime num_particles"]
    ancestors: Int[Array, "ntime num_particles"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]


class ParticleSmootherPosterior(NamedTuple):
    """Equal-weight draws from a discrete particle FFBS approximation.

    Attributes:
        smoothed_trajectories: Draw-major latent-state PyTree. Leaf ``[j, t]``
            is draw ``j`` at time ``t``.
        backward_indices: Int32 filtering-cloud indices with shape
            ``(num_draws, ntime)``. All ``-1`` marks traced degeneracy; then
            inexact trajectory leaves are NaN and exact leaves are
            placeholders.
    """

    smoothed_trajectories: PyTree[Shaped[Array, "num_draws ntime ..."]]
    backward_indices: Int32[Array, "num_draws ntime"]


class GaussianFilterPosterior(NamedTuple):
    r"""Gaussian filtering output.

    Attributes:
        marginal_loglik: $\log p(y_{1:T})$ as computed by the
            filtering method.
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
    r"""Gaussian filtering and smoothing output.

    The filtering fields are retained so a downstream method can consume
    one self-contained posterior without rerunning the forward pass.

    Attributes:
        marginal_loglik: $\log p(y_{1:T})$ as computed by the
            filtering method.
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

    Extends `smcx.containers.ParticleFilterPosterior` with parameter samples.
    The Liu-West filter (Liu & West, 2001) jointly estimates latent
    states and static parameters using kernel density smoothing.
    ``marginal_loglik`` uses Neumaier-compensated accumulation.

    Attributes:
        marginal_loglik: Scalar estimate of $\log p(y_{1:T})$.
        filtered_particles: Particle values at each time step,
            shape ``(ntime, num_particles, state_dim)``.
        filtered_log_weights: Normalized log weights at each step,
            shape ``(ntime, num_particles)``.
        ancestors: Resampled ancestor indices at each time step,
            shape ``(ntime, num_particles)``.
        ess: Effective sample size at each time step,
            shape ``(ntime,)``.
        log_evidence_increments: Per-step log marginal likelihood
            increments, shape ``(ntime,)``. Their compensated sum is
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

    ``particles`` form an equal-weight particle approximation to the target
    after the final resample and invariant moves, so ``log_weights`` is
    uniform. The field is kept for interface symmetry and as a reminder to
    compute summaries from weighted clouds when they are available.
    ``marginal_loglik`` is the Neumaier-compensated log-evidence
    estimate, and ``log_evidence_increments`` holds the per-stage
    reweighting normalizers whose compensated sum it is.
    """

    particles: Float[Array, "num_particles dim"]
    log_weights: Float[Array, " num_particles"]
    marginal_loglik: Float[Array, ""]
    temperatures: Float[Array, " num_stages"]
    ess: Float[Array, " num_stages"]
    acceptance_rates: Float[Array, " num_stages"]
    log_evidence_increments: Float[Array, " num_stages"]


class DLMFilterPosterior(NamedTuple):
    """Conjugate unknown-variance DLM filtering output.

    Scale-free moments: ``scale_estimates[:, None, None] *
    filtered_scale_free_covariances`` is the scale matrix of the
    filtered Student-t state marginal, not its covariance. The
    filtered covariance is ``n / (n - 2)`` times that matrix, where
    ``n = scale_shapes[t]``, and exists only for ``n > 2`` — with the
    default unit prior shape it does not exist at the first step.
    ``scale_shapes`` and ``scale_estimates`` are the Inverse-Gamma
    posterior degrees of freedom and point estimate of the unknown
    observational variance; ``marginal_loglik`` is the exact sum of
    Student-t one-step forecast log densities.
    """

    filtered_means: Float[Array, "ntime state_dim"]
    filtered_scale_free_covariances: Float[Array, "ntime state_dim state_dim"]
    scale_shapes: Float[Array, " ntime"]
    scale_estimates: Float[Array, " ntime"]
    marginal_loglik: Float[Array, ""]
    log_evidence_increments: Float[Array, " ntime"]


class DLMSmootherPosterior(NamedTuple):
    """Conjugate unknown-variance DLM smoothing output.

    The first six fields retain the filter record with canonical symmetric
    covariances. Conditional covariance is ``V * C_tilde_s[t]``; marginally
    the state is Student-t with final degrees of freedom and scale matrix
    ``scale_estimates[-1] * smoothed_scale_free_covariances[t]``.
    """

    filtered_means: Float[Array, "ntime state_dim"]
    filtered_scale_free_covariances: Float[Array, "ntime state_dim state_dim"]
    scale_shapes: Float[Array, " ntime"]
    scale_estimates: Float[Array, " ntime"]
    marginal_loglik: Float[Array, ""]
    log_evidence_increments: Float[Array, " ntime"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_scale_free_covariances: Float[Array, "ntime state_dim state_dim"]


class DGLMFilterPosterior(NamedTuple):
    """WHM dynamic generalized linear model filtering output.

    ``filtered_means`` and ``filtered_covariances`` are linear-Bayes
    moment summaries of the state, not parameters of a posterior
    distribution (the DGLM specifies the state by moments only).
    ``conjugate_alphas`` and ``conjugate_betas`` are the exact
    per-step posterior conjugate parameters of the observation
    family; ``marginal_loglik`` is the sum of the closed-form
    one-step forecast log densities.
    """

    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    conjugate_alphas: Float[Array, " ntime"]
    conjugate_betas: Float[Array, " ntime"]
    marginal_loglik: Float[Array, ""]
    log_evidence_increments: Float[Array, " ntime"]


class DGLMSmootherPosterior(NamedTuple):
    """DGLM retrospective state-moment output.

    The first six fields retain the filter record and its forward semantics,
    with a canonical symmetric covariance history. Conjugate alpha and beta
    remain filtering-time quantities; they are not smoothed parameters.
    For a general DGLM, the smoothed arrays are approximate retrospective
    linear-Bayes state moments, not a smoothing distribution or credible
    intervals.
    """

    filtered_means: Float[Array, "ntime state_dim"]
    filtered_covariances: Float[Array, "ntime state_dim state_dim"]
    conjugate_alphas: Float[Array, " ntime"]
    conjugate_betas: Float[Array, " ntime"]
    marginal_loglik: Float[Array, ""]
    log_evidence_increments: Float[Array, " ntime"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_covariances: Float[Array, "ntime state_dim state_dim"]


class SMC2Posterior(NamedTuple):
    """SMC² posterior over static parameters.

    The outer layer is an SMC sampler over ``num_theta`` parameter
    particles; each carries a ``num_x``-particle inner filter whose
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


class IBISPosterior(NamedTuple):
    """IBIS posterior over a dense static parameter vector.

    Parameter and weight histories contain the post-move cloud at each datum,
    or only the final row when ``store_history=False``. ``selection_ess`` is
    measured after correction and before resampling; ``ess`` is implied by the
    stored weights and is therefore N after resampling. Acceptance is zero
    where no resampling or invariant move ran.
    """

    marginal_loglik: Float[Array, ""]
    filtered_params: Float[Array, "ntime num_particles param_dim"]
    filtered_log_weights: Float[Array, "ntime num_particles"]
    ess: Float[Array, " ntime"]
    log_evidence_increments: Float[Array, " ntime"]
    acceptance_rates: Float[Array, " ntime"]
    selection_ess: Float[Array, " ntime"]
    resampled: Bool[Array, " ntime"]
