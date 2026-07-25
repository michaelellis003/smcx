# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: corrected Pareto-k and tail-ESS semantics,
# genealogy and scoring diagnostics, and structured-state support.

r"""Diagnostic utilities for particle filter posteriors.

Posterior summaries (Vehtari: *report posterior summaries with
uncertainty*; McElreath: *always report intervals, not just means*):

- `smcx.weighted_mean` — weighted posterior mean at each time step
- `smcx.weighted_variance` — weighted posterior variance
- `smcx.weighted_quantile` — weighted quantiles for credible
  intervals
- `smcx.param_weighted_mean` — weighted parameter mean (Liu-West)
- `smcx.param_weighted_quantile` — weighted parameter quantiles

Computational faithfulness (Vehtari: *can we trust the computation?*):

- `smcx.particle_diversity` — fraction of surviving time-zero lineages
- `smcx.reconstruct_trajectories` — genealogy-traced particle paths
- `smcx.log_ml_variance` — single-run log-evidence variance
- `smcx.log_ml_increments` — per-step evidence contributions
- `smcx.pareto_k_diagnostic` — per-step Pareto-k reliability
- `smcx.tail_ess` — ESS for tail quantiles
- `smcx.diagnose` — summary diagnostics with warnings

Model comparison:

- `smcx.log_bayes_factor` — log Bayes factor between two models
- `smcx.replicated_log_ml` — Monte Carlo variability of log-ML
- `smcx.cumulative_log_score` — running predictive log-score

Posterior predictive checks:

- `smcx.posterior_predictive_sample` — one-step-ahead predictions

Scoring rules:

- `smcx.crps` — Continuous Ranked Probability Score

Array-returning functions are pure, stateless, and JIT-compatible for
static shape and control arguments. Genealogy and predictive operations
preserve structured latent-state PyTrees; Euclidean summaries require a
dense ``(T, N, D)`` particle history. Parameter summaries accept
`smcx.containers.LiuWestPosterior` and
`smcx.containers.SMC2Posterior`. `smcx.diagnose` converts results to Python
scalars and strings, so it is intentionally host-only.
Axis-sensitive state, parameter, predictive, and genealogy diagnostics
validate the time, particle, and event axes they consume. State, parameter,
and predictive summaries accept final-only histories; genealogy diagnostics
require full history.
"""

import math
import operator
from numbers import Real
from typing import TYPE_CHECKING, Any, SupportsFloat, TypeAlias, cast

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree, vmap
from jax.core import Tracer
from jaxtyping import Array, Float, Int

from smcx._numerics import _neumaier_prefix_sum
from smcx._utils import (
    _array_tree_signature,
    _gather_particles,
    _validate_emission,
    _validate_particle_cloud,
    _validate_state_tree,
    _weighted_quantile_1d,
)
from smcx.containers import (
    LiuWestPosterior,
    ParticleFilterResult,
    SMC2Posterior,
)
from smcx.types import (
    EmissionSampler,
    ParticleCloud,
    ParticleHistory,
    PRNGKeyT,
    Scalar,
    TransitionSampler,
    _ReplicatedLogMLFn,
)
from smcx.weights import normalize

_PRIOR_STRENGTH = 10
"""Effective sample size of the weakly informative prior on k."""

_PRIOR_K = 0.5
"""Prior mean for k.  Vehtari et al. (2024) use 0.5."""

_EXC_FLOOR = 1e-100
"""Exceedance floor below which the tail is treated as degenerate."""

# Static contracts stay narrow; runtime validators admit malformed values.
if TYPE_CHECKING:
    _DiagnosticVector: TypeAlias = Float[Array, " num_values"]
    _DiagnosticScalar: TypeAlias = Scalar
    _RealArgument: TypeAlias = SupportsFloat
    _IntegerArgument: TypeAlias = int
else:
    _DiagnosticVector: TypeAlias = Any
    _DiagnosticScalar: TypeAlias = object
    _RealArgument: TypeAlias = object
    _IntegerArgument: TypeAlias = object


def _require_float_vector(
    value: object,
    *,
    name: str,
    dimension: str,
) -> Array:
    if not isinstance(value, (jax.Array, Tracer)):
        raise ValueError(f"{name} must be a JAX array")
    if value.ndim != 1 or value.shape[0] == 0:
        raise ValueError(
            f"{name} must have shape ({dimension},) with "
            f"{dimension} >= 1; got {value.shape}"
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError(
            f"{name} must have a floating dtype; got {value.dtype}"
        )
    return value


def _require_float_scalar(value: object, *, name: str) -> Array:
    if isinstance(value, float):
        scalar = jnp.asarray(value)
    elif isinstance(value, (jax.Array, Tracer)):
        scalar = value
    else:
        raise ValueError(f"{name} must be a floating scalar")
    if scalar.ndim != 0 or not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise ValueError(
            f"{name} must be a floating scalar; got shape {scalar.shape} "
            f"and dtype {scalar.dtype}"
        )
    return scalar


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    domain: str,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be {domain}; got {value!r}")
    try:
        integer = operator.index(cast(Any, value))
    except TypeError as error:
        raise ValueError(f"{name} must be {domain}; got {value!r}") from error
    if integer < minimum:
        raise ValueError(f"{name} must be {domain}; got {integer}")
    return integer


def _validate_quantile_levels(q: object) -> Array:
    levels = _require_float_vector(
        q,
        name="q",
        dimension="num_quantiles",
    )
    if not isinstance(levels, Tracer) and not bool(
        jnp.all((levels >= 0.0) & (levels <= 1.0))
    ):
        raise ValueError("q values must be in [0, 1]")
    return levels


def _require_dense_particle_history(
    posterior: ParticleFilterResult,
    *,
    diagnostic: str,
) -> Float[Array, "ntime num_particles state_dim"]:
    """Return dense particles or explain the structured-state boundary."""
    expected_axes = _validate_log_weight_history(posterior, diagnostic)
    particles = posterior.filtered_particles
    is_array = isinstance(particles, (jax.Array, Tracer))
    is_float = is_array and jnp.issubdtype(particles.dtype, jnp.floating)
    if not is_array or particles.ndim != 3 or not is_float:
        raise TypeError(
            f"{diagnostic} requires posterior.filtered_particles to be a "
            "dense array with floating dtype and shape (T, N, D); select or "
            "project a structured-state leaf before using this Euclidean "
            "diagnostic"
        )
    if particles.shape[2] == 0:
        raise ValueError(
            f"{diagnostic} requires posterior.filtered_particles to have "
            "shape (T, N, state_dim) with state_dim >= 1; "
            f"got {particles.shape}"
        )
    if particles.shape[:2] != expected_axes:
        raise ValueError(
            f"{diagnostic} requires posterior.filtered_particles time and "
            "particle axes to match posterior.filtered_log_weights "
            f"{expected_axes}; got {particles.shape[:2]}"
        )
    return particles


def _require_history_array(
    value: object,
    name: str,
    ndim: int,
    shape_contract: str,
    diagnostic: str,
    *,
    integer: bool = False,
) -> Array:
    """Require a nonempty floating or integer history array."""
    dtype_kind = jnp.integer if integer else jnp.floating
    dtype_name = "integer" if integer else "floating"
    if (
        not isinstance(value, (jax.Array, Tracer))
        or value.ndim != ndim
        or 0 in value.shape
        or not jnp.issubdtype(value.dtype, dtype_kind)
    ):
        raise ValueError(
            f"{diagnostic} requires {name} to be a nonempty {dtype_name} "
            f"JAX array with {shape_contract}; got shape "
            f"{getattr(value, 'shape', None)} and dtype "
            f"{getattr(value, 'dtype', None)}"
        )
    return value


def _validate_log_weight_history(
    posterior: ParticleFilterResult | SMC2Posterior,
    diagnostic: str,
) -> tuple[int, int]:
    """Validate the common nonempty ``(T, N)`` log-weight history."""
    log_weights = _require_history_array(
        posterior.filtered_log_weights,
        "posterior.filtered_log_weights",
        2,
        "shape (T, N)",
        diagnostic,
    )
    return log_weights.shape[0], log_weights.shape[1]


def _validate_float_trace(values: object, name: str, diagnostic: str) -> int:
    """Validate one nonempty floating time trace."""
    trace = _require_history_array(values, name, 1, "shape (T,)", diagnostic)
    return trace.shape[0]


def _validate_particle_axes(
    particles: object,
    expected_axes: tuple[int, int],
    diagnostic: str,
) -> None:
    """Require every latent-state leaf to share ``(T, N)`` leading axes."""
    history = _array_tree_signature(
        particles,
        name="posterior.filtered_particles",
    )
    for path, shape in zip(history.paths, history.shapes, strict=True):
        if len(shape) < 2:
            raise ValueError(
                f"{diagnostic} requires posterior.filtered_particles leaf "
                f"{path} to have leading time and particle axes; got {shape}"
            )
        if shape[:2] != expected_axes:
            raise ValueError(
                f"{diagnostic} requires posterior.filtered_particles leaf "
                f"{path} time and particle axes to match "
                f"posterior.filtered_log_weights {expected_axes}; "
                f"got {shape[:2]}"
            )


def _require_parameter_history(
    posterior: LiuWestPosterior | SMC2Posterior,
    diagnostic: str,
) -> Float[Array, "ntime num_particles param_dim"]:
    """Validate and return a parameter history aligned with its weights."""
    axes = _validate_log_weight_history(posterior, diagnostic)
    params = _require_history_array(
        posterior.filtered_params,
        "posterior.filtered_params",
        3,
        "shape (T, N, param_dim), param_dim >= 1",
        diagnostic,
    )
    if params.shape[:2] != axes:
        raise ValueError(
            f"{diagnostic} requires posterior.filtered_params time and "
            "particle axes to match posterior.filtered_log_weights "
            f"{axes}; got {params.shape[:2]}"
        )
    return params


def _validate_particle_result_axes(
    posterior: ParticleFilterResult,
    diagnostic: str,
    *,
    particles: bool = False,
    ancestors: bool = False,
    full_history: bool = False,
    ess: bool = False,
) -> tuple[int, int]:
    """Validate only the posterior axes consumed by one diagnostic."""
    axes = _validate_log_weight_history(posterior, diagnostic)
    trace_steps = axes[0]
    if full_history or ess:
        trace_steps = _validate_float_trace(
            posterior.log_evidence_increments,
            "posterior.log_evidence_increments",
            diagnostic,
        )
    if full_history and axes[0] != trace_steps:
        raise ValueError(
            f"{diagnostic} requires full particle history; rerun the filter "
            "with store_history=True"
        )
    if particles:
        _validate_particle_axes(posterior.filtered_particles, axes, diagnostic)
    if ancestors:
        ancestry = _require_history_array(
            posterior.ancestors,
            "posterior.ancestors",
            2,
            "shape (T, N)",
            diagnostic,
            integer=True,
        )
        if ancestry.shape != axes:
            raise ValueError(
                f"{diagnostic} requires posterior.ancestors time and "
                "particle axes to match posterior.filtered_log_weights "
                f"{axes}; got {ancestry.shape}"
            )
    if ess:
        ess_steps = _validate_float_trace(
            posterior.ess, "posterior.ess", diagnostic
        )
        if ess_steps != trace_steps:
            raise ValueError(
                f"{diagnostic} requires posterior.ess time axis to match "
                "posterior.log_evidence_increments "
                f"({trace_steps}); got {ess_steps}"
            )
    return axes


def _weighted_mean_field(
    log_weights: Float[Array, "ntime num_particles"],
    field: Float[Array, "ntime num_particles dim"],
) -> Float[Array, "ntime dim"]:
    """Compute weighted mean of a (ntime, N, D) field.

    Args:
        log_weights: Log weights, shape ``(ntime, num_particles)``.
        field: Values to average, shape ``(ntime, num_particles, D)``.

    Returns:
        Weighted means, shape ``(ntime, D)``.
    """
    weights = vmap(normalize)(log_weights)
    return jnp.einsum("tn,tnd->td", weights, field)


def _weighted_quantile_field(
    log_weights: Float[Array, "ntime num_particles"],
    field: Float[Array, "ntime num_particles dim"],
    q: Float[Array, " num_quantiles"],
) -> Float[Array, "ntime num_quantiles dim"]:
    """Compute weighted quantiles of a (ntime, N, D) field.

    Args:
        log_weights: Log weights, shape ``(ntime, num_particles)``.
        field: Values, shape ``(ntime, num_particles, D)``.
        q: Quantile levels in [0, 1].

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, D)``.
    """
    weights = vmap(normalize)(log_weights)

    def _quantile_one_time(
        field_t: Float[Array, "num_particles dim"],
        weights_t: Float[Array, " num_particles"],
    ) -> Float[Array, "num_quantiles dim"]:
        """Compute quantiles for one time step, all dims."""
        return vmap(_weighted_quantile_1d, in_axes=(1, None, None))(
            field_t, weights_t, q
        ).T

    return vmap(_quantile_one_time)(field, weights)


def weighted_mean(
    posterior: ParticleFilterResult,
) -> Float[Array, "ntime state_dim"]:
    r"""Compute the weighted mean of particles at each time step.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Weighted means, shape ``(ntime, state_dim)``.

    Raises:
        TypeError: The posterior has structured rather than dense particles.
        ValueError: Consumed posterior arrays are malformed or misaligned.
    """
    particles = _require_dense_particle_history(
        posterior, diagnostic="weighted_mean"
    )
    return _weighted_mean_field(posterior.filtered_log_weights, particles)


def weighted_variance(
    posterior: ParticleFilterResult,
) -> Float[Array, "ntime state_dim"]:
    r"""Compute the weighted variance of particles at each time step.

    Uses the formula $V = \sum_i w_i (x_i - \mu)^2$, where $\mu$ is the
    weighted mean.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Weighted variances, shape ``(ntime, state_dim)``.

    Raises:
        TypeError: The posterior has structured rather than dense particles.
        ValueError: Consumed posterior arrays are malformed or misaligned.
    """
    particles = _require_dense_particle_history(
        posterior, diagnostic="weighted_variance"
    )
    means = _weighted_mean_field(posterior.filtered_log_weights, particles)
    deviations = particles - means[:, None, :]
    return _weighted_mean_field(
        posterior.filtered_log_weights,
        deviations**2,
    )


def weighted_quantile(
    posterior: ParticleFilterResult,
    q: _DiagnosticVector,
) -> Float[Array, "ntime num_quantiles state_dim"]:
    r"""Compute weighted quantiles of particles at each time step.

    Uses a sorted resampling approach for JIT compatibility:
    sorts particles, computes cumulative weights, and interpolates.

    Args:
        posterior: Particle filter posterior output.
        q: Quantile levels in [0, 1], e.g. ``jnp.array([0.025, 0.975])``
            for a 95% credible interval. Values are checked eagerly;
            callers passing traced levels must preserve this domain.

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, state_dim)``.

    Raises:
        TypeError: The posterior has structured rather than dense particles.
        ValueError: Posterior arrays or ``q`` are malformed/misaligned, or an
            eager quantile lies outside [0, 1].
    """
    q = _validate_quantile_levels(q)
    particles = _require_dense_particle_history(
        posterior, diagnostic="weighted_quantile"
    )
    return _weighted_quantile_field(
        posterior.filtered_log_weights, particles, q
    )


def log_ml_increments(
    posterior: ParticleFilterResult,
) -> Float[Array, " ntime"]:
    r"""Extract per-step log marginal likelihood increments.

    The marginal log-likelihood can be decomposed as:

    $$
    \log p(y_{1:T}) = \sum_{t=1}^T
        \log p(y_t \mid y_{1:t-1})
    $$

    This function returns the individual increments, which diagnose
    which observations are hardest for the model.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Per-step evidence increments, shape ``(ntime,)``.  These sum
        to ``posterior.marginal_loglik``.

    Raises:
        ValueError: ``posterior.log_evidence_increments`` is not a nonempty
            floating array with shape ``(T,)``.
    """
    _validate_float_trace(
        posterior.log_evidence_increments,
        "posterior.log_evidence_increments",
        "log_ml_increments",
    )
    return posterior.log_evidence_increments


def particle_diversity(
    posterior: ParticleFilterResult,
) -> Float[Array, " ntime"]:
    r"""Compute the fraction of surviving time-zero particle lineages.

    The diagnostic composes parent indices across time to obtain each
    particle's time-zero ancestor, or Eve index. A value near 1 means
    most initial lineages survive; a value near 0 means the particle
    paths have coalesced onto few initial lineages. It does not measure
    uniqueness of current state values or only the latest parent array.

    Uses an indicator-based method (not ``jnp.unique``) for JIT
    compatibility.

    Args:
        posterior: Particle filter posterior with full history
            (``store_history=True``, the default).

    Returns:
        Fraction of represented time-zero lineages in [0, 1] at each
        time step, shape ``(ntime,)``. The first value is one by
        construction.

    Raises:
        ValueError: Consumed posterior arrays are malformed or misaligned,
            or the posterior retains only its final particle cloud.
    """
    _validate_particle_result_axes(
        posterior,
        "particle_diversity",
        ancestors=True,
        full_history=True,
    )
    eves = _eve_indices(posterior.ancestors)
    num_particles = eves.shape[1]

    def _diversity_one_step(
        eve: Int[Array, " num_particles"],
    ) -> Float[Array, ""]:
        """Count represented time-zero lineages at one time step."""
        sorted_eve = jnp.sort(eve)
        is_unique = jnp.concatenate([
            jnp.array([True]),
            sorted_eve[1:] != sorted_eve[:-1],
        ])
        return jnp.sum(is_unique) / num_particles

    return vmap(_diversity_one_step)(eves)


def reconstruct_trajectories(
    posterior: ParticleFilterResult,
) -> ParticleHistory:
    r"""Trace each surviving particle's full path through the genealogy.

    The filter's per-step particle clouds approximate the filtering
    distributions; the paths obtained by following each final
    particle's ancestor indices backwards approximate the smoothing
    distribution (with the usual path-degeneracy caveat: early
    segments collapse onto few distinct values as the genealogy
    coalesces). Output ``[t, n]`` is the state at time t of the
    lineage that ends at particle n at the final step, so weighting
    trajectories by the final-step weights gives smoothing
    expectations.

    Args:
        posterior: Particle filter posterior with full history
            (``store_history=True``, the default).

    Returns:
        A latent-state PyTree matching ``filtered_particles``. Every
        leaf has shape ``(ntime, num_particles, ...)``.

    Raises:
        ValueError: Consumed posterior arrays are malformed or misaligned,
            or the posterior retains only its final particle cloud.
    """
    _validate_particle_result_axes(
        posterior,
        "reconstruct_trajectories",
        particles=True,
        ancestors=True,
        full_history=True,
    )
    ancestors = posterior.ancestors
    num_particles = ancestors.shape[1]
    final_idx = jnp.arange(num_particles, dtype=ancestors.dtype)

    def _back(idx: Array, anc_t: Array) -> tuple[Array, Array]:
        """One backward step: parent indices of the current lineage."""
        return anc_t[idx], idx

    # Walking t = T-1, ..., 1 yields the selector for t-1 at each step.
    last, selectors = lax.scan(_back, final_idx, ancestors[1:], reverse=True)
    selectors = jnp.concatenate([last[None], selectors], axis=0)
    times = jnp.arange(ancestors.shape[0])[:, None]
    return tree.map(
        lambda leaf: leaf[times, selectors],
        posterior.filtered_particles,
    )


def _eve_indices(
    ancestors: Int[Array, "ntime num_particles"],
) -> Int[Array, "ntime num_particles"]:
    """Time-0 ancestor (Eve) index of every particle at every step."""
    num_particles = ancestors.shape[1]
    eve_0 = jnp.arange(num_particles, dtype=ancestors.dtype)

    def _fwd(eve: Array, anc_t: Array) -> tuple[Array, Array]:
        eve_t = eve[anc_t]
        return eve_t, eve_t

    _, eves = lax.scan(_fwd, eve_0, ancestors[1:])
    return jnp.concatenate([eve_0[None], eves], axis=0)


def _eve_class_mass_sq(
    log_w_t: Float[Array, " num_particles"],
    eve_t: Int[Array, " num_particles"],
) -> Float[Array, ""]:
    """Sum over Eve classes of squared normalized-weight mass."""
    num_particles = log_w_t.shape[0]
    weights = normalize(log_w_t)
    class_mass = jnp.zeros(num_particles).at[eve_t].add(weights)
    est = jnp.sum(class_mass**2)
    # One Eve class left: the run carries no variance information, so
    # return infinity rather than saturating the estimate at 1.
    sorted_eve = jnp.sort(eve_t)
    num_classes = 1 + jnp.sum(sorted_eve[1:] != sorted_eve[:-1])
    return jnp.where(num_classes > 1, est, jnp.inf)


def log_ml_variance(
    posterior: ParticleFilterResult,
    lag: _IntegerArgument | None = None,
) -> Float[Array, " ntime"]:
    r"""Estimate the variance of the log-evidence from a single run.

    Implements the genealogy-based estimator of Chan and Lai (2013)
    and Lee and Whiteley (2018): with Eve variables $B_t^n$
    tracing particle n at time t to its time-0 ancestor, the estimate
    at time t is

    $$
    \widehat{V}_t = \sum_{e} \Big( \sum_{n : B_t^n = e}
        W_t^n \Big)^2,
    $$

    the sum over Eve classes of squared normalized-weight mass. Under
    multinomial resampling, for large N this estimates the variance of
    ``marginal_loglik`` up to time t, and it costs one filter run where
    `smcx.replicated_log_ml` costs R. `ParticleFilterResult` does not record
    its resampling scheme, so callers must establish that precondition.
    Results from systematic, stratified, residual, or custom resampling are
    uncalibrated genealogy heuristics.

    The estimate degenerates as the genealogy coalesces; once a single Eve
    class remains the function returns ``inf`` for that step (the run carries
    no variance information there).

    Passing ``lag`` uses the ancestor at time ``t - lag`` instead of time 0.
    This ancestry-window heuristic is motivated by fixed-lag variance
    estimation, but smcx has no independent calibration result for applying
    it to this log-normalizer class-mass formula. Treat lagged values as
    exploratory diagnostics rather than calibrated variance estimates.

    Args:
        posterior: Particle filter posterior with full history
            (``store_history=True``, the default).
        lag: Optional ancestry-window lag. ``None`` uses time-0 Eves.

    Returns:
        Per-step variance estimates, shape ``(ntime,)``.

    Raises:
        ValueError: Consumed posterior arrays are malformed or misaligned,
            the lag is not a nonnegative integer, or the posterior retains
            only its final particle cloud.

    References:
        Chan, H. P., and Lai, T. L. (2013). A general theory of particle
        filters in hidden Markov models and some applications.
        https://arxiv.org/abs/1312.5114
        Lee, A., and Whiteley, N. (2018). Variance estimation in the particle
        filter. https://doi.org/10.1093/biomet/asy028
        Olsson, J., and Douc, R. (2019). Numerically stable online estimation
        of variance in particle filters. https://arxiv.org/abs/1701.01001
    """
    if lag is not None:
        lag = _require_integer(
            lag,
            name="lag",
            minimum=0,
            domain="nonnegative integer",
        )

    _validate_particle_result_axes(
        posterior,
        "log_ml_variance",
        ancestors=True,
        full_history=True,
    )
    ancestors = posterior.ancestors
    ntime, num_particles = ancestors.shape

    if lag is None or lag >= ntime:
        eves = _eve_indices(ancestors)
    else:
        exact = _eve_indices(ancestors)
        identity = jnp.arange(num_particles, dtype=ancestors.dtype)

        def _composed(t: Array) -> Array:
            idx = identity
            for j in range(lag):
                idx = ancestors[t - j][idx]
            return idx

        lagged = vmap(_composed)(jnp.arange(lag, ntime))
        eves = jnp.concatenate([exact[:lag], lagged], axis=0)

    return vmap(_eve_class_mass_sq)(posterior.filtered_log_weights, eves)


def log_bayes_factor(
    log_ml_1: _DiagnosticScalar,
    log_ml_2: _DiagnosticScalar,
) -> Scalar:
    r"""Compute the log Bayes factor between two models.

    $$
    \log BF_{12} = \log p(y_{1:T} \mid M_1)
                 - \log p(y_{1:T} \mid M_2)
    $$

    Positive values favour model 1; negative values favour model 2.

    !!! warning

        Marginal likelihoods are sensitive to the prior in ways that
        the posterior is not.  Bayes factors evaluate priors, not
        posteriors (Gelman, 2023).  With weakly informative priors the
        marginal likelihood is dominated by prior tails that have
        little effect on posterior inference, so a Bayes factor can
        reverse sign under prior changes that leave the posterior
        essentially unchanged. Inspect `smcx.cumulative_log_score` to see
        when evidence differences accrue, complement the comparison with
        a predictive criterion such as `smcx.crps`, and use
        `smcx.replicated_log_ml` to quantify Monte Carlo variability.

    Args:
        log_ml_1: Log marginal likelihood of model 1.
        log_ml_2: Log marginal likelihood of model 2.

    Returns:
        Scalar log Bayes factor.

    Raises:
        ValueError: Either log marginal likelihood is not a floating scalar.
    """
    first = _require_float_scalar(log_ml_1, name="log_ml_1")
    second = _require_float_scalar(log_ml_2, name="log_ml_2")
    return first - second


def replicated_log_ml(
    key: PRNGKeyT,
    filter_fn: _ReplicatedLogMLFn,
    num_replicates: _IntegerArgument,
) -> Float[Array, " num_replicates"]:
    r"""Run a particle filter multiple times to assess log-ML variability.

    Uses `jax.vmap` over PRNG keys for efficient parallel
    evaluation.  The resulting distribution of log-ML estimates
    quantifies Monte Carlo uncertainty in the evidence.

    Args:
        key: JAX PRNG key.
        filter_fn: Function ``(key) -> scalar`` that runs a particle
            filter and returns the marginal log-likelihood.
        num_replicates: Number of independent filter runs.

    Returns:
        Array of log-ML estimates, shape ``(num_replicates,)``.

    Raises:
        ValueError: ``num_replicates`` is not a positive integer, or
            ``filter_fn`` does not return a floating scalar.
    """
    num_replicates = _require_integer(
        num_replicates,
        name="num_replicates",
        minimum=1,
        domain="a positive integer",
    )
    keys = jr.split(key, num_replicates)

    def _run_one(key_i: PRNGKeyT) -> Array:
        return _require_float_scalar(
            filter_fn(key_i),
            name="filter_fn output",
        )

    return jnp.asarray(vmap(_run_one)(keys))


def param_weighted_mean(
    posterior: LiuWestPosterior | SMC2Posterior,
) -> Float[Array, "ntime param_dim"]:
    r"""Compute the weighted mean of parameter particles at each step.

    Args:
        posterior: Liu-West or SMC² posterior output.

    Returns:
        Weighted parameter means, shape ``(ntime, param_dim)``.

    Raises:
        ValueError: Consumed posterior arrays are malformed or misaligned.
    """
    params = _require_parameter_history(posterior, "param_weighted_mean")
    return _weighted_mean_field(posterior.filtered_log_weights, params)


def param_weighted_quantile(
    posterior: LiuWestPosterior | SMC2Posterior,
    q: _DiagnosticVector,
) -> Float[Array, "ntime num_quantiles param_dim"]:
    r"""Compute weighted quantiles of parameter particles at each step.

    Args:
        posterior: Liu-West or SMC² posterior output.
        q: Quantile levels in [0, 1], e.g. ``jnp.array([0.025, 0.975])``
            for a 95% credible interval. Values are checked eagerly;
            callers passing traced levels must preserve this domain.

    Returns:
        Weighted quantiles, shape ``(ntime, num_quantiles, param_dim)``.

    Raises:
        ValueError: Posterior arrays or ``q`` are malformed/misaligned, or an
            eager quantile lies outside [0, 1].
    """
    q = _validate_quantile_levels(q)
    params = _require_parameter_history(posterior, "param_weighted_quantile")
    return _weighted_quantile_field(posterior.filtered_log_weights, params, q)


# --- Posterior predictive checks -------------------------------------------


def posterior_predictive_sample(
    key: PRNGKeyT,
    posterior: ParticleFilterResult,
    transition_sampler: TransitionSampler,
    emission_sampler: EmissionSampler,
    num_samples: _IntegerArgument | None = None,
) -> Float[Array, "ntime num_samples emission_dim"]:
    r"""Draw one-step-ahead posterior predictive samples.

    At each time step $t$, we:

    1. Resample particle indices from the normalised weights.
    2. Propagate each resampled state through ``transition_sampler``.
    3. Draw an emission from ``emission_sampler``.

    This gives iid samples from the posterior predictive
    $p(y_{t+1} \mid y_{1:t})$, which can be compared with the actual
    observation $y_{t+1}$ for posterior predictive
    checking (Gelman et al., 2013, ch. 6).

    Args:
        key: JAX PRNG key.
        posterior: Particle filter posterior output.
        transition_sampler: Function ``(key, state) -> state``. ``state``
            may be a latent-state PyTree.
        emission_sampler: Function ``(key, state) -> emission`` accepting
            the same state PyTree and returning a nonempty floating vector.
        num_samples: Number of predictive draws per time step.
            Defaults to the number of particles.

    Returns:
        Predictive samples, shape
        ``(ntime, num_samples, emission_dim)``.

    Raises:
        ValueError: ``num_samples`` is not a positive integer, the posterior
            state is malformed, the transition changes its PyTree contract,
            or the emission is not a nonempty floating vector.
    """
    if num_samples is not None:
        num_samples = _require_integer(
            num_samples,
            name="num_samples",
            minimum=1,
            domain=">= 1 (a positive integer)",
        )
    axes = _validate_particle_result_axes(
        posterior,
        "posterior_predictive_sample",
        particles=True,
    )
    ntime, n_particles = axes
    num_samples = n_particles if num_samples is None else num_samples

    def _sample_one_step(
        log_weights_t: Float[Array, " num_particles"],
        particles_t: ParticleCloud,
        step_key: PRNGKeyT,
    ) -> Float[Array, "num_samples emission_dim"]:
        """Draw predictive samples at one time step."""
        k1, k2, k3 = jr.split(step_key, 3)
        weights = jnp.exp(log_weights_t - jnp.max(log_weights_t))
        weights = weights / jnp.sum(weights)
        indices = jr.choice(k1, n_particles, shape=(num_samples,), p=weights)
        resampled = _gather_particles(particles_t, indices)
        # Propagate through transition
        state_signature = _validate_particle_cloud(
            particles_t,
            n_particles,
            name="posterior.filtered_particles time slice",
        )

        def _transition(key_i, state_i):
            next_state = transition_sampler(key_i, state_i)
            _validate_state_tree(
                next_state,
                state_signature,
                name="transition_sampler output",
            )
            return next_state

        trans_keys = jr.split(k2, num_samples)
        propagated = vmap(_transition)(trans_keys, resampled)
        # Draw emissions
        emit_keys = jr.split(k3, num_samples)

        def _emit(key_i, state_i):
            emission = emission_sampler(key_i, state_i)
            _validate_emission(emission, name="emission_sampler output")
            return emission

        return vmap(_emit)(emit_keys, propagated)

    step_keys = jr.split(key, ntime)
    return vmap(_sample_one_step)(
        posterior.filtered_log_weights,
        posterior.filtered_particles,
        step_keys,
    )


def crps(
    predictions: _DiagnosticVector,
    observation: _DiagnosticScalar,
) -> Scalar:
    r"""Compute the Continuous Ranked Probability Score.

    CRPS is a proper scoring rule for probabilistic forecasts:

    $$
    \text{CRPS} = \mathbb{E}|Y - y|
                 - \tfrac{1}{2}\,\mathbb{E}|Y - Y'|
    $$

    where $Y, Y'$ are iid predictive samples and $y$ is the observation.

    Args:
        predictions: iid samples from the predictive distribution. The
            empirical distribution assigns each sample equal weight.
        observation: Observed scalar value.

    Returns:
        Scalar CRPS (lower is better, zero for perfect prediction).

    Raises:
        ValueError: Predictions are not a nonempty rank-one floating array,
            or the observation is not a floating scalar.

    Notes:
        The quantile-decomposition identity is evaluated on values centered
        at the observation. This retains :math:`O(N \log N)` complexity while
        avoiding cancellation between raw order statistics and integer
        scaling by :math:`N^2`.

    References:
        Matheson, J. E., and Winkler, R. L. (1976). Scoring rules for
        continuous probability distributions.
        https://doi.org/10.1287/mnsc.22.10.1087

        Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring
        rules, prediction, and estimation.
        https://doi.org/10.1198/016214506000001437

        Jordan, A., Krüger, F., and Lerch, S. (2019). Evaluating
        probabilistic forecasts with scoringRules.
        https://doi.org/10.18637/jss.v090.i12
    """
    predictions = _require_float_vector(
        predictions,
        name="predictions",
        dimension="num_samples",
    )
    obs = _require_float_scalar(observation, name="observation")
    centered = jnp.sort(predictions - obs)
    n = predictions.shape[0]
    reciprocal_n = jnp.reciprocal(jnp.asarray(n, dtype=centered.dtype))
    lower_midpoint_mass = (
        jnp.arange(n, dtype=centered.dtype) + 0.5
    ) * reciprocal_n
    upper_midpoint_mass = (
        jnp.arange(n, 0, -1, dtype=centered.dtype) - 0.5
    ) * reciprocal_n
    quantile_weight = jnp.where(
        centered < 0.0,
        lower_midpoint_mass,
        upper_midpoint_mass,
    )
    return jnp.asarray(
        2.0 * reciprocal_n * jnp.sum(jnp.abs(centered) * quantile_weight)
    )


# --- Pareto-k diagnostic ---------------------------------------------------


def _fit_generalized_pareto(
    x: Float[Array, " m"],
) -> Float[Array, ""]:
    r"""Fit GPD shape k via Zhang & Stephens (2009) with Vehtari prior.

    Implements the profile-likelihood estimator of Zhang and Stephens
    (2009) with Bayesian model averaging over a grid of candidate
    shape parameters, then applies the weakly informative prior from
    Vehtari, Simpson, Gelman, Yao, and Gabry (2024) that shrinks
    $\hat{k}$ toward 0.5.

    Matches the algorithm in NumPyro's ``_fit_generalized_pareto_impl``
    and ArviZ's ``gpdfitnew``.

    Args:
        x: Positive exceedances (sorted ascending), length m.

    Returns:
        Estimated GPD shape parameter k.
    """
    m = x.shape[0]
    num_candidates = 30 + math.isqrt(m)
    prior = 3  # Zhang-Stephens grid prior

    x_star = x[-1]  # max (sorted ascending)
    # First-quartile value for the grid denominator
    x_q25 = x[max(0, m // 4 - 1)]

    # Zhang-Stephens candidate b values:
    #   b_i = 1/x_star + (1 - sqrt(M / i)) / (prior * x_q25)
    # for i = 0.5, 1.5, ..., M - 0.5, where M is the CANDIDATE count,
    # not the sample size (using m here is a known porting bug: it
    # skews the grid and biases k upward, worst at small k).
    i = jnp.arange(1, num_candidates + 1) - 0.5
    b_grid = 1.0 / x_star + (1.0 - jnp.sqrt(num_candidates / i)) / (
        prior * x_q25
    )

    # Profile log-likelihood for each candidate:
    #   k_i = mean(log1p(-b_i * x))
    #   L_i = m * (log(-b_i / k_i) - k_i - 1)
    log_terms = jnp.log1p(-b_grid[:, None] * x[None, :])
    k_grid = jnp.mean(log_terms, axis=1)

    log_lik = m * (jnp.log(-b_grid / k_grid) - k_grid - 1.0)

    # Replace NaN/Inf with -inf before computing weights
    log_lik = jnp.where(jnp.isfinite(log_lik), log_lik, -jnp.inf)

    # Posterior weights via softmax (matches NumPyro's pairwise formulation)
    log_lik_max = jnp.max(log_lik)
    w = jnp.exp(log_lik - log_lik_max)
    w_sum = jnp.sum(w)
    w = w / (w_sum + _EXC_FLOOR)

    # Zero out negligible weights for numerical stability
    eps_threshold = 10.0 * jnp.finfo(w.dtype).eps
    w = jnp.where(w >= eps_threshold, w, 0.0)
    w = w / (jnp.sum(w) + _EXC_FLOOR)

    # Posterior mean for b, then derive k
    b_hat = jnp.sum(w * b_grid)
    k_hat = jnp.mean(jnp.log1p(-b_hat * x))

    # Vehtari et al. (2024) prior regularisation:
    #   k_final = k * m/(m+a) + a * 0.5/(m+a)
    a = _PRIOR_STRENGTH
    k_reg = k_hat * m / (m + a) + a * _PRIOR_K / (m + a)
    return jnp.asarray(k_reg)


def _fit_pareto_k(
    log_weights: Float[Array, " num_particles"],
) -> Float[Array, ""]:
    r"""Fit the shape parameter k of a generalised Pareto distribution.

    Extracts the upper tail of the importance weights and fits a GPD
    via the Zhang and Stephens (2009) profile-likelihood estimator
    with the Vehtari et al. (2024) weakly informative prior.

    The tail sample uses
    ``M = min(max(10, ceil(min(0.2 * N, 3 * sqrt(N)))), N - 1)``
    order statistics. The central heuristic follows ArviZ and NumPyro;
    smcx adds the explicit small-sample floor and cap.

    Args:
        log_weights: Normalised log importance weights at one step.

    Returns:
        Estimated shape parameter k, or NaN when fewer than two weights make
        the tail fit undefined.
    """
    n = log_weights.shape[0]
    if n < 2:
        return jnp.asarray(jnp.nan, dtype=log_weights.dtype)

    # Tail count: min(20% of N, 3*sqrt(N)), at least 10.
    m = max(10, math.ceil(min(0.2 * n, 3.0 * math.sqrt(n))))
    m = min(m, n - 1)

    # Sort ascending; tail is the last m elements.
    log_w_sorted = jnp.sort(log_weights)
    # Cutoff: the element just below the tail
    cutoff = log_w_sorted[n - m - 1]
    # Exceedances in the original weight scale:
    #   exp(log_w_tail) - exp(cutoff)
    # This matches ArviZ/NumPyro: exceedance above the threshold
    # weight, not above the threshold log-weight.
    log_tail = log_w_sorted[n - m :]
    exceedances = jnp.exp(log_tail) - jnp.exp(cutoff)
    # Ensure non-negative (numerical noise)
    exceedances = jnp.maximum(exceedances, 0.0)

    # When all weights are equal the exceedances are zero and the
    # GPD fit is degenerate.  Return the prior mean in that case.
    max_exc = jnp.max(exceedances)
    prior_mean = _PRIOR_K * _PRIOR_STRENGTH / (m + _PRIOR_STRENGTH)
    return jnp.asarray(
        jnp.where(
            max_exc > _EXC_FLOOR,
            _fit_generalized_pareto(exceedances),
            prior_mean,
        )
    )


def pareto_k_diagnostic(
    posterior: ParticleFilterResult,
) -> Float[Array, " ntime"]:
    r"""Compute the Pareto-k diagnostic at each time step.

    Fits a generalised Pareto distribution (GPD) to the upper tail
    of the importance weights using the Zhang and Stephens (2009)
    profile-likelihood estimator with the weakly informative prior
    from Vehtari, Simpson, Gelman, Yao, and Gabry (2024).

    The shape parameter $\hat{k}$ indicates reliability:

    - $\hat{k} < 0.5$: good; the IS estimate has finite
      variance
    - $0.5 \le \hat{k} < 0.7$: variance is infinite, but the
      PSIS convergence-rate results say estimates remain practically
      reliable
    - $0.7 \le \hat{k} < 1.0$: unreliable (the practical
      threshold of Vehtari et al. 2024)
    - $\hat{k} \ge 1.0$: very unreliable (infinite mean)

    The tail size is
    ``min(max(10, ceil(min(0.2 * N, 3 * sqrt(N)))), N - 1)`` order
    statistics, including the small-sample floor and cap.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Per-step Pareto-k estimates, shape ``(ntime,)``. Estimates are NaN
        when the posterior contains fewer than two particles.

    Raises:
        ValueError: The posterior log-weight history is malformed.
    """
    _validate_log_weight_history(posterior, "pareto_k_diagnostic")
    return vmap(_fit_pareto_k)(posterior.filtered_log_weights)


# --- Tail-ESS --------------------------------------------------------------


def tail_ess(
    posterior: ParticleFilterResult,
    q: _RealArgument = 0.05,
) -> Float[Array, " ntime"]:
    r"""Effective particles supporting the distribution tails.

    For each step, each state dimension, and each tail, restrict the
    normalized weights to the particles beyond the weighted q / 1-q
    quantile of the particle *values* and compute
    $(\sum w)^2 / \sum w^2$ — the effective number of particles
    estimating that tail. Returns the minimum over dimensions and both
    tails (in the spirit of the quantile tail-ESS of Vehtari, Gelman,
    Simpson, Carpenter & Burkner 2021).

    Uniform weights give roughly ``q * N`` (a tail only ever holds a
    ``q`` fraction of the mass); compare against ``q * N``, not ``N``.

    Note: this replaces the earlier smcjax quantity of the same name,
    which measured top-weight-mass concentration and did not examine
    particle values. It was carried forward from smcx's former MLX
    implementation.

    Args:
        posterior: Any `smcx.containers.ParticleFilterResult`.
        q: Tail fraction (each tail is the mass beyond the weighted
            ``q`` / ``1 - q`` quantile).

    Returns:
        Per-step minimum tail-ESS, shape ``(ntime,)``.

    Raises:
        TypeError: The posterior has structured rather than dense particles.
        ValueError: Malformed/misaligned posterior arrays or invalid ``q``.
    """
    if not isinstance(q, Real):
        raise ValueError(f"q must be finite and in (0, 0.5]; got {q!r}")
    q = float(q)
    if not math.isfinite(q) or not 0.0 < q <= 0.5:
        raise ValueError(f"q must be finite and in (0, 0.5]; got {q!r}")
    qs = jnp.array([q, 1.0 - q])
    particles = _require_dense_particle_history(
        posterior, diagnostic="tail_ess"
    )
    ntime, _, dim = particles.shape

    def n_eff(tw):
        s2 = jnp.sum(tw * tw)
        return jnp.where(
            s2 > 0.0, jnp.sum(tw) ** 2 / jnp.maximum(s2, 1e-30), 0.0
        )

    # Plain Python loops over the small time/dim axes — diagnostics
    # are not hot-loop code, and the loop keeps the per-dim weighted
    # quantile trivially readable.
    out = []
    for t in range(ntime):
        w = jnp.exp(posterior.filtered_log_weights[t])
        w = w / jnp.maximum(jnp.sum(w), 1e-30)
        per_dim = []
        for d in range(dim):
            vals = particles[t, :, d]
            order = jnp.argsort(vals)
            v_sorted = vals[order]
            w_sorted = w[order]
            cum = jnp.cumsum(w_sorted)
            # Midpoint CDF: centre each particle's mass in its
            # interval, normalized so the axis is [0, 1].
            mid = (jnp.concatenate([jnp.zeros(1), cum[:-1]]) + cum) / (
                2.0 * jnp.maximum(cum[-1], 1e-30)
            )
            edges = jnp.interp(qs, mid, v_sorted)
            lo = jnp.where(vals <= edges[0], w, 0.0)
            hi = jnp.where(vals >= edges[1], w, 0.0)
            per_dim.append(jnp.minimum(n_eff(lo), n_eff(hi)))
        out.append(jnp.min(jnp.stack(per_dim)))
    return jnp.stack(out)


def cumulative_log_score(
    posterior: ParticleFilterResult,
) -> Float[Array, " ntime"]:
    r"""Compute the cumulative one-step-ahead predictive log-score.

    The log-evidence increments $\log p(y_t \mid y_{1:t-1})$
    are already one-step-ahead predictive log-densities.  This
    function returns their running sum:

    $$
    S_t = \sum_{s=1}^{t} \log p(y_s \mid y_{1:s-1})
    $$

    so that $S_T$ equals the total marginal log-likelihood.
    Comparing $S_t$ across models shows when their log Bayes
    factor accrues. At the final time their difference is exactly the
    log Bayes factor, with the same prior sensitivity.

    Every prefix uses sequential Neumaier compensation in the input
    dtype, matching compensated posterior evidence accumulation. This
    corrects the ordinary cumulative sum used in earlier releases:
    fixed-input prefixes can change, especially for long or
    cancellation-heavy traces, while shape, dtype, and JIT behavior
    are unchanged.

    Args:
        posterior: Particle filter posterior output.

    Returns:
        Cumulative log-scores, shape ``(ntime,)``.

    Raises:
        ValueError: ``posterior.log_evidence_increments`` is not a nonempty
            floating array with shape ``(T,)``.

    References:
        Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring
        rules, prediction, and estimation.
        https://doi.org/10.1198/016214506000001437
        Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur
        Summation endlicher Summen.
        https://doi.org/10.1002/zamm.19740540106
    """
    _validate_float_trace(
        posterior.log_evidence_increments,
        "posterior.log_evidence_increments",
        "cumulative_log_score",
    )
    return _neumaier_prefix_sum(posterior.log_evidence_increments)


# --- Diagnostic summary ----------------------------------------------------


def diagnose(
    posterior: ParticleFilterResult,
    ess_threshold: float = 0.1,
    diversity_threshold: float = 0.1,
    pareto_k_threshold: float | None = None,
) -> dict[str, Any]:
    r"""Summarise filter health and flag potential problems.

    Runs a battery of diagnostics and returns a dictionary with
    scalar summaries and a list of plain-text warnings.  The
    thresholds are configurable; the defaults flag situations
    where the particle approximation is likely unreliable.

    This convenience summary is host-only because it converts arrays
    to Python scalars and constructs warning strings. Use the individual
    array-returning diagnostics inside `jax.jit`.

    Args:
        posterior: Particle filter posterior output.
        ess_threshold: Fraction of N below which ESS triggers a
            warning.
        diversity_threshold: Diversity below which a warning is
            triggered.
        pareto_k_threshold: Pareto-k above which a warning is
            triggered.

    Returns:
        Dictionary with keys:

        - ``min_ess``: minimum ESS across all time steps
        - ``min_diversity``: minimum particle diversity
        - ``max_pareto_k``: maximum finite Pareto-k across time steps,
            or NaN when every estimate is undefined
        - ``min_tail_ess``: minimum tail-ESS across time steps
        - ``ess_below_threshold``: count of steps where
            ESS < ``ess_threshold * N``
        - ``warnings``: list of diagnostic warning strings

    Raises:
        TypeError: The posterior has structured rather than dense particles.
        ValueError: Posterior arrays are malformed/misaligned or final-only.
    """
    _validate_particle_result_axes(
        posterior,
        "diagnose",
        ancestors=True,
        full_history=True,
        ess=True,
    )
    _require_dense_particle_history(
        posterior,
        diagnostic="diagnose",
    )
    n = posterior.filtered_log_weights.shape[1]
    ess_vals = posterior.ess
    diversity = particle_diversity(posterior)
    k_hat = pareto_k_diagnostic(posterior)
    t_ess = tail_ess(posterior)

    min_ess = float(jnp.min(ess_vals))
    min_div = float(jnp.min(diversity))
    finite_k = jnp.isfinite(k_hat)
    undefined_k_count = int(jnp.sum(~finite_k))
    if undefined_k_count == k_hat.size:
        max_k = math.nan
    else:
        max_k = float(jnp.max(jnp.where(finite_k, k_hat, -jnp.inf)))
    min_t_ess = float(jnp.min(t_ess))
    ess_count = int(jnp.sum(ess_vals < ess_threshold * n))

    warnings: list[str] = []
    if min_ess < ess_threshold * n:
        warnings.append(
            f"ESS dropped below {ess_threshold:.0%} of N "
            f"at {ess_count} step(s) (min ESS = {min_ess:.1f})"
        )
    if min_div < diversity_threshold:
        warnings.append(
            f"Particle diversity fell below "
            f"{diversity_threshold:.0%} (min = {min_div:.3f})"
        )
    if undefined_k_count:
        warnings.append(
            f"Pareto-k was undefined at {undefined_k_count} step(s) "
            f"(non-finite estimate); inspect particle weights at those steps"
        )
    if pareto_k_threshold is None:
        # Sample-size-dependent PSIS reliability threshold
        # (Vehtari et al.): min(1 - 1/log10(N), 0.7).
        n_particles = posterior.filtered_log_weights.shape[1]
        pareto_k_threshold = min(
            1.0 - 1.0 / math.log10(max(n_particles, 11)), 0.7
        )
    if max_k > pareto_k_threshold:
        warnings.append(
            f"Pareto-k exceeded {pareto_k_threshold:.2f} "
            f"(max k = {max_k:.3f}); estimates at those steps "
            f"are unreliable (PSIS practical threshold — weight "
            f"variance is already infinite for k >= 0.5)"
        )

    return {
        "min_ess": min_ess,
        "min_diversity": min_div,
        "max_pareto_k": max_k,
        "min_tail_ess": min_t_ess,
        "ess_below_threshold": ess_count,
        "warnings": warnings,
    }
