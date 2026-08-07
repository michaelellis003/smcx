# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Dynamic generalized linear filtering and retrospective state moments.

Exponential-family observations over a linear state evolution carried
by moments only. Each step matches a conjugate prior to the two
moments of the linear predictor $\lambda_t = F' x_t$, updates it
exactly on the observation, and feeds the posterior moments back to
the state by linear Bayes estimation:

$$
m_t = a_t + R_t F\,\frac{f_t^* - f_t}{q_t}, \qquad
C_t = R_t - R_t F F' R_t\,\frac{1 - q_t^*/q_t}{q_t}.
$$

The recursion is sequential, deterministic, and closed-form, and it
is approximate in exactly three places: the state prior is specified
by moments alone, the conjugate form is an assumption matched to two
moments, and the state feedback is the optimal *linear* Bayes
estimate (no error bound). Conditional on the matched conjugate
prior, the one-step forecast density, the conjugate update, and the
posterior moments are exact. For the normal family at
``dispersion_discount=1``, all three approximations vanish and the recursion
is exactly the Kalman filter [West, Harrison & Migon, 1985, sec. 2.2].

`smcx.dglm_smoother` adds a retrospective Gaussian/Bayes-linear projection of
the stored state moments. Its normal-family reduction is exact RTS. General
DGLM output remains an approximate state-moment summary, not a posterior
distribution or credible interval, and has no error bound. Particle filters
are the natural accuracy check.

Forecast densities and special functions come from `jax.scipy`; smcx
implements only the inverse moment-matching solves.

References:
    West, M., Harrison, P. J., and Migon, H. S. (1985). Dynamic
    generalized linear models and Bayesian forecasting. Journal of
    the American Statistical Association, 80(389), 73-83.
    https://doi.org/10.1080/01621459.1985.10477131
    West, M., and Harrison, J. (1997). Bayesian Forecasting and
    Dynamic Models, second edition, chapter 14.
    https://doi.org/10.1007/b98971
    Alves, M. B., Migon, H. S., Santos Jr, S. V., and Marotta, R. (2025).
    An Efficient Sequential Approach for k-Parametric Dynamic Generalised
    Linear Models, section 3.2 and Algorithm 2.
    https://arxiv.org/html/2201.05387v4#S3.SS2
"""

import math
from typing import TYPE_CHECKING, Any, NamedTuple, SupportsIndex, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import core, debug_infs, debug_nans, lax
from jax.nn import sigmoid as jax_sigmoid
from jax.nn import softplus
from jax.scipy.special import digamma, gammaln, polygamma
from jax.scipy.stats import betabinom, nbinom
from jaxtyping import Array, Float, Shaped

from smcx._numerics import _neumaier_add
from smcx._utils import _canonicalize_emissions, _positive_integer
from smcx.containers import (
    DGLMFilterPosterior,
    DGLMForecast,
    DGLMForecastPaths,
    DGLMSmootherPosterior,
)
from smcx.dlm import _validate_positive_scalar
from smcx.kalman import (
    _backward_pass,
    _canonicalize_filter_covariances,
    _check_covariance,
    _check_float_array,
    _check_sampling_factors,
    _sampling_covariance_factor,
    _sanitize_missing,
    _symmetrize,
    _time_matrix,
    _validate_emission_rows,
)
from smcx.types import (
    EmissionSequence,
    FamilyConjugateUpdate,
    FamilyEmissionSampler,
    FamilyLogForecast,
    FamilyMomentMatch,
    FamilyPosteriorMoments,
    PRNGKeyT,
    Scalar,
)

if TYPE_CHECKING:
    _CountArgument: TypeAlias = SupportsIndex
else:
    _CountArgument: TypeAlias = Any

_NEWTON_ITERATIONS = 12
_NEWTON_ITERATIONS_2D = 25


class DGLMFamily(NamedTuple):
    r"""Observation-family contract for `dglm_filter`.

    Four pure callables over scalars, in the notation of West,
    Harrison & Migon (1985). ``alpha`` and ``beta`` are the conjugate
    parameters of the family; for the normal family they degenerate
    to the moments themselves.

    Attributes:
        match_moments: ``(f, q) -> (alpha, beta)`` — conjugate prior
            parameters matched to the prior mean and variance of the
            linear predictor.
        log_forecast: ``(y, alpha, beta) -> log p(y)`` — the exact
            one-step forecast log density under the matched prior.
        update: ``(y, alpha, beta) -> (alpha_star, beta_star)`` — the
            exact conjugate posterior parameters.
        posterior_moments: ``(alpha_star, beta_star) -> (f_star,
            q_star)`` — posterior mean and variance of the linear
            predictor, fed back to the state by linear Bayes.

    Note:
        The built-in factories additionally attach an eager
        emission-support check that `dglm_filter` runs once on
        concrete emissions at its boundary (#283). The check lives
        outside the record because the record's released contract is
        this four-field sequence (#317). Emissions of a user-defined
        family are not support-checked. Path simulation
        (`smcx.dglm_forecast_sample`) takes a standalone
        ``sample_emission`` callable; the capability joins the record
        at 3.0 (ADR-0036 amendment).
    """

    match_moments: FamilyMomentMatch
    log_forecast: FamilyLogForecast
    update: FamilyConjugateUpdate
    posterior_moments: FamilyPosteriorMoments


# The built-in factories mark their own ``log_forecast`` closures
# with this attribute; the filter boundary reads it with ``getattr``,
# which places no hashability or weak-reference requirement on
# user-supplied callables (#317).
_VALIDATOR_ATTRIBUTE = "_smcx_validate_emissions"


_ASYMPTOTIC_CUTOFF = 1e8


def _digamma_safe(value: Scalar) -> Scalar:
    r"""Digamma with the asymptotic series above the cutoff.

    Past $10^8$ the dropped $1/(12x^2)$ term sits below f64 machine
    precision of $\log x$, and the series avoids the special-function
    kernel's large-argument failure modes (#282).
    """
    large = value > _ASYMPTOTIC_CUTOFF
    ordinary = jnp.where(large, jnp.ones_like(value), value)
    big = jnp.where(large, value, jnp.ones_like(value))
    return jnp.where(large, jnp.log(big) - 0.5 / big, digamma(ordinary))


def _trigamma_safe(value: Scalar) -> Scalar:
    r"""Trigamma with the asymptotic series above the cutoff (#282)."""
    large = value > _ASYMPTOTIC_CUTOFF
    ordinary = jnp.where(large, jnp.ones_like(value), value)
    big = jnp.where(large, value, jnp.ones_like(value))
    return jnp.where(
        large, 1.0 / big + 0.5 / (big * big), polygamma(1, ordinary)
    )


def _tetragamma_safe(value: Scalar) -> Scalar:
    r"""Tetragamma with the asymptotic series above the cutoff (#282)."""
    large = value > _ASYMPTOTIC_CUTOFF
    ordinary = jnp.where(large, jnp.ones_like(value), value)
    big = jnp.where(large, value, jnp.ones_like(value))
    squared = big * big
    return jnp.where(
        large, -1.0 / squared - 1.0 / (squared * big), polygamma(2, ordinary)
    )


def _tetragamma_times_arg_safe(value: Scalar) -> Scalar:
    r"""Return $\alpha\,\psi''(\alpha)$ without the intermediate square.

    ``_tetragamma_safe(a) * a`` forms ``1 / a**2`` before multiplying,
    which overflows float32 for the huge solutions of tiny target
    variances even though the fused product $-1/\alpha - 1/\alpha^2$ is
    representable (follow-up review R3b).
    """
    large = value > _ASYMPTOTIC_CUTOFF
    ordinary = jnp.where(large, jnp.ones_like(value), value)
    big = jnp.where(large, value, jnp.ones_like(value))
    return jnp.where(
        large,
        -1.0 / big - 1.0 / (big * big),
        polygamma(2, ordinary) * ordinary,
    )


def _log_ratio_directional(a: Scalar, b: Scalar) -> Scalar:
    r"""$\log(a/b)$ with the small difference in the numerator.

    ``log1p((a - b) / b)`` fails in one direction: when ``a`` is far
    below ``b``, the represented quotient can round to exactly ``-1``
    and produce ``log1p(-1) = -inf``. The helper therefore orders
    the operands, evaluates one ``log1p`` on the nonnegative
    magnitude, and applies the sign afterward. A two-branch
    ``jnp.where`` form is not equivalent: the inactive branch still
    evaluates ``log1p(-1)`` and its infinite partial derivative
    reaches reverse-mode differentiation as ``NaN``.
    """
    swap = a < b
    high = jnp.where(swap, b, a)
    low = jnp.where(swap, a, b)
    magnitude = jnp.log1p((high - low) / low)
    return jnp.where(swap, -magnitude, magnitude)


def _digamma_difference_safe(a: Scalar, b: Scalar) -> Scalar:
    r"""$\psi(a) - \psi(b)$ without large-argument cancellation.

    For large conjugate parameters the raw digamma difference cancels
    to the working epsilon times $\log a$, erasing representable
    posterior-moment differences (follow-up review R3b). The series
    form keeps the numerator ``a - b`` exact and truncates at
    $O(1/a^4)$.
    """
    large = jnp.minimum(a, b) > 1e3
    a_ord = jnp.where(large, jnp.ones_like(a), a)
    b_ord = jnp.where(large, jnp.ones_like(b), b)
    a_big = jnp.where(large, a, jnp.ones_like(a))
    b_big = jnp.where(large, b, jnp.ones_like(b))
    series = (
        _log_ratio_directional(a_big, b_big)
        - 1.0 / (2.0 * a_big)
        + 1.0 / (2.0 * b_big)
        - 1.0 / (12.0 * a_big * a_big)
        + 1.0 / (12.0 * b_big * b_big)
    )
    return jnp.where(large, series, digamma(a_ord) - digamma(b_ord))


def _digamma_minus_log_safe(a: Scalar, b: Scalar) -> Scalar:
    r"""$\psi(a) - \log(b)$ without large-argument cancellation."""
    large = jnp.minimum(a, b) > 1e3
    a_ord = jnp.where(large, jnp.ones_like(a), a)
    b_ord = jnp.where(large, jnp.ones_like(b), b)
    a_big = jnp.where(large, a, jnp.ones_like(a))
    b_big = jnp.where(large, b, jnp.ones_like(b))
    series = (
        _log_ratio_directional(a_big, b_big)
        - 1.0 / (2.0 * a_big)
        - 1.0 / (12.0 * a_big * a_big)
    )
    return jnp.where(large, series, digamma(a_ord) - jnp.log(b_ord))


# Log-space Newton steps larger than this are clamped. Far from the
# root the clamp bounds each multiplicative update (a standard Newton
# globalization); near the root steps are small, so the quadratic
# convergence is untouched.
_NEWTON_MAX_LOG_STEP = 4.0


def _inverse_trigamma(value: Scalar) -> Scalar:
    r"""Solve $\psi'(\alpha) = q$ for $\alpha > 0$.

    Log-space Newton with a fixed iteration count so the solve is
    jit-, vmap-, and grad-compatible. The initializer solves the
    two-term asymptote $1/\alpha + 1/\alpha^2 = q$ exactly,
    $\alpha_0 = (1 + \sqrt{1 + 4q}) / (2q)$, which is correct in both
    the $q \to 0$ ($\alpha \approx 1/q$) and $q \to \infty$
    ($\alpha \approx 1/\sqrt{q}$) limits, so the quadratic iteration
    reaches f64 machine precision across $q \in [10^{-6}, 10^{6}]$
    (#282; the previous small-$q$ initializer diverged past
    $q \approx 40$).
    """
    log_alpha = jnp.log((1.0 + jnp.sqrt(1.0 + 4.0 * value)) / (2.0 * value))
    for _ in range(_NEWTON_ITERATIONS):
        alpha = jnp.exp(log_alpha)
        residual = _trigamma_safe(alpha) - value
        slope = _tetragamma_times_arg_safe(alpha)
        step = jnp.clip(
            residual / slope, -_NEWTON_MAX_LOG_STEP, _NEWTON_MAX_LOG_STEP
        )
        log_alpha = log_alpha - step
    return jnp.exp(log_alpha)


def poisson() -> DGLMFamily:
    r"""Poisson observations with a log link.

    Conjugate gamma on the rate: the exact moment match solves
    $f = \psi(\alpha) - \log\beta$, $q = \psi'(\alpha)$
    (one inverse-trigamma Newton solve; $\beta$ then closed-form),
    the one-step forecast is the exact negative binomial
    $\mathrm{Nb}(\alpha, \beta/(1+\beta))$, and the conjugate update
    is $(\alpha + y, \beta + 1)$ [West, Harrison & Migon, 1985,
    sec. 5]. Emissions are nonnegative counts.
    """

    def match_moments(forecast_mean, forecast_variance):
        alpha = _inverse_trigamma(forecast_variance)
        beta = jnp.exp(_digamma_safe(alpha) - forecast_mean)
        return alpha, beta

    def log_forecast(emission, alpha, beta):
        # Branch rule. The matched negative binomial converges to
        # Poisson(e^f) as the predictor variance q shrinks. The
        # limit's truncation error is estimated by
        # q*((y - rate)^2 + y + rate + 1)/2. The estimate was
        # conservative on every high-precision reference probe used
        # to calibrate it, but it is not a proven bound. The exact
        # algebra keeps priority. The limit is taken only when the
        # estimate falls below a calibrated floor of the direct
        # evaluation's rounding error, 0.03*eps*alpha*log(alpha),
        # itself fitted to measured cases. Inactive operands are
        # clamped so the unused branch cannot poison gradients.
        eps = jnp.finfo(jnp.result_type(alpha)).eps
        predictor_variance = _trigamma_safe(alpha)
        predictor_mean = _digamma_minus_log_safe(alpha, beta)
        rate = jnp.exp(predictor_mean)
        limit_error_estimate = (
            0.5
            * predictor_variance
            * ((emission - rate) ** 2 + emission + rate + 1.0)
        )
        exact_floor = 0.03 * eps * alpha * jnp.log(alpha)
        use_limit = limit_error_estimate < exact_floor
        limit = emission * predictor_mean - rate - gammaln(emission + 1.0)
        alpha_safe = jnp.where(use_limit, jnp.ones_like(alpha), alpha)
        beta_safe = jnp.where(use_limit, jnp.ones_like(beta), beta)
        exact = nbinom.logpmf(
            emission, alpha_safe, beta_safe / (1.0 + beta_safe)
        )
        return jnp.where(use_limit, limit, exact)

    def update(emission, alpha, beta):
        return alpha + emission, beta + 1.0

    def posterior_moments(alpha, beta):
        return _digamma_minus_log_safe(alpha, beta), _trigamma_safe(alpha)

    def validate_emissions(emissions):
        values = np.asarray(emissions)
        if not (
            np.all(np.isfinite(values))
            and np.all(values >= 0)
            and np.all(values == np.floor(values))
        ):
            raise ValueError(
                "poisson emissions must be nonnegative integer counts"
            )

    setattr(log_forecast, _VALIDATOR_ATTRIBUTE, validate_emissions)
    return DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )


def _match_beta_moments(
    mean: Scalar, variance: Scalar
) -> tuple[Scalar, Scalar]:
    r"""Solve the beta moment-matching equations for the logit link.

    The system is $\psi(\alpha) - \psi(\beta) = f$ with
    $\psi'(\alpha) + \psi'(\beta) = q$.

    Two-dimensional log-space Newton with a fixed iteration count,
    initialized at WHM's mode/curvature closed forms
    $\alpha_0 = (1 + e^{f})/q$, $\beta_0 = (1 + e^{-f})/q$ — the
    $q \to 0$ asymptote of the exact system — evaluated as
    $\log \alpha_0 = \mathrm{softplus}(f) - \log q$ so a large
    predictor never overflows the initializer, with the asymptotic
    polygamma forms carrying the iteration at large arguments (#282).
    """
    log_alpha = softplus(mean) - jnp.log(variance)
    log_beta = softplus(-mean) - jnp.log(variance)
    for _ in range(_NEWTON_ITERATIONS_2D):
        alpha, beta = jnp.exp(log_alpha), jnp.exp(log_beta)
        residual_mean = _digamma_difference_safe(alpha, beta) - mean
        residual_var = _trigamma_safe(alpha) + _trigamma_safe(beta) - variance
        j11 = _trigamma_safe(alpha) * alpha
        j12 = -_trigamma_safe(beta) * beta
        j21 = _tetragamma_times_arg_safe(alpha)
        j22 = _tetragamma_times_arg_safe(beta)
        determinant = j11 * j22 - j12 * j21
        step_alpha = jnp.clip(
            (j22 * residual_mean - j12 * residual_var) / determinant,
            -_NEWTON_MAX_LOG_STEP,
            _NEWTON_MAX_LOG_STEP,
        )
        step_beta = jnp.clip(
            (-j21 * residual_mean + j11 * residual_var) / determinant,
            -_NEWTON_MAX_LOG_STEP,
            _NEWTON_MAX_LOG_STEP,
        )
        log_alpha = log_alpha - step_alpha
        log_beta = log_beta - step_beta
    return jnp.exp(log_alpha), jnp.exp(log_beta)


def binomial(*, trials: int) -> DGLMFamily:
    r"""Binomial observations with a logit link and known trials.

    Conjugate beta on the success probability: the exact moment match
    solves $f = \psi(\alpha) - \psi(\beta)$,
    $q = \psi'(\alpha) + \psi'(\beta)$ (2-D Newton), the one-step
    forecast is the exact beta-binomial, and the conjugate update is
    $(\alpha + y, \beta + n - y)$ [West, Harrison & Migon, 1985,
    sec. 5]. Emissions are counts in $\{0, \dots, n\}$; ``trials``
    is a static known $n \ge 1$ shared across steps.
    """
    if (
        isinstance(trials, bool)
        or not isinstance(trials, (int, np.integer))
        or trials < 1
    ):
        raise ValueError(f"trials must be a positive integer; got {trials!r}")

    def match_moments(forecast_mean, forecast_variance):
        return _match_beta_moments(forecast_mean, forecast_variance)

    def log_forecast(emission, alpha, beta):
        # Branch rule between the exact beta-binomial algebra and its
        # Binomial(trials, sigmoid(f)) limit. The floor construction
        # follows the Poisson family (R3a, R3c).
        eps = jnp.finfo(jnp.result_type(alpha)).eps
        predictor_variance = _trigamma_safe(alpha) + _trigamma_safe(beta)
        predictor_mean = _digamma_difference_safe(alpha, beta)
        success_probability = jax_sigmoid(predictor_mean)
        mean_count = trials * success_probability
        limit_error_estimate = (
            0.5
            * predictor_variance
            * (
                (emission - mean_count) ** 2
                + mean_count * (1.0 - success_probability)
                + 1.0
            )
        )
        total = alpha + beta
        exact_floor = 0.03 * eps * total * jnp.log(total)
        use_limit = limit_error_estimate < exact_floor
        limit = (
            gammaln(trials + 1.0)
            - gammaln(emission + 1.0)
            - gammaln(trials - emission + 1.0)
            + emission * predictor_mean
            - trials * softplus(predictor_mean)
        )
        alpha_safe = jnp.where(use_limit, jnp.ones_like(alpha), alpha)
        beta_safe = jnp.where(use_limit, jnp.ones_like(beta), beta)
        exact = betabinom.logpmf(emission, trials, alpha_safe, beta_safe)
        return jnp.where(use_limit, limit, exact)

    def update(emission, alpha, beta):
        return alpha + emission, beta + trials - emission

    def posterior_moments(alpha, beta):
        # Stable difference: the raw digamma subtraction erases
        # representable posterior-moment differences for the large
        # matched parameters of small predictor variances (R3b).
        return (
            _digamma_difference_safe(alpha, beta),
            _trigamma_safe(alpha) + _trigamma_safe(beta),
        )

    def validate_emissions(emissions):
        values = np.asarray(emissions)
        if not (
            np.all(np.isfinite(values))
            and np.all(values >= 0)
            and np.all(values <= trials)
            and np.all(values == np.floor(values))
        ):
            raise ValueError(
                f"binomial emissions must be integers in [0, {trials}]"
            )

    setattr(log_forecast, _VALIDATOR_ATTRIBUTE, validate_emissions)
    return DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )


def gamma(*, shape: Scalar) -> DGLMFamily:
    r"""Gamma observations with a log link and known shape.

    The response is $y_t \mid \mu_t \sim
    \mathrm{Gamma}(\nu, \nu/\mu_t)$ with known shape $\nu$ and
    mean $\mu_t = e^{\lambda_t}$, the positive-continuous member of
    the West--Harrison exponential-family programme (1997, chapter
    14). The conjugate Gamma analysis lives on the rate
    $\theta_t = \nu/\mu_t$: matching the linear predictor's
    moments gives $\alpha$ from the inverse trigamma (the Poisson
    family's solve) and
    $\beta = e^{\psi(\alpha) + f - \log\nu}$; observing $y$
    updates $(\alpha, \beta)$ to $(\alpha + \nu, \beta + y)$
    exactly; and the one-step forecast is the exact compound gamma.
    Emissions must be finite and strictly positive. For
    `smcx.dglm_forecast_sample`, the log-link commitment is
    ``lambda key, lam: jr.gamma(key, nu) * jnp.exp(lam) / nu``.

    Args:
        shape: Known positive observation shape $\nu$. Larger values
            mean less dispersed responses around the mean.

    Returns:
        The family record for `smcx.dglm_filter`.
    """
    shape_value = _validate_positive_scalar(shape, "shape")
    if not isinstance(shape_value, core.Tracer):
        shape_value = float(shape_value)  # ty: ignore[invalid-argument-type]
    nu = shape_value
    log_nu = jnp.log(nu) if isinstance(nu, core.Tracer) else math.log(nu)

    def match_moments(forecast_mean, forecast_variance):
        alpha = _inverse_trigamma(forecast_variance)
        beta = jnp.exp(_digamma_safe(alpha) + forecast_mean - log_nu)
        return alpha, beta

    def log_forecast(emission, alpha, beta):
        # The exact compound-gamma density. Its gammaln difference
        # stays well conditioned through the small-variance boundary:
        # the alpha = 1e6 golden reproduces at 1e-8 relative, so no
        # limit branch is needed (unlike the Poisson family's #309).
        return (
            gammaln(nu + alpha)
            - gammaln(jnp.asarray(nu, dtype=jnp.result_type(alpha)))
            - gammaln(alpha)
            + alpha * jnp.log(beta)
            + (nu - 1.0) * jnp.log(emission)
            - (nu + alpha) * jnp.log(beta + emission)
        )

    def update(emission, alpha, beta):
        return alpha + nu, beta + emission

    def posterior_moments(alpha, beta):
        predictor_mean = log_nu - _digamma_minus_log_safe(alpha, beta)
        return predictor_mean, _trigamma_safe(alpha)

    def validate_emissions(emissions):
        values = np.asarray(emissions)
        if not (np.all(np.isfinite(values)) and np.all(values > 0)):
            raise ValueError(
                "gamma emissions must be finite and strictly positive"
            )

    setattr(log_forecast, _VALIDATOR_ATTRIBUTE, validate_emissions)
    return DGLMFamily(
        match_moments=match_moments,
        log_forecast=log_forecast,
        update=update,
        posterior_moments=posterior_moments,
    )


def bernoulli() -> DGLMFamily:
    r"""Bernoulli observations with a logit link.

    The single-trial case of `smcx.binomial`; the one-step forecast
    is the exact beta-Bernoulli,
    $\Pr[y_t = 1] = \alpha_t/(\alpha_t + \beta_t)$. Emissions are in
    $\{0, 1\}$.
    """
    return binomial(trials=1)


class _DGLMCarry(NamedTuple):
    """Moment state carried through the scan."""

    mean: Shaped[Array, " state_dim"]
    covariance: Shaped[Array, "state_dim state_dim"]
    marginal_loglik: Shaped[Array, ""]
    log_evidence_compensation: Shaped[Array, ""]


def dglm_filter(
    initial_mean: Shaped[Array, "*initial_mean_shape"],
    initial_covariance: Shaped[Array, "*initial_cov_shape"],
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    emissions: EmissionSequence,
    *,
    family: DGLMFamily,
    transition_covariance: Shaped[Array, "*evolution_shape"] | None = None,
    discount: Scalar | None = None,
    dispersion_discount: Scalar = 1.0,
) -> DGLMFilterPosterior:
    r"""Run the WHM dynamic generalized linear model filter.

    Args:
        initial_mean: Prior mean $m_0$, shape ``(state_dim,)``.
        initial_covariance: Prior covariance $C_0$; positive
            semidefinite, shape ``(state_dim, state_dim)``. Moments
            only — no distributional form is assumed.
        transition_matrix: State evolution matrix $G$.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``, or a time-varying history with shape
            ``(ntime, state_dim)`` (dynamic regression: row ``t`` is
            the covariate vector at ``t``); the linear predictor is
            $F_t' x_t$.
        emissions: Univariate observations shaped ``(ntime,)`` or
            ``(ntime, 1)``; the family documents its domain (counts
            for `smcx.poisson`). A NaN entry marks that datum missing:
            it bypasses the family support check, the update is
            skipped while the prior construction (including the
            dispersion inflation) stands, the stored conjugates are
            the matched prior pair, and ``log_evidence_increments``
            carries an exact zero there (ADR-0034). NaN needs a float
            dtype, so integer emission arrays remain
            fully-observed-only; infinite entries are rejected
            eagerly.
        family: `DGLMFamily` record of the four conjugate callables.
            Required; there is no default because a mismatched family
            fails silently, not loudly.
        transition_covariance: Evolution covariance $W$ — static
            ``(state_dim, state_dim)`` or timed
            ``(ntime - 1, state_dim, state_dim)``. Supply exactly one
            of this and ``discount``.
        discount: Discount factor $\delta \in (0, 1]$ specifying
            $R_t = G C_{t-1} G' / \delta$.
        dispersion_discount: Berry and West's random-effects discount
            $\rho \in (0, 1]$: the linear-predictor variance is
            inflated to $q_t/\rho$ before moment matching, adding
            unpredictable extra-dispersion each step. The default 1
            recovers the plain DGLM.

    Returns:
        `smcx.containers.DGLMFilterPosterior`. ``filtered_means`` and
        ``filtered_covariances`` are linear-Bayes moment summaries,
        not parameters of a posterior distribution; distributional
        statements flow through the conjugate parameters.
        ``marginal_loglik`` is the exact evidence of the sequentially
        specified approximating model (the sum of closed-form
        forecast log densities under exact moment matching).

    Raises:
        ValueError: Malformed arrays or domains, multivariate
            emissions, an evolution specification that is not
            exactly one of the two forms, a linear predictor
            whose prior variance at the first step is zero — the
            conjugate moment match divides by that variance — or
            concrete emissions outside a built-in family's
            documented support. Value checks run eagerly and are
            skipped for traced arrays.

    Note:
        Only the first step's predictor variance is checkable at the
        boundary. A model whose predictor direction loses all variance
        at a later step (for example a nilpotent transition with zero
        evolution covariance) divides by zero there and returns NaN in
        every field from that step onward (follow-up review R3).
    """
    if (transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of transition_covariance and discount"
        )
    if discount is not None:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
    dispersion_value = _validate_positive_scalar(
        dispersion_discount, "dispersion_discount"
    )
    if not isinstance(dispersion_value, core.Tracer) and (
        float(dispersion_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"dispersion_discount must be in (0, 1]; got {dispersion_value}"
        )

    _check_float_array(initial_mean, "initial_mean")
    dtype = initial_mean.dtype
    state_dim = initial_mean.shape[0] if initial_mean.ndim == 1 else 0
    if state_dim == 0:
        raise ValueError("initial_mean must have shape (state_dim,)")

    emissions = _canonicalize_emissions(emissions)
    if emissions.ndim != 2 or emissions.shape[1] != 1:
        raise ValueError(
            "dglm_filter is univariate: emissions must have shape "
            f"(ntime,) or (ntime, 1); got {emissions.shape}"
        )
    # Support validation sees the concrete input values BEFORE the
    # lossy cast to the working dtype: a float64 1.00000001 truncates
    # to float32 1.0 and a float64 -1e-50 to negative zero, which
    # would defeat integer- and nonnegativity-support checks
    # (follow-up review R8).
    if not isinstance(emissions, core.Tracer):
        _validate_emission_rows(emissions)
    validator = getattr(family.log_forecast, _VALIDATOR_ATTRIBUTE, None)
    if validator is not None and not isinstance(emissions, core.Tracer):
        # Missing (all-NaN) rows bypass the family support check
        # (ADR-0034); every observed entry still faces it. Integer
        # emission arrays cannot encode NaN, so they remain
        # fully-observed-only.
        observed = emissions[~jnp.isnan(emissions)]
        if observed.size:
            validator(observed)
    emission_values = emissions.astype(dtype)
    num_timesteps = emissions.shape[0]

    for name, value, expected in (
        ("initial_covariance", initial_covariance, (state_dim, state_dim)),
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    _check_float_array(observation_vector, "observation_vector", dtype)
    if observation_vector.shape == (state_dim,):
        timed_observation = jnp.broadcast_to(
            observation_vector, (num_timesteps, state_dim)
        )
    elif observation_vector.shape == (num_timesteps, state_dim):
        timed_observation = observation_vector
    else:
        raise ValueError(
            f"observation_vector must have shape ({state_dim},) or "
            f"({num_timesteps}, {state_dim}); got "
            f"{observation_vector.shape}"
        )
    _check_covariance(
        initial_covariance, "initial_covariance", positive_definite=False
    )
    if transition_covariance is not None:
        value = transition_covariance
        _check_float_array(value, "transition_covariance", dtype)
        if value.shape not in (
            (state_dim, state_dim),
            (num_timesteps - 1, state_dim, state_dim),
        ):
            raise ValueError(
                "transition_covariance must have shape "
                f"({state_dim}, {state_dim}) or "
                f"({num_timesteps - 1}, {state_dim}, {state_dim}); "
                f"got {value.shape}"
            )
        _check_covariance(
            value, "transition_covariance", positive_definite=False
        )
        if value.ndim == 2:
            timed_evolution = jnp.broadcast_to(
                value, (num_timesteps - 1, state_dim, state_dim)
            )
        else:
            timed_evolution = value
    else:
        timed_evolution = jnp.zeros(
            (num_timesteps - 1, state_dim, state_dim), dtype=dtype
        )

    use_discount = discount is not None
    inverse_discount = (
        1.0 / jnp.asarray(discount_value, dtype=dtype)
        if use_discount
        else jnp.asarray(1.0, dtype=dtype)
    )

    inverse_dispersion = 1.0 / jnp.asarray(dispersion_value, dtype=dtype)

    def _update(
        carry: _DGLMCarry,
        prior_mean,
        prior_cov,
        emission_t,
        observation_vector,
    ):
        # An all-NaN emission marks the datum missing (ADR-0034): the
        # prior construction (including the dispersion inflation inside
        # the moment match) stands, the observation update is skipped,
        # the stored conjugates are the matched prior pair, and the
        # increment is exactly zero. Zero-filling keeps the unselected
        # branch finite for every shipped family.
        missing, safe_emission = _sanitize_missing(emission_t)
        forecast_mean = observation_vector @ prior_mean
        forecast_variance = (
            observation_vector @ prior_cov @ observation_vector
        ) * inverse_dispersion
        alpha, beta = family.match_moments(forecast_mean, forecast_variance)
        log_density = jnp.asarray(
            family.log_forecast(safe_emission[0], alpha, beta), dtype=dtype
        )
        alpha_post, beta_post = family.update(safe_emission[0], alpha, beta)
        post_mean, post_variance = family.posterior_moments(
            alpha_post, beta_post
        )
        # Measure the conjugate moment change on one represented
        # parameter grid: subtracting the recovered matched moments
        # rather than the supplied forecast moments cancels the moment
        # matcher's solve residual, which otherwise swamps updates
        # smaller than that residual. For the normal family the
        # recovery is the identity, so the Kalman reduction is
        # unchanged.
        matched_mean, matched_variance = family.posterior_moments(alpha, beta)
        gain = prior_cov @ observation_vector / forecast_variance
        new_mean = jnp.where(
            missing,
            prior_mean,
            prior_mean + gain * (post_mean - matched_mean),
        )
        new_cov = jnp.where(
            missing,
            prior_cov,
            prior_cov
            - jnp.outer(gain, gain) * (matched_variance - post_variance),
        )
        alpha_post = jnp.where(
            missing, jnp.asarray(alpha, dtype=dtype), alpha_post
        )
        beta_post = jnp.where(
            missing, jnp.asarray(beta, dtype=dtype), beta_post
        )
        log_density = jnp.where(
            missing, jnp.zeros_like(log_density), log_density
        )
        loglik, compensation = _neumaier_add(
            carry.marginal_loglik,
            carry.log_evidence_compensation,
            log_density,
        )
        new_carry = _DGLMCarry(new_mean, new_cov, loglik, compensation)
        return new_carry, (
            new_mean,
            new_cov,
            jnp.asarray(alpha_post, dtype=dtype),
            jnp.asarray(beta_post, dtype=dtype),
            log_density,
        )

    def _step(carry: _DGLMCarry, args):
        emission_t, evolution_t, observation_t = args
        prior_mean = transition_matrix @ carry.mean
        propagated = (
            transition_matrix @ carry.covariance
        ) @ transition_matrix.T
        if use_discount:
            prior_cov = propagated * inverse_discount
        else:
            prior_cov = propagated + evolution_t
        return _update(carry, prior_mean, prior_cov, emission_t, observation_t)

    init = _DGLMCarry(
        initial_mean.astype(dtype),
        initial_covariance.astype(dtype),
        jnp.zeros((), dtype=dtype),
        jnp.zeros((), dtype=dtype),
    )
    # Library convention: emissions[0] conditions the prior directly;
    # evolution (or discounting) applies between observations. The
    # conjugate moment match divides by the predictor's prior
    # variance, so a direction with zero initial variance is rejected
    # here rather than surfacing as NaN throughout the run (#282).
    first_variance = (
        timed_observation[0] @ init.covariance @ timed_observation[0]
    )
    if not isinstance(first_variance, core.Tracer) and (
        float(first_variance) <= 0.0
    ):
        raise ValueError(
            "the linear predictor has zero prior variance at the first "
            "step (observation_vector @ initial_covariance @ "
            "observation_vector == 0); the conjugate moment match "
            "divides by this variance. Give the predictor direction "
            "positive initial covariance"
        )
    carry_0, first = _update(
        init,
        init.mean,
        init.covariance,
        emission_values[0],
        timed_observation[0],
    )
    final, rest = lax.scan(
        _step,
        carry_0,
        (emission_values[1:], timed_evolution, timed_observation[1:]),
    )
    outputs = tuple(
        jnp.concatenate([first_leaf[None], rest_leaf])
        for first_leaf, rest_leaf in zip(first, rest, strict=True)
    )
    means, covariances, alphas, betas, increments = outputs

    return DGLMFilterPosterior(
        filtered_means=means,
        filtered_covariances=covariances,
        conjugate_alphas=alphas,
        conjugate_betas=betas,
        marginal_loglik=final.marginal_loglik + final.log_evidence_compensation,
        log_evidence_increments=increments,
    )


def _validate_dglm_filter_posterior(
    posterior: DGLMFilterPosterior,
) -> tuple[int, int, jnp.dtype, Float[Array, "ntime state_dim state_dim"]]:
    """Validate a DGLM record and canonicalize its covariance history."""
    means = posterior.filtered_means
    if means.ndim != 2 or means.shape[0] == 0 or means.shape[1] == 0:
        raise ValueError("filtered_means must have shape (T, d) with T, d > 0")
    num_timesteps, state_dim = means.shape
    dtype = means.dtype
    _check_float_array(means, "filtered_means")
    names = (
        "filtered_covariances",
        "conjugate_alphas",
        "conjugate_betas",
        "log_evidence_increments",
    )
    shapes = ((num_timesteps, state_dim, state_dim), *((num_timesteps,),) * 3)
    for name, shape in zip(names, shapes, strict=True):
        value = getattr(posterior, name)
        _check_float_array(value, name, dtype)
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
    marginal = jnp.asarray(posterior.marginal_loglik)
    if marginal.ndim != 0:
        raise ValueError("marginal_loglik must be scalar")
    _check_float_array(marginal, "marginal_loglik", dtype)
    name = names[0]
    covariance = _canonicalize_filter_covariances(
        getattr(posterior, name), name
    )
    return num_timesteps, state_dim, dtype, covariance


def dglm_smoother(
    filtered_posterior: DGLMFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_shape"],
    *,
    transition_covariance: Shaped[Array, "*evolution_shape"] | None = None,
    discount: Scalar | None = None,
) -> DGLMSmootherPosterior:
    r"""Compute retrospective DGLM state moments (Alves et al., 2025).

    Args:
        filtered_posterior: Result of `smcx.dglm_filter` with full history.
        transition_matrix: Static state evolution matrix $G$, shape
            ``(state_dim, state_dim)``.
        transition_covariance: State evolution covariance $W_t$, positive
            semidefinite and either static ``(state_dim, state_dim)`` or timed
            ``(T - 1, state_dim, state_dim)``. Supply exactly one of this and
            ``discount``.
        discount: Scalar state discount $\delta \in (0, 1]$. Supply exactly one
            of this and ``transition_covariance``.

    Returns:
        `smcx.containers.DGLMSmootherPosterior`. For a general DGLM, its
        smoothed arrays are approximate retrospective linear-Bayes state
        moments, not a smoothing distribution or credible intervals.

    Raises:
        ValueError: Invalid record, evolution shape, dtype, or covariance.

    Note:
        Resupply the filter's $G$, $W$, or $\delta$. The record cannot verify
        this fact; a mismatch can silently describe the wrong moments. The
        smoother needs no observation family, observation vector, emissions,
        or dispersion discount. Retained conjugate parameters remain
        filtering-time quantities and are not rematched.

        The result stores a canonical symmetric filtering covariance history;
        the caller's record is unchanged. Positive-time reconstructed priors
        must be positive definite because the shared kernel factors them.
        Filtered covariances may be semidefinite, and $T=1$ takes no
        factorization. A traced factorization failure can produce NaNs or a
        JAX debug error. This covariance-form Joseph update is not a
        square-root method. Reconstructed noise can have roundoff-sized
        negative modes, and an ill-conditioned transition can amplify error
        by roughly its squared condition number per step.

        With a normal-family filter record and ``dispersion_discount=1``, the
        result is the exact RTS posterior. General DGLMs additionally inherit
        the filter's three approximation layers and add a retrospective
        Gaussian/Bayes-linear state projection, with no error bound.

    References:
        Alves, M. B., Migon, H. S., Santos Jr, S. V., and Marotta, R. (2025).
        An Efficient Sequential Approach for k-Parametric Dynamic Generalised
        Linear Models, section 3.2 and Algorithm 2.
        https://arxiv.org/html/2201.05387v4#S3.SS2
    """
    if (transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of transition_covariance and discount"
        )
    num_timesteps, state_dim, dtype, canonical_covariances = (
        _validate_dglm_filter_posterior(filtered_posterior)
    )
    _check_float_array(transition_matrix, "transition_matrix", dtype)
    expected = (state_dim, state_dim)
    if transition_matrix.shape != expected:
        raise ValueError(
            f"transition_matrix shape {transition_matrix.shape} != {expected}"
        )

    if transition_covariance is not None:
        evolution = transition_covariance
        _check_float_array(evolution, "transition_covariance", dtype)
        timed_evolution = _time_matrix(
            evolution,
            num_timesteps - 1,
            state_dim,
            state_dim,
            "transition_covariance",
        )
        _check_covariance(
            evolution, "transition_covariance", positive_definite=False
        )
    else:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(f"discount outside (0, 1]: {discount_value}")
        inverse_discount = 1.0 / jnp.asarray(discount_value, dtype=dtype)

    propagated = (
        transition_matrix @ canonical_covariances[:-1]
    ) @ transition_matrix.T
    if transition_covariance is None:
        next_predicted_covariances = _symmetrize(propagated * inverse_discount)
    else:
        next_predicted_covariances = _symmetrize(propagated + timed_evolution)
    _check_covariance(
        next_predicted_covariances,
        "reconstructed_predicted_covariances",
        positive_definite=True,
    )
    predicted_means = jnp.concatenate((
        filtered_posterior.filtered_means[:1],
        filtered_posterior.filtered_means[:-1] @ transition_matrix.T,
    ))
    predicted_covariances = jnp.concatenate((
        canonical_covariances[:1],
        next_predicted_covariances,
    ))
    smoothed_means, smoothed_covariances = _backward_pass(
        filtered_posterior.filtered_means,
        canonical_covariances,
        predicted_means,
        predicted_covariances,
        jnp.broadcast_to(
            transition_matrix, (num_timesteps - 1, state_dim, state_dim)
        ),
    )
    return DGLMSmootherPosterior(
        filtered_means=filtered_posterior.filtered_means,
        filtered_covariances=canonical_covariances,
        conjugate_alphas=filtered_posterior.conjugate_alphas,
        conjugate_betas=filtered_posterior.conjugate_betas,
        marginal_loglik=filtered_posterior.marginal_loglik,
        log_evidence_increments=filtered_posterior.log_evidence_increments,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covariances,
    )


def dglm_forecast(
    filtered_posterior: DGLMFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    *,
    family: DGLMFamily,
    num_steps: _CountArgument,
    transition_covariance: Shaped[Array, "*evolution_shape"] | None = None,
    discount: Scalar | None = None,
    dispersion_discount: Scalar = 1.0,
) -> DGLMForecast:
    r"""Iterate k-step DGLM forecast states from the filtering frontier.

    Starting at the terminal linear-Bayes moments $(m_T, C_T)$, each
    horizon applies the Gaussian state prediction
    $a(k) = G\,a(k-1)$ and $R(k) = G\,R(k-1)\,G^\top + W_{T+k}$, maps
    to the linear predictor $f(k) = F^\top a(k)$ with variance
    $q(k) = F^\top R(k)\,F / \rho$ (the dispersion inflation), and
    moment-matches the family's conjugate pair at $(f(k), q(k))$ —
    the filter's own one-step forecast construction applied at each
    horizon, honest as an approximation with exactly the filter's
    linear-Bayes caveats. The observation forecast at horizon $k$ is
    the family's conjugate forecast at the matched pair (its density
    is ``family.log_forecast``).

    Under a ``discount`` specification the evolution variance is
    frozen at its frontier value
    $W = G\,C_T\,G^\top (1 - \delta)/\delta$ for every horizon (West
    and Harrison 1997, section 6.3.3), matching `smcx.dlm_forecast`:
    a discount forecast agrees with filtering through an all-NaN gap
    at horizon one and diverges beyond it, while an explicit
    evolution covariance agrees at every horizon.

    Args:
        filtered_posterior: Output of `smcx.dglm_filter`. The
            forecast cannot verify that the resupplied pieces below
            match the filter run.
        transition_matrix: Static state evolution matrix $G$, shape
            ``(state_dim, state_dim)``.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``.
        family: The `smcx.DGLMFamily` used by the filter.
        num_steps: Positive integer forecast horizon. This controls an
            output shape and must be closed over or marked static
            through an outer ``jax.jit`` boundary.
        transition_covariance: Static ``(state_dim, state_dim)`` or
            per-horizon ``(num_steps, state_dim, state_dim)``
            evolution covariance. Supply exactly one of this and
            ``discount``.
        discount: State discount $\delta \in (0, 1]$ used by the
            filter.
        dispersion_discount: Berry--West random-effects discount
            $\rho \in (0, 1]$ used by the filter; each horizon's
            linear-predictor variance is inflated by $1/\rho$ inside
            the moment match, exactly as in the filter.

    Returns:
        Per-horizon Gaussian state moments, linear-predictor moments,
        and moment-matched conjugate pairs.

    Raises:
        ValueError: The posterior or a model piece has an invalid
            shape, dtype, count, or domain, the evolution
            specification is not exactly one of covariance and
            discount, or the linear predictor has nonpositive
            forecast variance at some horizon (the conjugate moment
            match divides by it).

    Note:
        No data enter a forecast: the conjugate pairs are matched
        priors, never updated, and the horizon-one pair reproduces the
        pair a continued filter run would match, so
        ``family.log_forecast`` at that pair reproduces the filter's
        evidence increment for a realized emission.
    """
    if (transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of transition_covariance and discount"
        )
    num_steps = _positive_integer(num_steps, name="num_steps")
    _, state_dim, dtype, canonical_covariances = (
        _validate_dglm_filter_posterior(filtered_posterior)
    )
    del canonical_covariances
    for name, value, expected in (
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    use_discount = discount is not None
    if use_discount:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
        inverse_discount = 1.0 / jnp.asarray(discount_value, dtype=dtype)
    else:
        value = transition_covariance
        assert value is not None  # XOR-checked above
        _check_float_array(value, "transition_covariance", dtype)
        timed_evolution = _time_matrix(
            value,
            num_steps,
            state_dim,
            state_dim,
            "transition_covariance",
        )
        _check_covariance(
            value,
            "transition_covariance",
            positive_definite=False,
        )
    dispersion_value = _validate_positive_scalar(
        dispersion_discount, "dispersion_discount"
    )
    if not isinstance(dispersion_value, core.Tracer) and (
        float(dispersion_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"dispersion_discount must be in (0, 1]; got {dispersion_value}"
        )
    inverse_dispersion = 1.0 / jnp.asarray(dispersion_value, dtype=dtype)

    mean = filtered_posterior.filtered_means[-1]
    # The raw stored covariance IS the filter's carry; symmetrizing it
    # here would break the exact agreement with a continued filter run.
    covariance = filtered_posterior.filtered_covariances[-1]

    propagated_frontier = (transition_matrix @ covariance) @ transition_matrix.T
    if use_discount:
        first_covariance = propagated_frontier * inverse_discount
        frontier_w = propagated_frontier * (inverse_discount - 1.0)
        rest_evolution = jnp.broadcast_to(
            frontier_w, (num_steps - 1, state_dim, state_dim)
        )
    else:
        first_covariance = propagated_frontier + timed_evolution[0]
        rest_evolution = timed_evolution[1:]
    first_mean = transition_matrix @ mean

    def _match(mean_k, covariance_k):
        # Scalar ops in the filter's exact order, so the horizon-one
        # pair reproduces a continued filter run bitwise.
        forecast_mean = observation_vector @ mean_k
        forecast_variance = (
            observation_vector @ covariance_k @ observation_vector
        ) * inverse_dispersion
        alpha, beta = family.match_moments(forecast_mean, forecast_variance)
        return forecast_mean, forecast_variance, alpha, beta

    first_match = _match(first_mean, first_covariance)

    def _horizon(carry, evolution_k):
        mean_k, covariance_k = carry
        next_mean = transition_matrix @ mean_k
        next_covariance = (
            transition_matrix @ covariance_k
        ) @ transition_matrix.T + evolution_k
        return (next_mean, next_covariance), (
            next_mean,
            next_covariance,
            *_match(next_mean, next_covariance),
        )

    _, rest = lax.scan(_horizon, (first_mean, first_covariance), rest_evolution)
    first_row = (first_mean, first_covariance, *first_match)
    outputs = tuple(
        jnp.concatenate([jnp.asarray(first_leaf)[None], rest_leaf])
        for first_leaf, rest_leaf in zip(first_row, rest, strict=True)
    )
    (
        state_means,
        state_covariances,
        predictor_means,
        predictor_variances,
        alphas,
        betas,
    ) = outputs
    if not isinstance(predictor_variances, core.Tracer):
        concrete_variances = np.asarray(predictor_variances)
        if not np.all(concrete_variances > 0.0):
            # The filter names this domain at its own boundary; the
            # forecast used to return NaN conjugate parameters instead
            # (2026-08-06 review, P2-3).
            raise ValueError(
                "the linear predictor has nonpositive forecast "
                "variance at some horizon (observation_vector @ "
                "state_covariance @ observation_vector <= 0); the "
                "conjugate moment match divides by this variance. "
                "Give the predictor direction positive covariance "
                "through the frontier or the evolution noise"
            )
    return DGLMForecast(
        state_means=state_means,
        state_covariances=state_covariances,
        linear_predictor_means=predictor_means,
        linear_predictor_variances=predictor_variances,
        conjugate_alphas=alphas,
        conjugate_betas=betas,
    )


def dglm_forecast_sample(
    key: PRNGKeyT,
    filtered_posterior: DGLMFilterPosterior,
    transition_matrix: Shaped[Array, "*transition_shape"],
    observation_vector: Shaped[Array, "*observation_shape"],
    *,
    sample_emission: FamilyEmissionSampler,
    num_steps: _CountArgument,
    num_draws: _CountArgument,
    transition_covariance: Shaped[Array, "*evolution_shape"] | None = None,
    discount: Scalar | None = None,
    dispersion_discount: Scalar = 1.0,
) -> DGLMForecastPaths:
    r"""Simulate joint DGLM forecast paths from the filtering frontier.

    Each path draws a terminal state from the Gaussian frontier,
    iterates the state equation with per-horizon evolution noise
    (frozen at its frontier value under a ``discount``, matching
    `smcx.dglm_forecast`), and samples each horizon's emission through
    the family's link via the standalone ``sample_emission``. The construction
    is honest as an approximation with exactly the filter's
    linear-Bayes caveats: state-path marginals reproduce
    `dglm_forecast`'s Gaussians, while emission marginals follow the
    link-mixture rather than the moment-matched conjugate forecast.

    Under a ``dispersion_discount`` $\rho < 1$ each horizon adds an
    independent Gaussian shock to the linear predictor with variance
    $(1/\rho - 1)\,F^\top R(k)\,F$ — the Berry--West random-effects
    reading of unpredictable extra dispersion — so the predictor
    variance reproduces the closed-form inflation exactly.

    Args:
        key: JAX PRNG key. Split once into a terminal-state key and
            one key per horizon; each horizon key splits into a
            transition-noise key, a dispersion-shock key, and a
            per-draw emission key root.
        filtered_posterior: Output of `smcx.dglm_filter`. The forecast
            cannot verify that the resupplied pieces below match the
            filter run.
        transition_matrix: Static state evolution matrix $G$, shape
            ``(state_dim, state_dim)``.
        observation_vector: Observation vector $F$, shape
            ``(state_dim,)``.
        sample_emission: ``(key, linear_predictor) -> emission`` —
            the family's link commitment for simulation, for example
            ``lambda key, lam: jr.poisson(key, jnp.exp(lam))`` for the
            Poisson log link. Standalone by design: the released
            `smcx.DGLMFamily` contract is a four-field sequence, so
            the capability joins the record only at 3.0.
        num_steps: Positive integer forecast horizon. This controls an
            output shape and must be closed over or marked static
            through an outer ``jax.jit`` boundary.
        num_draws: Positive integer path count; also an output shape.
        transition_covariance: Static ``(state_dim, state_dim)`` or
            per-horizon ``(num_steps, state_dim, state_dim)``
            evolution covariance. Supply exactly one of this and
            ``discount``.
        discount: State discount $\delta \in (0, 1]$ used by the
            filter.
        dispersion_discount: Berry--West random-effects discount
            $\rho \in (0, 1]$ used by the filter.

    Returns:
        Draw-major state and emission trajectories.

    Raises:
        ValueError: The posterior or a model piece has an invalid
            shape, dtype, count, or domain, the evolution
            specification is not exactly one of covariance and
            discount, or a concrete covariance is not factorable on
            the active backend.
    """
    if (transition_covariance is None) == (discount is None):
        raise ValueError(
            "supply exactly one of transition_covariance and discount"
        )
    num_steps = _positive_integer(num_steps, name="num_steps")
    num_draws = _positive_integer(num_draws, name="num_draws")
    _, state_dim, dtype, canonical_covariances = (
        _validate_dglm_filter_posterior(filtered_posterior)
    )
    del canonical_covariances
    for name, value, expected in (
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
            )
    use_discount = discount is not None
    if use_discount:
        discount_value = _validate_positive_scalar(discount, "discount")
        if not isinstance(discount_value, core.Tracer) and (
            float(discount_value) > 1.0  # ty: ignore[invalid-argument-type]
        ):
            raise ValueError(
                f"discount must be in (0, 1]; got {discount_value}"
            )
        inverse_discount = 1.0 / jnp.asarray(discount_value, dtype=dtype)
    else:
        value = transition_covariance
        assert value is not None  # XOR-checked above
        _check_float_array(value, "transition_covariance", dtype)
        timed_evolution = _time_matrix(
            value,
            num_steps,
            state_dim,
            state_dim,
            "transition_covariance",
        )
        _check_covariance(
            value,
            "transition_covariance",
            positive_definite=False,
        )
    dispersion_value = _validate_positive_scalar(
        dispersion_discount, "dispersion_discount"
    )
    if not isinstance(dispersion_value, core.Tracer) and (
        float(dispersion_value) > 1.0  # ty: ignore[invalid-argument-type]
    ):
        raise ValueError(
            f"dispersion_discount must be in (0, 1]; got {dispersion_value}"
        )
    excess_dispersion = 1.0 / jnp.asarray(
        dispersion_value, dtype=dtype
    ) - jnp.asarray(1.0, dtype=dtype)

    mean = filtered_posterior.filtered_means[-1]
    # The raw stored covariance IS the filter's carry; symmetrizing it
    # here would break coherence with a continued filter run.
    covariance = filtered_posterior.filtered_covariances[-1]

    propagated_frontier = (transition_matrix @ covariance) @ transition_matrix.T
    if use_discount:
        frontier_w = propagated_frontier * (inverse_discount - 1.0)
        noise_covariances = jnp.broadcast_to(
            frontier_w, (num_steps, state_dim, state_dim)
        )
    else:
        noise_covariances = timed_evolution
    with debug_nans(False), debug_infs(False):
        terminal_factor = _sampling_covariance_factor(_symmetrize(covariance))
        noise_factors = lax.map(_sampling_covariance_factor, noise_covariances)
    _check_sampling_factors(terminal_factor[None])
    _check_sampling_factors(noise_factors)

    keys = jr.split(key, num_steps + 1)
    terminal_noise = jr.normal(keys[0], (num_draws, state_dim), dtype=dtype)
    states_0 = mean + terminal_noise @ terminal_factor.T

    def _path_step(carry, args):
        states, closed_covariance = carry
        step_key, transition_noise_cov, noise_factor = args
        state_key, shock_key, emission_root = jr.split(step_key, 3)
        state_noise = jr.normal(state_key, (num_draws, state_dim), dtype=dtype)
        next_states = states @ transition_matrix.T + (
            state_noise @ noise_factor.T
        )
        next_closed = (
            transition_matrix @ closed_covariance
        ) @ transition_matrix.T + transition_noise_cov
        predictors = next_states @ observation_vector
        shock_variance = excess_dispersion * (
            observation_vector @ next_closed @ observation_vector
        )
        shock = jnp.sqrt(shock_variance) * jr.normal(
            shock_key, (num_draws,), dtype=dtype
        )
        predictors = predictors + shock
        emission_keys = jr.split(emission_root, num_draws)
        emissions = jax.vmap(sample_emission)(emission_keys, predictors)
        return (next_states, next_closed), (next_states, emissions)

    _, (state_paths, emission_paths) = lax.scan(
        _path_step,
        (states_0, covariance),
        (keys[1:], noise_covariances, noise_factors),
    )
    return DGLMForecastPaths(
        state_paths=jnp.swapaxes(state_paths, 0, 1),
        emission_paths=jnp.swapaxes(emission_paths, 0, 1),
    )
