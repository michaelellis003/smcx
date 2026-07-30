# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""Dynamic generalized linear model filtering (West, Harrison & Migon).

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
posterior moments are exact. For the normal family both
approximations vanish and the recursion is exactly the Kalman filter
[West, Harrison & Migon, 1985, sec. 2.2] — the primary test gate.
This filter sits between `smcx.dlm_filter` (exact) and the particle
filters (asymptotically exact), which are the natural accuracy check.

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
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import core, lax
from jax.nn import sigmoid as jax_sigmoid
from jax.nn import softplus
from jax.scipy.special import digamma, gammaln, polygamma
from jax.scipy.stats import betabinom, nbinom
from jaxtyping import Array, Shaped

from smcx._numerics import _neumaier_add
from smcx._utils import _canonicalize_emissions
from smcx.containers import DGLMFilterPosterior
from smcx.dlm import _validate_positive_scalar
from smcx.kalman import _check_covariance, _check_float_array
from smcx.types import (
    EmissionSequence,
    FamilyConjugateUpdate,
    FamilyLogForecast,
    FamilyMomentMatch,
    FamilyPosteriorMoments,
    Scalar,
)

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
        family are not support-checked.
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
            ``(state_dim,)``; the linear predictor is $F' x_t$.
        emissions: Univariate observations shaped ``(ntime,)`` or
            ``(ntime, 1)``; the family documents its domain (counts
            for `smcx.poisson`).
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
            documented support.

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
    validator = getattr(family.log_forecast, _VALIDATOR_ATTRIBUTE, None)
    if validator is not None and not isinstance(emissions, core.Tracer):
        validator(emissions)
    emission_values = emissions.astype(dtype)
    num_timesteps = emissions.shape[0]

    for name, value, expected in (
        ("initial_covariance", initial_covariance, (state_dim, state_dim)),
        ("transition_matrix", transition_matrix, (state_dim, state_dim)),
        ("observation_vector", observation_vector, (state_dim,)),
    ):
        _check_float_array(value, name, dtype)
        if value.shape != expected:
            raise ValueError(
                f"{name} must have shape {expected}; got {value.shape}"
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

    def _update(carry: _DGLMCarry, prior_mean, prior_cov, emission_t):
        forecast_mean = observation_vector @ prior_mean
        forecast_variance = (
            observation_vector @ prior_cov @ observation_vector
        ) * inverse_dispersion
        alpha, beta = family.match_moments(forecast_mean, forecast_variance)
        log_density = jnp.asarray(
            family.log_forecast(emission_t[0], alpha, beta), dtype=dtype
        )
        alpha_post, beta_post = family.update(emission_t[0], alpha, beta)
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
        new_mean = prior_mean + gain * (post_mean - matched_mean)
        new_cov = prior_cov - jnp.outer(gain, gain) * (
            matched_variance - post_variance
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
        emission_t, evolution_t = args
        prior_mean = transition_matrix @ carry.mean
        propagated = (
            transition_matrix @ carry.covariance
        ) @ transition_matrix.T
        if use_discount:
            prior_cov = propagated * inverse_discount
        else:
            prior_cov = propagated + evolution_t
        return _update(carry, prior_mean, prior_cov, emission_t)

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
    first_variance = observation_vector @ init.covariance @ observation_vector
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
        init, init.mean, init.covariance, emission_values[0]
    )
    final, rest = lax.scan(
        _step, carry_0, (emission_values[1:], timed_evolution)
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
