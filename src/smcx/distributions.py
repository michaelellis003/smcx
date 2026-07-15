# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Distribution log-densities and samplers (ADR-0012, design §7).

Flat functions, not distribution objects — enough for the closures
users write. Scalars broadcast per MLX rules; densities are
per-event (vmap for particle batches). Numerics follow
docs/research/numerical-methods.md: ``-log1p(-u)`` for exponentials
(uniform can return exactly 0), ``log1p`` forms for Bernoulli and
Student-t, and a guarded factorization for MvNormal so the hot loop
is matmul-only (no ``mx.linalg``, no CPU stream).

References: Lanczos (1964) & Godfrey (2001) g=7, n=9 coefficient set
as used in GSL/Boost/CPython (unencumbered; see
docs/research/licensing.md); Marsaglia & Tsang (2000, ACM TOMS
26(3)); Wilson & Hilferty (1931); Devroye (1986, Ch. V); Higham
(1988) nearest-PSD projection.
"""

import math
from typing import NamedTuple

import mlx.core as mx
import numpy as np
from jaxtyping import Float

from smcx.resampling import _searchsorted, _sorted_uniforms
from smcx.types import KeyT

_LOG_2PI = math.log(2.0 * math.pi)
_LANCZOS_G = 7.0
_LANCZOS_COEF = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def lgamma(x: mx.array) -> mx.array:
    """Log-gamma via the Lanczos approximation (g=7, 9 coefficients).

    Max relative error 1.3e-6 in float32 over x in [0.01, 1e4],
    including the reflection formula for x < 0.5
    (docs/research/mlx-audit.md; upstream lgamma was declined,
    ADR-0006).
    """

    def _positive(z: mx.array) -> mx.array:
        z = z - 1.0
        base = z + _LANCZOS_G + 0.5
        s = mx.full(z.shape, _LANCZOS_COEF[0])
        for i, c in enumerate(_LANCZOS_COEF[1:], start=1):
            s = s + c / (z + i)
        return (
            0.5 * math.log(2.0 * math.pi)
            + (z + 0.5) * mx.log(base)
            - base
            + mx.log(s)
        )

    xa = mx.where(x < 0.5, 1.0 - x, x)
    lg = _positive(xa)
    reflected = math.log(math.pi) - mx.log(mx.abs(mx.sin(math.pi * x))) - lg
    return mx.where(x < 0.5, reflected, lg)


# --- univariate log-densities ----------------------------------------


def normal_logpdf(x: mx.array, loc: mx.array, scale: mx.array) -> mx.array:
    """Normal log-density."""
    z = (x - loc) / scale
    return -0.5 * (_LOG_2PI + z * z) - mx.log(scale)


def exponential_logpdf(x: mx.array, rate: mx.array) -> mx.array:
    """Exponential (rate parameterization); -inf below 0."""
    return mx.where(x >= 0, mx.log(rate) - rate * x, mx.array(-mx.inf))


def laplace_logpdf(x: mx.array, loc: mx.array, scale: mx.array) -> mx.array:
    """Laplace log-density."""
    return -mx.log(2.0 * scale) - mx.abs(x - loc) / scale


def uniform_logpdf(x: mx.array, low: mx.array, high: mx.array) -> mx.array:
    """Uniform on [low, high); -inf outside the support."""
    inside = (x >= low) & (x < high)
    return mx.where(inside, -mx.log(high - low), mx.array(-mx.inf))


def bernoulli_logpmf(k: mx.array, p: mx.array) -> mx.array:
    """Bernoulli log-pmf; uses log1p(-p) (stable at extreme p)."""
    return k * mx.log(p) + (1.0 - k) * mx.log1p(-p)


def categorical_logpmf(k: mx.array, logits: mx.array) -> mx.array:
    """Categorical log-pmf from unnormalized logits."""
    return mx.take(logits, k) - mx.logsumexp(logits)


def gamma_logpdf(x: mx.array, shape: mx.array, rate: mx.array) -> mx.array:
    """Gamma log-density, shape/rate parameterization; -inf at x<=0."""
    dens = (
        shape * mx.log(rate)
        - lgamma(shape)
        + (shape - 1.0) * mx.log(x)
        - rate * x
    )
    return mx.where(x > 0, dens, mx.array(-mx.inf))


def studentt_logpdf(
    x: mx.array, df: mx.array, loc: mx.array, scale: mx.array
) -> mx.array:
    """Student-t log-density; log1p keeps the tail term stable."""
    z = (x - loc) / scale
    return (
        lgamma((df + 1.0) / 2.0)
        - lgamma(df / 2.0)
        - 0.5 * mx.log(df * math.pi)
        - mx.log(scale)
        - (df + 1.0) / 2.0 * mx.log1p(z * z / df)
    )


# --- multivariate normal (guarded factorization; design §7) ----------


class CholFactors(NamedTuple):
    """One-time MvNormal factorization: hot loop stays matmul-only."""

    scale_tril: Float[mx.array, "dim dim"]
    inv_scale_tril: Float[mx.array, "dim dim"]
    half_log_det: Float[mx.array, ""]


def chol_factor(cov) -> CholFactors:
    """Factor a covariance once, with the silent-failure guard.

    Factorizes in numpy float64 (which raises on non-PD input —
    unlike MLX's f32 cholesky, which returns finite garbage silently;
    docs/research/numerical-methods.md). On failure, escalates jitter
    from 1e-6*(trace/d) by 10x; the final fallback is an
    eigenvalue-clip (Higham 1988) recomposition. The returned f32
    factors make ``mvnormal_logpdf`` matmul-only.

    Args:
        cov: (d, d) covariance (numpy or MLX array).

    Returns:
        CholFactors(scale_tril, inv_scale_tril, half_log_det).
    """
    cov64 = np.array(cov, dtype=np.float64)
    d = cov64.shape[0]
    base = 1e-6 * np.trace(cov64) / d
    lower = None
    for attempt in range(-1, 6):
        jitter = 0.0 if attempt < 0 else base * 10.0**attempt
        try:
            lower = np.linalg.cholesky(cov64 + jitter * np.eye(d))
            break
        except np.linalg.LinAlgError:
            continue
    if lower is None:
        # eigh-clip fallback: nearest-PSD recomposition, then factor.
        w, v = np.linalg.eigh((cov64 + cov64.T) / 2.0)
        floor = max(base, 1e-12)
        cov_psd = (v * np.maximum(w, floor)) @ v.T
        lower = np.linalg.cholesky(cov_psd + floor * np.eye(d))
    inv_lower = np.linalg.inv(lower)
    half_log_det = float(np.sum(np.log(np.diag(lower))))
    return CholFactors(
        mx.array(lower.astype(np.float32)),
        mx.array(inv_lower.astype(np.float32)),
        mx.array(np.float32(half_log_det)),
    )


def mvnormal_logpdf(
    x: Float[mx.array, " dim"],
    mean: Float[mx.array, " dim"],
    factors: CholFactors,
) -> Float[mx.array, ""]:
    """MvNormal log-density from precomputed factors (matmul-only)."""
    d = x.shape[0]
    z = factors.inv_scale_tril @ (x - mean)
    return -0.5 * d * _LOG_2PI - factors.half_log_det - 0.5 * mx.sum(z * z)


def mvnormal_sample(
    key: KeyT,
    mean: Float[mx.array, " dim"],
    factors: CholFactors,
    shape: tuple = (),
) -> mx.array:
    """Draw MvNormal samples: ``mean + z @ L.T`` (reparameterized)."""
    d = mean.shape[0]
    z = mx.random.normal((*shape, d), key=key)
    return mean + z @ factors.scale_tril.T


# --- samplers ---------------------------------------------------------


def exponential_sample(key: KeyT, rate: float, shape: tuple) -> mx.array:
    """Exponential draws via ``-log1p(-u)/rate`` (never log(0))."""
    return -mx.log1p(-mx.random.uniform(shape=shape, key=key)) / rate


def categorical_sample(
    key: KeyT, logits: mx.array, num_samples: int
) -> mx.array:
    """Iid categorical draws via inverse-CDF.

    Never uses ``mx.random.categorical(num_samples=...)`` — O(N*M)
    memory (mlx-audit hazard 1). Sorted uniforms keep the CDF search
    and any downstream gather monotone.
    """
    cdf = mx.cumsum(mx.softmax(logits))
    cdf = cdf / cdf[-1]
    u = _sorted_uniforms(key, num_samples)
    return _searchsorted(cdf, u)


def gamma_sample(
    key: KeyT, shape_param: float, rate: float, shape: tuple
) -> mx.array:
    """Gamma draws: fixed-round masked Marsaglia-Tsang (2000).

    Eight rejection rounds resolve all but ~(0.05)^8 ~ 4e-11 of
    particles at the alpha=1 worst case (round-1 acceptance >= 0.951
    for alpha >= 1, verified); stragglers fall back to
    Wilson-Hilferty (1931), whose bias at that probability is
    unmeasurable. For shape < 1, the boost
    ``X_a = X_{a+1} * U^(1/a)`` runs in log space (the prob-space
    power underflows f32).

    Args:
        key: PRNG key.
        shape_param: Gamma shape alpha (Python float, > 0).
        rate: Rate beta (Python float, > 0).
        shape: Output shape.

    Returns:
        Positive gamma draws with mean alpha/beta.
    """
    boosted = shape_param < 1.0
    alpha = shape_param + 1.0 if boosted else shape_param
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    rounds = 8
    keys = mx.random.split(key, 2 * rounds + 2)
    result = mx.zeros(shape)
    # (comparison yields a bool array; the generated mlx stub lacks
    # mx.bool_ even though runtime has it)
    done = mx.zeros(shape) > 1.0
    for r in range(rounds):
        zn = mx.random.normal(shape, key=keys[2 * r])
        u = mx.random.uniform(shape=shape, key=keys[2 * r + 1])
        v = (1.0 + c * zn) ** 3
        ok = (v > 0) & (
            mx.log(mx.maximum(u, 1e-37))
            < 0.5 * zn * zn + d - d * v + d * mx.log(mx.maximum(v, 1e-37))
        )
        take_it = ok & ~done
        result = mx.where(take_it, d * v, result)
        done = done | ok
    # Wilson-Hilferty straggler fill (P(used) ~ 4e-11 per particle).
    zw = mx.random.normal(shape, key=keys[2 * rounds])
    wh = alpha * (
        mx.maximum(
            1.0 - 1.0 / (9.0 * alpha) + zw / (3.0 * math.sqrt(alpha)),
            1e-3,
        )
        ** 3
    )
    x = mx.where(done, result, wh)
    if boosted:
        u_boost = mx.random.uniform(shape=shape, key=keys[2 * rounds + 1])
        x = mx.exp(mx.log(x) + mx.log1p(-u_boost) / shape_param)
    return x / rate
