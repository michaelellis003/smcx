# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

r"""Adaptive tempered SMC for static targets.

Anneals from the prior to the posterior
:math:`\pi_\phi \propto p(x)\, L(x)^\phi` along an adaptive schedule
[Del Moral, Doucet & Jasra, 2006]: the next temperature solves
``ESS(phi) = target_ess * N`` by bisection on the *resident*
log-likelihood vector — a deterministic solve, no fresh sampling
(Jasra et al., 2011). Each stage reweights by
:math:`\ell \cdot \Delta\phi` (evidence increment at the reweight,
pre-move — the Del Moral et al. collapse), resamples, and applies
:math:`\pi_{\phi'}`-invariant random-walk Metropolis moves whose
proposal covariance is :math:`2.38^2/d \cdot \hat\Sigma` from the
*weighted* pre-resample cloud (Roberts & Rosenthal, 2001) — two-pass
in f64 on the host (single-pass cancels catastrophically at ordinary
posterior offsets) through the guarded ``chol_factor``.

This is the thesis workload: N-wide likelihood evaluations and RWM
sweeps are pure batched compute; the whole sweep is one
``mx.compile``d function with acceptance via ``where`` (never a
Python branch on array values).
"""

import math

import mlx.core as mx
import numpy as np
from jaxtyping import Float

from smcx.containers import TemperedPosterior
from smcx.distributions import chol_factor
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import systematic
from smcx.types import InitialSampler, KeyT
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

_BISECT_ITERS = 60
_RWM_SCALE = 2.38


def _weighted_cov_f64(particles: mx.array, weights: mx.array) -> np.ndarray:
    """Two-pass weighted covariance on the host in f64.

    One call per temperature stage — setup cost, not hot loop. The
    single-pass form is 1% wrong at a 100-sigma mean offset
    (docs/research/numerical-methods.md §5); two-pass in f64 is
    exact to ~3e-7.
    """
    x = np.array(particles, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    mu = w @ x
    xc = x - mu
    return (xc * w[:, None]).T @ xc


def temper(
    key: KeyT,
    initial_sampler: InitialSampler,
    log_prior_fn,
    log_likelihood_fn,
    num_particles: int,
    num_mcmc_steps: int = 5,
    target_ess: float = 0.5,
    resampling_fn=systematic,
    *,
    max_stages: int = 1000,
) -> TemperedPosterior:
    r"""Sample a static target by adaptive tempered SMC.

    Args:
        key: PRNG key.
        initial_sampler: ``(key, num_particles) -> (N, d)`` drawing
            from the prior.
        log_prior_fn: Per-particle ``(state) -> scalar`` log-prior;
            vmapped internally.
        log_likelihood_fn: Per-particle ``(state) -> scalar``
            log-likelihood; vmapped internally.
        num_particles: Number of particles N.
        num_mcmc_steps: RWM sweeps per temperature stage.
        target_ess: The bisection solves ``ESS = target_ess * N``
            for each stage's temperature increment.
        resampling_fn: ADR-0004 contract resampler (applied at every
            stage — the schedule drives ESS to the target by
            construction).
        max_stages: Safety cap on the number of stages.

    Returns:
        :class:`~smcx.containers.TemperedPosterior` with equal-weight
        posterior draws, the log-evidence estimate, and per-stage
        temperature/ESS/acceptance traces.

    Raises:
        DegenerateWeightsError: The likelihood is ``-inf`` on the
            whole cloud (no tempering step is possible).
        RuntimeError: ``max_stages`` exceeded before reaching
            ``phi = 1``.
    """
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}")
    n = num_particles
    log_n = math.log(n)
    key, k_init = mx.random.split(key)
    particles = initial_sampler(k_init, n)
    dim = particles.shape[1]
    scale2 = _RWM_SCALE**2 / dim

    def batch_lik(x):
        return mx.vmap(log_likelihood_fn)(x)

    def batch_prior(x):
        return mx.vmap(log_prior_fn)(x)

    loglik = batch_lik(particles)
    logprior = batch_prior(particles)
    log_w = mx.full((n,), -log_n)  # normalized (LSE == 0)

    def _rwm_sweep(key, particles, loglik, logprior, phi_arr, l_prop):
        """num_mcmc_steps RWM sweeps, branchless acceptance."""
        acc = mx.zeros(())
        for _ in range(num_mcmc_steps):
            kz, ku, key = mx.random.split(key, 3)
            z = mx.random.normal((n, dim), key=kz)
            prop = particles + z @ l_prop.T
            lp = batch_prior(prop)
            ll = batch_lik(prop)
            log_alpha = (lp + phi_arr * ll) - (logprior + phi_arr * loglik)
            u = mx.random.uniform(shape=(n,), key=ku)
            accept = mx.log(mx.maximum(u, 1e-37)) < log_alpha
            particles = mx.where(accept[:, None], prop, particles)
            loglik = mx.where(accept, ll, loglik)
            logprior = mx.where(accept, lp, logprior)
            acc = acc + mx.mean(accept)
        return particles, loglik, logprior, acc / num_mcmc_steps

    sweep = mx.compile(_rwm_sweep)

    def ess_at(phi_new: float, phi: float) -> float:
        return compute_ess(log_w + (phi_new - phi) * loglik).item()

    phi = 0.0
    temps: list[float] = []
    ess_trace: list[float] = []
    acc_trace: list[mx.array] = []
    total = mx.array(0.0)
    comp = mx.array(0.0)
    target = target_ess * n

    for _ in range(max_stages):
        # --- adaptive schedule: bisect ESS(phi') = target ----------
        e_full = ess_at(1.0, phi)
        if math.isnan(e_full) and math.isnan(ess_at(phi + 1e-6, phi)):
            raise DegenerateWeightsError(
                "likelihood is -inf across the whole cloud; "
                "no tempering step is possible"
            )
        if e_full >= target:
            phi_new = 1.0
        else:
            lo, hi = phi, 1.0
            for _ in range(_BISECT_ITERS):
                mid = 0.5 * (lo + hi)
                e_mid = ess_at(mid, phi)
                if math.isnan(e_mid) or e_mid < target:
                    hi = mid
                else:
                    lo = mid
            phi_new = lo if lo > phi else 0.5 * (phi + hi)
        delta = phi_new - phi

        # --- reweight; increment at the reweight stage --------------
        lw_norm, log_sum = log_normalize(log_w + delta * loglik)
        stage_ess = compute_ess(lw_norm).item()
        # Neumaier-compensated evidence accumulation (ADR-0003).
        t = total + log_sum
        comp = comp + mx.where(
            mx.abs(total) >= mx.abs(log_sum),
            (total - t) + log_sum,
            (log_sum - t) + total,
        )
        total = t

        # --- adapt proposal from the weighted pre-resample cloud ----
        cov = _weighted_cov_f64(particles, mx.exp(lw_norm))
        factors = chol_factor(scale2 * cov)

        # --- resample (always) + pi_{phi'}-invariant moves ----------
        key, kr, km = mx.random.split(key, 3)
        idx = resampling_fn(kr, mx.exp(lw_norm), n)
        particles = mx.take(particles, idx, axis=0)
        loglik = mx.take(loglik, idx)
        logprior = mx.take(logprior, idx)
        particles, loglik, logprior, acc = sweep(
            km,
            particles,
            loglik,
            logprior,
            mx.array(phi_new),
            factors.scale_tril,
        )
        log_w = mx.full((n,), -log_n)

        temps.append(phi_new)
        ess_trace.append(stage_ess)
        acc_trace.append(acc)
        phi = phi_new
        if phi >= 1.0:
            break
    else:
        raise RuntimeError(
            f"tempering did not reach phi=1 within {max_stages} stages"
        )

    marginal: Float[mx.array, ""] = total + comp
    return TemperedPosterior(
        particles=particles,
        log_weights=log_w,
        marginal_loglik=marginal,
        temperatures=mx.array(temps),
        ess=mx.array(ess_trace),
        acceptance_rates=mx.stack(acc_trace),
    )
