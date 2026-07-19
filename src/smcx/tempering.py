# Copyright 2026 Michael Ellis
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
in float64 on the host (single-pass cancels catastrophically at
ordinary posterior offsets).

The adaptive schedule is host-driven (bisection reads ESS values), so
``temper`` itself is not jittable; the per-stage RWM sweep is jitted.
"""

import math
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, vmap
from jaxtyping import Array, Float

from smcx.containers import TemperedPosterior
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import systematic
from smcx.types import (
    DenseInitialSampler,
    PRNGKeyT,
    ResamplingFn,
    StaticLogDensity,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

_BISECT_ITERS = 60
_RWM_SCALE = 2.38


@runtime_checkable
class _RWMSweep(Protocol):
    """Execute one fixed-count, vectorized RWM mutation stage."""

    def __call__(
        self,
        key: PRNGKeyT,
        particles: Float[Array, "num_particles state_dim"],
        loglik: Float[Array, " num_particles"],
        logprior: Float[Array, " num_particles"],
        phi_arr: Float[Array, ""],
        l_prop: Float[Array, "state_dim state_dim"],
        /,
    ) -> tuple[
        Float[Array, "num_particles state_dim"],
        Float[Array, " num_particles"],
        Float[Array, " num_particles"],
        Float[Array, ""],
    ]: ...


def _weighted_cov_f64(particles: Array, weights: Array) -> np.ndarray:
    """Two-pass weighted covariance on the host in float64.

    One call per temperature stage — setup cost, not hot loop. The
    single-pass form cancels catastrophically at ordinary posterior
    offsets; two-pass in f64 is exact to rounding.
    """
    x = np.asarray(particles, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mu = w @ x
    xc = x - mu
    return (xc * w[:, None]).T @ xc


def _chol_with_jitter(cov: np.ndarray) -> jnp.ndarray:
    """Cholesky factor with escalating diagonal jitter on failure."""
    d = cov.shape[0]
    base = np.trace(cov) / max(d, 1)
    for jitter_scale in (0.0, 1e-8, 1e-6, 1e-4):
        jitter = base * jitter_scale
        try:
            lower = np.linalg.cholesky(cov + jitter * np.eye(d))
            return jnp.asarray(lower)
        except np.linalg.LinAlgError:
            continue
    # Last resort: eigenvalue clip.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, base * 1e-6, None)
    return jnp.asarray(
        np.linalg.cholesky(
            (eigvecs * eigvals) @ eigvecs.T + base * 1e-6 * np.eye(d)
        )
    )


def _build_rwm_sweep(
    log_prior_fn: StaticLogDensity,
    log_likelihood_fn: StaticLogDensity,
    n: int,
    dim: int,
    num_mcmc_steps: int,
) -> _RWMSweep:
    """Build one jitted RWM sweep for fixed callbacks and static sizes."""
    batch_lik = vmap(log_likelihood_fn)
    batch_prior = vmap(log_prior_fn)

    @jit
    def _rwm_sweep(
        key: PRNGKeyT,
        particles: Float[Array, "num_particles state_dim"],
        loglik: Float[Array, " num_particles"],
        logprior: Float[Array, " num_particles"],
        phi_arr: Float[Array, ""],
        l_prop: Float[Array, "state_dim state_dim"],
    ) -> tuple[
        Float[Array, "num_particles state_dim"],
        Float[Array, " num_particles"],
        Float[Array, " num_particles"],
        Float[Array, ""],
    ]:
        """Run fixed-count RWM sweeps with branchless acceptance."""
        acc = jnp.zeros(())
        for _ in range(num_mcmc_steps):
            kz, ku, key = jr.split(key, 3)
            z = jr.normal(kz, (n, dim))
            prop = particles + z @ l_prop.T
            lp = batch_prior(prop)
            ll = batch_lik(prop)
            log_alpha = (lp + phi_arr * ll) - (logprior + phi_arr * loglik)
            u = jr.uniform(ku, (n,))
            accept = jnp.log(jnp.maximum(u, 1e-300)) < log_alpha
            particles = jnp.where(accept[:, None], prop, particles)
            loglik = jnp.where(accept, ll, loglik)
            logprior = jnp.where(accept, lp, logprior)
            acc = acc + jnp.mean(accept)
        return particles, loglik, logprior, acc / num_mcmc_steps

    return _rwm_sweep


def temper(
    key: PRNGKeyT,
    initial_sampler: DenseInitialSampler,
    log_prior_fn: StaticLogDensity,
    log_likelihood_fn: StaticLogDensity,
    num_particles: int,
    num_mcmc_steps: int = 5,
    target_ess: float = 0.5,
    resampling_fn: ResamplingFn = systematic,
    *,
    max_stages: int = 1000,
) -> TemperedPosterior:
    r"""Sample a static target by adaptive tempered SMC.

    Args:
        key: JAX PRNG key.
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
    key, k_init = jr.split(key)
    particles = initial_sampler(k_init, n)
    dim = particles.shape[1]
    scale2 = _RWM_SCALE**2 / dim

    batch_lik = vmap(log_likelihood_fn)
    batch_prior = vmap(log_prior_fn)

    loglik: Float[Array, " num_particles"] = jnp.asarray(batch_lik(particles))
    logprior: Float[Array, " num_particles"] = jnp.asarray(
        batch_prior(particles)
    )
    log_w = jnp.full((n,), -log_n)  # normalized (LSE == 0)
    rwm_sweep = _build_rwm_sweep(
        log_prior_fn,
        log_likelihood_fn,
        n,
        dim,
        num_mcmc_steps,
    )

    def ess_at(phi_new: float, phi: float) -> float:
        return float(compute_ess(log_w + (phi_new - phi) * loglik))

    phi = 0.0
    temps: list[float] = []
    ess_trace: list[float] = []
    acc_trace: list[Array] = []
    total = jnp.zeros(())
    comp = jnp.zeros(())
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
        stage_ess = float(compute_ess(lw_norm))
        # Neumaier-compensated evidence accumulation.
        t = total + log_sum
        comp = comp + jnp.where(
            jnp.abs(total) >= jnp.abs(log_sum),
            (total - t) + log_sum,
            (log_sum - t) + total,
        )
        total = t

        # --- adapt proposal from the weighted pre-resample cloud ----
        cov = _weighted_cov_f64(particles, jnp.exp(lw_norm))
        l_prop = _chol_with_jitter(scale2 * cov)

        # --- resample (always) + pi_{phi'}-invariant moves ----------
        key, kr, km = jr.split(key, 3)
        idx = resampling_fn(kr, jnp.exp(lw_norm), n)
        particles = particles[idx]
        loglik = loglik[idx]
        logprior = logprior[idx]
        particles, loglik, logprior, acc = rwm_sweep(
            km,
            particles,
            loglik,
            logprior,
            jnp.asarray(phi_new),
            l_prop,
        )
        log_w = jnp.full((n,), -log_n)

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

    marginal: Float[Array, ""] = total + comp
    return TemperedPosterior(
        particles=particles,
        log_weights=log_w,
        marginal_loglik=marginal,
        temperatures=jnp.asarray(temps),
        ess=jnp.asarray(ess_trace),
        acceptance_rates=jnp.stack(acc_trace),
    )
