# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""SMC² — nested parameter inference for state-space models.

``num_theta`` parameter particles form an outer SMC sampler; each
carries an ``num_x``-particle inner bootstrap filter whose unbiased
likelihood estimate drives the outer weights. As each datum arrives
every inner filter advances one step, its incremental likelihood
updates the outer weights, and when the outer ESS degrades the
parameter cloud is rejuvenated with a particle-marginal
Metropolis-Hastings (PMMH) move (Chopin, Jacob & Papaspiliopoulos
2013).

The live state is the ``(num_theta, num_x, d_x)`` inner tensor plus
the outer weights. All weights are log-domain, log-Z carries are
Neumaier-compensated. The inner filter resamples every step (the
simplest unbiased estimator); adaptive inner resampling and adaptive
``num_x`` are not implemented. The PMMH move is an internal
rejuvenation kernel, not a user-facing PMMH sampler. The
outer schedule branches on host-side ESS reads, so ``smc2`` itself is
not jittable; the batched inner kernels are jitted.
"""

import math
from typing import cast

import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap
from jaxtyping import Array, Float

from smcx.containers import SMC2Posterior
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import _BELOW_ONE, _TINY, systematic
from smcx.tempering import _chol_with_jitter, _weighted_cov_f64
from smcx.types import (
    ParamInitialSampler,
    ParamInitialStateSampler,
    ParamLogObservationFn,
    ParamTransitionSampler,
    PRNGKeyT,
    ResamplingFn,
    StaticLogDensity,
)
from smcx.weights import ess as compute_ess

_RWM_SCALE = 2.38


def _lse_rows(x: Array) -> Array:
    """Max-shifted log-sum-exp along axis 1; all -inf -> -inf, not NaN."""
    m = jnp.max(x, axis=1, keepdims=True)
    m_safe = jnp.where(jnp.isinf(m), 0.0, m)
    return (
        m_safe + jnp.log(jnp.sum(jnp.exp(x - m_safe), axis=1, keepdims=True))
    ).squeeze(1)


def _normalize_rows(log_w: Array) -> tuple[Array, Array]:
    """Row-wise log-normalize; returns (normalized, per-row LSE)."""
    lse = _lse_rows(log_w)
    return log_w - lse[:, None], lse


def _neumaier_add(total: Array, comp: Array, x: Array) -> tuple[Array, Array]:
    """One Neumaier compensated-summation step (branchless where)."""
    t = total + x
    comp = comp + jnp.where(
        jnp.abs(total) >= jnp.abs(x), (total - t) + x, (x - t) + total
    )
    return t, comp


def _batched_inner_resample(
    key: PRNGKeyT, weights_2d: Array, n_x: int
) -> Array:
    """Systematic resample within each of the outer filters.

    One uniform offset per filter (never a shared constant), vmapped
    right-bisect per row, with the sub-1 endpoint clamp shared with
    ``smcx.resampling``.
    """
    cdf = jnp.cumsum(weights_2d, axis=1)
    cdf = cdf / jnp.maximum(cdf[:, -1:], _TINY)
    n_theta = weights_2d.shape[0]
    u0 = jr.uniform(key, (n_theta, 1))
    positions = jnp.minimum((jnp.arange(n_x) + u0) / n_x, _BELOW_ONE)

    def _row(cdf_row, q_row):
        idx = jnp.searchsorted(cdf_row, q_row, side="right")
        return jnp.clip(idx, 0, n_x - 1).astype(jnp.int32)

    return vmap(_row)(cdf, positions)


def smc2(
    key: PRNGKeyT,
    param_initial_sampler: ParamInitialSampler,
    log_prior_fn: StaticLogDensity,
    initial_sampler: ParamInitialStateSampler,
    transition_sampler: ParamTransitionSampler,
    log_observation_fn: ParamLogObservationFn,
    emissions: Float[Array, "ntime emission_dim"],
    num_theta: int,
    num_x: int,
    *,
    ess_threshold: float = 0.5,
    num_pmmh_steps: int = 1,
    resampling_fn: ResamplingFn = systematic,
    store_history: bool = True,
) -> SMC2Posterior:
    r"""Run SMC² for joint state-and-parameter inference.

    Args:
        key: JAX PRNG key.
        param_initial_sampler: ``(key, num_theta) -> (num_theta,
            param_dim)`` prior draw over the static parameters.
        log_prior_fn: ``(theta) -> scalar`` parameter log-prior;
            vmapped internally (used in the PMMH accept ratio).
        initial_sampler: inner ``(key, num_x, theta) -> (num_x,
            state_dim)`` drawing the initial cloud given a parameter.
        transition_sampler: inner ``(key, state, theta) -> state``,
            per-particle; vmapped internally.
        log_observation_fn: inner ``(emission, state, theta) ->
            scalar``, per-particle; vmapped internally.
        emissions: Observations ``(T, D)`` (or ``(T,)``,
            canonicalized).
        num_theta: Number of outer parameter particles.
        num_x: Fixed number of inner particles.
        ess_threshold: Rejuvenate the parameter cloud when the outer
            ESS drops below ``ess_threshold * num_theta``. Set 0 to
            disable rejuvenation (a pure forward pass).
        num_pmmh_steps: PMMH moves applied per rejuvenation.
        resampling_fn: Resampler for the outer parameter cloud at
            rejuvenation (the inner filters use a fixed vmapped
            systematic kernel).
        store_history: When False, only the
            final parameter cloud is returned (time axis length 1).

    Returns:
        An :class:`~smcx.containers.SMC2Posterior`.

    Raises:
        DegenerateWeightsError: The outer weights collapse (every
            parameter particle assigned an all--inf inner likelihood).
    """
    if emissions.ndim == 1:
        emissions = emissions[:, None]
    n_time = emissions.shape[0]
    log_n_theta = math.log(num_theta)

    k_theta, k_loop = jr.split(key)
    theta = param_initial_sampler(k_theta, num_theta)
    d_theta = theta.shape[-1]
    scale2 = _RWM_SCALE**2 / d_theta
    batch_prior = vmap(log_prior_fn)
    log_n_x = math.log(num_x)

    # --- batched inner kernels (flatten -> single vmap) ---------------
    @jit
    def inner_init(
        k0: PRNGKeyT,
        th: Float[Array, "num_theta param_dim"],
        y0: Float[Array, " emission_dim"],
    ) -> tuple[
        Float[Array, "num_theta num_x state_dim"],
        Float[Array, "num_theta num_x"],
        Float[Array, " num_theta"],
    ]:
        k_init = jr.split(k0, num_theta)
        inner = vmap(lambda k, p: initial_sampler(k, num_x, p))(k_init, th)
        flat = inner.reshape(-1, inner.shape[-1])
        th_flat = jnp.repeat(th, num_x, axis=0)
        flat_log_g = cast(
            Float[Array, " flat_particle"],
            vmap(lambda s, p: log_observation_fn(y0, s, p))(flat, th_flat),
        )
        log_g = flat_log_g.reshape(num_theta, num_x)
        inner_log_w, log_g_lse = _normalize_rows(log_g)
        return inner, inner_log_w, log_g_lse - log_n_x

    @jit
    def inner_step(
        kr: PRNGKeyT,
        kt: PRNGKeyT,
        inner: Float[Array, "num_theta num_x state_dim"],
        inner_log_w: Float[Array, "num_theta num_x"],
        th: Float[Array, "num_theta param_dim"],
        y_t: Float[Array, " emission_dim"],
    ) -> tuple[
        Float[Array, "num_theta num_x state_dim"],
        Float[Array, "num_theta num_x"],
        Float[Array, " num_theta"],
    ]:
        idx = _batched_inner_resample(kr, jnp.exp(inner_log_w), num_x)
        parents = jnp.take_along_axis(inner, idx[:, :, None], axis=1)
        keys_flat = jr.split(kt, num_theta * num_x)
        flat = parents.reshape(-1, parents.shape[-1])
        th_flat = jnp.repeat(th, num_x, axis=0)
        moved = vmap(lambda k, s, p: transition_sampler(k, s, p))(
            keys_flat, flat, th_flat
        ).reshape(num_theta, num_x, -1)
        flat_log_g = cast(
            Float[Array, " flat_particle"],
            vmap(lambda s, p: log_observation_fn(y_t, s, p))(
                moved.reshape(-1, moved.shape[-1]), th_flat
            ),
        )
        log_g = flat_log_g.reshape(num_theta, num_x)
        inner_log_w, log_g_lse = _normalize_rows(log_g)
        return moved, inner_log_w, log_g_lse - log_n_x

    def inner_forward(fwd_key, th, upto):
        """Fresh inner filter over emissions[0:upto]; resolved logZ."""
        keys = jr.split(fwd_key, max(upto, 1))
        inner, inner_log_w, log_ell = inner_init(keys[0], th, emissions[0])
        lz, lz_c = log_ell, jnp.zeros_like(log_ell)
        for tp in range(1, upto):
            kr, kt = jr.split(keys[tp])
            inner, inner_log_w, log_ell = inner_step(
                kr, kt, inner, inner_log_w, th, emissions[tp]
            )
            lz, lz_c = _neumaier_add(lz, lz_c, log_ell)
        return inner, inner_log_w, lz + lz_c

    def rejuvenate(rkey, t, th, log_omega, inner, inner_log_w, log_z):
        k_res, k_move = jr.split(rkey)
        # Proposal scale from the weighted theta cloud (pre-resample).
        cov = _weighted_cov_f64(th, jnp.exp(log_omega))
        scale_tril = _chol_with_jitter(scale2 * cov)
        # Resample theta with its attached inner state (monotone gather).
        idx = resampling_fn(k_res, jnp.exp(log_omega), num_theta)
        th = th[idx]
        inner = inner[idx]
        inner_log_w = inner_log_w[idx]
        log_z = log_z[idx]
        logprior = batch_prior(th)
        acc_sum = jnp.zeros(())
        for _ in range(num_pmmh_steps):
            k_prop, k_run, k_u, k_move = jr.split(k_move, 4)
            z = jr.normal(k_prop, (num_theta, d_theta))
            th_star = th + z @ scale_tril.T
            logprior_star = batch_prior(th_star)
            inner_s, inner_log_w_s, log_z_s = inner_forward(
                k_run, th_star, t + 1
            )
            log_alpha = (logprior_star + log_z_s) - (logprior + log_z)
            u = jr.uniform(k_u, (num_theta,))
            accept = jnp.log(jnp.maximum(u, _TINY)) < log_alpha
            th = jnp.where(accept[:, None], th_star, th)
            logprior = jnp.where(accept, logprior_star, logprior)
            log_z = jnp.where(accept, log_z_s, log_z)
            inner = jnp.where(accept[:, None, None], inner_s, inner)
            inner_log_w = jnp.where(accept[:, None], inner_log_w_s, inner_log_w)
            acc_sum = acc_sum + jnp.mean(accept.astype(jnp.float32))
        rate = acc_sum / max(num_pmmh_steps, 1)
        log_omega = jnp.full((num_theta,), -log_n_theta)
        return th, log_omega, inner, inner_log_w, log_z, rate

    def _check(t: int, inc_val: Array) -> None:
        v = float(inc_val)
        if v == float("-inf") or v != v:
            raise DegenerateWeightsError(
                f"outer weights collapsed at step {t} "
                f"(log-evidence increment {v})"
            )

    # --- t = 0: init inner clouds, first reweight ---------------------
    # Disjoint streams: the init key must not collide with the
    # rejuvenation key.
    step_keys = jr.split(k_loop, max(n_time, 1))
    k_init0, k_rej0 = jr.split(step_keys[0], 2)
    inner, inner_log_w, log_ell = inner_init(k_init0, theta, emissions[0])
    log_omega, inc0 = _normalize_rows((-log_n_theta + log_ell)[None, :])
    log_omega = log_omega.squeeze(0)
    m_tot, m_comp = inc0.squeeze(0), jnp.zeros(())
    lz_tot, lz_comp = log_ell, jnp.zeros_like(log_ell)

    _check(0, m_tot)
    threshold = ess_threshold * num_theta

    # Rejuvenate at t=0 too (a collapsed initial cloud must be
    # refreshed before more data arrives).
    rate0 = jnp.zeros(())
    if threshold > 0.0 and float(compute_ess(log_omega)) < threshold:
        theta, log_omega, inner, inner_log_w, lz_res, rate0 = rejuvenate(
            k_rej0, 0, theta, log_omega, inner, inner_log_w, lz_tot + lz_comp
        )
        lz_tot, lz_comp = lz_res, jnp.zeros_like(lz_res)

    params_hist = [theta]
    omega_hist = [log_omega]
    ess_hist = [jnp.asarray(compute_ess(log_omega))]
    inc_hist = [m_tot]
    accept_hist = [rate0]

    # --- t >= 1: advance every inner filter one datum -----------------
    for t in range(1, n_time):
        # Distinct streams for step noise vs the rejuvenation move.
        kr, kt, k_rej = jr.split(step_keys[t], 3)
        inner, inner_log_w, log_ell = inner_step(
            kr, kt, inner, inner_log_w, theta, emissions[t]
        )
        log_omega, inc = _normalize_rows((log_omega + log_ell)[None, :])
        log_omega = log_omega.squeeze(0)
        inc = inc.squeeze(0)
        m_tot, m_comp = _neumaier_add(m_tot, m_comp, inc)
        lz_tot, lz_comp = _neumaier_add(lz_tot, lz_comp, log_ell)

        rate = jnp.zeros(())
        if threshold > 0.0 and float(compute_ess(log_omega)) < threshold:
            theta, log_omega, inner, inner_log_w, lz_resolved, rate = (
                rejuvenate(
                    k_rej,
                    t,
                    theta,
                    log_omega,
                    inner,
                    inner_log_w,
                    lz_tot + lz_comp,
                )
            )
            lz_tot, lz_comp = lz_resolved, jnp.zeros_like(lz_resolved)

        if store_history:
            params_hist.append(theta)
            omega_hist.append(log_omega)
        ess_hist.append(jnp.asarray(compute_ess(log_omega)))
        inc_hist.append(inc)
        accept_hist.append(rate)
        _check(t, inc)

    if not store_history:
        params_hist = [theta]
        omega_hist = [log_omega]

    return SMC2Posterior(
        marginal_loglik=m_tot + m_comp,
        filtered_params=jnp.stack(params_hist),
        filtered_log_weights=jnp.stack(omega_hist),
        ess=jnp.stack(ess_hist),
        log_evidence_increments=jnp.stack(inc_hist),
        acceptance_rates=jnp.stack(accept_hist),
    )
