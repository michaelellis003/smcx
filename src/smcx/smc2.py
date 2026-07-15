# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

r"""SMC² — nested parameter inference for state-space models (ADR-0014).

N_theta parameter particles form an outer SMC sampler; each carries an
N_x-particle inner bootstrap filter whose unbiased likelihood estimate
drives the outer weights. As each datum arrives every inner filter
advances one step, its incremental likelihood updates the outer
weights, and when the outer ESS degrades the parameter cloud is
rejuvenated with a particle-marginal Metropolis-Hastings (PMMH) move
(Chopin, Jacob & Papaspiliopoulos 2013).

The live state is the (N_theta, N_x, d_x) inner tensor plus the outer
weights — the dense, batch-shaped workload ADR-0014 targets for
unified memory. All weights are f32 log-domain, log-Z carries are
Neumaier-compensated. The inner filter resamples every step (the
simplest unbiased estimator); adaptive inner resampling and adaptive
N_x are deferred (ADR-0014). The PMMH move is an internal
rejuvenation kernel, not a user-facing PMMH sampler (ADR-0014).
"""

import math

import mlx.core as mx
import numpy as np
from jaxtyping import Float

from smcx.containers import SMC2Posterior
from smcx.distributions import chol_factor
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import _searchsorted_take, systematic
from smcx.types import (
    InitialSampler,
    KeyT,
    ParamInitialSampler,
    ParamLogObservationFn,
    ParamTransitionSampler,
    PerParticleLogDensity,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess

# f32 min-normal is 1.175e-38; a floor below it is flushed to zero
# (Metal FTZ) and the CDF guard degrades to 0/0 = NaN. Match
# resampling.py's 1e-30.
_TINY = 1e-30
_RWM_SCALE = 2.38


def _lse_rows(x: mx.array) -> mx.array:
    """Max-shifted log-sum-exp along axis 1; all -inf -> -inf, not NaN."""
    m = mx.max(x, axis=1, keepdims=True)
    m_safe = mx.where(mx.isinf(m), 0.0, m)
    return (
        m_safe + mx.log(mx.sum(mx.exp(x - m_safe), axis=1, keepdims=True))
    ).squeeze(1)


def _normalize_rows(log_w: mx.array) -> tuple[mx.array, mx.array]:
    """Row-wise log-normalize; returns (normalized, per-row LSE)."""
    lse = _lse_rows(log_w)
    return log_w - lse[:, None], lse


def _neumaier_add(
    total: mx.array, comp: mx.array, x: mx.array
) -> tuple[mx.array, mx.array]:
    """One Neumaier compensated-summation step (branchless where)."""
    t = total + x
    comp = comp + mx.where(
        mx.abs(total) >= mx.abs(x), (total - t) + x, (x - t) + total
    )
    return t, comp


def _weighted_cov_f64(particles: mx.array, weights: mx.array) -> np.ndarray:
    """Two-pass weighted covariance on the host in f64 (numerics §5)."""
    x = np.array(particles, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    mu = w @ x
    xc = x - mu
    return (xc * w[:, None]).T @ xc


def _batched_inner_resample(
    key: KeyT, weights_2d: mx.array, n_x: int
) -> mx.array:
    """Systematic resample within each of N_theta filters (vmap-safe).

    One uniform offset per filter, take-based bisection (the Metal
    kernel has no vmap — ADR-0009/mlx-constraints). Single vmap over
    the theta axis only; the bisection is a plain take loop, so this
    is not the nested-vmap-over-take hazard.
    """
    cdf = mx.cumsum(weights_2d, axis=1)
    cdf = cdf / mx.maximum(cdf[:, -1:], _TINY)
    n_theta = weights_2d.shape[0]
    u0 = mx.random.uniform(shape=(n_theta, 1), key=key)
    positions = (mx.arange(n_x) + u0) / n_x
    return mx.vmap(_searchsorted_take)(cdf, positions)


# --- Batched inner-filter kernels (module-level so they are testable
# apart from the smc2 driver; ADR-0013). Densities use flatten +
# single vmap over the theta*N_x axis to avoid the nested-vmap-over-
# take hazard; each of the N_theta filters is advanced independently.


def _batched_logobs(y_t, particles, theta, log_observation_fn, n_theta, n_x):
    flat = particles.reshape(-1, particles.shape[-1])
    lg = mx.vmap(lambda s, p: log_observation_fn(y_t, s, p))(
        flat, mx.repeat(theta, n_x, axis=0)
    )
    return lg.reshape(n_theta, n_x)


def _batched_trans(
    keys_flat, particles, theta, transition_sampler, n_theta, n_x
):
    flat = particles.reshape(-1, particles.shape[-1])
    moved = mx.vmap(lambda k, s, p: transition_sampler(k, s, p))(
        keys_flat, flat, mx.repeat(theta, n_x, axis=0)
    )
    return moved.reshape(n_theta, n_x, -1)


def _batched_inner_init(
    k0, theta, y0, initial_sampler, log_observation_fn, n_theta, n_x, log_n_x
):
    """Init N_theta inner clouds and first reweight (one datum)."""
    k_init = mx.random.split(k0, n_theta)
    inner = mx.vmap(lambda k, p: initial_sampler(k, n_x, p))(k_init, theta)
    log_g = _batched_logobs(y0, inner, theta, log_observation_fn, n_theta, n_x)
    inner_log_w, _ = _normalize_rows(log_g)
    return inner, inner_log_w, _lse_rows(log_g) - log_n_x


def _batched_inner_step(
    kr,
    kt,
    inner,
    inner_log_w,
    theta,
    y_t,
    transition_sampler,
    log_observation_fn,
    n_theta,
    n_x,
    log_n_x,
):
    """Advance N_theta inner filters one datum.

    Resample, propagate, reweight. Each filter is independent — the
    theta axis never couples (validated batched-vs-independent,
    tier-2).
    """
    idx = _batched_inner_resample(kr, mx.exp(inner_log_w), n_x)
    parents = mx.take_along_axis(inner, idx[:, :, None], axis=1)
    keys_flat = mx.random.split(kt, n_theta * n_x)
    inner = _batched_trans(
        keys_flat, parents, theta, transition_sampler, n_theta, n_x
    )
    log_g = _batched_logobs(y_t, inner, theta, log_observation_fn, n_theta, n_x)
    inner_log_w, _ = _normalize_rows(log_g)
    return inner, inner_log_w, _lse_rows(log_g) - log_n_x


def smc2(
    key: KeyT,
    param_initial_sampler: InitialSampler,
    log_prior_fn: PerParticleLogDensity,
    initial_sampler: ParamInitialSampler,
    transition_sampler: ParamTransitionSampler,
    log_observation_fn: ParamLogObservationFn,
    emissions: Float[mx.array, "ntime emission_dim"]
    | Float[mx.array, " ntime"],
    num_theta: int,
    num_x: int,
    *,
    ess_threshold: float = 0.5,
    num_pmmh_steps: int = 1,
    resampling_fn: ResamplingFn = systematic,
    store_history: bool = True,
) -> SMC2Posterior:
    r"""Run SMC² for joint state-and-parameter inference (ADR-0014).

    Args:
        key: PRNG key.
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
        emissions: Observations ``(T, D)`` (or ``(T,)``, canonicalized).
        num_theta: Number of outer parameter particles N_theta.
        num_x: Number of inner particles N_x (fixed; ADR-0014).
        ess_threshold: Rejuvenate the parameter cloud when the outer
            ESS drops below ``ess_threshold * num_theta``. Set 0 to
            disable rejuvenation (a pure forward pass).
        num_pmmh_steps: PMMH moves applied per rejuvenation.
        resampling_fn: ADR-0004 resampler for the OUTER theta cloud at
            rejuvenation (the inner filters use a fixed vmap-safe
            systematic kernel).
        store_history: ADR-0011 memory option; when False only the
            final parameter cloud is returned (time axis length 1).

    Returns:
        An :class:`~smcx.containers.SMC2Posterior`.

    Raises:
        DegenerateWeightsError: if the outer weights collapse (every
            parameter particle assigned an all-inf inner likelihood).
    """
    if emissions.ndim == 1:
        emissions = emissions[:, None]
    n_time = emissions.shape[0]
    log_n_x = math.log(num_x)
    log_n_theta = math.log(num_theta)

    k_theta, k_loop = mx.random.split(key)
    theta = param_initial_sampler(k_theta, num_theta)  # (N_theta, d_theta)
    d_theta = theta.shape[-1]
    scale2 = _RWM_SCALE**2 / d_theta

    # --- batched inner-density helpers (flatten -> single vmap) -------
    # One compiled step per algorithm (mlx-constraints): the batched
    # inner advance is the hot kernel — the forward loop calls it T
    # times and every PMMH re-run calls it once per prefix datum, so
    # rejuvenation dominates. All randomness is explicitly keyed
    # (kr/kt/k0 are arguments), so compile does not freeze it. The
    # kernels live at module scope (testable apart from the driver);
    # these thin closures bind the model and dimensions.
    def _inner_init(k0, th, y0):
        return _batched_inner_init(
            k0,
            th,
            y0,
            initial_sampler,
            log_observation_fn,
            num_theta,
            num_x,
            log_n_x,
        )

    def _inner_step(kr, kt, inner, inner_log_w, th, y_t):
        return _batched_inner_step(
            kr,
            kt,
            inner,
            inner_log_w,
            th,
            y_t,
            transition_sampler,
            log_observation_fn,
            num_theta,
            num_x,
            log_n_x,
        )

    inner_init = mx.compile(_inner_init)
    inner_step = mx.compile(_inner_step)

    def inner_forward(fwd_key, th, upto):
        """Fresh inner filter over emissions[0:upto]; returns resolved logZ."""
        keys = mx.random.split(fwd_key, max(upto, 1))
        inner, inner_log_w, log_ell = inner_init(keys[0], th, emissions[0])
        lz, lz_c = log_ell, mx.zeros_like(log_ell)
        for tp in range(1, upto):
            kr, kt = mx.random.split(keys[tp])
            inner, inner_log_w, log_ell = inner_step(
                kr, kt, inner, inner_log_w, th, emissions[tp]
            )
            lz, lz_c = _neumaier_add(lz, lz_c, log_ell)
        return inner, inner_log_w, lz + lz_c

    def rejuvenate(rkey, t, th, log_omega, inner, inner_log_w, log_z):
        k_res, k_move = mx.random.split(rkey)
        # Proposal scale from the weighted theta cloud (pre-resample).
        cov = _weighted_cov_f64(th, mx.exp(log_omega))
        scale_tril = chol_factor(scale2 * cov).scale_tril
        # Resample theta with its attached inner state (monotone gather).
        idx = resampling_fn(k_res, mx.exp(log_omega), num_theta)
        th = mx.take(th, idx, axis=0)
        inner = mx.take(inner, idx, axis=0)
        inner_log_w = mx.take(inner_log_w, idx, axis=0)
        log_z = mx.take(log_z, idx, axis=0)
        logprior = mx.vmap(log_prior_fn)(th)
        acc_sum = mx.zeros(())
        for _ in range(num_pmmh_steps):
            k_prop, k_run, k_u, k_move = mx.random.split(k_move, 4)
            z = mx.random.normal((num_theta, d_theta), key=k_prop)
            th_star = th + z @ scale_tril.T
            logprior_star = mx.vmap(log_prior_fn)(th_star)
            inner_s, inner_log_w_s, log_z_s = inner_forward(
                k_run, th_star, t + 1
            )
            log_alpha = (logprior_star + log_z_s) - (logprior + log_z)
            u = mx.random.uniform(shape=(num_theta,), key=k_u)
            accept = mx.log(mx.maximum(u, _TINY)) < log_alpha
            th = mx.where(accept[:, None], th_star, th)
            logprior = mx.where(accept, logprior_star, logprior)
            log_z = mx.where(accept, log_z_s, log_z)
            inner = mx.where(accept[:, None, None], inner_s, inner)
            inner_log_w = mx.where(accept[:, None], inner_log_w_s, inner_log_w)
            acc_sum = acc_sum + mx.mean(accept.astype(mx.float32))
        rate = acc_sum / max(num_pmmh_steps, 1)
        log_omega = mx.full((num_theta,), -log_n_theta)
        return th, log_omega, inner, inner_log_w, log_z, rate

    def _check(t: int, inc_val: mx.array) -> None:
        v = inc_val.item()
        if v == float("-inf") or v != v:
            raise DegenerateWeightsError(
                f"outer weights collapsed at step {t} "
                f"(log-evidence increment {v})"
            )

    # --- t = 0: init inner clouds, first reweight ---------------------
    # Split disjoint streams: inner_init splits k_init0 into num_theta
    # keys, which must not collide with the rejuvenation key —
    # split(step_keys[0], 3)[2] would share threefry counter positions
    # with an init subkey.
    step_keys = mx.random.split(k_loop, max(n_time, 1))
    k_init0, k_rej0 = mx.random.split(step_keys[0], 2)
    inner, inner_log_w, log_ell = inner_init(k_init0, theta, emissions[0])
    log_omega, inc0 = _normalize_rows((-log_n_theta + log_ell)[None, :])
    log_omega = log_omega.squeeze(0)
    m_tot, m_comp = inc0.squeeze(0), mx.zeros(())
    lz_tot, lz_comp = log_ell, mx.zeros_like(log_ell)

    mx.eval(inner, log_omega, m_tot, lz_tot)
    _check(0, m_tot)
    threshold = ess_threshold * num_theta

    # Rejuvenate at t=0 too (a collapsed initial cloud must be
    # refreshed before more data arrives, and is the only opportunity
    # in a single-observation run).
    rate0 = mx.array(0.0)
    if threshold > 0.0 and compute_ess(log_omega).item() < threshold:
        theta, log_omega, inner, inner_log_w, lz_res, rate0 = rejuvenate(
            k_rej0, 0, theta, log_omega, inner, inner_log_w, lz_tot + lz_comp
        )
        lz_tot, lz_comp = lz_res, mx.zeros_like(lz_res)

    params_hist = [theta]
    omega_hist = [log_omega]
    ess_hist = [compute_ess(log_omega)]
    inc_hist = [m_tot]
    accept_hist = [rate0]

    # --- t >= 1: advance every inner filter one datum -----------------
    for t in range(1, n_time):
        # Distinct streams: inner-step noise (kr, kt) must be
        # independent of the rejuvenation move's noise (k_rej) — a
        # 2-way split would make k_rej byte-identical to kt.
        kr, kt, k_rej = mx.random.split(step_keys[t], 3)
        inner, inner_log_w, log_ell = inner_step(
            kr, kt, inner, inner_log_w, theta, emissions[t]
        )
        log_omega, inc = _normalize_rows((log_omega + log_ell)[None, :])
        log_omega = log_omega.squeeze(0)
        inc = inc.squeeze(0)
        m_tot, m_comp = _neumaier_add(m_tot, m_comp, inc)
        lz_tot, lz_comp = _neumaier_add(lz_tot, lz_comp, log_ell)

        rate = mx.array(0.0)
        if threshold > 0.0 and compute_ess(log_omega).item() < threshold:
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
            lz_tot, lz_comp = lz_resolved, mx.zeros_like(lz_resolved)

        if store_history:
            params_hist.append(theta)
            omega_hist.append(log_omega)
        ess_hist.append(compute_ess(log_omega))
        inc_hist.append(inc)
        accept_hist.append(rate)

        mx.eval(inner, log_omega, m_tot, lz_tot, theta)
        _check(t, inc)

    if not store_history:
        params_hist = [theta]
        omega_hist = [log_omega]

    return SMC2Posterior(
        marginal_loglik=m_tot + m_comp,
        filtered_params=mx.stack(params_hist),
        filtered_log_weights=mx.stack(omega_hist),
        ess=mx.stack(ess_hist),
        log_evidence_increments=mx.stack(inc_hist),
        acceptance_rates=mx.stack(accept_hist),
    )
