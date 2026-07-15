# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

#
# Ported to MLX from smcjax (https://github.com/michaelellis003/smcjax,
# frozen @ e93d527), Apache-2.0. Modified: MLX arrays; guarded
# host-side parameter-covariance factorization; shifted shrinkage
# form; store_history.

r"""Liu-West filter: joint state-parameter estimation.

Augments the auxiliary particle filter with kernel-smoothed parameter
particles [Liu & West, 2001]: at each step, parameters are shrunk
toward their weighted mean, :math:`m_i = \bar\theta + a(\theta_i -
\bar\theta)`, and jittered with covariance :math:`h^2 V` where
:math:`h^2 = 1 - a^2` and :math:`V` is the weighted parameter
covariance — the discount construction that avoids inflating the
marginal parameter variance.

**Labeled approximate**: the method carries non-vanishing bias,
accumulating over-dispersion, and discount sensitivity (Kantas et
al. 2015); SMC^2 is the exact alternative on the roadmap. Store
parameters roughly centered/standardized — the f32 constraint bites
when :math:`|\theta| / \mathrm{sd}(\theta) \gtrsim 10^4`
(docs/research/numerical-methods.md §5).

Implementation notes: parameter collapse (all particles sharing one
value) is Liu-West's known failure mode and exactly where MLX's f32
Cholesky returns silent garbage, so the per-step factorization of
:math:`h^2 V` runs through the guarded host-side
:func:`smcx.distributions.chol_factor` — at the cost of one host
sync per step (this filter does not pipeline; it is the
convenience/legacy option, not the throughput path). The ``inputs``
channel is not yet wired for Liu-West.
"""

import math

import mlx.core as mx
import numpy as np
from jaxtyping import Float

from smcx import _utils
from smcx.containers import LiuWestPosterior
from smcx.distributions import chol_factor
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import systematic
from smcx.types import (
    InitialSampler,
    KeyT,
    ParamLogObservationFn,
    ParamTransitionSampler,
    ResamplingFn,
)
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize, normalize


def liu_west_filter(
    key: KeyT,
    initial_sampler: InitialSampler,
    transition_sampler: ParamTransitionSampler,
    log_observation_fn: ParamLogObservationFn,
    log_auxiliary_fn: ParamLogObservationFn,
    param_initial_sampler: InitialSampler,
    emissions: Float[mx.array, "ntime emission_dim"]
    | Float[mx.array, " ntime"],
    num_particles: int,
    shrinkage: float = 0.95,
    resampling_fn: ResamplingFn = systematic,
    resampling_threshold: float = 0.5,
    *,
    store_history: bool = True,
) -> LiuWestPosterior:
    r"""Run a Liu-West filter (states + static parameters).

    Args:
        key: PRNG key.
        initial_sampler: ``(key, num_particles) -> (N, state_dim)``.
        transition_sampler: ``(key, state, params) -> state``;
            vmapped internally.
        log_observation_fn: ``(emission, state, params) -> logp``;
            vmapped internally.
        log_auxiliary_fn: APF look-ahead
            ``(emission, state, params) -> logp``, evaluated at the
            *shrunk* parameters; vmapped internally.
        param_initial_sampler: ``(key, num_particles) ->
            (N, param_dim)`` prior draw for the parameters.
        emissions: Observations ``(T, D)`` (or ``(T,)``,
            canonicalized).
        num_particles: Number of particles N.
        shrinkage: Discount :math:`a \in (0, 1)`; jitter scale is
            :math:`h^2 = 1 - a^2`. Larger a = weaker shrinkage =
            wider parameter posterior.
        resampling_fn: ADR-0004 contract resampler.
        resampling_threshold: Resample when the first-stage ESS
            drops below ``threshold * N``.
        store_history: ADR-0011 memory option.

    Returns:
        :class:`~smcx.containers.LiuWestPosterior`.

    Raises:
        DegenerateWeightsError: All weights collapsed at some step.
        ValueError: Malformed shapes/arguments.
    """
    if num_particles < 1:
        raise ValueError(f"num_particles must be >= 1; got {num_particles}")
    if not 0.0 < shrinkage < 1.0:
        raise ValueError(f"shrinkage must be in (0, 1); got {shrinkage}")
    emissions = _utils.canonicalize_emissions(emissions)
    for fn, name in (
        (transition_sampler, "transition_sampler"),
        (log_observation_fn, "log_observation_fn"),
        (log_auxiliary_fn, "log_auxiliary_fn"),
    ):
        _utils.check_callback_arity(fn, name, 3, False)

    n = num_particles
    num_timesteps = emissions.shape[0]
    log_n = math.log(n)
    a = shrinkage
    h_sq = 1.0 - a * a
    identity = mx.arange(n, dtype=mx.int32)
    threshold = resampling_threshold * n

    batch_obs = mx.vmap(log_observation_fn, in_axes=(None, 0, 0))
    batch_aux = mx.vmap(log_auxiliary_fn, in_axes=(None, 0, 0))
    batch_trans = mx.vmap(transition_sampler, in_axes=(0, 0, 0))

    # --- t = 0: init-as-if-resampled ----------------------------------
    key, k_s, k_p = mx.random.split(key, 3)
    particles = initial_sampler(k_s, n)
    params = param_initial_sampler(k_p, n)
    log_w, log_sum = log_normalize(batch_obs(emissions[0], particles, params))
    increment = log_sum - log_n
    ess_t = compute_ess(log_w)

    def _check(t, inc):
        v = inc.item()  # per-step host sync (see module docstring)
        if v == float("-inf") or v != v:
            raise DegenerateWeightsError(
                f"all particle weights collapsed at step {t} "
                f"(log-evidence increment {v})"
            )

    _check(0, increment)

    all_particles = [particles]
    all_params = [params]
    all_log_w = [log_w]
    all_ancestors = [identity]
    all_ess = [ess_t]
    all_inc = [increment]
    log_ml = increment

    step_keys = mx.random.split(key, max(num_timesteps - 1, 1))
    for t in range(1, num_timesteps):
        k1, k2, k3 = mx.random.split(step_keys[t - 1], 3)
        y_t = emissions[t]

        # Weighted parameter moments + guarded factorization of
        # h^2 V, host-side in f64 (two-pass; chol_factor's jitter
        # ladder covers the collapsed-parameter case).
        w64 = np.array(normalize(log_w), dtype=np.float64)
        p64 = np.array(params, dtype=np.float64)
        w64 = w64 / w64.sum()
        mean64 = w64 @ p64
        dev = p64 - mean64
        cov64 = (dev * w64[:, None]).T @ dev
        factors = chol_factor(h_sq * cov64)
        param_mean = mx.array(mean64.astype(np.float32))

        # Shifted shrinkage form (numerical-methods §5).
        shrunk = param_mean + a * (params - param_mean)

        # First-stage weights at the shrunk parameters.
        log_aux = batch_aux(y_t, particles, shrunk)
        log_first_norm, log_first_sum = log_normalize(log_w + log_aux)
        ess_first = compute_ess(log_first_norm)
        do_resample = ess_first < threshold
        idx = resampling_fn(k1, mx.exp(log_first_norm), n)
        ancestors = mx.where(do_resample, idx, identity)

        # Kernel-smoothed parameter propagation.
        eps = mx.random.normal((n, params.shape[1]), key=k2)
        new_params = (
            mx.take(shrunk, ancestors, axis=0) + eps @ factors.scale_tril.T
        )

        keys = mx.random.split(k3, n)
        propagated = batch_trans(
            keys, mx.take(particles, ancestors, axis=0), new_params
        )

        # Second-stage correction (eta divided out at the ancestor
        # in the resampled branch only; plain W*g otherwise).
        log_obs = batch_obs(y_t, propagated, new_params)
        log_w_unnorm = mx.where(
            do_resample,
            log_obs - mx.take(log_aux, ancestors),
            log_w + log_obs,
        )
        log_w, log_sum = log_normalize(log_w_unnorm)
        inc = mx.where(do_resample, log_first_sum + log_sum - log_n, log_sum)
        ess_t = compute_ess(log_w)
        particles, params = propagated, new_params
        log_ml = log_ml + inc

        _check(t, inc)
        if store_history:
            all_particles.append(particles)
            all_params.append(params)
            all_log_w.append(log_w)
            all_ancestors.append(ancestors)
        all_ess.append(ess_t)
        all_inc.append(inc)

    if not store_history:
        all_particles = [particles]
        all_params = [params]
        all_log_w = [log_w]
        all_ancestors = [ancestors] if num_timesteps > 1 else all_ancestors

    # Neumaier-compensated total (ADR-0003).
    total = mx.array(0.0)
    comp = mx.array(0.0)
    for inc in all_inc:
        s = total + inc
        comp = comp + mx.where(
            mx.abs(total) >= mx.abs(inc), (total - s) + inc, (inc - s) + total
        )
        total = s

    return LiuWestPosterior(
        marginal_loglik=total + comp,
        filtered_particles=mx.stack(all_particles),
        filtered_log_weights=mx.stack(all_log_w),
        ancestors=mx.stack(all_ancestors),
        ess=mx.stack(all_ess),
        log_evidence_increments=mx.stack(all_inc),
        filtered_params=mx.stack(all_params),
    )
