# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Internal Feynman-Kac core: one loop for every filter (ADR-0002).

Per-step order is load-bearing (design §2): conditional resample on
carried weights (ancestors logged; identity when skipped) -> mutate
-> reweight -> evidence increment at the reweight stage. Branch
conventions: ``log_w = where(resampled, log_g, log_W + log_g)`` with
increment ``LSE(log_w) - (log N if resampled else 0)``; t=0 is
init-as-if-resampled. No resample ever follows the final reweight.

This module is internal API (free to change): the FK model here is a
*data-sliced* form — mutate/log-potential callables receive per-step
data ``(y_t, input_t)`` as arrays rather than a Python time index, so
one ``mx.compile``d step serves every t without retracing (compile
treats captured Python scalars as cache keys; mlx-performance.md).
The guided filter's parent dependence is already threaded:
``log_g(prev_particles, particles, data)``.

Loop shell (measured cadence, mlx-performance.md): ``mx.async_eval``
each step + a blocking ``mx.eval`` on the carry from ``lag`` steps
back — full pipelining, bounded memory, and the lagged eval is where
degeneracy raises ``DegenerateWeightsError`` (detection latency <=
``lag`` steps; ADR-0003).
"""

from collections import deque
from collections.abc import Callable
from typing import NamedTuple

import mlx.core as mx

from smcx.containers import ParticleFilterPosterior, ParticleState
from smcx.exceptions import DegenerateWeightsError
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

_EVAL_LAG = 4


class FKModel(NamedTuple):
    """Data-sliced Feynman-Kac model (internal).

    Fields:
        m0: ``(key, num_particles) -> (N, d)`` initial cloud.
        m: ``(key, particles, data) -> particles`` mutation, already
            batched over particles (constructors vmap user callbacks).
        log_g: ``(prev_particles, particles, data) -> (N,)`` log
            potential; ``prev_particles`` are the post-resample
            parents (ignored by bootstrap/APF; needed by guided).
    """

    m0: Callable
    m: Callable
    log_g: Callable


def _neumaier_add(
    total: mx.array, comp: mx.array, x: mx.array
) -> tuple[mx.array, mx.array]:
    """One Neumaier compensated-summation step (branchless where)."""
    t = total + x
    comp = comp + mx.where(
        mx.abs(total) >= mx.abs(x), (total - t) + x, (x - t) + total
    )
    return t, comp


def run_filter(
    key: mx.array,
    fk: FKModel,
    data: tuple[mx.array, ...],
    num_particles: int,
    resampling_fn: Callable,
    resampling_threshold: float,
    store_history: bool = True,
) -> ParticleFilterPosterior:
    """Run the generic SMC loop over ``T`` steps of per-step data.

    Args:
        key: PRNG key (split internally; smcjax's split order).
        fk: The Feynman-Kac model.
        data: Tuple of arrays each with leading dimension T; sliced
            per step and passed to ``m``/``log_g`` (e.g.
            ``(emissions,)`` or ``(emissions, inputs)``).
        num_particles: N.
        resampling_fn: ADR-0004 contract resampler.
        resampling_threshold: Resample when ESS < threshold * N.
        store_history: When False (ADR-0011), the per-step particle,
            weight, and ancestor histories are not retained — the
            returned arrays cover only the final step (time axis
            length 1) while ``ess``/``log_evidence_increments`` stay
            full — dropping memory from O(T*N) to O(N).

    Returns:
        ParticleFilterPosterior (smcjax field parity).

    Raises:
        DegenerateWeightsError: All weights collapsed at some step
            (detected at the lagged eval boundary).
    """
    num_timesteps = data[0].shape[0]
    n = num_particles
    key, init_key = mx.random.split(key)
    log_n = mx.log(mx.array(float(n)))
    identity_ancestors = mx.arange(n, dtype=mx.int32)
    threshold = resampling_threshold * n

    # --- t = 0: init-as-if-resampled (design §2) ----------------------
    data_0 = tuple(d[0] for d in data)
    particles = fk.m0(init_key, n)
    log_g0 = fk.log_g(particles, particles, data_0)
    log_w, log_sum = log_normalize(log_g0)
    increment = log_sum - log_n
    ess_t = compute_ess(log_w)
    state = ParticleState(particles, log_w, increment)

    def _step(state: ParticleState, step_key: mx.array, *data_t: mx.array):
        k1, k2 = mx.random.split(step_key)
        ess_prev = compute_ess(state.log_weights)
        do_resample = ess_prev < threshold
        # Branchless conditional resample (playbook: correct at all N;
        # the value-branch optimization for large N is a bake-off item).
        idx = resampling_fn(k1, mx.exp(state.log_weights), n)
        ancestors = mx.where(do_resample, idx, identity_ancestors)
        parents = mx.take(state.particles, ancestors, axis=0)
        propagated = fk.m(k2, parents, data_t)
        log_g = fk.log_g(parents, propagated, data_t)
        # where-rule: carried weights fold in only when NOT resampled.
        log_w_unnorm = mx.where(do_resample, log_g, state.log_weights + log_g)
        log_w_norm, log_sum = log_normalize(log_w_unnorm)
        log_ev_inc = mx.where(do_resample, log_sum - log_n, log_sum)
        new_state = ParticleState(
            propagated,
            log_w_norm,
            state.log_marginal_likelihood + log_ev_inc,
        )
        ess_t = compute_ess(log_w_norm)
        return new_state, ancestors, ess_t, log_ev_inc

    step = mx.compile(_step)

    # --- Loop shell: async + lag-k eval, degeneracy at the boundary ---
    step_keys = mx.random.split(key, max(num_timesteps - 1, 1))
    all_particles = [particles]
    all_log_w = [log_w]
    all_ancestors = [identity_ancestors]
    all_ess = [ess_t]
    all_inc = [increment]

    pending: deque[tuple[int, mx.array, mx.array]] = deque()
    pending.append((0, increment, ess_t))

    def _check(t: int, inc: mx.array, ess_val: mx.array) -> None:
        mx.eval(inc, ess_val)
        v = inc.item()
        if v == float("-inf") or v != v:
            raise DegenerateWeightsError(
                f"all particle weights collapsed at step {t} "
                f"(log-evidence increment {v})"
            )

    for t in range(1, num_timesteps):
        data_t = tuple(d[t] for d in data)
        state, ancestors, ess_t, inc = step(state, step_keys[t - 1], *data_t)
        if store_history:
            all_particles.append(state.particles)
            all_log_w.append(state.log_weights)
            all_ancestors.append(ancestors)
        all_ess.append(ess_t)
        all_inc.append(inc)
        mx.async_eval(state.particles, state.log_weights, ancestors)
        pending.append((t, inc, ess_t))
        if len(pending) > _EVAL_LAG:
            _check(*pending.popleft())
    while pending:
        _check(*pending.popleft())
    if not store_history:
        # Final step only (time axis length 1); histories were never
        # retained, so their buffers freed as the pipeline advanced.
        all_particles = [state.particles]
        all_log_w = [state.log_weights]
        all_ancestors = [ancestors] if num_timesteps > 1 else all_ancestors

    # Neumaier-compensated total (ADR-0003); increments returned so
    # users can re-sum in f64.
    total = mx.array(0.0)
    comp = mx.array(0.0)
    for inc in all_inc:
        total, comp = _neumaier_add(total, comp, inc)

    return ParticleFilterPosterior(
        marginal_loglik=total + comp,
        filtered_particles=mx.stack(all_particles),
        filtered_log_weights=mx.stack(all_log_w),
        ancestors=mx.stack(all_ancestors),
        ess=mx.stack(all_ess),
        log_evidence_increments=mx.stack(all_inc),
    )
