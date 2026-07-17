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

# Value-branch conditional resampling (perf-analysis.md #1): above
# this N, log_eta-free filters branch host-side on the previous
# step's already-materialized ESS and skip the resample pipeline
# entirely on skip steps (43-45% of a 1e6 step; smcjax's lax.cond
# does the same, so this is fairness-restoring). Below it, the
# branchless where-select wins (the sync tax dominates the skipped
# work). Results are bit-identical either way (explicit keys: the
# unconsumed resample key shifts nothing). APF keeps branchless:
# its trigger (first-stage W*eta ESS) only exists inside the step.
_VALUE_BRANCH_MIN_N = 50_000


def _select_loop_mode(
    has_log_eta: bool,
    num_particles: int,
    resampling_threshold: float,
) -> str:
    """Choose the loop-shell route (ADR-0016).

    Degenerate thresholds need no trigger, so they run dedicated
    pipelined steps with no host sync at any N. One documented edge:
    the where-rule resamples iff ``ess < threshold * N``, so at
    ``threshold >= 1.0`` it would skip only when the ESS equals N
    *exactly* (exactly uniform weights) or is NaN (degenerate, about
    to raise); the always-resample route resamples there too. APF's
    trigger (first-stage W*eta ESS) exists only inside the step, so
    ``log_eta`` pins the branchless route.
    """
    if has_log_eta:
        return "branchless"
    if resampling_threshold >= 1.0:
        return "always_resample"
    if resampling_threshold <= 0.0:
        return "never_resample"
    if num_particles >= _VALUE_BRANCH_MIN_N:
        return "value_branch"
    return "branchless"


class FKModel(NamedTuple):
    """Data-sliced Feynman-Kac model (internal).

    Fields:
        m0: ``(key, num_particles) -> (N, d)`` initial cloud.
        m: ``(key, particles, data) -> particles`` mutation, already
            batched over particles (constructors vmap user callbacks).
        log_g: ``(prev_particles, particles, data) -> (N,)`` log
            potential; ``prev_particles`` are the post-resample
            parents (ignored by bootstrap/APF; needed by guided).
        log_eta: optional APF twist ``(particles, data) -> (N,)``
            (the look-ahead, batched). Enters ONLY at the resample
            stage — first-stage weights W*eta drive the trigger and
            draw — and is divided out at the ancestor in the
            resampled branch only; when resampling is skipped, eta
            is not applied anywhere (ADR-0002; smcjax
            ``auxiliary.py`` semantics).
    """

    m0: Callable
    m: Callable
    log_g: Callable
    log_eta: Callable | None = None
    # t=0 potential override ``(particles, data) -> (N,)``: the
    # general guided potential g*f/q is undefined at t=0 (no
    # transition into the initial cloud), so guided supplies the
    # observation-only weighting here. Default: log_g with
    # prev = particles (bootstrap/APF semantics).
    log_g0: Callable | None = None


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
    if fk.log_g0 is not None:
        log_g0 = fk.log_g0(particles, data_0)
    else:
        log_g0 = fk.log_g(particles, particles, data_0)
    log_w, log_sum = log_normalize(log_g0)
    increment = log_sum - log_n
    ess_t = compute_ess(log_w)
    state = ParticleState(particles, log_w, increment)

    def _step(
        state: ParticleState,
        prev_ess: mx.array,
        step_key: mx.array,
        *data_t: mx.array,
    ):
        k1, k2 = mx.random.split(step_key)
        # APF twist (trace-time branch; fk is a compile capture):
        # first-stage weights W*eta drive the trigger AND the draw.
        if fk.log_eta is not None:
            log_aux = fk.log_eta(state.particles, data_t)
            log_first_norm, log_first_sum = log_normalize(
                state.log_weights + log_aux
            )
            # The first-stage ESS is a different quantity (W*eta), not
            # a recompute of the carried value.
            ess_prev = compute_ess(log_first_norm)
        else:
            log_aux = None
            log_first_norm = state.log_weights
            log_first_sum = mx.array(0.0)
            # ADR-0016: the carry already holds this step's trigger —
            # compute_ess(state.log_weights) is bit-identical to the
            # ess_t the previous step computed from the same array.
            ess_prev = prev_ess
        do_resample = ess_prev < threshold
        # Branchless conditional resample (playbook: correct at all N;
        # the value-branch optimization for large N is a bake-off item).
        idx = resampling_fn(k1, mx.exp(log_first_norm), n)
        ancestors = mx.where(do_resample, idx, identity_ancestors)
        parents = mx.take(state.particles, ancestors, axis=0)
        propagated = fk.m(k2, parents, data_t)
        log_g = fk.log_g(parents, propagated, data_t)
        if log_aux is not None:
            # Resampled branch: divide eta out at the ancestor
            # (second-stage correction). Skip branch: plain W*g —
            # eta appears NOWHERE (the bias the nontrivial-eta
            # threshold=0 test catches; design §2).
            log_second = log_g - mx.take(log_aux, ancestors)
            log_w_unnorm = mx.where(
                do_resample, log_second, state.log_weights + log_g
            )
            log_w_norm, log_sum = log_normalize(log_w_unnorm)
            # Two-factor increment when resampled:
            # log(sum W*eta) + LSE(second stage) - log N.
            log_ev_inc = mx.where(
                do_resample, log_first_sum + log_sum - log_n, log_sum
            )
        else:
            # where-rule: carried weights fold in only when NOT
            # resampled.
            log_w_unnorm = mx.where(
                do_resample, log_g, state.log_weights + log_g
            )
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

    # --- Value-branch step variants (log_eta-free path only) ----------
    def _step_resample(state: ParticleState, step_key, *data_t):
        k1, k2 = mx.random.split(step_key)
        idx = resampling_fn(k1, mx.exp(state.log_weights), n)
        parents = mx.take(state.particles, idx, axis=0)
        propagated = fk.m(k2, parents, data_t)
        log_g = fk.log_g(parents, propagated, data_t)
        log_w_norm, log_sum = log_normalize(log_g)
        log_ev_inc = log_sum - log_n
        new_state = ParticleState(
            propagated,
            log_w_norm,
            state.log_marginal_likelihood + log_ev_inc,
        )
        return new_state, idx, compute_ess(log_w_norm), log_ev_inc

    def _step_skip(state: ParticleState, step_key, *data_t):
        _, k2 = mx.random.split(step_key)  # same key discipline
        propagated = fk.m(k2, state.particles, data_t)
        log_g = fk.log_g(state.particles, propagated, data_t)
        log_w_norm, log_sum = log_normalize(state.log_weights + log_g)
        new_state = ParticleState(
            propagated,
            log_w_norm,
            state.log_marginal_likelihood + log_sum,
        )
        return new_state, identity_ancestors, compute_ess(log_w_norm), log_sum

    step_resample = mx.compile(_step_resample)
    step_skip = mx.compile(_step_skip)
    # ADR-0016: degenerate thresholds run the matching variant
    # pipelined (no host sync); mx.compile traces lazily, so unused
    # variants cost nothing.
    mode = _select_loop_mode(fk.log_eta is not None, n, resampling_threshold)

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
        if mode == "value_branch":
            # Host branch on the previous step's ESS — the same
            # array the branchless where-select compares in-graph,
            # so the decision (and the results) are identical.
            # NaN ESS compares False => skip, matching where().
            e = ess_t.item()
            if e < threshold:
                state, ancestors, ess_t, inc = step_resample(
                    state, step_keys[t - 1], *data_t
                )
            else:
                state, ancestors, ess_t, inc = step_skip(
                    state, step_keys[t - 1], *data_t
                )
        elif mode == "always_resample":
            state, ancestors, ess_t, inc = step_resample(
                state, step_keys[t - 1], *data_t
            )
        elif mode == "never_resample":
            state, ancestors, ess_t, inc = step_skip(
                state, step_keys[t - 1], *data_t
            )
        else:
            state, ancestors, ess_t, inc = step(
                state, ess_t, step_keys[t - 1], *data_t
            )
        if store_history:
            all_particles.append(state.particles)
            all_log_w.append(state.log_weights)
            all_ancestors.append(ancestors)
        all_ess.append(ess_t)
        all_inc.append(inc)
        # The checked scalars ride the same async pass: if inc/ess
        # stay out of the schedule, the lagged _check's blocking eval
        # recomputes step t-lag's reductions synchronously — measured
        # at ~0.17 ms/step, the dominant shell cost at small N.
        mx.async_eval(state.particles, state.log_weights, ancestors, inc, ess_t)
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
