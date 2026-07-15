# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# H1: per-step phase breakdown per workload at N=1e5, 1e6.
# Phases compiled separately (the filter compiles the whole step; this
# isolates each phase's contribution). Timing: warm-up, then reps
# batches of B async-dispatched calls fenced by mx.synchronize;
# median of >=10 reps reported per phase.
import json
import pathlib
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/michaelellis/Projects/smcx/benchmarks/killtest")
from mlx_side import make_lgssm, make_sv, make_track

import smcx
from smcx.resampling import systematic
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

SCRATCH = pathlib.Path(__file__).parent


def timeit(f, args, batch, reps=12, warm=3):
    for _ in range(warm):
        out = f(*args)
        mx.eval(out)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch):
            out = f(*args)
            mx.async_eval(out)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / batch)
    return statistics.median(ts)


def phases_for(name, make, n):
    init, trans, logobs, y = make()
    key = mx.random.key(0)
    particles = init(mx.random.key(1), n)
    d = particles.shape[1]
    y_t = y[0]
    # realistic mildly-uneven weights
    lw = -0.5 * mx.random.normal((n,), key=mx.random.key(2)) ** 2
    log_w, _ = log_normalize(lw)
    mx.eval(particles, log_w, y_t)
    identity = mx.arange(n, dtype=mx.int32)
    threshold = 0.5 * n
    batch = 20 if n >= 10**6 else 50

    res = {}

    # (a) per-particle key split
    f_split = mx.compile(lambda k: mx.random.split(k, n))
    res["a_split"] = timeit(f_split, (key,), batch)

    # (b) vmapped transition draw, as the filter does (keys precomputed)
    keys_n = mx.random.split(key, n)
    mx.eval(keys_n)
    f_trans = mx.compile(lambda ks, p: mx.vmap(trans)(ks, p))
    res["b_mutate"] = timeit(f_trans, (keys_n, particles), batch)

    # (c) vmapped log-likelihood
    f_logg = mx.compile(lambda yy, p: mx.vmap(logobs, in_axes=(None, 0))(yy, p))
    res["c_logg"] = timeit(f_logg, (y_t, particles), batch)

    # (d) ESS + weight update (trigger ESS, where-fold, normalize,
    # increment, post ESS) given log_g
    log_g = f_logg(y_t, particles)
    mx.eval(log_g)
    log_n = mx.log(mx.array(float(n)))

    def weight_update(lww, lg):
        ess_prev = compute_ess(lww)
        do = ess_prev < threshold
        lw_un = mx.where(do, lg, lww + lg)
        lw_norm, log_sum = log_normalize(lw_un)
        inc = mx.where(do, log_sum - log_n, log_sum)
        ess_t = compute_ess(lw_norm)
        return lw_norm, inc, ess_t

    f_wu = mx.compile(weight_update)
    res["d_weights_ess"] = timeit(f_wu, (log_w, log_g), batch)

    # (e) full resample pipeline as dispatched by default: systematic
    # counting kernel + branchless where + ancestor take
    def resample(k, lww, p):
        ess_prev = compute_ess(lww)
        do = ess_prev < threshold
        idx = systematic(k, mx.exp(lww), n)
        anc = mx.where(do, idx, identity)
        parents = mx.take(p, anc, axis=0)
        return parents, anc

    f_rs = mx.compile(resample)
    res["e_resample"] = timeit(f_rs, (key, log_w, particles), batch)

    # whole compiled step, exactly the filter's _step (no APF)
    def step(p, lww, step_key, yy):
        k1, k2 = mx.random.split(step_key)
        ess_prev = compute_ess(lww)
        do = ess_prev < threshold
        idx = systematic(k1, mx.exp(lww), n)
        anc = mx.where(do, idx, identity)
        parents = mx.take(p, anc, axis=0)
        keys = mx.random.split(k2, n)
        prop = mx.vmap(trans)(keys, parents)
        lg = mx.vmap(logobs, in_axes=(None, 0))(yy, prop)
        lw_un = mx.where(do, lg, lww + lg)
        lw_norm, log_sum = log_normalize(lw_un)
        inc = mx.where(do, log_sum - log_n, log_sum)
        ess_t = compute_ess(lw_norm)
        return prop, lw_norm, anc, ess_t, inc

    f_step = mx.compile(step)

    # steady-state carried-loop timing with async+lag4 cadence
    def whole_step_loop(reps=12, t_steps=40):
        from collections import deque

        for _ in range(3):
            out = f_step(particles, log_w, key, y_t)
            mx.eval(out)
        mx.synchronize()
        ts = []
        for r in range(reps):
            p, lw2 = particles, log_w
            sks = mx.random.split(mx.random.key(100 + r), t_steps)
            mx.eval(sks)
            pending = deque()
            mx.synchronize()
            t0 = time.perf_counter()
            for t in range(t_steps):
                p, lw2, anc, ess_t, inc = f_step(p, lw2, sks[t], y_t)
                mx.async_eval(p, lw2, anc)
                pending.append((inc, ess_t))
                if len(pending) > 4:
                    i, e = pending.popleft()
                    mx.eval(i, e)
                    i.item()
            mx.synchronize()
            ts.append((time.perf_counter() - t0) / t_steps)
        return statistics.median(ts)

    res["whole_step_replica"] = whole_step_loop()

    # actual filter end-to-end per-step (cross-check vs kill test)
    t_data = y.shape[0]

    def run_filter():
        out = smcx.bootstrap_filter(mx.random.key(7), init, trans, logobs, y, n)
        mx.eval(out.marginal_loglik)

    run_filter()
    mx.synchronize()
    fts = []
    for _ in range(5 if n >= 10**6 else 10):
        mx.synchronize()
        t0 = time.perf_counter()
        run_filter()
        mx.synchronize()
        fts.append((time.perf_counter() - t0) / t_data)
    res["actual_filter_per_step"] = statistics.median(fts)
    res["T"] = t_data
    return res


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))  # burn Metal JIT
    out = {}
    for name, make in [
        ("lgssm", make_lgssm),
        ("sv", make_sv),
        ("track", make_track),
    ]:
        for n in (100_000, 1_000_000):
            print(f"{name} N={n}", flush=True)
            out[f"{name}/{n}"] = phases_for(name, make, n)
    (SCRATCH / "h1_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
