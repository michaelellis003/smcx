# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# End-to-end measured: branchless filter replica vs exact value-branch
# (Python branch on previous step's ESS via .item(), two compiled
# steps) over the REAL datasets. Also TRACK batched-model variants.
# Reports total wall time per config; logZ printed as sanity check.
import json
import pathlib
import statistics
import sys
import time
from collections import deque

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/michaelellis/Projects/smcx/benchmarks/killtest")
from gen_data import track_matrices
from mlx_side import make_lgssm, make_sv, make_track

from smcx.resampling import systematic
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

SCRATCH = pathlib.Path(__file__).parent


def build(make, n, batched_track=False):
    init, trans, logobs, y = make()
    if batched_track:
        f_mat, q_mat, _h, r_diag, _rf = track_matrices()
        fjT = mx.array(f_mat.astype(np.float32)).T
        lqT = mx.array(np.linalg.cholesky(q_mat).astype(np.float32)).T
        r_inv = mx.array(np.linalg.inv(r_diag).astype(np.float32))
        _, logdet = np.linalg.slogdet(2 * np.pi * r_diag)
        const = float(-0.5 * logdet)

        def mutate(k, p):
            return p @ fjT + mx.random.normal((n, 4), key=k) @ lqT

        def logg(yy, p):
            v = yy - p[:, :2]
            return const - 0.5 * mx.sum((v @ r_inv) * v, axis=1)

    else:

        def mutate(k, p):
            return mx.vmap(trans)(mx.random.split(k, n), p)

        def logg(yy, p):
            return mx.vmap(logobs, in_axes=(None, 0))(yy, p)

    identity = mx.arange(n, dtype=mx.int32)
    threshold = 0.5 * n
    log_n = mx.log(mx.array(float(n)))

    def step_branchless(p, lw, sk, yy, lml):
        k1, k2 = mx.random.split(sk)
        ess_prev = compute_ess(lw)
        do = ess_prev < threshold
        idx = systematic(k1, mx.exp(lw), n)
        anc = mx.where(do, idx, identity)
        parents = mx.take(p, anc, axis=0)
        prop = mutate(k2, parents)
        lg = logg(yy, prop)
        lw_un = mx.where(do, lg, lw + lg)
        lw_norm, log_sum = log_normalize(lw_un)
        inc = mx.where(do, log_sum - log_n, log_sum)
        ess_t = compute_ess(lw_norm)
        return prop, lw_norm, ess_t, lml + inc

    def step_resample(p, lw, sk, yy, lml):
        k1, k2 = mx.random.split(sk)
        anc = systematic(k1, mx.exp(lw), n)
        parents = mx.take(p, anc, axis=0)
        prop = mutate(k2, parents)
        lg = logg(yy, prop)
        lw_norm, log_sum = log_normalize(lg)
        ess_t = compute_ess(lw_norm)
        return prop, lw_norm, ess_t, lml + log_sum - log_n

    def step_skip(p, lw, sk, yy, lml):
        _k1, k2 = mx.random.split(sk)
        prop = mutate(k2, p)
        lg = logg(yy, prop)
        lw_norm, log_sum = log_normalize(lw + lg)
        ess_t = compute_ess(lw_norm)
        return prop, lw_norm, ess_t, lml + log_sum

    return (
        init,
        y,
        mx.compile(step_branchless),
        mx.compile(step_resample),
        mx.compile(step_skip),
        threshold,
    )


def run_branchless(built, n, seed):
    init, y, f_bl, _fr, _fs, _th = built
    t_steps = y.shape[0]
    key = mx.random.key(seed)
    key, ik = mx.random.split(key)
    sks = mx.random.split(key, t_steps)
    p = init(ik, n)
    lg0 = None  # t=0: weight by logobs, as the filter does
    # reuse skip-step machinery inline for t=0 simplicity:
    lw, _ = log_normalize(mx.zeros((n,)))
    lml = mx.array(0.0)
    pending = deque()
    for t in range(t_steps):
        p, lw, ess_t, lml = f_bl(p, lw, sks[t], y[t], lml)
        mx.async_eval(p, lw)
        pending.append(ess_t)
        if len(pending) > 4:
            mx.eval(pending.popleft())
    mx.eval(lml)
    return lml.item()


def run_valuebranch(built, n, seed):
    init, y, _fbl, f_r, f_s, th = built
    t_steps = y.shape[0]
    key = mx.random.key(seed)
    key, ik = mx.random.split(key)
    sks = mx.random.split(key, t_steps)
    p = init(ik, n)
    lw, _ = log_normalize(mx.zeros((n,)))
    lml = mx.array(0.0)
    ess_t = compute_ess(lw)
    n_trig = 0
    for t in range(t_steps):
        if ess_t.item() < th:  # exact trigger; syncs previous step
            n_trig += 1
            p, lw, ess_t, lml = f_r(p, lw, sks[t], y[t], lml)
        else:
            p, lw, ess_t, lml = f_s(p, lw, sks[t], y[t], lml)
        mx.async_eval(p, lw, ess_t, lml)
    mx.eval(lml)
    return lml.item(), n_trig


def bench(fn, reps):
    fn(999)  # warm
    mx.synchronize()
    ts, outs = [], []
    for s in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        o = fn(s)
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
        outs.append(o)
    return statistics.median(ts), outs[0]


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    out = {}
    configs = [
        ("lgssm", make_lgssm, False),
        ("sv", make_sv, False),
        ("track", make_track, False),
        ("track_batched", make_track, True),
    ]
    for name, make, bt in configs:
        for n in (100_000, 1_000_000):
            built = build(make, n, batched_track=bt)
            reps = 5 if n >= 10**6 else 10
            t_bl, z_bl = bench(lambda s: run_branchless(built, n, s), reps)
            t_vb, z_vb = bench(lambda s: run_valuebranch(built, n, s), reps)
            row = {
                "branchless_s": t_bl,
                "valuebranch_s": t_vb,
                "logz_bl": z_bl,
                "logz_vb": z_vb[0],
                "triggers": z_vb[1],
                "T": built[1].shape[0],
            }
            out[f"{name}/{n}"] = row
            print(name, n, row, flush=True)
    (SCRATCH / "h7_valuebranch.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
