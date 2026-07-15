# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# H4: TRACK mutation — vmapped per-particle F@s + L@normal((4,))
# vs batched particles@F.T + normal((N,4))@L.T. Also the vmapped
# log-obs vs batched log-obs, and a full batched-model step to
# project end-to-end TRACK gains.
import json
import pathlib
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/michaelellis/Projects/smcx/benchmarks/killtest")
from gen_data import track_matrices
from mlx_side import make_track

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


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    f_mat, q_mat, _h, r_diag, _rf = track_matrices()
    fj = mx.array(f_mat.astype(np.float32))
    lq = mx.array(np.linalg.cholesky(q_mat).astype(np.float32))
    r_inv = mx.array(np.linalg.inv(r_diag).astype(np.float32))
    _, logdet = np.linalg.slogdet(2 * np.pi * r_diag)
    const = float(-0.5 * logdet)
    fjT = fj.T
    lqT = lq.T

    init, trans, logobs, y = make_track()
    key = mx.random.key(0)
    out = {}
    for n in (100_000, 1_000_000):
        batch = 20 if n >= 10**6 else 50
        particles = init(mx.random.key(1), n)
        keys_n = mx.random.split(key, n)
        y_t = y[0]
        mx.eval(particles, keys_n, y_t)
        row = {}

        # current: vmapped per-particle
        f_v = mx.compile(lambda ks, p: mx.vmap(trans)(ks, p))
        row["mutate_vmap"] = timeit(f_v, (keys_n, particles), batch)

        # batched: one matmul + one batched draw (single key)
        def trans_batched(k, p):
            return p @ fjT + mx.random.normal((n, 4), key=k) @ lqT

        f_b = mx.compile(trans_batched)
        row["mutate_batched"] = timeit(f_b, (key, particles), batch)

        # batched but still per-particle keys folded (split included
        # in vmap arm for fairness)
        f_v2 = mx.compile(lambda k, p: mx.vmap(trans)(mx.random.split(k, n), p))
        row["mutate_vmap_incl_split"] = timeit(f_v2, (key, particles), batch)

        # log-obs: vmapped quadratic form vs batched
        f_lv = mx.compile(
            lambda yy, p: mx.vmap(logobs, in_axes=(None, 0))(yy, p)
        )
        row["logg_vmap"] = timeit(f_lv, (y_t, particles), batch)

        def logg_batched(yy, p):
            v = yy - p[:, :2]
            return const - 0.5 * mx.sum((v @ r_inv) * v, axis=1)

        f_lb = mx.compile(logg_batched)
        row["logg_batched"] = timeit(f_lb, (y_t, particles), batch)

        # full batched-model step (what a batched fast path buys)
        identity = mx.arange(n, dtype=mx.int32)
        threshold = 0.5 * n
        log_n = mx.log(mx.array(float(n)))
        lw0 = -0.5 * mx.random.normal((n,), key=mx.random.key(2)) ** 2
        log_w, _ = log_normalize(lw0)
        mx.eval(log_w)

        def step_batched(p, lww, step_key, yy):
            k1, k2 = mx.random.split(step_key)
            ess_prev = compute_ess(lww)
            do = ess_prev < threshold
            idx = systematic(k1, mx.exp(lww), n)
            anc = mx.where(do, idx, identity)
            parents = mx.take(p, anc, axis=0)
            prop = trans_batched(k2, parents)
            lg = logg_batched(yy, prop)
            lw_un = mx.where(do, lg, lww + lg)
            lw_norm, log_sum = log_normalize(lw_un)
            inc = mx.where(do, log_sum - log_n, log_sum)
            ess_t = compute_ess(lw_norm)
            return prop, lw_norm, anc, ess_t, inc

        f_step = mx.compile(step_batched)

        from collections import deque

        for _ in range(3):
            o = f_step(particles, log_w, key, y_t)
            mx.eval(o)
        mx.synchronize()
        ts = []
        for r in range(12):
            p, lw2 = particles, log_w
            sks = mx.random.split(mx.random.key(100 + r), 40)
            mx.eval(sks)
            pending = deque()
            mx.synchronize()
            t0 = time.perf_counter()
            for t in range(40):
                p, lw2, anc, ess_t, inc = f_step(p, lw2, sks[t], y_t)
                mx.async_eval(p, lw2, anc)
                pending.append((inc, ess_t))
                if len(pending) > 4:
                    i, e = pending.popleft()
                    mx.eval(i, e)
                    i.item()
            mx.synchronize()
            ts.append((time.perf_counter() - t0) / 40)
        row["whole_step_batched"] = statistics.median(ts)
        out[str(n)] = row
        print(n, {k: f"{v * 1e6:.1f}us" for k, v in row.items()}, flush=True)
    (SCRATCH / "h4_track.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
