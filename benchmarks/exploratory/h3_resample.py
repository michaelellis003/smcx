# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# H3: (1) how often does resampling trigger per workload at N=1e6
# (threshold 0.5, from the real filter's returned ESS history);
# (2) step time with resample pipeline forced ON vs forced OFF
# (standalone step replicas); expected saving of a lagged-ESS
# Python value-branch given observed trigger rates.
import json
import pathlib
import statistics
import sys
import time
from collections import deque

import mlx.core as mx

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "killtest")
)
from mlx_side import make_lgssm, make_sv, make_track

import smcx
from smcx.resampling import systematic
from smcx.weights import ess as compute_ess
from smcx.weights import log_normalize

SCRATCH = pathlib.Path(__file__).parent
N = 1_000_000


def loop_time(f_step, particles, log_w, y_t, t_steps=40, reps=12):
    key = mx.random.key(0)
    for _ in range(3):
        o = f_step(particles, log_w, key, y_t)
        mx.eval(o)
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
            p, lw2, ess_t, inc = f_step(p, lw2, sks[t], y_t)
            mx.async_eval(p, lw2)
            pending.append((inc, ess_t))
            if len(pending) > 4:
                i, e = pending.popleft()
                mx.eval(i, e)
                i.item()
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / t_steps)
    return statistics.median(ts)


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    out = {}
    for name, make in [
        ("lgssm", make_lgssm),
        ("sv", make_sv),
        ("track", make_track),
    ]:
        init, trans, logobs, y = make()
        # (1) trigger rate from a real run: trigger at step t uses
        # ESS of the *carried* weights = recorded ess[t-1]
        post = smcx.bootstrap_filter(
            mx.random.key(0),
            init,
            trans,
            logobs,
            y,
            N,
            store_history=False,
        )
        ess_hist = post.ess
        mx.eval(ess_hist)
        import numpy as np

        e = np.array(ess_hist)
        triggers = int((e[:-1] < 0.5 * N).sum())
        rate = triggers / (len(e) - 1)

        # (2) forced-on / forced-off / branchless step replicas
        particles = init(mx.random.key(1), N)
        lw0 = -0.5 * mx.random.normal((N,), key=mx.random.key(2)) ** 2
        log_w, _ = log_normalize(lw0)
        y_t = y[0]
        mx.eval(particles, log_w, y_t)
        identity = mx.arange(N, dtype=mx.int32)
        threshold = 0.5 * N
        log_n = mx.log(mx.array(float(N)))

        def mk_step(mode):
            def step(p, lww, step_key, yy):
                k1, k2 = mx.random.split(step_key)
                ess_prev = compute_ess(lww)
                do = ess_prev < threshold
                if mode == "on":
                    anc = systematic(k1, mx.exp(lww), N)
                    parents = mx.take(p, anc, axis=0)
                    lg_carry = mx.zeros_like(lww)
                elif mode == "off":
                    parents = p
                    lg_carry = lww
                else:  # branchless (current)
                    idx = systematic(k1, mx.exp(lww), N)
                    anc = mx.where(do, idx, identity)
                    parents = mx.take(p, anc, axis=0)
                    lg_carry = mx.where(do, mx.zeros_like(lww), lww)
                keys = mx.random.split(k2, N)
                prop = mx.vmap(trans)(keys, parents)
                lg = mx.vmap(logobs, in_axes=(None, 0))(yy, prop)
                lw_norm, log_sum = log_normalize(lg_carry + lg)
                inc = log_sum
                ess_t = compute_ess(lw_norm)
                return prop, lw_norm, ess_t, inc

            return mx.compile(step)

        row = {"triggers": triggers, "steps": len(e) - 1, "rate": rate}
        for mode in ("branchless", "on", "off"):
            row[f"step_{mode}"] = loop_time(
                mk_step(mode), particles, log_w, y_t
            )
        # expected value-branch step time given trigger rate
        row["value_branch_expected"] = (
            rate * row["step_on"] + (1 - rate) * row["step_off"]
        )
        out[name] = row
        print(name, row, flush=True)
    (SCRATCH / "h3_resample.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
