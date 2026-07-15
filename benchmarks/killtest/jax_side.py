# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Kill-test JAX-CPU baseline (run in the smcjax venv).

PROTOCOL.md pinning: x64 DISABLED (jax default), the whole filter
jitted as one program (jax.jit over the key; closures captured),
warm-up run before timing, block_until_ready fencing, R=20 runs per
(workload, N) — every run timed, so median/min/IQR come from 20
repeats (>= the protocol's 5) and the same log-Zs feed the gates.
Writes JSON to benchmarks/data/jax_results.json.
"""

import json
import pathlib
import resource
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import jax
import jax.numpy as jnp
import jax.random as jr
from gen_data import LGSSM, SV, track_matrices
from smcjax import bootstrap_filter

DATA = pathlib.Path(__file__).parent.parent / "data"
R_KEYS = 20
GRID = (10_000, 100_000, 1_000_000)


def make_lgssm():
    p = LGSSM
    sq, sp = jnp.sqrt(p["q"]), jnp.sqrt(p["p0"])

    def init(key, n):
        return p["m0"] + sp * jr.normal(key, (n, 1))

    def trans(key, s):
        return p["a"] * s + sq * jr.normal(key, s.shape)

    def logobs(y, s):
        return -0.5 * (
            jnp.log(2 * jnp.pi * p["r"]) + (y[0] - s[0]) ** 2 / p["r"]
        )

    y = jnp.asarray(np.load(DATA / "lgssm_y.npy"), dtype=jnp.float32)[:, None]
    return init, trans, logobs, y


def make_sv():
    s = SV
    s0 = s["sigma"] / np.sqrt(1 - s["phi"] ** 2)

    def init(key, n):
        return s0 * jr.normal(key, (n, 1))

    def trans(key, x):
        return s["phi"] * x + s["sigma"] * jr.normal(key, x.shape)

    def logobs(y, x):
        return -0.5 * (jnp.log(2 * jnp.pi) + x[0] + y[0] ** 2 * jnp.exp(-x[0]))

    y = jnp.asarray(np.load(DATA / "sv_y.npy"), dtype=jnp.float32)[:, None]
    return init, trans, logobs, y


def make_track(full_cov=False):
    f_mat, q_mat, _h_mat, r_diag, r_full = track_matrices()
    fj = jnp.asarray(f_mat, dtype=jnp.float32)
    lq = jnp.asarray(np.linalg.cholesky(q_mat), dtype=jnp.float32)
    lp0 = jnp.asarray(
        np.linalg.cholesky(np.diag([1.0, 1.0, 0.25, 0.25])),
        dtype=jnp.float32,
    )
    r_mat = r_full if full_cov else r_diag
    r_inv = jnp.asarray(np.linalg.inv(r_mat), dtype=jnp.float32)
    _, logdet = np.linalg.slogdet(2 * np.pi * r_mat)
    const = jnp.float32(-0.5 * logdet)

    def init(key, n):
        return (lp0 @ jr.normal(key, (n, 4)).T).T

    def trans(key, s):
        return fj @ s + lq @ jr.normal(key, (4,))

    def logobs(y, s):
        v = y - s[:2]
        return const - 0.5 * (v @ r_inv @ v)

    name = "track_y_full.npy" if full_cov else "track_y.npy"
    y = jnp.asarray(np.load(DATA / name), dtype=jnp.float32)
    return init, trans, logobs, y


def bench(make, n):
    init, trans, logobs, y = make()
    fn = jax.jit(lambda key: bootstrap_filter(key, init, trans, logobs, y, n))
    fn(jr.PRNGKey(999)).marginal_loglik.block_until_ready()  # warm/trace
    logzs, times = [], []
    for s in range(R_KEYS):
        t0 = time.perf_counter()
        out = fn(jr.PRNGKey(s)).marginal_loglik.block_until_ready()
        times.append(time.perf_counter() - t0)
        logzs.append(float(out))
    return {"logz": logzs, "times_s": times}


def run_workloads():
    return {
        "lgssm": make_lgssm,
        "sv": make_sv,
        "track": make_track,
        "track_full": lambda: make_track(full_cov=True),
    }


def main():
    if len(sys.argv) > 2:  # fresh-process-per-cell (protocol 07-15)
        wname, n = sys.argv[1], int(sys.argv[2])
        cell = bench(run_workloads()[wname], n)
        out = DATA / "cells"
        out.mkdir(exist_ok=True)
        (out / f"jax_{wname}_{n}.json").write_text(json.dumps(cell))
        print("done", wname, n)
        return

    results = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "x64": jax.config.read("jax_enable_x64"),
        "cpu_count": __import__("os").cpu_count(),
        "cells": {},
    }
    workloads = {
        "lgssm": make_lgssm,
        "sv": make_sv,
        "track": make_track,
        "track_full": lambda: make_track(full_cov=True),
    }
    for wname, make in workloads.items():
        for n in GRID:
            print(f"jax {wname} N={n}", flush=True)
            results["cells"][f"{wname}/{n}"] = bench(make, n)
    results["ru_maxrss_mb"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    )
    (DATA / "jax_results.json").write_text(json.dumps(results))
    print("done")


if __name__ == "__main__":
    main()
