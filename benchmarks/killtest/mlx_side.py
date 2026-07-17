# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Kill-test MLX side (run in the smcx venv).

Primary arm: GPU, async+lag-4 cadence (the shipped default). Cadence
sweep per PROTOCOL: per-step (lag 0), lag 2/4/8, pure async (lag
inf; peak memory reported). MLX-CPU recorded on the primary cadence.
One throwaway compile is burned first (first-in-process Metal JIT).
R=20 timed runs on the primary arm feed the gates; sweep arms get 5.
Writes JSON to benchmarks/data/mlx_results.json.
"""

import json
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import smcx._fk as fk
from gen_data import LGSSM, SV, track_matrices

import smcx

DATA = pathlib.Path(__file__).parent.parent / "data"
R_KEYS = 20
SWEEP_REPS = 5
GRID = (10_000, 100_000, 1_000_000)
LAGS = {"lag0": 0, "lag2": 2, "lag4": 4, "lag8": 8, "async": 10**9}


def make_lgssm():
    p = LGSSM
    sq, sp = p["q"] ** 0.5, p["p0"] ** 0.5
    c = float(np.log(2 * np.pi * p["r"]))

    def init(key, n):
        return p["m0"] + sp * mx.random.normal((n, 1), key=key)

    def trans(key, s):
        return p["a"] * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s):
        return -0.5 * (c + (y[0] - s[0]) ** 2 / p["r"])

    y = mx.array(np.load(DATA / "lgssm_y.npy").astype(np.float32))[:, None]
    return init, trans, logobs, y


def make_sv():
    s = SV
    s0 = s["sigma"] / (1 - s["phi"] ** 2) ** 0.5
    c = float(np.log(2 * np.pi))

    def init(key, n):
        return s0 * mx.random.normal((n, 1), key=key)

    def trans(key, x):
        return s["phi"] * x + s["sigma"] * mx.random.normal(x.shape, key=key)

    def logobs(y, x):
        return -0.5 * (c + x[0] + y[0] ** 2 * mx.exp(-x[0]))

    y = mx.array(np.load(DATA / "sv_y.npy").astype(np.float32))[:, None]
    return init, trans, logobs, y


def make_track_batched(full_cov=False):
    """ADR-0013 batched closures: transition as one GEMM.

    Disclosed in the results file; XLA's vmap already fuses, MLX's
    does not.
    """
    f_mat, q_mat, _h_mat, r_diag, r_full = track_matrices()
    fj = mx.array(f_mat.astype(np.float32))
    lq = mx.array(np.linalg.cholesky(q_mat).astype(np.float32))
    lp0 = mx.array(
        np.linalg.cholesky(np.diag([1.0, 1.0, 0.25, 0.25])).astype(np.float32)
    )
    r_mat = r_full if full_cov else r_diag
    r_inv = mx.array(np.linalg.inv(r_mat).astype(np.float32))
    _, logdet = np.linalg.slogdet(2 * np.pi * r_mat)
    const = float(-0.5 * logdet)

    def init(key, n):
        return mx.random.normal((n, 4), key=key) @ lp0.T

    def trans(key, particles):
        return (
            particles @ fj.T + mx.random.normal(particles.shape, key=key) @ lq.T
        )

    def logobs(y, particles):
        v = y[None, :] - particles[:, :2]
        z = v @ r_inv
        return const - 0.5 * mx.sum(z * v, axis=1)

    name = "track_y_full.npy" if full_cov else "track_y.npy"
    y = mx.array(np.load(DATA / name).astype(np.float32))
    return init, trans, logobs, y


def make_track(full_cov=False):
    f_mat, q_mat, _h_mat, r_diag, r_full = track_matrices()
    fj = mx.array(f_mat.astype(np.float32))
    lq = mx.array(np.linalg.cholesky(q_mat).astype(np.float32))
    lp0 = mx.array(
        np.linalg.cholesky(np.diag([1.0, 1.0, 0.25, 0.25])).astype(np.float32)
    )
    r_mat = r_full if full_cov else r_diag
    # Precomputed inverse: matmul-only hot loop (design §7).
    r_inv = mx.array(np.linalg.inv(r_mat).astype(np.float32))
    _, logdet = np.linalg.slogdet(2 * np.pi * r_mat)
    const = float(-0.5 * logdet)

    def init(key, n):
        return (lp0 @ mx.random.normal((n, 4), key=key).T).T

    def trans(key, s):
        return fj @ s + lq @ mx.random.normal((4,), key=key)

    def logobs(y, s):
        v = y - s[:2]
        return const - 0.5 * (v @ r_inv @ v)

    name = "track_y_full.npy" if full_cov else "track_y.npy"
    y = mx.array(np.load(DATA / name).astype(np.float32))
    return init, trans, logobs, y


def bench(make, n, reps, lag, store_history=True, batched=False):
    fk._EVAL_LAG = lag
    init, trans, logobs, y = make()
    # warm-up (also traces the compiled step for this N)
    out = smcx.bootstrap_filter(
        mx.random.key(999),
        init,
        trans,
        logobs,
        y,
        n,
        store_history=store_history,
        batched=batched,
    )
    mx.eval(out.marginal_loglik)
    mx.synchronize()
    logzs, times, peaks = [], [], []
    for s in range(reps):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = smcx.bootstrap_filter(
            mx.random.key(s),
            init,
            trans,
            logobs,
            y,
            n,
            store_history=store_history,
            batched=batched,
        )
        mx.eval(out.marginal_loglik)
        mx.synchronize()
        times.append(time.perf_counter() - t0)
        logzs.append(out.marginal_loglik.item())
        peaks.append(mx.get_peak_memory() / 1e6)
    return {"logz": logzs, "times_s": times, "peak_mb": peaks}


def run_cell(wname, n, workloads):
    make = workloads[wname]
    batched = wname.startswith("track")  # ADR-0013, disclosed
    cell = {}
    cell["gpu_lag4"] = bench(make, n, R_KEYS, 4, batched=batched)
    for arm, lag in LAGS.items():
        if lag == 4:
            continue
        cell[f"gpu_{arm}"] = bench(make, n, SWEEP_REPS, lag, batched=batched)
    cell["gpu_lag4_nohist"] = bench(
        make, n, SWEEP_REPS, 4, store_history=False, batched=batched
    )
    mx.set_default_device(mx.Device(mx.cpu))
    try:
        cell["cpu_lag4"] = bench(make, n, SWEEP_REPS, 4, batched=batched)
    finally:
        mx.set_default_device(mx.Device(mx.gpu))
    return cell


def main():
    # Burn first-in-process Metal JIT (~68 ms) before any timing.
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))

    workloads = {
        "lgssm": make_lgssm,
        "sv": make_sv,
        "track": lambda: make_track_batched(),
        "track_full": lambda: make_track_batched(full_cov=True),
    }
    if len(sys.argv) > 2:  # fresh-process-per-cell (protocol 07-15)
        wname, n = sys.argv[1], int(sys.argv[2])
        cell = run_cell(wname, n, workloads)
        out = DATA / "cells"
        out.mkdir(exist_ok=True)
        (out / f"mlx_{wname}_{n}.json").write_text(json.dumps(cell))
        print("done", wname, n)
        return

    results = {"mlx": mx.__version__, "cells": {}}
    for wname, make in workloads.items():
        for n in GRID:
            print(f"mlx {wname} N={n}", flush=True)
            cell = {}
            # primary arm: GPU, lag-4, R=20 (feeds gates + timing)
            cell["gpu_lag4"] = bench(make, n, R_KEYS, 4)
            # cadence sweep (timing only, 5 reps)
            for arm, lag in LAGS.items():
                if lag == 4:
                    continue
                cell[f"gpu_{arm}"] = bench(make, n, SWEEP_REPS, lag)
            # store_history=False arm (ADR-0011; report-only per the
            # 2026-07-15 protocol amendment)
            cell["gpu_lag4_nohist"] = bench(
                make, n, SWEEP_REPS, 4, store_history=False
            )
            # MLX-CPU on the primary cadence
            mx.set_default_device(mx.Device(mx.cpu))
            try:
                cell["cpu_lag4"] = bench(make, n, SWEEP_REPS, 4)
            finally:
                mx.set_default_device(mx.Device(mx.gpu))
            results["cells"][f"{wname}/{n}"] = cell
    fk._EVAL_LAG = 4
    (DATA / "mlx_results.json").write_text(json.dumps(results))
    print("done")


if __name__ == "__main__":
    main()
