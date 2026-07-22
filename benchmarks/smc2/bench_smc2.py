# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""SMC² device benchmark: MLX-GPU vs MLX-CPU on the same smcx code.

This second kill test uses SMC²'s live (N_theta x N_x) tensor — the
densest, most batch-shaped workload in
the SMC literature — so it is the strongest case for the unified-
memory thesis. This benchmark isolates the hardware: identical smcx
code, only the MLX default device changes.

Model: LGSSM with an unknown AR coefficient a (z_t = a z_{t-1} + q,
y_t = z_t + r), so the marginal likelihood has an exact Kalman-grid
reference for the correctness gate. Chopin's `particles` supplies an
external baseline in ``particles_side.py``.

Fresh process per cell (protocol amendment 2026-07-15): the driver
re-invokes this script with ``--cell`` so no compile cache or device
state leaks between cells. Run: ``python bench_smc2.py`` (driver) or
``python bench_smc2.py --cell gpu 512 512 100`` (one cell).
"""

import json
import math
import os
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

A_TRUE, Q, R, P0 = 0.9, 0.5, 0.3, 1.0
CELLS = [(512, 512, 100), (1024, 1024, 100)]
REPS = 5
# |log Zhat - exact log Z|; ~6 SE at these N, catches breakage.
# Overridable so CI can tighten the gate.
GATE_TOL = float(os.environ.get("SMC2_BENCH_GATE_TOL", "0.5"))


def _data(t_len, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(t_len)
    x[0] = rng.normal(0.0, math.sqrt(P0))
    for t in range(1, t_len):
        x[t] = A_TRUE * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return (x + rng.normal(0, math.sqrt(R), t_len)).astype(np.float32)


def _kalman_loglik(y, a):
    """Exact LGSSM log-likelihood at a fixed a (1-D Kalman)."""
    m, p, ll = 0.0, P0, 0.0
    for t in range(len(y)):
        if t > 0:
            m, p = a * m, a * a * p + Q
        s = p + R
        ll += -0.5 * (math.log(2 * math.pi * s) + (y[t] - m) ** 2 / s)
        k = p / s
        m, p = m + k * (y[t] - m), (1 - k) * p
    return ll


def _exact_logz(y):
    grid = np.linspace(0.5, 1.3, 2001)
    da = grid[1] - grid[0]
    ll = np.array([_kalman_loglik(y, a) for a in grid])
    shifted = ll + math.log(1.0 / 0.8) + math.log(da)
    m = shifted.max()
    return float(m + math.log(np.exp(shifted - m).sum()))


def _model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def pinit(k, n):
        return 0.5 + 0.8 * mx.random.uniform(shape=(n, 1), key=k)

    def lprior(th):
        a = th[0]
        return mx.where((a >= 0.5) & (a <= 1.3), math.log(1.0 / 0.8), -mx.inf)

    def iinit(k, nx, th):
        return sp * mx.random.normal((nx, 1), key=k)

    def itrans(k, s, th):
        return th[0] * s + sq * mx.random.normal(s.shape, key=k)

    def ilogobs(y, s, th):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    return pinit, lprior, iinit, itrans, ilogobs


def run_cell(device, n_theta, n_x, t_len):
    import smcx

    mx.set_default_device(mx.Device(mx.gpu if device == "gpu" else mx.cpu))
    y = _data(t_len)
    em = mx.array(y)[:, None]
    pinit, lprior, iinit, itrans, ilogobs = _model()

    def one():
        post = smcx.smc2(
            mx.random.key(0),
            pinit,
            lprior,
            iinit,
            itrans,
            ilogobs,
            em,
            n_theta,
            n_x,
            ess_threshold=0.5,
            num_pmmh_steps=3,
        )
        mx.eval(post.marginal_loglik)
        return post

    one()  # warm up the compile
    mx.synchronize()
    mx.reset_peak_memory()
    times = []
    for _ in range(REPS):
        mx.synchronize()
        t0 = time.time()
        post = one()
        mx.synchronize()
        times.append(time.time() - t0)
    logz = post.marginal_loglik.item()
    exact = _exact_logz(y)
    return {
        "device": device,
        "n_theta": n_theta,
        "n_x": n_x,
        "t_len": t_len,
        "median_s": float(np.median(times)),
        "peak_gb": mx.get_peak_memory() / 1e9,
        "logz": logz,
        "exact_logz": exact,
        "gate_pass": abs(logz - exact) < GATE_TOL,
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cell":
        device = sys.argv[2]
        n_theta = int(sys.argv[3])
        n_x = int(sys.argv[4])
        t_len = int(sys.argv[5])
        print(json.dumps(run_cell(device, n_theta, n_x, t_len)))
        return

    rows = []
    for n_theta, n_x, t_len in CELLS:
        cell = {}
        for device in ("gpu", "cpu"):
            out = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--cell",
                    device,
                    str(n_theta),
                    str(n_x),
                    str(t_len),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            cell[device] = json.loads(out.stdout.strip().splitlines()[-1])
        speedup = cell["cpu"]["median_s"] / cell["gpu"]["median_s"]
        gate = cell["gpu"]["gate_pass"] and cell["cpu"]["gate_pass"]
        rows.append((n_theta, n_x, cell, speedup, gate))
        print(
            f"N_theta={n_theta} N_x={n_x} "
            f"({n_theta * n_x / 1e6:.2f}M inner particles): "
            f"GPU {cell['gpu']['median_s']:.2f}s / "
            f"CPU {cell['cpu']['median_s']:.2f}s = {speedup:.1f}x  "
            f"| gate {'PASS' if gate else 'FAIL'} "
            f"(logZ {cell['gpu']['logz']:.2f} vs exact "
            f"{cell['gpu']['exact_logz']:.2f})"
        )

    # The gate must be able to fail the run: a speed number from a
    # numerically broken filter is worthless, so exit non-zero if any
    # cell's log Zhat drifts from the exact Kalman-grid reference.
    failed = [
        (nth, nx, dev, cell[dev]["logz"], cell[dev]["exact_logz"])
        for nth, nx, cell, _, gate in rows
        if not gate
        for dev in ("gpu", "cpu")
        if not cell[dev]["gate_pass"]
    ]
    if failed:
        raise SystemExit(
            "correctness gate FAILED (|log Zhat - exact| >= "
            f"{GATE_TOL}): "
            + "; ".join(
                f"N_theta={n} N_x={x} {d}: {lz:.2f} vs {ex:.2f}"
                for n, x, d, lz, ex in failed
            )
        )
    return rows


if __name__ == "__main__":
    main()
