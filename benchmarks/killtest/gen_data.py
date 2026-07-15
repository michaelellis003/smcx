# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Kill-test data generation (benchmarks/PROTOCOL.md).

Generates the three pre-registered workload datasets with fixed
seeds, computes exact Kalman-oracle log-likelihoods (numpy f64)
where linear-Gaussian, and writes sha256 hashes so both library
sides provably consume identical observations.

Workloads:
    LGSSM-1D: x_t = 0.9 x_{t-1} + N(0, 0.5); y_t = x_t + N(0, 0.3);
        x_0 ~ N(0, 1). T=100.
    SV-1: x_t = 0.98 x_{t-1} + N(0, 0.16^2);
        x_0 ~ N(0, 0.16^2/(1-0.98^2)); y_t = exp(x_t/2) N(0, 1).
        T=500. (No oracle; cross-library gate.)
    TRACK-4: constant-velocity 2-D tracking, d=4, dt=1,
        process q=0.1, emission R=diag(0.5, 0.5) (+ full-cov
        variant rho=0.5, report-only). T=200.
"""

import hashlib
import json
import pathlib

import numpy as np

DATA = pathlib.Path(__file__).parent.parent / "data"

# --- model constants (single source of truth for both sides) --------
LGSSM = {"a": 0.9, "q": 0.5, "r": 0.3, "m0": 0.0, "p0": 1.0, "T": 100}
SV = {"phi": 0.98, "sigma": 0.16, "T": 500}
TRACK = {"q": 0.1, "r": 0.5, "rho": 0.5, "T": 200, "dt": 1.0}


def track_matrices():
    """F, Q, H, R_diag, R_full for the tracking model (f64)."""
    dt = TRACK["dt"]
    f_mat = np.eye(4)
    f_mat[0, 2] = f_mat[1, 3] = dt
    q = TRACK["q"]
    qq = np.zeros((4, 4))
    for i in (0, 1):
        qq[i, i] = q * dt**3 / 3
        qq[i, i + 2] = qq[i + 2, i] = q * dt**2 / 2
        qq[i + 2, i + 2] = q * dt
    h_mat = np.zeros((2, 4))
    h_mat[0, 0] = h_mat[1, 1] = 1.0
    r = TRACK["r"]
    r_diag = np.diag([r, r])
    r_full = np.array([[r, TRACK["rho"] * r], [TRACK["rho"] * r, r]])
    return f_mat, qq, h_mat, r_diag, r_full


def kalman_general(y, f_mat, q_mat, h_mat, r_mat, m0, p0):
    """Exact multivariate Kalman log-likelihood (f64).

    t=0 uses the prior as the predictive (no transition into t=0).
    """
    loglik = 0.0
    mean, cov = m0.copy(), p0.copy()
    for t in range(y.shape[0]):
        if t > 0:
            mean = f_mat @ mean
            cov = f_mat @ cov @ f_mat.T + q_mat
        s = h_mat @ cov @ h_mat.T + r_mat
        v = y[t] - h_mat @ mean
        _, logdet = np.linalg.slogdet(2.0 * np.pi * s)
        loglik += -0.5 * (logdet + v @ np.linalg.solve(s, v))
        k_gain = cov @ h_mat.T @ np.linalg.inv(s)
        mean = mean + k_gain @ v
        cov = (np.eye(len(m0)) - k_gain @ h_mat) @ cov
    return float(loglik)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    meta = {"oracles": {}}

    rng = np.random.default_rng(20260714)
    # LGSSM-1D
    p = LGSSM
    x = np.empty(p["T"])
    x[0] = rng.normal(p["m0"], np.sqrt(p["p0"]))
    for t in range(1, p["T"]):
        x[t] = p["a"] * x[t - 1] + rng.normal(0, np.sqrt(p["q"]))
    y = x + rng.normal(0, np.sqrt(p["r"]), p["T"])
    np.save(DATA / "lgssm_y.npy", y)
    meta["oracles"]["lgssm"] = kalman_general(
        y[:, None],
        np.array([[p["a"]]]),
        np.array([[p["q"]]]),
        np.array([[1.0]]),
        np.array([[p["r"]]]),
        np.array([p["m0"]]),
        np.array([[p["p0"]]]),
    )

    # SV-1 (no oracle)
    s = SV
    x = np.empty(s["T"])
    x[0] = rng.normal(0, s["sigma"] / np.sqrt(1 - s["phi"] ** 2))
    for t in range(1, s["T"]):
        x[t] = s["phi"] * x[t - 1] + rng.normal(0, s["sigma"])
    y = np.exp(x / 2) * rng.normal(0, 1, s["T"])
    np.save(DATA / "sv_y.npy", y)

    # TRACK-4 (diag emission; full-cov variant reuses the same path)
    f_mat, q_mat, h_mat, r_diag, r_full = track_matrices()
    lq = np.linalg.cholesky(q_mat)
    m0 = np.zeros(4)
    p0 = np.diag([1.0, 1.0, 0.25, 0.25])
    x = np.empty((TRACK["T"], 4))
    x[0] = rng.multivariate_normal(m0, p0)
    for t in range(1, TRACK["T"]):
        x[t] = f_mat @ x[t - 1] + lq @ rng.normal(size=4)
    y_diag = x[:, :2] + rng.normal(0, np.sqrt(TRACK["r"]), (TRACK["T"], 2))
    np.save(DATA / "track_y.npy", y_diag)
    lr_full = np.linalg.cholesky(r_full)
    y_full = x[:, :2] + (lr_full @ rng.normal(size=(2, TRACK["T"]))).T
    np.save(DATA / "track_y_full.npy", y_full)
    meta["oracles"]["track"] = kalman_general(
        y_diag, f_mat, q_mat, h_mat, r_diag, m0, p0
    )
    meta["oracles"]["track_full"] = kalman_general(
        y_full, f_mat, q_mat, h_mat, r_full, m0, p0
    )

    meta["sha256"] = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        for f in sorted(DATA.glob("*.npy"))
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
