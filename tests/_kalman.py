# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Exact Kalman filter oracle for 1-D LGSSM tests (numpy float64).

Model: x_0 ~ N(m0, p0); x_t = a x_{t-1} + b u_t + N(0, q);
y_t = x_t + N(0, r). Supports missing observations (NaN y_t skips
the update — the exact treatment, matching the masking recipe users
apply in log_observation_fn) and an optional control input u_t
applied to the transition INTO t (inputs[0] unused by the
transition; an input-conditioned initial distribution belongs in
``m0`` and ``p0`` for this oracle).
"""

import numpy as np


def kalman_1d(
    y: np.ndarray,
    a: float,
    q: float,
    r: float,
    m0: float,
    p0: float,
    b: float = 0.0,
    u: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact log-likelihood and filtered moments.

    Returns:
        (log_likelihood, filtered_means, filtered_variances) with
        one entry per time step.
    """
    t_len = len(y)
    u = np.zeros(t_len) if u is None else np.asarray(u, dtype=np.float64)
    means = np.empty(t_len)
    variances = np.empty(t_len)
    loglik = 0.0
    # t = 0: prior is the initial distribution (no transition).
    mean_pred, var_pred = m0, p0
    for t in range(t_len):
        if t > 0:
            mean_pred = a * means[t - 1] + b * u[t]
            var_pred = a * a * variances[t - 1] + q
        if np.isnan(y[t]):
            means[t], variances[t] = mean_pred, var_pred
            continue
        s = var_pred + r
        loglik += -0.5 * (np.log(2.0 * np.pi * s) + (y[t] - mean_pred) ** 2 / s)
        k_gain = var_pred / s
        means[t] = mean_pred + k_gain * (y[t] - mean_pred)
        variances[t] = (1.0 - k_gain) * var_pred
    return float(loglik), means, variances
