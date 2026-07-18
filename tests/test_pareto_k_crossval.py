# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Cross-validate the Pareto-k fit against ArviZ (ADR-0020).

The docstring of :func:`smcx.diagnostics._fit_generalized_pareto`
claims to match ArviZ's ``gpdfitnew`` / NumPyro's implementation of
the Zhang & Stephens (2009) estimator with the Vehtari et al. (2024)
prior. This test enforces the claim: the same log-weight vectors go
through smcx's fit and through ``arviz.psislw``, and the fitted shape
parameters must agree. ArviZ is a dev-only dependency (ADR-0020: it
never enters the runtime requirements).
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

# ArviZ 1.x moved the PSIS machinery out of the top-level namespace
# into arviz-stats; array_stats is its array-facing implementation.
from arviz_stats.base import array_stats

from smcx.containers import ParticleFilterPosterior
from smcx.diagnostics import pareto_k_diagnostic
from smcx.weights import log_normalize


def _khat_arviz(log_weights: np.ndarray) -> float:
    """Fit k on one weight vector via ArviZ's PSIS machinery.

    ``psislw`` follows the LOO convention: its argument holds
    log-likelihood values and the importance log-weights are their
    negation, so raw log-weights enter negated.
    """
    _, khat = array_stats.psislw(-np.asarray(log_weights))
    return float(khat)


def _posterior_from_log_weights(log_w_rows):
    """Wrap (T, N) normalized log-weights in a minimal posterior."""
    lw = jnp.stack([log_normalize(row)[0] for row in log_w_rows])
    t, n = lw.shape
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(0.0),
        filtered_particles=jnp.zeros((t, n, 1)),
        filtered_log_weights=lw,
        ancestors=jnp.zeros((t, n), dtype=jnp.int32),
        ess=jnp.ones(t),
        log_evidence_increments=jnp.zeros(t),
    )


@pytest.mark.parametrize(
    "tail_index",
    [0.25, 0.5, 0.8, 1.2],
    ids=lambda a: f"true-k-{a}",
)
def test_pareto_k_matches_arviz(key, tail_index):
    """Fitted k agrees with ArviZ across light to heavy tails.

    Log-weights are generated so the importance ratios are Pareto
    with shape 1/tail_index, giving a known true k. The two
    implementations share the algorithm but differ in minor
    numerical details, so agreement is required to 0.1 in k — far
    tighter than the 0.5/0.7 decision boundaries the statistic
    feeds.
    """
    n = 4_000
    keys = jr.split(key, 3)
    rows = []
    for k_i in keys:
        u = jr.uniform(k_i, (n,), minval=1e-12, maxval=1.0)
        # Inverse-CDF Pareto: ratios = u^(-tail_index) have shape
        # parameter 1/tail_index, so the GPD tail index is tail_index.
        rows.append(-tail_index * jnp.log(u))
    post = _posterior_from_log_weights(rows)

    ours = np.asarray(pareto_k_diagnostic(post))
    theirs = np.array([
        _khat_arviz(np.asarray(r, dtype=np.float64)) for r in rows
    ])

    np.testing.assert_allclose(ours, theirs, atol=0.1)


def test_pareto_k_matches_arviz_on_filter_output(lgssm_params, lgssm_data):
    """Agreement holds on real filter weights, not just synthetic."""
    import jax

    import smcx

    _, emissions = lgssm_data
    a = float(lgssm_params["dynamics_weights"][0, 0])
    q_sd = float(jnp.sqrt(lgssm_params["dynamics_cov"][0, 0]))
    r_var = float(lgssm_params["emissions_cov"][0, 0])

    def init(key, n):
        return jr.normal(key, (n, 1))

    def trans(key, z):
        return a * z + q_sd * jr.normal(key, z.shape)

    def log_obs(y, z):
        return -0.5 * (jnp.log(2 * jnp.pi * r_var) + (y[0] - z[0]) ** 2 / r_var)

    post = smcx.bootstrap_filter(
        jax.random.PRNGKey(3), init, trans, log_obs, emissions, 2_000
    )

    ours = np.asarray(pareto_k_diagnostic(post))
    lw = np.asarray(post.filtered_log_weights, dtype=np.float64)
    theirs = np.array([_khat_arviz(lw[t]) for t in range(lw.shape[0])])

    np.testing.assert_allclose(ours, theirs, atol=0.1)
