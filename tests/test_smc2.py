# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""SMC² tests (ported from the MLX suite; ADR-0014).

LGSSM with unknown AR coefficient ``a``; exact reference from the
Kalman log-likelihood on a fine a-grid integrated against the
U(0.5, 1.3) prior.
"""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

A_TRUE, Q, R, P0 = 0.9, 0.5, 0.3, 1.0
T = 40


def _np_lse(a):
    a = np.asarray(a, dtype=np.float64)
    m = a.max()
    return float(m + math.log(np.exp(a - m).sum()))


def _model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def param_init(key, n_theta):
        return 0.5 + 0.8 * jr.uniform(key, (n_theta, 1))

    def log_prior(theta):
        a = theta[0]
        inside = (a >= 0.5) & (a <= 1.3)
        return jnp.where(inside, math.log(1.0 / 0.8), -jnp.inf)

    def inner_init(key, n_x, theta):
        return sp * jr.normal(key, (n_x, 1))

    def inner_trans(key, state, theta):
        return theta[0] * state + sq * jr.normal(key, state.shape)

    def inner_logobs(y, state, theta):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - state[0]) ** 2 / R)

    return param_init, log_prior, inner_init, inner_trans, inner_logobs


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(0.0, math.sqrt(P0))
    for t in range(1, T):
        x[t] = A_TRUE * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return x + rng.normal(0, math.sqrt(R), T)


Y = _data()
Y_JX = jnp.asarray(Y)[:, None]
PARAM_INIT, LOG_PRIOR, INNER_INIT, INNER_TRANS, INNER_LOGOBS = _model()


def _exact_reference():
    y = Y.astype(np.float64)
    grid = np.linspace(0.5, 1.3, 2001)
    da = grid[1] - grid[0]
    ll = np.array([kalman_1d(y, a, Q, R, 0.0, P0)[0] for a in grid])
    log_prior = math.log(1.0 / 0.8)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    exact_mean = float((w * grid).sum())
    exact_logz = _np_lse(ll + log_prior + math.log(da))
    return exact_mean, exact_logz


EXACT_MEAN, EXACT_LOGZ = _exact_reference()


def _run(seed, n_theta=64, n_x=128, ess_threshold=0.0, **kw):
    return smcx.smc2(
        jr.key(seed),
        PARAM_INIT,
        LOG_PRIOR,
        INNER_INIT,
        INNER_TRANS,
        INNER_LOGOBS,
        Y_JX,
        n_theta,
        n_x,
        ess_threshold=ess_threshold,
        **kw,
    )


class TestStructure:
    """Shapes, invariants, determinism, degeneracy."""

    def test_container_shapes(self):
        post = _run(0)
        assert post.filtered_params.shape == (T, 64, 1)
        assert post.filtered_log_weights.shape == (T, 64)
        assert post.ess.shape == (T,)
        assert post.log_evidence_increments.shape == (T,)
        assert post.acceptance_rates.shape == (T,)

    def test_evidence_increments_sum_to_marginal(self):
        post = _run(1)
        assert float(jnp.sum(post.log_evidence_increments)) == pytest.approx(
            float(post.marginal_loglik), rel=1e-8
        )

    def test_outer_ess_in_range(self):
        post = _run(2)
        e = np.array(post.ess)
        assert np.all(e > 0) and np.all(e <= 64 + 1e-6)

    def test_deterministic_per_key(self):
        a = _run(3)
        b = _run(3)
        assert np.array_equal(
            np.array(a.marginal_loglik), np.array(b.marginal_loglik)
        )
        assert np.array_equal(
            np.array(a.filtered_params), np.array(b.filtered_params)
        )

    def test_store_history_false_matches_evidence(self):
        a = _run(4)
        b = _run(4, store_history=False)
        assert np.array_equal(
            np.array(a.marginal_loglik), np.array(b.marginal_loglik)
        )
        assert b.filtered_params.shape == (1, 64, 1)
        assert np.array_equal(
            np.array(a.filtered_params[-1]), np.array(b.filtered_params[0])
        )

    def test_degenerate_raises(self):
        def impossible(y, state, theta):
            return jnp.array(-jnp.inf)

        with pytest.raises(smcx.DegenerateWeightsError):
            smcx.smc2(
                jr.key(5),
                PARAM_INIT,
                LOG_PRIOR,
                INNER_INIT,
                INNER_TRANS,
                impossible,
                Y_JX,
                32,
                32,
                ess_threshold=0.0,
            )


class TestPosteriorRecovery:
    """The parameter posterior matches the exact grid reference."""

    def test_posterior_mean_and_logz_gate(self):
        r_keys = 8
        means, logzs = [], []
        for s in range(r_keys):
            post = _run(s, n_theta=128, n_x=256, ess_threshold=0.5)
            w = np.exp(np.array(post.filtered_log_weights[-1], np.float64))
            w /= w.sum()
            th = np.array(post.filtered_params[-1, :, 0], np.float64)
            means.append(float(w @ th))
            logzs.append(float(post.marginal_loglik))
        means, logzs = np.array(means), np.array(logzs)
        # MC-calibrated gates (R=8): mean within 5 SE; log Z within the
        # one-sided Jensen budget.
        se = means.std(ddof=1) / math.sqrt(r_keys)
        assert abs(means.mean() - EXACT_MEAN) < 5 * max(se, 1e-4)
        sd = logzs.std(ddof=1)
        err = logzs.mean() - EXACT_LOGZ
        upper = 3 * sd / math.sqrt(r_keys)
        lower = -(upper + 0.5 * sd**2)
        assert lower <= err <= upper, (err, sd, EXACT_LOGZ)


class TestReduction:
    """A point-mass prior reduces SMC² to a bank of bootstrap filters."""

    def test_logz_matches_bootstrap_at_point_mass(self):
        def point_init(key, n_theta):
            return jnp.full((n_theta, 1), A_TRUE)

        post = smcx.smc2(
            jr.key(9),
            point_init,
            LOG_PRIOR,
            INNER_INIT,
            INNER_TRANS,
            INNER_LOGOBS,
            Y_JX,
            16,
            512,
            ess_threshold=0.0,
        )
        # Exact Kalman log-lik at the point mass is the target.
        ll_true = kalman_1d(Y.astype(np.float64), A_TRUE, Q, R, 0.0, P0)[0]
        assert float(post.marginal_loglik) == pytest.approx(
            ll_true, abs=3.0 * math.sqrt(T) / math.sqrt(512)
        )


class TestRejuvenation:
    """PMMH rejuvenation behavior."""

    def test_rejuvenation_keeps_outer_ess_healthy(self):
        low = _run(10, ess_threshold=0.0)
        high = _run(10, ess_threshold=0.5)
        assert float(jnp.min(high.ess)) >= float(jnp.min(low.ess)) - 1e-6

    def test_pmmh_moves_fire_and_accept(self):
        post = _run(11, ess_threshold=0.9, num_pmmh_steps=2)
        acc = np.array(post.acceptance_rates)
        fired = acc[acc > 0]
        assert fired.size > 0
        assert np.all(fired <= 1.0)

    def test_rejuvenation_deterministic_per_key(self):
        a = _run(12, ess_threshold=0.5)
        b = _run(12, ess_threshold=0.5)
        assert np.array_equal(
            np.array(a.filtered_params), np.array(b.filtered_params)
        )

    def test_evidence_increments_sum_under_rejuvenation(self):
        post = _run(13, ess_threshold=0.5)
        assert float(jnp.sum(post.log_evidence_increments)) == pytest.approx(
            float(post.marginal_loglik), rel=1e-8
        )


class TestBatchedIndependence:
    """The theta axis never couples the inner filters."""

    def test_batched_resample_routes_each_row_independently(self):
        from smcx.smc2 import _batched_inner_resample

        w = jnp.stack([
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
        ])
        idx = _batched_inner_resample(jr.key(0), w, 4)
        assert np.all(np.array(idx[0]) == 0)
        assert np.all(np.array(idx[1]) == 3)
