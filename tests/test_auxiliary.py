# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Auxiliary filter tests (spec: feat-6-auxiliary; ADR-0002 twist)."""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

A, Q, R = 0.9, 0.5, 0.3
M0, P0 = 0.0, 1.0
T = 50


def _model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def init(key, n):
        return M0 + sp * mx.random.normal((n, 1), key=key)

    def trans(key, s):
        return A * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    def logaux_exact(y, s):
        # Exact predictive p(y_t | x_{t-1}) = N(y; A*x, Q + R) — the
        # "fully adapted" look-ahead for this LGSSM.
        v = Q + R
        return -0.5 * (math.log(2 * math.pi * v) + (y[0] - A * s[0]) ** 2 / v)

    return init, trans, logobs, logaux_exact


def _data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(M0, math.sqrt(P0))
    for t in range(1, T):
        x[t] = A * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return x + rng.normal(0, math.sqrt(R), T)


Y = _data()
LOGZ_TRUE, _, _ = kalman_1d(Y, A, Q, R, M0, P0)
Y_MX = mx.array(Y.astype(np.float32))[:, None]


class TestReductions:
    """The two structural equivalences from design §2/§9."""

    def test_flat_auxiliary_matches_bootstrap(self):
        # eta == 1: first-stage weights equal carried weights up to a
        # renormalization ulp, so results match to f32 tolerance (not
        # bit-exact: log_normalize of already-normalized weights).
        init, trans, logobs, _ = _model()

        def flat(y, s):
            return mx.array(0.0)

        a = smcx.auxiliary_filter(
            mx.random.key(1), init, trans, logobs, flat, Y_MX, 2000
        )
        b = smcx.bootstrap_filter(
            mx.random.key(1), init, trans, logobs, Y_MX, 2000
        )
        assert a.marginal_loglik.item() == pytest.approx(
            b.marginal_loglik.item(), abs=1e-3
        )
        assert np.allclose(np.array(a.ess), np.array(b.ess), rtol=1e-3)

    def test_nontrivial_eta_never_resampling_is_bootstrap_exact(self):
        # threshold=0 => the skip branch runs every step; eta must
        # appear NOWHERE (spurious eta-division would bias weights,
        # and the flat-aux test above cannot catch it). Same key,
        # same RNG consumption => bit-identical.
        init, trans, logobs, logaux = _model()
        a = smcx.auxiliary_filter(
            mx.random.key(2),
            init,
            trans,
            logobs,
            logaux,
            Y_MX,
            1000,
            resampling_threshold=0.0,
        )
        b = smcx.bootstrap_filter(
            mx.random.key(2),
            init,
            trans,
            logobs,
            Y_MX,
            1000,
            resampling_threshold=0.0,
        )
        assert a.marginal_loglik.item() == b.marginal_loglik.item()
        assert np.array_equal(
            np.array(a.filtered_log_weights),
            np.array(b.filtered_log_weights),
        )
        assert np.array_equal(
            np.array(a.log_evidence_increments),
            np.array(b.log_evidence_increments),
        )


class TestKalmanGate:
    """PROTOCOL-semantics gate with the exact-predictive look-ahead."""

    def test_log_ml_gate_r20(self):
        init, trans, logobs, logaux = _model()
        r_keys = 20
        vals = np.array([
            smcx.auxiliary_filter(
                mx.random.key(s), init, trans, logobs, logaux, Y_MX, 10_000
            ).marginal_loglik.item()
            for s in range(r_keys)
        ])
        sd = vals.std(ddof=1)
        err = vals.mean() - LOGZ_TRUE
        upper = 3 * sd / math.sqrt(r_keys)
        assert -(upper + 0.5 * sd**2) <= err <= upper, (err, sd)


class TestStructure:
    """Invariants, store_history, and entry validation."""

    def test_invariants_and_protocol(self):
        init, trans, logobs, logaux = _model()
        post = smcx.auxiliary_filter(
            mx.random.key(3), init, trans, logobs, logaux, Y_MX, 1000
        )
        assert isinstance(post, smcx.ParticleFilterResult)
        total = np.array(post.log_evidence_increments, dtype=np.float64).sum()
        assert post.marginal_loglik.item() == pytest.approx(total, abs=5e-4)
        e = np.array(post.ess)
        assert np.all(e >= 1 - 1e-4) and np.all(e <= 1000 * (1 + 1e-4))

    def test_store_history_final_only(self):
        init, trans, logobs, logaux = _model()
        post = smcx.auxiliary_filter(
            mx.random.key(4),
            init,
            trans,
            logobs,
            logaux,
            Y_MX,
            500,
            store_history=False,
        )
        assert post.filtered_particles.shape == (1, 500, 1)
        assert post.ess.shape == (T,)

    def test_arity_mismatch_raises(self):
        init, _, _, logaux = _model()
        with pytest.raises(TypeError, match="log_auxiliary_fn"):
            smcx.auxiliary_filter(
                mx.random.key(5),
                init,
                lambda k, s, u: s,
                lambda y, s, u: mx.array(0.0),
                logaux,  # 2-arg while inputs supplied
                Y_MX,
                100,
                inputs=mx.zeros((T,)),
            )
