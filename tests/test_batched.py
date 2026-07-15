# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Batched-model fast path tests (ADR-0013).

Batched and per-particle modes consume RNG differently, so
equivalence is tier-2 statistical (design §9b): mean log-Z within
3*sqrt(SD_a^2/R + SD_b^2/R) over R keys on the same data.
"""

import math

import mlx.core as mx
import numpy as np

import smcx

A, Q, R = 0.9, 0.5, 0.3
T = 50
R_KEYS = 10


def _emissions(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal()
    for t in range(1, T):
        x[t] = A * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return mx.array((x + rng.normal(0, math.sqrt(R), T)).astype(np.float32))


Y = _emissions()


def _lgssm(batched):
    sq = math.sqrt(Q)

    def init(key, n):
        return mx.random.normal((n, 1), key=key)

    if batched:

        def trans(key, particles):
            return A * particles + sq * mx.random.normal(
                particles.shape, key=key
            )

        def logobs(y, particles):
            return -0.5 * (
                math.log(2 * math.pi * R) + (y[0] - particles[:, 0]) ** 2 / R
            )

    else:

        def trans(key, s):
            return A * s + sq * mx.random.normal(s.shape, key=key)

        def logobs(y, s):
            return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    return init, trans, logobs


def _cross_bound(a, b):
    return 3 * math.sqrt(
        a.std(ddof=1) ** 2 / len(a) + b.std(ddof=1) ** 2 / len(b)
    )


class TestBootstrapBatched:
    """Statistical equivalence between calling conventions."""

    def test_logz_equivalent_to_vmapped(self):
        vals = {}
        for mode in (False, True):
            init, trans, logobs = _lgssm(mode)
            vals[mode] = np.array([
                smcx.bootstrap_filter(
                    mx.random.key(s),
                    init,
                    trans,
                    logobs,
                    Y,
                    2000,
                    batched=mode,
                ).marginal_loglik.item()
                for s in range(R_KEYS)
            ])
        diff = vals[True].mean() - vals[False].mean()
        assert abs(diff) <= _cross_bound(vals[True], vals[False]), diff

    def test_deterministic_and_shapes(self):
        init, trans, logobs = _lgssm(True)
        a = smcx.bootstrap_filter(
            mx.random.key(1), init, trans, logobs, Y, 500, batched=True
        )
        b = smcx.bootstrap_filter(
            mx.random.key(1), init, trans, logobs, Y, 500, batched=True
        )
        assert a.marginal_loglik.item() == b.marginal_loglik.item()
        assert a.filtered_particles.shape == (T, 500, 1)


class TestTrackGemmPath:
    """The motivating case: matrix-valued transitions as one GEMM."""

    def _track(self, batched):
        f_np = np.eye(4, dtype=np.float32)
        f_np[0, 2] = f_np[1, 3] = 1.0
        f_mx = mx.array(f_np)
        lq = mx.array((0.3 * np.eye(4)).astype(np.float32))
        r_noise = 0.5
        c = math.log(2 * math.pi * r_noise)

        def init(key, n):
            return mx.random.normal((n, 4), key=key)

        if batched:

            def trans(key, particles):
                return (
                    particles @ f_mx.T
                    + mx.random.normal(particles.shape, key=key) @ lq.T
                )

            def logobs(y, particles):
                v0 = y[0] - particles[:, 0]
                v1 = y[1] - particles[:, 1]
                return -0.5 * (2 * c + (v0 * v0 + v1 * v1) / r_noise)

        else:

            def trans(key, s):
                return f_mx @ s + lq @ mx.random.normal((4,), key=key)

            def logobs(y, s):
                v = y - s[:2]
                return -0.5 * (2 * c + mx.sum(v * v) / r_noise)

        return init, trans, logobs

    def _track_data(self):
        rng = np.random.default_rng(3)
        f_np = np.eye(4)
        f_np[0, 2] = f_np[1, 3] = 1.0
        x = np.zeros((40, 4))
        x[0] = rng.normal(size=4)
        for t in range(1, 40):
            x[t] = f_np @ x[t - 1] + 0.3 * rng.normal(size=4)
        return mx.array(
            (x[:, :2] + rng.normal(0, math.sqrt(0.5), (40, 2))).astype(
                np.float32
            )
        )

    def test_batched_equivalent_to_vmapped(self):
        y = self._track_data()
        vals = {}
        for mode in (False, True):
            init, trans, logobs = self._track(mode)
            vals[mode] = np.array([
                smcx.bootstrap_filter(
                    mx.random.key(s),
                    init,
                    trans,
                    logobs,
                    y,
                    2000,
                    batched=mode,
                ).marginal_loglik.item()
                for s in range(R_KEYS)
            ])
        diff = vals[True].mean() - vals[False].mean()
        assert abs(diff) <= _cross_bound(vals[True], vals[False]), diff


class TestGuidedAuxiliaryBatched:
    """Reductions hold in batched mode.

    Tier-2 statistical: ulp-level weight differences can flip a
    resample trigger at a single key, so same-key comparison is
    fragile by construction.
    """

    def _guided_vals(self):
        init, _, logobs = _lgssm(True)
        sq = math.sqrt(Q)

        def prop(key, particles, y):
            return A * particles + sq * mx.random.normal(
                particles.shape, key=key
            )

        def log_trans(new, old):
            return -0.5 * (
                math.log(2 * math.pi * Q) + (new[:, 0] - A * old[:, 0]) ** 2 / Q
            )

        def log_prop(y, new, old):
            return log_trans(new, old)

        return np.array([
            smcx.guided_filter(
                mx.random.key(s),
                init,
                prop,
                log_prop,
                log_trans,
                logobs,
                Y,
                1000,
                batched=True,
            ).marginal_loglik.item()
            for s in range(R_KEYS)
        ])

    def _bootstrap_vals(self):
        init, trans, logobs = _lgssm(True)
        return np.array([
            smcx.bootstrap_filter(
                mx.random.key(s), init, trans, logobs, Y, 1000, batched=True
            ).marginal_loglik.item()
            for s in range(R_KEYS)
        ])

    def test_guided_prior_proposal_matches_batched_bootstrap(self):
        a = self._guided_vals()
        b = self._bootstrap_vals()
        diff = a.mean() - b.mean()
        assert abs(diff) <= _cross_bound(a, b), (diff, _cross_bound(a, b))

    def test_auxiliary_flat_matches_batched_bootstrap(self):
        init, trans, logobs = _lgssm(True)

        def flat(y, particles):
            return mx.zeros((particles.shape[0],))

        a = np.array([
            smcx.auxiliary_filter(
                mx.random.key(s),
                init,
                trans,
                logobs,
                flat,
                Y,
                1000,
                batched=True,
            ).marginal_loglik.item()
            for s in range(R_KEYS)
        ])
        b = self._bootstrap_vals()
        diff = a.mean() - b.mean()
        assert abs(diff) <= _cross_bound(a, b), (diff, _cross_bound(a, b))
