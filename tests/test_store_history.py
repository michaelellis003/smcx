# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""store_history=False tests (spec: feat-4-store-history; ADR-0011)."""

import math

import mlx.core as mx
import numpy as np

import smcx

A, Q, R = 0.9, 0.5, 0.3
T = 60


def _model():
    sq = math.sqrt(Q)

    def init(key, n):
        return mx.random.normal((n, 1), key=key)

    def trans(key, s):
        return A * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    return init, trans, logobs


def _emissions():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=T)) * 0.3
    return mx.array((x + rng.normal(0, math.sqrt(R), T)).astype(np.float32))


Y = _emissions()


def _run(store, n=1000, seed=0):
    init, trans, logobs = _model()
    return smcx.bootstrap_filter(
        mx.random.key(seed), init, trans, logobs, Y, n, store_history=store
    )


def test_final_only_shapes():
    post = _run(False)
    assert post.filtered_particles.shape == (1, 1000, 1)
    assert post.filtered_log_weights.shape == (1, 1000)
    assert post.ancestors.shape == (1, 1000)
    assert post.ess.shape == (T,)
    assert post.log_evidence_increments.shape == (T,)


def test_marginal_loglik_bit_identical():
    a = _run(True)
    b = _run(False)
    assert a.marginal_loglik.item() == b.marginal_loglik.item()
    assert np.array_equal(np.array(a.ess), np.array(b.ess))
    assert np.array_equal(
        np.array(a.log_evidence_increments),
        np.array(b.log_evidence_increments),
    )


def test_final_step_matches_full_history_run():
    a = _run(True)
    b = _run(False)
    assert np.array_equal(
        np.array(a.filtered_particles[-1]), np.array(b.filtered_particles[0])
    )
    assert np.array_equal(
        np.array(a.filtered_log_weights[-1]),
        np.array(b.filtered_log_weights[0]),
    )


def test_still_satisfies_protocol():
    assert isinstance(_run(False), smcx.ParticleFilterResult)


def test_peak_memory_drops():
    # History at N=2e5, T=60: ~48 MB particles + 48 MB weights +
    # 48 MB ancestors vs final-only ~2 MB; pipeline overhead shared.
    # A 2x margin is far inside the ~10x expected gap.
    n = 200_000
    mx.reset_peak_memory()
    post = _run(True, n=n)
    mx.eval(post.filtered_particles, post.marginal_loglik)
    peak_full = mx.get_peak_memory()
    post = None  # release the full histories before the lean run
    del post
    mx.reset_peak_memory()
    lean = _run(False, n=n)
    mx.eval(lean.filtered_particles, lean.marginal_loglik)
    peak_lean = mx.get_peak_memory()
    assert peak_lean * 2 < peak_full, (peak_lean / 1e6, peak_full / 1e6)
