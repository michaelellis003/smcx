# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""store_history=False tests (ADR-0011, ported from the MLX suite).

The MLX suite also asserted a device peak-memory drop; on CPU JAX
there is no comparable counter, and the memory property is structural
here — the final-only scan simply never stacks the (T, N) histories.
"""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import smcx

A, Q, R = 0.9, 0.5, 0.3
T = 60


def _model():
    sq = math.sqrt(Q)

    def init(key, n):
        return jr.normal(key, (n, 1))

    def trans(key, z):
        return A * z + sq * jr.normal(key, z.shape)

    def logobs(y, z):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - z[0]) ** 2 / R)

    return init, trans, logobs


def _emissions():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=T)) * 0.3
    return jnp.asarray(x + rng.normal(0, math.sqrt(R), T))[:, None]


Y = _emissions()


def _run(store, n=1000, seed=0):
    init, trans, logobs = _model()
    return smcx.bootstrap_filter(
        jr.key(seed), init, trans, logobs, Y, n, store_history=store
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
    assert np.array_equal(
        np.array(a.marginal_loglik), np.array(b.marginal_loglik)
    )
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
    assert np.array_equal(np.array(a.ancestors[-1]), np.array(b.ancestors[0]))


def test_still_satisfies_protocol():
    assert isinstance(_run(False), smcx.ParticleFilterResult)
