# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Value-branch conditional resampling (spec: feat-11-value-branch)."""

import math

import mlx.core as mx
import numpy as np

import smcx
from smcx import _fk

A, Q, R = 0.9, 0.5, 0.3
T = 40


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
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=T)) * 0.4
    return mx.array((x + rng.normal(0, math.sqrt(R), T)).astype(np.float32))


Y = _emissions()


def _run(n, seed=0):
    init, trans, logobs = _model()
    return smcx.bootstrap_filter(mx.random.key(seed), init, trans, logobs, Y, n)


def test_modes_bit_identical(monkeypatch):
    # Same key, both modes: explicit-key RNG means the unconsumed
    # resample key on skip steps shifts nothing, and the decision
    # rule compares the same ESS array — so results must be
    # BIT-identical, not just close.
    n = 2000
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 1)  # force value
    a = _run(n)
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 10**12)  # force branchless
    b = _run(n)
    assert a.marginal_loglik.item() == b.marginal_loglik.item()
    assert np.array_equal(np.array(a.ancestors), np.array(b.ancestors))
    assert np.array_equal(
        np.array(a.filtered_log_weights), np.array(b.filtered_log_weights)
    )
    assert np.array_equal(
        np.array(a.log_evidence_increments),
        np.array(b.log_evidence_increments),
    )


def test_apf_keeps_branchless(monkeypatch):
    # APF's trigger lives inside the step (first-stage weights):
    # the value branch must not engage even above the threshold.
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 1)
    init, trans, logobs = _model()

    def logaux(y, s):
        v = Q + R
        return -0.5 * (math.log(2 * math.pi * v) + (y[0] - A * s[0]) ** 2 / v)

    post = smcx.auxiliary_filter(
        mx.random.key(1), init, trans, logobs, logaux, Y, 500
    )
    assert math.isfinite(post.marginal_loglik.item())


def test_degenerate_still_raises_in_value_mode(monkeypatch):
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 1)
    init, trans, _ = _model()

    def impossible(y, s):
        return mx.array(-mx.inf)

    import pytest

    with pytest.raises(smcx.DegenerateWeightsError):
        smcx.bootstrap_filter(mx.random.key(2), init, trans, impossible, Y, 200)
