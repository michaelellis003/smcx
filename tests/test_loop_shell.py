# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Loop shell v2 (ADR-0016; spec: perf-loop-shell).

Three anchors. The route policy is a pure function the shell must
expose (`_select_loop_mode`). Step bodies compute the ESS once — the
branchless trigger reuses the carried value instead of recomputing it
(observable at trace time: `compute_ess` runs while `mx.compile`
traces, so call counts count traces, not steps). And every route must
produce bit-identical posteriors for the same key: the grid below is
the contract any new fast path has to satisfy.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from smcx import _fk

A, Q, R = 0.9, 0.5, 0.3
M0, P0 = 0.0, 1.0
T = 24
N = 500


def _bootstrap_model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def init(key, n):
        return M0 + sp * mx.random.normal((n, 1), key=key)

    def trans(key, s):
        return A * s + sq * mx.random.normal(s.shape, key=key)

    def logobs(y, s):
        return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

    return init, trans, logobs


def _guided_extras():
    # Locally optimal proposal (Doucet et al. 2000), as in test_guided.
    s_star = 1.0 / (1.0 / Q + 1.0 / R)
    sd_star = math.sqrt(s_star)

    def prop_sample(key, s, y):
        m = s_star * (A * s / Q + y / R)
        return m + sd_star * mx.random.normal(s.shape, key=key)

    def log_prop(y, new, old):
        m = s_star * (A * old[0] / Q + y[0] / R)
        return -0.5 * (
            math.log(2 * math.pi * s_star) + (new[0] - m) ** 2 / s_star
        )

    def log_trans(new, old):
        return -0.5 * (
            math.log(2 * math.pi * Q) + (new[0] - A * old[0]) ** 2 / Q
        )

    return prop_sample, log_prop, log_trans


def _emissions(seed=7):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=T)) * 0.4
    return mx.array((x + rng.normal(0, math.sqrt(R), T)).astype(np.float32))


Y = _emissions()


def _run_bootstrap(threshold, store_history, seed=0):
    init, trans, logobs = _bootstrap_model()
    return smcx.bootstrap_filter(
        mx.random.key(seed),
        init,
        trans,
        logobs,
        Y,
        N,
        resampling_threshold=threshold,
        store_history=store_history,
    )


def _run_guided(threshold, store_history, seed=0):
    init, _, logobs = _bootstrap_model()
    prop_sample, log_prop, log_trans = _guided_extras()
    return smcx.guided_filter(
        mx.random.key(seed),
        init,
        prop_sample,
        log_prop,
        log_trans,
        logobs,
        Y,
        N,
        resampling_threshold=threshold,
        store_history=store_history,
    )


def _assert_posteriors_identical(a, b):
    assert a.marginal_loglik.item() == b.marginal_loglik.item()
    for field in (
        "filtered_particles",
        "filtered_log_weights",
        "ancestors",
        "ess",
        "log_evidence_increments",
    ):
        left = np.array(getattr(a, field))
        right = np.array(getattr(b, field))
        assert np.array_equal(left, right), field


# --- route policy (ADR-0016 decisions 2 and 3) ------------------------


def test_select_loop_mode_always_resample_at_threshold_one():
    # threshold >= 1.0 needs no trigger and no host sync, at any N.
    assert _fk._select_loop_mode(False, 100, 1.0) == "always_resample"
    assert _fk._select_loop_mode(False, 10**6, 1.0) == "always_resample"
    assert _fk._select_loop_mode(False, 10**6, 1.5) == "always_resample"


def test_select_loop_mode_never_resample_at_threshold_zero():
    assert _fk._select_loop_mode(False, 10**6, 0.0) == "never_resample"


def test_select_loop_mode_value_branch_is_size_and_threshold_gated():
    # Bake-off (perf-analysis.md, ADR-0016 decision 3): the sync only
    # pays for itself at large N and low trigger rates.
    assert (
        _fk._select_loop_mode(False, _fk._VALUE_BRANCH_MIN_N, 0.5)
        == "value_branch"
    )
    assert (
        _fk._select_loop_mode(False, _fk._VALUE_BRANCH_MIN_N - 1, 0.5)
        == "branchless"
    )
    assert (
        _fk._select_loop_mode(False, _fk._VALUE_BRANCH_MIN_N, 0.75)
        == "branchless"
    )


def test_select_loop_mode_apf_stays_branchless():
    # The APF trigger (first-stage W*eta ESS) only exists inside the
    # step, so log_eta forces the branchless route regardless of size
    # or threshold.
    assert _fk._select_loop_mode(True, 10**6, 0.5) == "branchless"
    assert _fk._select_loop_mode(True, 10**6, 1.0) == "branchless"


# --- single ESS per step (ADR-0016 decision 1) -------------------------


@pytest.fixture
def ess_counter(monkeypatch):
    calls = {"n": 0}
    real = _fk.compute_ess

    def counting(log_w):
        calls["n"] += 1
        return real(log_w)

    monkeypatch.setattr(_fk, "compute_ess", counting)
    return calls


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_log_eta_free_shell_computes_ess_once_per_trace(
    ess_counter, monkeypatch, threshold
):
    # compute_ess runs at trace time: t=0 once, plus once per traced
    # step body. The old branchless step traced two (ess_prev + ess_t
    # — ess_prev recomputes what the carry already holds), so the old
    # total was 3; a single-ESS shell traces 2.
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 10**12)
    _run_bootstrap(threshold, store_history=False)
    assert ess_counter["n"] == 2


def test_apf_keeps_its_first_stage_ess(ess_counter):
    # The APF first-stage ESS is a different quantity (W*eta), not a
    # recompute: t=0 once + step trace twice stays correct.
    init, trans, logobs = _bootstrap_model()

    def logaux(y, s):
        v = Q + R
        return -0.5 * (math.log(2 * math.pi * v) + (y[0] - A * s[0]) ** 2 / v)

    smcx.auxiliary_filter(mx.random.key(1), init, trans, logobs, logaux, Y, N)
    assert ess_counter["n"] == 3


# --- bit-identity: every route agrees with forced-branchless ----------


@pytest.mark.parametrize("store_history", [True, False])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("runner", [_run_bootstrap, _run_guided])
def test_routes_bit_identical(monkeypatch, runner, threshold, store_history):
    # The stochastic LGSSM never reaches exactly-uniform weights, so
    # the threshold=1.0 edge (ESS == N exactly, where the where-rule
    # would skip) is unreachable here; any fast path must match the
    # branchless reference bit-for-bit on this grid.
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 10**12)
    reference = runner(threshold, store_history)
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", 1)
    other = runner(threshold, store_history)
    _assert_posteriors_identical(reference, other)


@pytest.mark.parametrize("store_history", [True, False])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("runner", [_run_bootstrap, _run_guided])
def test_natural_route_matches_branchless_reference(
    monkeypatch, runner, threshold, store_history
):
    # The where-rule branchless step is the semantic reference: the
    # always-/never-resample fast paths and the value branch must all
    # reproduce it bit-for-bit (same stochastic-model caveat as above
    # for the threshold=1.0 exactly-uniform edge).
    natural = runner(threshold, store_history)
    monkeypatch.setattr(_fk, "_select_loop_mode", lambda *args: "branchless")
    reference = runner(threshold, store_history)
    _assert_posteriors_identical(reference, natural)


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("min_n", [1, 10**12])
def test_degenerate_raises_on_every_route(monkeypatch, min_n, threshold):
    # The threshold routes to never_resample / value_branch-or-
    # branchless / always_resample respectively; crossed with min_n
    # this covers every route's degeneracy path (reviewer finding).
    monkeypatch.setattr(_fk, "_VALUE_BRANCH_MIN_N", min_n)
    init, trans, _ = _bootstrap_model()

    def impossible(y, s):
        return mx.array(-mx.inf)

    with pytest.raises(smcx.DegenerateWeightsError):
        smcx.bootstrap_filter(
            mx.random.key(2),
            init,
            trans,
            impossible,
            Y,
            200,
            resampling_threshold=threshold,
        )
