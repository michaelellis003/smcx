# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap filter tests (spec: feat-3-bootstrap).

Correctness gates are MC-error-calibrated per design §9 and
benchmarks/PROTOCOL.md: over R keys,
-(3*SD/sqrt(R) + SD**2/2) <= mean(logZ) - logZ_kalman <= 3*SD/sqrt(R)
— the Jensen budget (E[log Zhat] ~ log Z - Var/2) is one-sided; an
upward deviation of that size indicates a bug and is not excused.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

# 1-D LGSSM fixture: x_t = A x_{t-1} + N(0, Q); y_t = x_t + N(0, R).
A, Q, R_NOISE = 0.9, 0.5, 0.3
M0, P0 = 0.0, 1.0
T = 50


def _make_model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def initial_sampler(key, n):
        return M0 + sp * mx.random.normal((n, 1), key=key)

    def transition_sampler(key, state):
        return A * state + sq * mx.random.normal(state.shape, key=key)

    def log_observation_fn(y, state):
        return -0.5 * (
            math.log(2 * math.pi * R_NOISE) + (y[0] - state[0]) ** 2 / R_NOISE
        )

    return initial_sampler, transition_sampler, log_observation_fn


def _simulate_data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(T)
    x[0] = rng.normal(M0, math.sqrt(P0))
    for t in range(1, T):
        x[t] = A * x[t - 1] + rng.normal(0, math.sqrt(Q))
    y = x + rng.normal(0, math.sqrt(R_NOISE), size=T)
    return y


Y = _simulate_data()
LOGZ_TRUE, KMEANS, KVARS = kalman_1d(Y, A, Q, R_NOISE, M0, P0)
Y_MX = mx.array(Y.astype(np.float32))[:, None]


def _run(key_seed, n=10_000, emissions=Y_MX, **kw):
    init, trans, logobs = _make_model()
    return smcx.bootstrap_filter(
        mx.random.key(key_seed), init, trans, logobs, emissions, n, **kw
    )


class TestKalmanGate:
    """PROTOCOL-semantics correctness gate against the exact oracle."""

    def test_log_ml_gate_r20(self):
        r_keys = 20
        vals = np.array([_run(s).marginal_loglik.item() for s in range(r_keys)])
        sd = vals.std(ddof=1)
        err = vals.mean() - LOGZ_TRUE
        upper = 3 * sd / math.sqrt(r_keys)
        lower = -(upper + 0.5 * sd**2)
        assert lower <= err <= upper, (err, sd)

    def test_estimator_spread_shrinks_with_n(self):
        # Var(log Zhat) ~ 1/N, so SD at N=100 vs N=10^4 differ ~10x —
        # decisive at R=10 keys, unlike mean-error comparisons which
        # are MC-noise-vs-MC-noise at this R (both biases are ~SD^2/2
        # and tiny on this easy fixture).
        r_keys = 10
        sds = []
        for n in (100, 10_000):
            vals = np.array([
                _run(s, n=n).marginal_loglik.item() for s in range(r_keys)
            ])
            sds.append(vals.std(ddof=1))
        assert sds[1] < sds[0] / 2

    def test_filtered_means_track_kalman(self):
        post = _run(0)
        w = np.exp(np.array(post.filtered_log_weights, dtype=np.float64))
        means = (w * np.array(post.filtered_particles)[:, :, 0]).sum(axis=1)
        # MC error of the weighted mean ~ sqrt(Var_filt/ESS); with
        # ESS >~ 2000 and Var <= P0 this is < 0.03; atol 0.15 = ~5x.
        assert np.allclose(means, KMEANS, atol=0.15)


class TestStructure:
    """Container invariants and loop conventions."""

    def test_shapes_and_t0_identity_ancestors(self):
        post = _run(1, n=256)
        assert post.filtered_particles.shape == (T, 256, 1)
        assert post.filtered_log_weights.shape == (T, 256)
        assert post.ancestors.shape == (T, 256)
        assert np.array_equal(np.array(post.ancestors[0]), np.arange(256))

    def test_increments_sum_to_marginal(self):
        post = _run(2, n=1000)
        total = np.array(post.log_evidence_increments, dtype=np.float64).sum()
        assert post.marginal_loglik.item() == pytest.approx(
            total, abs=5e-4
        )  # Neumaier vs f64 re-sum at T=50: ulp-scale slack

    def test_weights_normalized_every_step(self):
        post = _run(3, n=1000)
        lse = np.array([
            mx.logsumexp(post.filtered_log_weights[t]).item() for t in range(T)
        ])
        assert np.allclose(lse, 0.0, atol=1e-5)

    def test_ess_bounds(self):
        post = _run(4, n=1000)
        e = np.array(post.ess)
        assert np.all(e >= 1.0 - 1e-4) and np.all(e <= 1000 * (1 + 1e-4))

    def test_deterministic_per_key(self):
        a = _run(5, n=500)
        b = _run(5, n=500)
        assert np.array_equal(
            np.array(a.filtered_particles), np.array(b.filtered_particles)
        )
        assert a.marginal_loglik.item() == b.marginal_loglik.item()

    def test_result_satisfies_protocol(self):
        assert isinstance(_run(6, n=64), smcx.ParticleFilterResult)


class TestEdgeCases:
    """Degeneracy, missing data, univariate emissions, inputs."""

    def test_degenerate_weights_raise(self):
        init, trans, _ = _make_model()

        def impossible(y, state):
            return mx.array(-mx.inf)

        with pytest.raises(smcx.DegenerateWeightsError, match="step"):
            smcx.bootstrap_filter(
                mx.random.key(0), init, trans, impossible, Y_MX, 100
            )

    def test_univariate_emissions_canonicalized(self):
        y_flat = mx.array(Y.astype(np.float32))
        a = _run(7, n=500, emissions=y_flat)
        b = _run(7, n=500, emissions=Y_MX)
        assert a.marginal_loglik.item() == b.marginal_loglik.item()

    def test_missing_observations_match_gapped_kalman(self):
        # 20% missing: mask in log_observation_fn (design §4 recipe);
        # the Kalman oracle skips those updates exactly.
        rng = np.random.default_rng(42)
        y_gap = Y.copy()
        y_gap[rng.choice(T, size=T // 5, replace=False)] = np.nan
        logz_gap, _, _ = kalman_1d(y_gap, A, Q, R_NOISE, M0, P0)
        init, trans, _ = _make_model()

        def masked_logobs(y, state):
            dens = -0.5 * (
                math.log(2 * math.pi * R_NOISE)
                + (y[0] - state[0]) ** 2 / R_NOISE
            )
            return mx.where(mx.isnan(y[0]), mx.array(0.0), dens)

        y_mx = mx.array(y_gap.astype(np.float32))[:, None]
        r_keys = 10
        vals = np.array([
            smcx.bootstrap_filter(
                mx.random.key(s), init, trans, masked_logobs, y_mx, 10_000
            ).marginal_loglik.item()
            for s in range(r_keys)
        ])
        sd = vals.std(ddof=1)
        err = vals.mean() - logz_gap
        upper = 3 * sd / math.sqrt(r_keys)
        assert -(upper + 0.5 * sd**2) <= err <= upper

    def test_inputs_channel_matches_kalman_with_control(self):
        # x_t = A x_{t-1} + B u_t + noise; u known. ADR-0008 trailing
        # input_t on both callbacks (observation ignores it here).
        b_ctrl = 0.7
        rng = np.random.default_rng(9)
        u = rng.normal(size=T)
        x = np.empty(T)
        x[0] = rng.normal(M0, math.sqrt(P0))
        for t in range(1, T):
            x[t] = A * x[t - 1] + b_ctrl * u[t] + rng.normal(0, math.sqrt(Q))
        y = x + rng.normal(0, math.sqrt(R_NOISE), size=T)
        logz_true, _, _ = kalman_1d(y, A, Q, R_NOISE, M0, P0, b=b_ctrl, u=u)

        init, _, _ = _make_model()
        sq = math.sqrt(Q)

        def trans_u(key, state, input_t):
            return (
                A * state
                + b_ctrl * input_t
                + sq * mx.random.normal(state.shape, key=key)
            )

        def logobs_u(yv, state, input_t):
            return -0.5 * (
                math.log(2 * math.pi * R_NOISE)
                + (yv[0] - state[0]) ** 2 / R_NOISE
            )

        y_mx = mx.array(y.astype(np.float32))[:, None]
        u_mx = mx.array(u.astype(np.float32))
        r_keys = 10
        vals = np.array([
            smcx.bootstrap_filter(
                mx.random.key(s),
                init,
                trans_u,
                logobs_u,
                y_mx,
                10_000,
                inputs=u_mx,
            ).marginal_loglik.item()
            for s in range(r_keys)
        ])
        sd = vals.std(ddof=1)
        err = vals.mean() - logz_true
        upper = 3 * sd / math.sqrt(r_keys)
        assert -(upper + 0.5 * sd**2) <= err <= upper

    def test_arity_mismatch_raises_named_error(self):
        init, trans, logobs = _make_model()  # two-arg forms
        with pytest.raises(TypeError, match="input_t"):
            smcx.bootstrap_filter(
                mx.random.key(0),
                init,
                trans,
                logobs,
                Y_MX,
                100,
                inputs=mx.zeros((T,)),
            )


class TestSimulate:
    """Dual-arity initial sampler + inputs reuse (ADR-0008 item 3)."""

    def test_cloud_form_initial_sampler(self):
        init, trans, _ = _make_model()

        def emit(key, state):
            return state + math.sqrt(R_NOISE) * mx.random.normal(
                state.shape, key=key
            )

        states, ems = smcx.simulate(mx.random.key(0), init, trans, emit, 25)
        assert states.shape == (25, 1) and ems.shape == (25, 1)

    def test_single_draw_form_initial_sampler(self):
        _, trans, _ = _make_model()

        def init_single(key):
            return M0 + math.sqrt(P0) * mx.random.normal((1,), key=key)

        def emit(key, state):
            return state

        states, _ = smcx.simulate(
            mx.random.key(1), init_single, trans, emit, 10
        )
        assert states.shape == (10, 1)

    def test_simulated_data_is_filterable(self):
        init, trans, logobs = _make_model()

        def emit(key, state):
            return state + math.sqrt(R_NOISE) * mx.random.normal(
                state.shape, key=key
            )

        _, ems = smcx.simulate(mx.random.key(2), init, trans, emit, 30)
        post = smcx.bootstrap_filter(
            mx.random.key(3), init, trans, logobs, ems, 1000
        )
        assert math.isfinite(post.marginal_loglik.item())
