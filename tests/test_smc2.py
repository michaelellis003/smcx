# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""SMC² tests against a numerical LGSSM oracle.

Algorithm: Chopin, Jacob, and Papaspiliopoulos (2013),
https://doi.org/10.1111/j.1467-9868.2012.01046.x.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

A_TRUE, Q, R, P0 = 0.9, 0.5, 0.3, 1.0
T = 40


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


GRID_MEAN = 0.870239461175306
GRID_VARIANCE = 0.007183951188524291
GRID_LOGZ = -55.458652497463525


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


def _small_model():
    emissions = jnp.array([[0.25], [-0.4], [0.1]], dtype=jnp.float64)

    def param_init(key, n_theta):
        return 0.7 + 0.2 * jr.uniform(key, (n_theta, 1), dtype=jnp.float64)

    def log_prior(theta):
        return -0.5 * jnp.sum(theta**2)

    def inner_init(key, n_x, theta):
        return theta[0] + 0.3 * jr.normal(key, (n_x, 1), dtype=jnp.float64)

    def inner_trans(key, state, theta):
        return theta[0] * state + 0.2 * jr.normal(
            key, state.shape, dtype=jnp.float64
        )

    def inner_logobs(y, state, theta):
        del theta
        return -0.5 * (
            jnp.log(2.0 * jnp.pi * 0.4) + (y[0] - state[0]) ** 2 / 0.4
        )

    return (
        param_init,
        log_prior,
        inner_init,
        inner_trans,
        inner_logobs,
        emissions,
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


class TestCallbackFreshness:
    """Public calls observe current callback-object behavior."""

    def test_mutated_observation_callback_matches_fresh_equivalent(self):
        (
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            _,
            emissions,
        ) = _small_model()

        class MutableObservation:
            def __init__(self, variance):
                self.variance = variance

            def __call__(self, emission, state, theta):
                del theta
                return -0.5 * (
                    jnp.log(2.0 * jnp.pi * self.variance)
                    + (emission[0] - state[0]) ** 2 / self.variance
                )

        observation = MutableObservation(0.2)
        smcx.smc2(
            jr.key(40),
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            observation,
            emissions,
            3,
            4,
            ess_threshold=0.0,
        )
        observation.variance = 0.9
        actual = smcx.smc2(
            jr.key(41),
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            observation,
            emissions,
            3,
            4,
            ess_threshold=0.0,
        )
        expected = smcx.smc2(
            jr.key(41),
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            MutableObservation(0.9),
            emissions,
            3,
            4,
            ess_threshold=0.0,
        )

        for expected_value, actual_value in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual_value),
                np.asarray(expected_value),
            )


class TestFixedKeyRegression:
    """Invocation-local inner JIT kernels preserve exact behavior."""

    @pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="frozen CPU/x64 arithmetic contract",
    )
    def test_preserves_frozen_fixed_key_output(self):
        (
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            inner_logobs,
            emissions,
        ) = _small_model()
        posterior = smcx.smc2(
            jr.key(314159),
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            inner_logobs,
            emissions,
            3,
            4,
            ess_threshold=0.0,
        )

        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik),
            np.asarray(-3.421747990559213),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_params),
            np.array([
                [
                    [0.8690271497469142],
                    [0.892344905318535],
                    [0.7275650823743653],
                ],
                [
                    [0.8690271497469142],
                    [0.892344905318535],
                    [0.7275650823743653],
                ],
                [
                    [0.8690271497469142],
                    [0.892344905318535],
                    [0.7275650823743653],
                ],
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights),
            np.array([
                [-1.1037195976548424, -0.9457871459341566, -1.2729977505620504],
                [-1.1812879768008746, -0.9261661263302763, -1.2138632483444947],
                [-1.4378978303439447, -0.7854789880395324, -1.1819752775167212],
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess),
            np.array([
                2.948017030946551,
                2.9473711233956394,
                2.7912284636963074,
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments),
            np.array([
                -0.9132325220568566,
                -1.8150181135800527,
                -0.6934973549223041,
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.acceptance_rates),
            np.zeros(3),
        )

    @pytest.mark.skipif(
        jax.default_backend() != "cpu",
        reason="frozen CPU/x64 arithmetic contract",
    )
    def test_preserves_frozen_rejuvenation_output(self):
        (
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            inner_logobs,
            emissions,
        ) = _small_model()
        posterior = smcx.smc2(
            jr.key(271828),
            param_init,
            log_prior,
            inner_init,
            inner_trans,
            inner_logobs,
            emissions,
            3,
            4,
            ess_threshold=1.1,
            num_pmmh_steps=2,
        )

        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik),
            np.asarray(-3.0331045486577697),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_params),
            np.array([
                [
                    [0.7586950197518261],
                    [0.8187408484682641],
                    [0.7589583281677461],
                ],
                [
                    [0.8105932981742988],
                    [0.6976200744967381],
                    [0.7609658911048357],
                ],
                [
                    [0.8249431380290069],
                    [0.4310136770781425],
                    [0.7311638511464534],
                ],
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights),
            np.full((3, 3), -1.0986122886681098),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess),
            np.array([3.0, 3.0, 3.0]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments),
            np.array([
                -0.8723455471462007,
                -1.4975149224137319,
                -0.6632440790978372,
            ]),
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.acceptance_rates),
            np.array([
                0.8333333432674408,
                1.0,
                0.6666666865348816,
            ]),
        )
        # A threshold above one forces rejuvenation at every time, while
        # positive acceptance records prove that each PMMH loop did work.
        assert np.all(np.asarray(posterior.acceptance_rates) > 0.0)


class TestPosteriorRecovery:
    """The parameter posterior matches the exact grid reference."""

    def test_posterior_mean_and_logz_gate(self):
        r_keys = 8
        means, variances, evidence_ratios = [], [], []
        for s in range(r_keys):
            post = _run(s, n_theta=128, n_x=256, ess_threshold=0.5)
            w = np.exp(np.array(post.filtered_log_weights[-1], np.float64))
            w /= w.sum()
            th = np.array(post.filtered_params[-1, :, 0], np.float64)
            mean = float(w @ th)
            means.append(mean)
            variances.append(float(w @ ((th - mean) ** 2)))
            evidence_ratios.append(
                math.exp(float(post.marginal_loglik) - GRID_LOGZ)
            )
        values = np.column_stack((means, variances, evidence_ratios))
        expected = np.array([GRID_MEAN, GRID_VARIANCE, 1.0])
        # R=8 independent complete SMC² runs, hence SE(mean) = sd/sqrt(R).
        estimator_se = values.std(axis=0, ddof=1) / math.sqrt(r_keys)
        np.testing.assert_array_less(
            np.abs(values.mean(axis=0) - expected),
            5 * estimator_se + 2e-5,
        )


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
        total = float(jnp.sum(post.log_evidence_increments))
        marginal = float(post.marginal_loglik)
        if post.log_evidence_increments.dtype == jnp.float64:
            assert total == pytest.approx(marginal, rel=1e-8)
        else:
            # The Metal path sums T float32 increments separately from the
            # scan carry; their reduction orders can differ by several ulps.
            assert total == pytest.approx(marginal, rel=1e-5)


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
