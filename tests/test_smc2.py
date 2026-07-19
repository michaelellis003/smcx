# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""SMC² tests against a converged numerical oracle and outside evidence.

For an LGSSM with unknown AR coefficient ``a``, a 20,001-point trapezoidal
integration of the exact Kalman likelihood supplies the reference posterior.
The retained refinement test bounds the numerical grid error.

One-time isolated validation (2026-07-18; N_theta=128, N_x=256, eight fixed
seeds) gave smcx log evidence/mean/variance ``-55.425010 (.056962)``,
``.873361 (.002665)``, and ``.00721287 (.000250)``; particles 0.4 gave
``-55.426805 (.064977)``, ``.873256 (.003133)``, and ``.00692804
(.000318)``. The grid targets are ``-55.458652497463525``,
``.870239461175306``, and ``.007183951188524291``; all passed five-SE gates.

TFP 0.25.0's experimental ``smc_squared`` was also investigated. The matched
call's trace was consistent with omitting the terminal observation, and its
unmodified numerical output disagreed with the grid target. Its rejuvenation
branch also explicitly resets outer log-weights to zero (uniform), rather
than preserving its incoming weights. It was therefore rejected as a full
SMC² authority. A disclosed
diagnostic run with rejuvenation disabled and one unused terminal sentinel
(N_theta=512, N_x=256, eight seeds) recovered log evidence ``-55.44813
(.03615)``, posterior mean ``.869165 (.00134)``, and variance ``.00700898
(.00025917)``, validating only its nested-weighting target.

Pinned sources and licenses (no outside code copied or imported here):

* particles 0.4, f71e94a21a11c73b58e2d694775b1b1d379b8854, MIT:
  https://github.com/nchopin/particles/blob/f71e94a21a11c73b58e2d694775b1b1d379b8854/particles/smc_samplers.py#L1052-L1181
  https://github.com/nchopin/particles/blob/f71e94a21a11c73b58e2d694775b1b1d379b8854/LICENSE
* TFP 0.25.0, 9709569d9c1159dc54154044f679edc4a15bd26b, Apache-2.0:
  https://github.com/tensorflow/probability/blob/9709569d9c1159dc54154044f679edc4a15bd26b/tensorflow_probability/python/experimental/mcmc/particle_filter.py#L766-L967
  https://github.com/tensorflow/probability/blob/9709569d9c1159dc54154044f679edc4a15bd26b/LICENSE

Algorithm: Chopin, Jacob, and Papaspiliopoulos (2013),
https://doi.org/10.1111/j.1467-9868.2012.01046.x
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


def _grid_reference(num_points):
    """Integrate the exact Kalman likelihood by stabilized trapezoids."""
    y = Y.astype(np.float64)
    grid = np.linspace(0.5, 1.3, num_points)
    ll = np.array([kalman_1d(y, a, Q, R, 0.0, P0)[0] for a in grid])
    shifted_density = np.exp(ll - ll.max()) / 0.8
    shifted_z = np.trapezoid(shifted_density, grid)
    density = shifted_density / shifted_z
    mean = float(np.trapezoid(density * grid, grid))
    second = float(np.trapezoid(density * grid**2, grid))
    logz = float(ll.max() + math.log(shifted_z))
    return mean, second - mean**2, logz


GRID_POINTS = 20_001
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


class TestNumericalReference:
    """The retained high-resolution grid constants are converged."""

    def test_coarse_grid_reproduces_promoted_constants(self):
        mean, variance, logz = _grid_reference(2_001)
        # Difference between 2,001 and the promoted 20,001-point trapezoidal
        # grids; 2e-9 therefore bounds the observed quadrature refinement.
        assert mean == pytest.approx(GRID_MEAN, abs=2e-9)
        assert variance == pytest.approx(GRID_VARIANCE, abs=2e-9)
        assert logz == pytest.approx(GRID_LOGZ, abs=2e-9)


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
