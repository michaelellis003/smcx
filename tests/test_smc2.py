# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""SMC² tests (spec: feat-14-smc2; ADR-0014).

SMC² nests an inner particle filter (N_x particles) inside an outer
SMC sampler over N_theta parameter particles. Tests target the
contract: container/evidence invariants, reduction to a single
bootstrap filter at a point-mass prior, rejuvenation behavior, and
exact recovery on a conjugate LGSSM. Correctness gates are
MC-error-honest (AGENTS.md), not bit-exact.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import smcx
from tests._kalman import kalman_1d

# LGSSM with unknown AR coefficient a: z_t = a z_{t-1} + q eps,
# y_t = z_t + r eta. a is the learned parameter; q, r, P0 known.
A_TRUE, Q, R, P0 = 0.9, 0.5, 0.3, 1.0
T = 40


def _np_lse(a):
    a = np.asarray(a, dtype=np.float64)
    m = a.max()
    return float(m + math.log(np.exp(a - m).sum()))


def _model():
    sq, sp = math.sqrt(Q), math.sqrt(P0)

    def param_init(key, n_theta):
        # U(0.5, 1.3) prior over the AR coefficient.
        return 0.5 + 0.8 * mx.random.uniform(shape=(n_theta, 1), key=key)

    def log_prior(theta):
        a = theta[0]
        inside = (a >= 0.5) & (a <= 1.3)
        return mx.where(inside, math.log(1.0 / 0.8), -mx.inf)

    def inner_init(key, n_x, theta):
        return sp * mx.random.normal((n_x, 1), key=key)

    def inner_trans(key, state, theta):
        return theta[0] * state + sq * mx.random.normal(state.shape, key=key)

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
Y_MX = mx.array(Y.astype(np.float32))[:, None]
PARAM_INIT, LOG_PRIOR, INNER_INIT, INNER_TRANS, INNER_LOGOBS = _model()


def _exact_reference():
    """Exact posterior mean of a and the marginal likelihood.

    From the Kalman log-likelihood on a fine a-grid integrated
    against the U(0.5, 1.3) prior — SMC²'s target is this integral.
    """
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
        mx.random.key(seed),
        PARAM_INIT,
        LOG_PRIOR,
        INNER_INIT,
        INNER_TRANS,
        INNER_LOGOBS,
        Y_MX,
        n_theta,
        n_x,
        ess_threshold=ess_threshold,
        **kw,
    )


class TestStructure:
    """Container shapes, evidence invariant, determinism."""

    def test_container_shapes(self):
        post = _run(0, n_theta=64, n_x=128)
        assert isinstance(post, smcx.SMC2Posterior)
        assert post.filtered_params.shape == (T, 64, 1)
        assert post.filtered_log_weights.shape == (T, 64)
        assert post.ess.shape == (T,)
        assert post.log_evidence_increments.shape == (T,)
        assert post.acceptance_rates.shape == (T,)

    def test_evidence_increments_sum_to_marginal(self):
        post = _run(0, n_theta=64, n_x=128)
        total = np.array(post.log_evidence_increments, dtype=np.float64).sum()
        assert post.marginal_loglik.item() == pytest.approx(total, abs=5e-4)

    def test_outer_ess_in_range(self):
        post = _run(0, n_theta=64, n_x=128)
        e = np.array(post.ess)
        assert np.all(e >= 1 - 1e-4) and np.all(e <= 64 * (1 + 1e-4))

    def test_deterministic_per_key(self):
        a = _run(0, n_theta=64, n_x=128)
        b = _run(0, n_theta=64, n_x=128)
        assert a.marginal_loglik.item() == b.marginal_loglik.item()
        assert np.array_equal(
            np.array(a.filtered_params), np.array(b.filtered_params)
        )


class TestRejuvenation:
    """The PMMH move: the documented rejuvenation API must act."""

    def test_rejuvenation_keeps_outer_ess_healthy(self):
        # A forward-only run's outer ESS collapses as the parameter
        # cloud degenerates; rejuvenation resamples + moves theta and
        # resets the outer weights, so the final ESS is far healthier.
        rej = _run(0, n_theta=128, n_x=64, ess_threshold=0.5)
        fwd = _run(0, n_theta=128, n_x=64, ess_threshold=0.0)
        assert rej.ess[-1].item() > fwd.ess[-1].item()

    def test_pmmh_moves_fire_and_accept(self):
        rej = _run(0, n_theta=128, n_x=64, ess_threshold=0.5)
        rates = np.array(rej.acceptance_rates)
        # At least one move accepted somewhere (informative data + a
        # diffuse prior make some proposals near good theta accept).
        assert rates.sum() > 0
        # Acceptance rates are valid probabilities.
        assert np.all(rates >= 0) and np.all(rates <= 1)

    def test_rejuvenation_deterministic_per_key(self):
        a = _run(0, n_theta=96, n_x=64, ess_threshold=0.5)
        b = _run(0, n_theta=96, n_x=64, ess_threshold=0.5)
        assert a.marginal_loglik.item() == b.marginal_loglik.item()
        assert np.array_equal(
            np.array(a.filtered_params), np.array(b.filtered_params)
        )

    def test_evidence_increments_sum_under_rejuvenation(self):
        # The marginal likelihood is a running product across steps;
        # rejuvenation resets the outer weights but must not reset the
        # evidence accumulation.
        post = _run(0, n_theta=128, n_x=64, ess_threshold=0.5)
        total = np.array(post.log_evidence_increments, dtype=np.float64).sum()
        assert post.marginal_loglik.item() == pytest.approx(total, abs=5e-4)


_RECOVERY_KEYS = 10


@pytest.fixture(scope="module")
def recovery_runs():
    """R independent SMC² runs (shared across the recovery gates)."""
    means, logzs = [], []
    for s in range(_RECOVERY_KEYS):
        post = _run(
            s, n_theta=512, n_x=128, ess_threshold=0.5, num_pmmh_steps=3
        )
        means.append(float(np.array(smcx.param_weighted_mean(post))[-1, 0]))
        logzs.append(post.marginal_loglik.item())
    return np.array(means), np.array(logzs)


class TestExactRecovery:
    """SMC² against an exact Kalman-grid reference (spec test 1, 2).

    The reference is the Kalman log-likelihood integrated against the
    prior on a fine a-grid — SMC²'s parameter posterior and marginal
    likelihood estimate this exactly. Gates are MC-error-honest over
    R independent keys.
    """

    def test_param_posterior_mean_matches_exact(self, recovery_runs):
        # Tier-2 moment gate: |bias| < 5 * SE of the R-key mean.
        means, _ = recovery_runs
        se = means.std(ddof=1) / math.sqrt(_RECOVERY_KEYS)
        assert abs(means.mean() - EXACT_MEAN) < 5 * se, (
            means.mean(),
            EXACT_MEAN,
        )

    def test_marginal_likelihood_matches_exact(self, recovery_runs):
        # log Zhat is downward-biased (Jensen); one-sided budget
        # 0.5 * sd^2 on the low side, 3 * SE band (k=3, as tempering
        # and liu_west).
        _, logzs = recovery_runs
        sd = logzs.std(ddof=1)
        se = sd / math.sqrt(_RECOVERY_KEYS)
        err = logzs.mean() - EXACT_LOGZ
        assert -(3 * se + 0.5 * sd**2) <= err <= 3 * se, (err, se, sd)

    def test_evidence_estimator_unbiased(self, recovery_runs):
        # E[exp(log Zhat)] = Z (the pseudo-marginal contract): the
        # log of the average of Zhat_s recovers the exact log Z.
        _, logzs = recovery_runs
        logz_hat = _np_lse(logzs) - math.log(_RECOVERY_KEYS)
        assert abs(logz_hat - EXACT_LOGZ) < 0.1, (logz_hat, EXACT_LOGZ)
