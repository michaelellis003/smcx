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

    def test_store_history_false_matches_evidence(self):
        # store_history only changes what is retained, not the
        # computation: the marginal likelihood is bit-identical and
        # the parameter cloud collapses to the final step (ADR-0011).
        full = _run(0, n_theta=64, n_x=128, store_history=True)
        final = _run(0, n_theta=64, n_x=128, store_history=False)
        assert full.marginal_loglik.item() == final.marginal_loglik.item()
        assert final.filtered_params.shape == (1, 64, 1)
        assert np.array_equal(
            np.array(final.filtered_params[0]),
            np.array(full.filtered_params[-1]),
        )

    def test_degenerate_raises(self):
        def impossible(y, state, theta):
            return mx.array(-mx.inf)

        with pytest.raises(smcx.DegenerateWeightsError):
            smcx.smc2(
                mx.random.key(0),
                PARAM_INIT,
                LOG_PRIOR,
                INNER_INIT,
                INNER_TRANS,
                impossible,
                Y_MX,
                32,
                32,
            )


class TestReduction:
    """At a point-mass prior, SMC² reduces to a bootstrap filter."""

    def test_logz_matches_bootstrap_at_point_mass(self):
        # N_theta=1 with theta fixed at A_TRUE: the single inner
        # filter IS a bootstrap filter, so SMC²'s log-evidence must
        # match a standalone bootstrap_filter's. RNG consumption
        # differs, so this is a tier-2 statistical gate (design §9b).
        sq, sp = math.sqrt(Q), math.sqrt(P0)

        def point_mass(key, n_theta):
            return mx.full((n_theta, 1), A_TRUE)

        def flat_prior(theta):
            return mx.array(0.0)

        def b_init(key, n):
            return sp * mx.random.normal((n, 1), key=key)

        def b_trans(key, s):
            return A_TRUE * s + sq * mx.random.normal(s.shape, key=key)

        def b_logobs(y, s):
            return -0.5 * (math.log(2 * math.pi * R) + (y[0] - s[0]) ** 2 / R)

        r_keys = 12
        smc2_lz = np.array([
            smcx.smc2(
                mx.random.key(s),
                point_mass,
                flat_prior,
                INNER_INIT,
                INNER_TRANS,
                INNER_LOGOBS,
                Y_MX,
                1,
                2000,
            ).marginal_loglik.item()
            for s in range(r_keys)
        ])
        boot_lz = np.array([
            smcx.bootstrap_filter(
                mx.random.key(s), b_init, b_trans, b_logobs, Y_MX, 2000
            ).marginal_loglik.item()
            for s in range(r_keys)
        ])
        diff = smc2_lz.mean() - boot_lz.mean()
        bound = 3 * math.sqrt(
            smc2_lz.std(ddof=1) ** 2 / r_keys
            + boot_lz.std(ddof=1) ** 2 / r_keys
        )
        assert abs(diff) <= bound, (diff, bound)


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

    def test_rejuvenation_fires_at_t0(self):
        # ess_threshold must be honored at the initial observation:
        # a T=1 run is the only case where t=0 is the sole chance to
        # rejuvenate, and it would be skipped by a range(1, n_time)
        # gate. With threshold=1.0 the non-uniform t=0 weights trigger
        # the move, which resamples and moves the parameter cloud.
        y1 = Y_MX[:1]

        def one(seed, ess_threshold):
            return smcx.smc2(
                mx.random.key(seed),
                PARAM_INIT,
                LOG_PRIOR,
                INNER_INIT,
                INNER_TRANS,
                INNER_LOGOBS,
                y1,
                128,
                64,
                ess_threshold=ess_threshold,
                num_pmmh_steps=2,
            )

        rej = one(0, 1.0)
        fwd = one(0, 0.0)
        assert rej.filtered_params.shape == (1, 128, 1)
        assert rej.acceptance_rates.shape == (1,)
        # A move ran at t=0, so the cloud differs from the untouched
        # forward-pass draw.
        assert not np.array_equal(
            np.array(rej.filtered_params), np.array(fwd.filtered_params)
        )


class TestBatchedStep:
    """The batched inner step vs N_theta independent single steps.

    Spec test 6 (ADR-0013): one batched inner step advancing N_theta
    filters must behave as N_theta *independent* single-filter steps.
    Independence is established by perturbation (exact); the
    correctness of a batched filter against the single-filter
    distribution is a separate tier-2 check.
    """

    def test_batched_step_filters_are_independent(self):
        # The defining property: perturbing one filter's input leaves
        # every OTHER filter's output bit-identical (same keys). A
        # coupled implementation would leak the change across the
        # theta axis. Exact, not statistical.
        from smcx.smc2 import _batched_inner_step

        n_theta, n_x = 5, 64
        log_n_x = math.log(n_x)
        clouds = mx.random.normal((n_theta, n_x, 1), key=mx.random.key(3))
        thetas = mx.array([[0.6], [0.7], [0.8], [0.9], [1.0]])
        ilw = mx.full((n_theta, n_x), -log_n_x)
        kr, kt = mx.random.split(mx.random.key(4))
        tail = (INNER_TRANS, INNER_LOGOBS, n_theta, n_x, log_n_x)

        inner0, _, ell0 = _batched_inner_step(
            kr, kt, clouds, ilw, thetas, Y_MX[1], *tail
        )
        # Perturb only filter j's inner cloud.
        j = 2
        pert = np.array(clouds)
        pert[j] += 3.0
        inner1, _, ell1 = _batched_inner_step(
            kr, kt, mx.array(pert), ilw, thetas, Y_MX[1], *tail
        )

        ell0, ell1 = np.array(ell0), np.array(ell1)
        in0, in1 = np.array(inner0), np.array(inner1)
        for m in range(n_theta):
            if m == j:
                assert ell0[m] != ell1[m]
                assert not np.array_equal(in0[m], in1[m])
            else:  # every other filter untouched, bit-for-bit
                assert ell0[m] == ell1[m], m
                assert np.array_equal(in0[m], in1[m]), m

    def test_batched_filter_increment_matches_single_filter(self):
        # Correctness (not independence): a filter advanced by the
        # batched step has the same increment distribution as an
        # independent single-filter step of the same problem. RNG
        # threading differs, so this is tier-2 (design §9b).
        from smcx.smc2 import _batched_inner_step

        n_theta, n_x = 64, 256
        log_n_x = math.log(n_x)
        cloud = mx.random.normal((n_x, 1), key=mx.random.key(10))
        theta1 = mx.array([[A_TRUE]])
        ulw = mx.full((n_x,), -log_n_x)
        model = (INNER_TRANS, INNER_LOGOBS)

        kr, kt = mx.random.split(mx.random.key(20))
        _, _, ell_b = _batched_inner_step(
            kr,
            kt,
            mx.broadcast_to(cloud, (n_theta, n_x, 1)),
            mx.broadcast_to(ulw, (n_theta, n_x)),
            mx.broadcast_to(theta1, (n_theta, 1)),
            Y_MX[1],
            *model,
            n_theta,
            n_x,
            log_n_x,
        )
        ell_b = np.array(ell_b)

        r = 256
        ell_s = np.empty(r)
        for i in range(r):
            kr1, kt1 = mx.random.split(mx.random.key(1000 + i))
            _, _, ell = _batched_inner_step(
                kr1,
                kt1,
                cloud[None],
                ulw[None],
                theta1,
                Y_MX[1],
                *model,
                1,
                n_x,
                log_n_x,
            )
            ell_s[i] = float(np.array(ell)[0])

        se = math.sqrt(ell_b.var(ddof=1) / n_theta + ell_s.var(ddof=1) / r)
        assert abs(ell_b.mean() - ell_s.mean()) < 4 * se, (
            ell_b.mean(),
            ell_s.mean(),
            se,
        )

    def test_lse_rows_matches_per_row_loop(self):
        from smcx.smc2 import _lse_rows

        x = mx.random.normal((7, 13), key=mx.random.key(0))
        batched = np.array(_lse_rows(x))
        looped = np.array([_np_lse(np.array(x[i])) for i in range(7)])
        assert np.allclose(batched, looped, atol=1e-5)

    def test_batched_resample_routes_each_row_independently(self):
        from smcx.smc2 import _batched_inner_resample

        # One-hot weight per row: systematic resampling must pick that
        # row's peak for every draw, and each row is independent.
        n_x = 16
        peaks = [2, 7, 15]
        w = np.zeros((3, n_x), dtype=np.float32)
        for i, j in enumerate(peaks):
            w[i, j] = 1.0
        idx = np.array(
            _batched_inner_resample(mx.random.key(0), mx.array(w), n_x)
        )
        for i, j in enumerate(peaks):
            assert np.all(idx[i] == j), (i, j, idx[i])


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
