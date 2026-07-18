# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Genealogy diagnostics (ADR-0021).

Covers trajectory reconstruction and single-run log-ML variance.

The variance estimator follows Chan & Lai (2013) / Lee & Whiteley
(2018) as implemented by Chopin's `particles` (``Var_logLt``): with
Eve variables tracing each particle to its time-0 ancestor, the
estimate at time t is the sum over Eve classes of the squared
normalized-weight mass. The reference formula is small enough to
restate in NumPy inside these tests, which is how the semantics are
pinned without adding `particles` (and numba) as a dependency.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import smcx
from smcx import log_ml_variance, reconstruct_trajectories
from smcx.containers import ParticleFilterPosterior


def _posterior(particles, log_weights, ancestors):
    t, _n = log_weights.shape
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(0.0),
        filtered_particles=jnp.asarray(particles, dtype=float),
        filtered_log_weights=jnp.asarray(log_weights, dtype=float),
        ancestors=jnp.asarray(ancestors, dtype=jnp.int32),
        ess=jnp.ones(t),
        log_evidence_increments=jnp.zeros(t),
    )


def _lgssm_fns():
    def init(key, n):
        return jr.normal(key, (n, 1))

    def trans(key, z):
        return 0.9 * z + 0.5 * jr.normal(key, z.shape)

    def log_obs(y, z):
        return -0.5 * (y[0] - z[0]) ** 2

    return init, trans, log_obs


class TestReconstructTrajectories:
    """Trajectory reconstruction against hand-traced genealogies."""

    def test_hand_computed_ancestry(self):
        """Trace a 3-step, 3-particle genealogy by hand.

        With ancestors A[1] = [2, 0, 0] and A[2] = [1, 1, 2], the
        lineage of final particle n passes through index sel[t]:
        sel[2] = [0, 1, 2], sel[1] = A[2][sel[2]] = [1, 1, 2],
        sel[0] = A[1][sel[1]] = [0, 0, 0].
        """
        particles = np.arange(9, dtype=float).reshape(3, 3, 1)
        ancestors = np.array([[0, 1, 2], [2, 0, 0], [1, 1, 2]])
        post = _posterior(particles, np.zeros((3, 3)), ancestors)

        traj = reconstruct_trajectories(post)

        assert traj.shape == (3, 3, 1)
        expected_sel = np.array([[0, 0, 0], [1, 1, 2], [0, 1, 2]])
        expected = np.stack([particles[t, expected_sel[t]] for t in range(3)])
        np.testing.assert_array_equal(np.asarray(traj), expected)

    def test_identity_without_resampling(self, key):
        """Never resampling leaves trajectories equal to the particles.

        With threshold 0 the ancestors stay identity throughout.
        """
        init, trans, log_obs = _lgssm_fns()
        emissions = jr.normal(key, (8, 1))
        post = smcx.bootstrap_filter(
            key,
            init,
            trans,
            log_obs,
            emissions,
            num_particles=50,
            resampling_threshold=0.0,
        )
        traj = reconstruct_trajectories(post)
        np.testing.assert_array_equal(
            np.asarray(traj), np.asarray(post.filtered_particles)
        )

    def test_jit_compatible(self):
        particles = np.zeros((4, 5, 2))
        ancestors = np.zeros((4, 5), dtype=int)
        post = _posterior(particles, np.zeros((4, 5)), ancestors)
        traj = jax.jit(reconstruct_trajectories)(post)
        assert traj.shape == (4, 5, 2)


def _var_reference(log_weights, ancestors):
    """NumPy restatement of particles' Var_logLt (Chan & Lai form)."""
    lw = np.asarray(log_weights, dtype=np.float64)
    anc = np.asarray(ancestors)
    t_len, n = lw.shape
    eve = np.arange(n)
    out = np.zeros(t_len)
    for t in range(t_len):
        if t > 0:
            eve = eve[anc[t]]
        w = np.exp(lw[t] - lw[t].max())
        w = w / w.sum()
        s = np.zeros(n)
        np.add.at(s, eve, w)
        out[t] = np.sum(s**2)
    return out


class TestLogMlVariance:
    """Single-run variance estimator against reference and replicates."""

    def test_matches_reference_formula(self, key):
        """Agree with the restated reference formula on filter output."""
        init, trans, log_obs = _lgssm_fns()
        emissions = jr.normal(key, (20, 1))
        post = smcx.bootstrap_filter(
            key, init, trans, log_obs, emissions, num_particles=200
        )
        ours = np.asarray(log_ml_variance(post))
        ref = _var_reference(post.filtered_log_weights, post.ancestors)
        finite = np.isfinite(ours)
        assert finite.all(), "no coalescence expected at T=20, N=200"
        # float32 (Metal) carries ~7 significant digits.
        rtol = 1e-6 if ours.dtype == np.float64 else 1e-5
        np.testing.assert_allclose(ours, ref, rtol=rtol)

    def test_coalesced_genealogy_returns_inf(self):
        """One Eve class carries no variance information (ADR-0021)."""
        t_len, n = 4, 6
        ancestors = np.zeros((t_len, n), dtype=int)
        ancestors[0] = np.arange(n)  # identity at t=0 by construction
        post = _posterior(
            np.zeros((t_len, n, 1)), np.zeros((t_len, n)), ancestors
        )
        est = np.asarray(log_ml_variance(post))
        assert np.isfinite(est[0])
        assert np.isinf(est[1:]).all()

    def test_lag_bounds_and_exactness(self, key):
        """Long lags reproduce the exact estimator; short ones stay sane.

        A lag of at least T is exactly the time-0 Eve estimator;
        short lags must stay finite and nonnegative.
        """
        init, trans, log_obs = _lgssm_fns()
        emissions = jr.normal(key, (15, 1))
        post = smcx.bootstrap_filter(
            key, init, trans, log_obs, emissions, num_particles=100
        )
        exact = np.asarray(log_ml_variance(post))
        lagged_full = np.asarray(log_ml_variance(post, lag=15))
        rtol = 1e-6 if exact.dtype == np.float64 else 1e-5
        np.testing.assert_allclose(lagged_full, exact, rtol=rtol)

        lag2 = np.asarray(log_ml_variance(post, lag=2))
        assert np.isfinite(lag2).all()
        assert (lag2 >= 0).all()

    def test_calibrates_against_replicates(self, lgssm_params, lgssm_data):
        """The single-run estimate agrees with replicated variance.

        Averaged over independent runs, the final-time single-run
        estimate must land within a factor of the empirical variance
        of the log-ML across those runs. The factor-of-three gate is
        loose because both sides are noisy at R=40; the point is
        catching order-of-magnitude wrongness, not decimals.
        """
        _, emissions = lgssm_data
        init, trans, log_obs = _lgssm_fns()

        def run(k):
            return smcx.bootstrap_filter(
                k, init, trans, log_obs, emissions, num_particles=300
            )

        keys = jr.split(jr.PRNGKey(11), 40)
        posts = [run(k) for k in keys]
        singles = np.array([float(log_ml_variance(p)[-1]) for p in posts])
        logmls = np.array([float(p.marginal_loglik) for p in posts])

        empirical = logmls.var(ddof=1)
        mean_single = singles[np.isfinite(singles)].mean()
        ratio = mean_single / empirical
        assert 1 / 3 < ratio < 3, (
            f"single-run {mean_single:.4g} vs empirical "
            f"{empirical:.4g} (ratio {ratio:.2f})"
        )
