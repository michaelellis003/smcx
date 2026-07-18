# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for smcx.diagnostics.

Cross-validates against Dynamax Kalman filter and verifies
mathematical properties of diagnostic functions.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from smcx.bootstrap import bootstrap_filter
from smcx.diagnostics import (
    cumulative_log_score,
    diagnose,
    log_bayes_factor,
    log_ml_increments,
    pareto_k_diagnostic,
    particle_diversity,
    posterior_predictive_sample,
    replicated_log_ml,
    tail_ess,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)
from tests.conftest import _mvn_logpdf, _mvn_sample


def _make_smcx_fns(lgssm_params):
    """Build (initial_sampler, transition_sampler, log_obs_fn)."""
    m0 = lgssm_params["initial_mean"]
    P0 = lgssm_params["initial_cov"]
    F = lgssm_params["dynamics_weights"]
    Q = lgssm_params["dynamics_cov"]
    H = lgssm_params["emissions_weights"]
    R = lgssm_params["emissions_cov"]

    def initial_sampler(key, n):
        return _mvn_sample(key, m0, P0, shape=(n,))

    def transition_sampler(key, state):
        mean = (F @ state[:, None]).squeeze(-1)
        return _mvn_sample(key, mean, Q)

    def log_observation_fn(emission, state):
        mean = (H @ state[:, None]).squeeze(-1)
        return _mvn_logpdf(emission, mean, R)

    return initial_sampler, transition_sampler, log_observation_fn


def _run_bootstrap(lgssm_params, lgssm_data, n=10_000, seed=0):
    """Run bootstrap filter and return posterior."""
    _, emissions = lgssm_data
    init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)
    return bootstrap_filter(
        key=jr.PRNGKey(seed),
        initial_sampler=init_fn,
        transition_sampler=trans_fn,
        log_observation_fn=obs_fn,
        emissions=emissions,
        num_particles=n,
    )


class TestWeightedMean:
    """Tests for weighted_mean."""

    def test_weighted_mean_matches_kalman(self, lgssm_params, lgssm_data):
        """PF weighted means should track Kalman filtered means."""
        from dynamax.linear_gaussian_ssm.inference import (
            lgssm_filter,
            make_lgssm_params,
        )

        _, emissions = lgssm_data
        params = make_lgssm_params(**lgssm_params)
        kalman_post = lgssm_filter(params, emissions)
        kalman_means = kalman_post.filtered_means

        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        pf_means = weighted_mean(pf_post)

        assert jnp.allclose(pf_means, kalman_means, atol=0.15), (
            f"Max error: {float(jnp.max(jnp.abs(pf_means - kalman_means))):.4f}"
        )


class TestWeightedVariance:
    """Tests for weighted_variance."""

    def test_weighted_variance_uniform_weights(self, lgssm_params, lgssm_data):
        """With uniform weights, matches unweighted variance."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)

        # Create uniform-weight posterior for comparison
        n = pf_post.filtered_particles.shape[1]
        uniform_log_w = jnp.full_like(pf_post.filtered_log_weights, -jnp.log(n))
        from smcx.containers import ParticleFilterPosterior

        uniform_post = ParticleFilterPosterior(
            marginal_loglik=pf_post.marginal_loglik,
            filtered_particles=pf_post.filtered_particles,
            filtered_log_weights=uniform_log_w,
            ancestors=pf_post.ancestors,
            ess=pf_post.ess,
            log_evidence_increments=pf_post.log_evidence_increments,
        )

        wvar = weighted_variance(uniform_post)
        # Unweighted variance
        uvar = jnp.var(pf_post.filtered_particles, axis=1)

        assert jnp.allclose(wvar, uvar, atol=1e-6)


class TestWeightedQuantile:
    """Tests for weighted_quantile."""

    def test_weighted_quantile_median_near_mean(self, lgssm_params, lgssm_data):
        """For roughly symmetric posterior, median ≈ mean."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        means = weighted_mean(pf_post)
        medians = weighted_quantile(pf_post, jnp.array([0.5]))

        # medians shape: (ntime, 1, state_dim), squeeze quantile dim
        assert jnp.allclose(medians[:, 0, :], means, atol=0.2)

    def test_weighted_quantile_zero_weight_particles(self):
        """Quantiles should be correct even when some weights are zero.

        With particles [1, 2, 3] and weights [0, 0.5, 0.5], the
        median should be near 2.5 (only particles 2 and 3 matter).
        """
        from smcx.containers import ParticleFilterPosterior

        particles = jnp.array([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        log_w = jnp.array([
            [jnp.finfo(jnp.float64).min, jnp.log(0.5), jnp.log(0.5)]
        ])
        posterior = ParticleFilterPosterior(
            marginal_loglik=jnp.float64(0.0),
            filtered_particles=particles,
            filtered_log_weights=log_w,
            ancestors=jnp.array([[0, 1, 2]], dtype=jnp.int32),
            ess=jnp.array([2.0]),
            log_evidence_increments=jnp.array([0.0]),
        )
        q = jnp.array([0.5])
        result = weighted_quantile(posterior, q)
        # Median of {2, 3} with equal weight = 2.5
        assert float(result[0, 0, 0]) == pytest.approx(2.5, abs=0.1)

    def test_weighted_quantile_interval_contains_truth(
        self, lgssm_params, lgssm_data
    ):
        """95% credible interval should cover true state most of time."""
        states, _ = lgssm_data
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)

        quantiles = weighted_quantile(
            pf_post, jnp.array([0.025, 0.975])
        )  # (ntime, 2, state_dim)
        lower = quantiles[:, 0, :]
        upper = quantiles[:, 1, :]

        covered = jnp.all((states >= lower) & (states <= upper), axis=-1)
        coverage = float(jnp.mean(covered))

        # With T=50, expect ~95% coverage but allow Monte Carlo
        # variation: anything above 70% is acceptable
        assert coverage > 0.70, f"Coverage {coverage:.2%} too low"


class TestLogMLIncrements:
    """Tests for log_ml_increments."""

    def test_log_ml_increments_sum_to_total(self, lgssm_params, lgssm_data):
        """Increments should sum to total marginal log-likelihood."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data)
        increments = log_ml_increments(pf_post)

        # float32 (Metal) carries ~7 significant digits; float64 gets
        # the sharp absolute bound.
        f64 = jnp.asarray(pf_post.marginal_loglik).dtype == jnp.float64
        if f64:
            assert float(jnp.sum(increments)) == pytest.approx(
                float(pf_post.marginal_loglik), abs=1e-6
            )
        else:
            assert float(jnp.sum(increments)) == pytest.approx(
                float(pf_post.marginal_loglik), rel=1e-5
            )


class TestParticleDiversity:
    """Tests for particle_diversity."""

    def test_particle_diversity_bounded(self, lgssm_params, lgssm_data):
        """Diversity should be in [0, 1] at every time step."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        diversity = particle_diversity(pf_post)

        assert jnp.all(diversity >= 0.0)
        assert jnp.all(diversity <= 1.0)
        # With 1000 particles, first step should have high diversity
        assert float(diversity[0]) > 0.5


class TestDiagnosticsJIT:
    """All diagnostics should be JIT-compatible."""

    def test_diagnostics_jit_compatible(self, lgssm_params, lgssm_data):
        """Diagnostics compile and run under jax.jit."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)

        jax.jit(weighted_mean)(pf_post)
        jax.jit(weighted_variance)(pf_post)
        jax.jit(lambda p: weighted_quantile(p, jnp.array([0.5])))(pf_post)
        jax.jit(log_ml_increments)(pf_post)
        jax.jit(particle_diversity)(pf_post)


class TestLogBayesFactor:
    """Tests for log_bayes_factor."""

    def test_log_bayes_factor_symmetric(self):
        """BF(M1, M2) = -BF(M2, M1)."""
        bf = log_bayes_factor(jnp.float64(-70.0), jnp.float64(-75.0))
        bf_rev = log_bayes_factor(jnp.float64(-75.0), jnp.float64(-70.0))
        assert float(bf) == pytest.approx(-float(bf_rev), abs=1e-10)

    def test_log_bayes_factor_value(self):
        """BF is difference of log-MLs."""
        bf = log_bayes_factor(jnp.float64(-70.0), jnp.float64(-75.0))
        assert float(bf) == pytest.approx(5.0, abs=1e-10)


class TestReplicatedLogML:
    """Tests for replicated_log_ml."""

    def test_replicated_log_ml_shape(self, lgssm_params, lgssm_data):
        """Should return array of shape (num_replicates,)."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)

        def filter_fn(key):
            return bootstrap_filter(
                key=key,
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=500,
            ).marginal_loglik

        result = replicated_log_ml(jr.PRNGKey(0), filter_fn, num_replicates=10)
        assert result.shape == (10,)
        assert jnp.all(jnp.isfinite(result))

    def test_replicated_log_ml_variability(self, lgssm_params, lgssm_data):
        """Replicates should have non-zero variance."""
        _, emissions = lgssm_data
        init_fn, trans_fn, obs_fn = _make_smcx_fns(lgssm_params)

        def filter_fn(key):
            return bootstrap_filter(
                key=key,
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                log_observation_fn=obs_fn,
                emissions=emissions,
                num_particles=200,
            ).marginal_loglik

        result = replicated_log_ml(jr.PRNGKey(1), filter_fn, num_replicates=20)
        assert float(jnp.var(result)) > 0.0


class TestParamWeightedMean:
    """Tests for param_weighted_mean."""

    def test_param_weighted_mean_shape(self, lgssm_params, lgssm_data):
        """Output shape should be (ntime, param_dim)."""
        from smcx.diagnostics import param_weighted_mean
        from smcx.liu_west import liu_west_filter

        _, emissions = lgssm_data
        m0 = lgssm_params["initial_mean"]
        P0 = lgssm_params["initial_cov"]
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        post = liu_west_filter(
            key=jr.PRNGKey(42),
            initial_sampler=init,
            transition_sampler=trans,
            log_observation_fn=obs,
            log_auxiliary_fn=aux,
            param_initial_sampler=param_init,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )

        result = param_weighted_mean(post)
        ntime = emissions.shape[0]
        assert result.shape == (ntime, 1)
        assert jnp.all(jnp.isfinite(result))

    def test_param_weighted_mean_finite_values(self, lgssm_params, lgssm_data):
        """All param mean values should be finite."""
        from smcx.diagnostics import param_weighted_mean
        from smcx.liu_west import liu_west_filter

        _, emissions = lgssm_data
        m0 = lgssm_params["initial_mean"]
        P0 = lgssm_params["initial_cov"]
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        post = liu_west_filter(
            key=jr.PRNGKey(7),
            initial_sampler=init,
            transition_sampler=trans,
            log_observation_fn=obs,
            log_auxiliary_fn=aux,
            param_initial_sampler=param_init,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )

        param_means = param_weighted_mean(post)
        ntime = emissions.shape[0]
        assert param_means.shape == (ntime, 1)
        assert jnp.all(jnp.isfinite(param_means))


class TestParamWeightedQuantile:
    """Tests for param_weighted_quantile."""

    def test_param_weighted_quantile_monotone(self, lgssm_params, lgssm_data):
        """Lower quantile <= upper quantile at every step."""
        from smcx.diagnostics import param_weighted_quantile
        from smcx.liu_west import liu_west_filter

        _, emissions = lgssm_data
        m0 = lgssm_params["initial_mean"]
        P0 = lgssm_params["initial_cov"]
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state, params):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state, params):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state, params):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        def param_init(key, n):
            return jnp.zeros((n, 1))

        post = liu_west_filter(
            key=jr.PRNGKey(42),
            initial_sampler=init,
            transition_sampler=trans,
            log_observation_fn=obs,
            log_auxiliary_fn=aux,
            param_initial_sampler=param_init,
            emissions=emissions,
            num_particles=500,
            shrinkage=0.95,
        )

        q = jnp.array([0.025, 0.5, 0.975])
        result = param_weighted_quantile(post, q)
        ntime = emissions.shape[0]
        assert result.shape == (ntime, 3, 1)
        # Monotonicity: q025 <= q50 <= q975
        assert jnp.all(result[:, 0, :] <= result[:, 1, :])
        assert jnp.all(result[:, 1, :] <= result[:, 2, :])


class TestCRPS:
    """Tests for crps."""

    def test_crps_nonnegative(self):
        """CRPS should always be non-negative."""
        from smcx.diagnostics import crps

        predictions = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = crps(predictions, jnp.float64(3.0))
        assert float(result) >= 0.0

    def test_crps_zero_for_perfect_prediction(self):
        """CRPS = 0 when all predictions equal observation."""
        from smcx.diagnostics import crps

        obs = jnp.float64(5.0)
        predictions = jnp.full(100, 5.0)
        result = crps(predictions, obs)
        assert float(result) == pytest.approx(0.0, abs=1e-10)

    def test_crps_known_value(self):
        """CRPS for known distribution matches analytical result."""
        from smcx.diagnostics import crps

        # For predictions = {0, 1} with equal weight, obs = 0.5:
        # E|Y - y| = 0.5*(|0-0.5| + |1-0.5|) = 0.5
        # E|Y - Y'| = 0.5*(|0-0| + |0-1| + |1-0| + |1-1|)/2
        #           = 0.5*(0 + 1 + 1 + 0)/2 but actually:
        # E|Y-Y'| = mean of all |yi-yj| = (0+1+1+0)/4 = 0.5
        # CRPS = 0.5 - 0.5*0.5 = 0.25
        predictions = jnp.array([0.0, 1.0])
        result = crps(predictions, jnp.float64(0.5))
        assert float(result) == pytest.approx(0.25, abs=1e-10)

    def test_crps_large_sample_matches_formula(self):
        """Sort-based CRPS matches brute-force on N=500.

        Cross-checks that the O(N log N) implementation gives the same
        answer as the naive O(N^2) all-pairs formula.
        """
        from smcx.diagnostics import crps

        key = jr.PRNGKey(77)
        predictions = jr.normal(key, (500,))
        obs = jnp.float64(0.5)

        # Brute-force reference
        term1 = jnp.mean(jnp.abs(predictions - obs))
        diffs = jnp.abs(predictions[:, None] - predictions[None, :])
        term2 = jnp.mean(diffs)
        expected = float(term1 - 0.5 * term2)

        result = float(crps(predictions, obs))
        assert result == pytest.approx(expected, abs=1e-6)

    def test_crps_jit_compatible(self):
        """CRPS should work under jax.jit."""
        from smcx.diagnostics import crps

        predictions = jnp.array([1.0, 2.0, 3.0])
        result = jax.jit(crps)(predictions, jnp.float64(2.0))
        assert jnp.isfinite(result)


class TestPosteriorPredictiveSample:
    """Tests for posterior_predictive_sample."""

    def test_shape(self, lgssm_params, lgssm_data):
        """Output shape is (ntime, num_samples, emission_dim)."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

        def trans(key, state):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def emit(key, state):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, R)

        result = posterior_predictive_sample(
            jr.PRNGKey(99), pf_post, trans, emit, num_samples=100
        )
        ntime = pf_post.filtered_particles.shape[0]
        assert result.shape == (ntime, 100, 1)

    def test_finite(self, lgssm_params, lgssm_data):
        """All predictive samples should be finite."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

        def trans(key, state):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def emit(key, state):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, R)

        result = posterior_predictive_sample(
            jr.PRNGKey(42), pf_post, trans, emit
        )
        assert jnp.all(jnp.isfinite(result))

    def test_default_num_samples(self, lgssm_params, lgssm_data):
        """Default num_samples should equal num_particles."""
        n = 200
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=n)
        F = lgssm_params["dynamics_weights"]
        Q = lgssm_params["dynamics_cov"]
        H = lgssm_params["emissions_weights"]
        R = lgssm_params["emissions_cov"]

        def trans(key, state):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def emit(key, state):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, R)

        result = posterior_predictive_sample(
            jr.PRNGKey(0), pf_post, trans, emit
        )
        assert result.shape[1] == n


class TestParetoKDiagnostic:
    """Tests for pareto_k_diagnostic."""

    def test_pareto_k_shape(self, lgssm_params, lgssm_data):
        """Output shape matches (ntime,)."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        k_hat = pareto_k_diagnostic(pf_post)
        assert k_hat.shape == (pf_post.filtered_log_weights.shape[0],)

    def test_pareto_k_finite(self, lgssm_params, lgssm_data):
        """All k-hat values should be finite."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        k_hat = pareto_k_diagnostic(pf_post)
        assert jnp.all(jnp.isfinite(k_hat))

    def test_pareto_k_uniform_weights_low(self):
        """Uniform weights should give low k (no heavy tail)."""
        from smcx.containers import ParticleFilterPosterior

        n = 1000
        log_w = jnp.full((5, n), -jnp.log(n))
        particles = jnp.zeros((5, n, 1))
        posterior = ParticleFilterPosterior(
            marginal_loglik=jnp.float64(0.0),
            filtered_particles=particles,
            filtered_log_weights=log_w,
            ancestors=jnp.zeros((5, n), dtype=jnp.int32),
            ess=jnp.full(5, float(n)),
            log_evidence_increments=jnp.zeros(5),
        )
        k_hat = pareto_k_diagnostic(posterior)
        # Uniform weights have no tail: k should be small
        assert jnp.all(k_hat < 0.7)

    def test_pareto_k_jit_compatible(self, lgssm_params, lgssm_data):
        """pareto_k_diagnostic compiles under jax.jit."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)
        result = jax.jit(pareto_k_diagnostic)(pf_post)
        assert jnp.all(jnp.isfinite(result))

    def test_pareto_k_ordering_by_tail_heaviness(self):
        """Cauchy log-weights produce a higher k than t_3 or Gaussian.

        Following the Stan posterior vignette (Vehtari et al. 2024),
        Cauchy (df=1) has the heaviest tails of the three and should
        be detected as such by the Pareto-k estimator.  We average
        over several seeds so the test depends on the property and
        not on a particular RNG implementation.

        Strict ordering between t_3 and Gaussian is not asserted: the
        estimator saturates near the same value for both at realistic
        sample sizes, so distinguishing them is below the noise floor.
        """
        from smcx.diagnostics import _fit_pareto_k

        n = 5000
        num_seeds = 8

        def _t_log_weights(key, df):
            k1, k2 = jr.split(key)
            z = jr.normal(k1, (n,), dtype=jnp.float64)
            v = jnp.sum(
                jr.normal(k2, (n, df), dtype=jnp.float64) ** 2,
                axis=1,
            )
            return jnp.log(jnp.abs(z / jnp.sqrt(v / df)))

        def _mean_k(seeds, sampler):
            ks = [float(_fit_pareto_k(sampler(jr.PRNGKey(s)))) for s in seeds]
            return sum(ks) / len(ks)

        seeds = list(range(num_seeds))
        k_cauchy = _mean_k(seeds, lambda k: _t_log_weights(k, 1))
        k_t3 = _mean_k(seeds, lambda k: _t_log_weights(k, 3))
        k_gauss = _mean_k(
            seeds, lambda k: jr.normal(k, (n,), dtype=jnp.float64)
        )

        assert k_cauchy > k_t3, (
            f"Expected k_cauchy > k_t3, got {k_cauchy:.3f}, {k_t3:.3f}"
        )
        assert k_cauchy > k_gauss, (
            f"Expected k_cauchy > k_gauss, got {k_cauchy:.3f}, {k_gauss:.3f}"
        )

    def test_pareto_k_cauchy_above_unreliable(self):
        """Cauchy log-weights give k above 0.7 (unreliable).

        The Cauchy distribution has tail index 1, so k ~ 1.0.
        Even after Vehtari prior shrinkage, k should exceed 0.7.
        """
        from smcx.diagnostics import _fit_pareto_k

        key = jr.PRNGKey(1)
        n = 2000
        k1, k2 = jr.split(key)
        z = jr.normal(k1, (n,), dtype=jnp.float64)
        v = jr.normal(k2, (n,), dtype=jnp.float64) ** 2
        log_w = jnp.log(jnp.abs(z / jnp.sqrt(v)))

        k_hat = float(_fit_pareto_k(log_w))
        assert k_hat > 0.7, f"Expected k > 0.7 for Cauchy, got {k_hat}"

    def test_pareto_k_gaussian_below_unreliable(self):
        """Gaussian log-weights give k below the 0.7 threshold.

        The Vehtari prior pulls toward 0.5, so light-tailed data
        gives k ~ 0.5-0.6.  The key property is that it stays
        below 0.7 (the "unreliable" cutoff).
        """
        from smcx.diagnostics import _fit_pareto_k

        key = jr.PRNGKey(123)
        n = 1000
        log_w = jr.normal(key, (n,), dtype=jnp.float64)

        k_hat = float(_fit_pareto_k(log_w))
        assert k_hat < 0.7, f"Expected k < 0.7 for Gaussian, got {k_hat}"


class TestTailESS:
    """Tests for tail_ess."""

    def test_tail_ess_shape(self, lgssm_params, lgssm_data):
        """Output shape matches (ntime,)."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        result = tail_ess(pf_post)
        assert result.shape == (pf_post.filtered_log_weights.shape[0],)

    def test_tail_ess_bounded(self, lgssm_params, lgssm_data):
        """Tail-ESS should be in [0, num_particles]."""
        n = 1_000
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=n)
        result = tail_ess(pf_post)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= n)

    def test_tail_ess_uniform_is_q_fraction(self):
        """Uniform weights: each tail holds ~q*N effective particles."""
        import jax.random as jr

        from smcx.containers import ParticleFilterPosterior

        n = 4000
        log_w = jnp.full((1, n), -jnp.log(n))
        particles = jr.normal(jr.key(2), (1, n, 1))
        posterior = ParticleFilterPosterior(
            marginal_loglik=jnp.float64(0.0),
            filtered_particles=particles,
            filtered_log_weights=log_w,
            ancestors=jnp.zeros((1, n), dtype=jnp.int32),
            ess=jnp.full((1,), float(n)),
            log_evidence_increments=jnp.zeros((1,)),
        )
        te = float(tail_ess(posterior, q=0.05)[0])
        assert te == pytest.approx(0.05 * n, rel=0.15)

    def test_tail_ess_leq_standard_ess(self, lgssm_params, lgssm_data):
        """Tail-ESS <= standard ESS (tails are harder to estimate)."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        t_ess = tail_ess(pf_post)
        s_ess = pf_post.ess
        # Allow small numerical tolerance
        assert jnp.all(t_ess <= s_ess + 1.0)

    def test_tail_ess_jit_compatible(self, lgssm_params, lgssm_data):
        """tail_ess compiles under jax.jit."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)
        result = jax.jit(tail_ess)(pf_post)
        assert jnp.all(jnp.isfinite(result))


class TestCumulativeLogScore:
    """Tests for cumulative_log_score."""

    def test_cumulative_log_score_shape(self, lgssm_params, lgssm_data):
        """Output shape matches (ntime,)."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        result = cumulative_log_score(pf_post)
        assert result.shape == (pf_post.log_evidence_increments.shape[0],)

    def test_cumulative_log_score_final_equals_marginal_loglik(
        self, lgssm_params, lgssm_data
    ):
        """Last element should equal marginal_loglik."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        result = cumulative_log_score(pf_post)
        assert float(result[-1]) == pytest.approx(
            float(pf_post.marginal_loglik), abs=1e-6
        )

    def test_cumulative_log_score_monotone_structure(
        self, lgssm_params, lgssm_data
    ):
        """Cumulative scores are a running cumsum of increments."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        result = cumulative_log_score(pf_post)
        expected = jnp.cumsum(pf_post.log_evidence_increments)
        assert jnp.allclose(result, expected, atol=1e-10)

    def test_cumulative_log_score_jit_compatible(
        self, lgssm_params, lgssm_data
    ):
        """cumulative_log_score compiles under jax.jit."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)
        result = jax.jit(cumulative_log_score)(pf_post)
        assert jnp.all(jnp.isfinite(result))


class TestDiagnose:
    """Tests for diagnose."""

    def test_diagnose_returns_dict(self, lgssm_params, lgssm_data):
        """Diagnose returns a dict with expected keys."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        result = diagnose(pf_post)
        assert "min_ess" in result
        assert "min_diversity" in result
        assert "max_pareto_k" in result
        assert "ess_below_threshold" in result
        assert "warnings" in result

    def test_diagnose_healthy_filter_no_warnings(
        self, lgssm_params, lgssm_data
    ):
        """A well-behaved filter should produce few or no warnings."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=5_000)
        result = diagnose(pf_post)
        # With 5000 particles on a simple LGSSM, ESS should be OK
        assert result["min_ess"] > 1.0

    def test_pareto_k_warning_states_reliability_not_variance(
        self, lgssm_params, lgssm_data
    ):
        """The k-threshold warning is about reliability, not variance.

        Weight variance is infinite for every k >= 0.5; the 0.7-ish
        threshold is the PSIS practical-reliability boundary (Vehtari
        et al. 2024). The warning must not present the threshold as
        the infinite-variance boundary.
        """
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=500)
        # Force the warning regardless of the fitted k values.
        result = diagnose(pf_post, pareto_k_threshold=-1.0)
        k_warnings = [w for w in result["warnings"] if "Pareto-k" in w]
        assert k_warnings, "expected a Pareto-k warning"
        msg = k_warnings[0]
        assert "unreliable" in msg
        assert "0.5" in msg
        assert "infinite variance at some steps" not in msg

    def test_diagnose_collapsed_ess_warns(self):
        """When ESS = 1, diagnose should warn."""
        from smcx.containers import ParticleFilterPosterior

        n = 100
        # One particle has all the weight
        log_w = jnp.full((3, n), jnp.finfo(jnp.float64).min)
        log_w = log_w.at[:, 0].set(0.0)
        particles = jnp.zeros((3, n, 1))
        posterior = ParticleFilterPosterior(
            marginal_loglik=jnp.float64(0.0),
            filtered_particles=particles,
            filtered_log_weights=log_w,
            ancestors=jnp.zeros((3, n), dtype=jnp.int32),
            ess=jnp.ones(3),
            log_evidence_increments=jnp.zeros(3),
        )
        result = diagnose(posterior)
        assert len(result["warnings"]) > 0
        assert result["ess_below_threshold"] > 0
