# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for smcx.diagnostics against frozen exact references."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from smcx.bootstrap import bootstrap_filter
from smcx.containers import (
    LiuWestPosterior,
    ParticleFilterPosterior,
    ParticleFilterRecord,
    SMC2Posterior,
)
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
from smcx.runner import run_particle_filter
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


def _make_posterior():
    """Return a small posterior with exact uniform-weight summaries."""
    n = 32
    base = 2.0 * jnp.arange(n) - (n - 1)
    particles = jnp.stack([base, base + 100.0, base - 100.0])[:, :, None]
    increments = jnp.array([-1.0, -2.0, -3.0])
    return ParticleFilterPosterior(
        marginal_loglik=jnp.sum(increments),
        filtered_particles=particles,
        filtered_log_weights=jnp.full((3, n), -jnp.log(n)),
        ancestors=jnp.broadcast_to(jnp.arange(n), (3, n)),
        ess=jnp.full((3,), float(n)),
        log_evidence_increments=increments,
    )


def _neumaier_prefix_oracle(values: np.ndarray) -> np.ndarray:
    """Return sequential f32 Neumaier prefixes from NumPy scalars."""
    values = np.asarray(values)
    cast = values.dtype.type
    total = cast(0.0)
    correction = cast(0.0)
    prefixes = np.empty_like(values)
    for index, value in enumerate(values):
        updated = cast(total + value)
        if abs(total) >= abs(value):
            lost = cast(cast(total - updated) + value)
        else:
            lost = cast(cast(value - updated) + total)
        correction = cast(correction + lost)
        total = updated
        prefixes[index] = cast(total + correction)
    return prefixes


def _exact_crps_oracle(
    predictions: np.ndarray,
    observation: np.float32,
) -> Fraction:
    """Return the equal-weight empirical CRPS as an exact rational."""
    samples = [Fraction.from_float(float(value)) for value in predictions]
    observed = Fraction.from_float(float(observation))
    sample_count = len(samples)
    absolute_error = sum(abs(value - observed) for value in samples)
    pairwise_error = sum(
        abs(left - right) for left in samples for right in samples
    )
    return absolute_error / sample_count - pairwise_error / (
        2 * sample_count * sample_count
    )


def _posterior_for_increment_contract(
    increments: jax.Array,
) -> ParticleFilterPosterior:
    """Build a valid final-only posterior with compensated evidence."""
    increments = jnp.asarray(increments)
    prefixes = _neumaier_prefix_oracle(np.asarray(increments))
    num_timesteps = increments.shape[0]
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(prefixes[-1]),
        filtered_particles=jnp.zeros((1, 1, 1), dtype=increments.dtype),
        filtered_log_weights=jnp.zeros((1, 1), dtype=increments.dtype),
        ancestors=jnp.zeros((1, 1), dtype=jnp.int32),
        ess=jnp.ones((num_timesteps,), dtype=increments.dtype),
        log_evidence_increments=increments,
    )


def _make_liu_west_posterior():
    """Add a deterministic parameter cloud to the small posterior."""
    posterior = _make_posterior()
    return LiuWestPosterior(
        *posterior,
        filtered_params=posterior.filtered_particles / 10.0,
    )


def _make_smc2_posterior() -> SMC2Posterior:
    """Return the same deterministic parameter cloud as an SMC² result."""
    posterior = _make_liu_west_posterior()
    return SMC2Posterior(
        marginal_loglik=posterior.marginal_loglik,
        filtered_params=posterior.filtered_params,
        filtered_log_weights=posterior.filtered_log_weights,
        ess=posterior.ess,
        log_evidence_increments=posterior.log_evidence_increments,
        acceptance_rates=jnp.zeros_like(posterior.ess),
    )


class TestWeightedMean:
    """Tests for weighted_mean."""

    def test_weighted_mean_exact(self):
        posterior = _make_posterior()
        expected = jnp.array([[0.0], [100.0], [-100.0]])
        assert jnp.array_equal(weighted_mean(posterior), expected)


class TestWeightedVariance:
    """Tests for weighted_variance."""

    def test_weighted_variance_uniform_weights(self):
        """With uniform weights, matches unweighted variance."""
        result = weighted_variance(_make_posterior())
        assert jnp.array_equal(result, jnp.full((3, 1), 341.0))


class TestWeightedQuantile:
    """Tests for weighted_quantile."""

    def test_weighted_quantile_median_exact(self):
        posterior = _make_posterior()
        result = weighted_quantile(posterior, jnp.array([0.5]))
        expected = jnp.array([[[0.0]], [[100.0]], [[-100.0]]])
        assert jnp.allclose(result, expected, rtol=0.0, atol=1e-6)

    def test_weighted_quantile_zero_weight_particles(self):
        """Quantiles should be correct even when some weights are zero.

        With particles [1, 2, 3] and weights [0, 0.5, 0.5], the
        median should be near 2.5 (only particles 2 and 3 matter).
        """
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

    def test_tracks_cumulative_time_zero_lineages(self):
        """State values and one-step parents do not define path diversity."""
        num_particles = 4
        particles = jnp.array([
            [[0.0], [0.0], [0.0], [0.0]],
            [[1.0], [2.0], [3.0], [4.0]],
            [[5.0], [6.0], [7.0], [8.0]],
        ])
        ancestors = jnp.array(
            [
                [0, 1, 2, 3],
                [0, 0, 0, 1],
                [1, 2, 1, 2],
            ],
            dtype=jnp.int32,
        )
        posterior = ParticleFilterPosterior(
            marginal_loglik=jnp.asarray(0.0),
            filtered_particles=particles,
            filtered_log_weights=jnp.full(
                (3, num_particles), -jnp.log(num_particles)
            ),
            ancestors=ancestors,
            ess=jnp.full((3,), float(num_particles)),
            log_evidence_increments=jnp.zeros(3),
        )

        expected = jnp.array([1.0, 0.5, 0.25])
        assert jnp.array_equal(particle_diversity(posterior), expected)

    def test_particle_diversity_bounded(self, lgssm_params, lgssm_data):
        """Diversity should be in [0, 1] at every time step."""
        pf_post = _run_bootstrap(lgssm_params, lgssm_data, n=1_000)
        diversity = particle_diversity(pf_post)

        assert jnp.all(diversity >= 0.0)
        assert jnp.all(diversity <= 1.0)
        # With 1000 particles, first step should have high diversity
        assert float(diversity[0]) > 0.5


class TestDiagnosticsJIT:
    """Array-returning diagnostics should be JIT-compatible."""

    def test_diagnostics_jit_compatible(self, lgssm_params, lgssm_data):
        """Array diagnostics compile and run under jax.jit."""
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

    def test_replicated_log_ml_splits_keys_and_vectorizes(self):
        def filter_fn(key):
            return jr.normal(key)

        key = jr.PRNGKey(0)
        result = replicated_log_ml(key, filter_fn, num_replicates=10)
        expected = jax.vmap(filter_fn)(jr.split(key, 10))
        assert jnp.array_equal(result, expected)
        assert float(jnp.var(result)) > 0.0


class TestParamWeightedMean:
    """Tests for param_weighted_mean."""

    def test_param_weighted_mean_exact(self):
        from smcx.diagnostics import param_weighted_mean

        post = _make_liu_west_posterior()
        result = param_weighted_mean(post)
        expected = jnp.array([[0.0], [10.0], [-10.0]])
        assert jnp.allclose(result, expected, rtol=0.0, atol=1e-6)

    def test_smc2_parameter_summaries(self):
        from smcx.diagnostics import (
            param_weighted_mean,
            param_weighted_quantile,
        )

        posterior = _make_smc2_posterior()
        mean = param_weighted_mean(posterior)
        quantiles = param_weighted_quantile(posterior, jnp.array([0.5]))

        expected = jnp.array([[0.0], [10.0], [-10.0]])
        assert jnp.allclose(mean, expected, rtol=0.0, atol=1e-6)
        assert jnp.allclose(quantiles[:, 0, :], expected, rtol=0.0, atol=1e-6)


class TestParamWeightedQuantile:
    """Tests for param_weighted_quantile."""

    def test_param_weighted_quantile_monotone(self):
        """Lower quantile <= upper quantile at every step."""
        from smcx.diagnostics import param_weighted_quantile

        q = jnp.array([0.025, 0.5, 0.975])
        result = param_weighted_quantile(_make_liu_west_posterior(), q)
        assert result.shape == (3, 3, 1)
        assert jnp.all(result[:, 0, :] <= result[:, 1, :])
        assert jnp.all(result[:, 1, :] <= result[:, 2, :])
        expected_median = jnp.array([[0.0], [10.0], [-10.0]])
        assert jnp.allclose(
            result[:, 1, :], expected_median, rtol=0.0, atol=1e-6
        )


class TestCRPS:
    """Tests for crps."""

    @pytest.mark.parametrize(
        ("num_samples", "value"),
        [(100, 1e10), (1000, 1e5)],
    )
    def test_crps_constant_large_offset_is_exactly_zero(
        self, num_samples, value
    ):
        """A perfect empirical forecast stays nonnegative at large offsets."""
        from smcx.diagnostics import crps

        predictions = jnp.full((num_samples,), value, dtype=jnp.float32)
        result = crps(predictions, jnp.float32(value))
        assert jnp.array_equal(result, jnp.float32(0.0))

    def test_crps_is_invariant_to_represented_translation(self):
        """Centering an already represented sample leaves CRPS unchanged."""
        from smcx.diagnostics import crps

        offset = jnp.float32(1e7)
        pattern = jnp.asarray(
            [-8.0, -3.0, -1.0, 0.0, 2.0, 5.0, 9.0],
            dtype=jnp.float32,
        )
        shifted = jnp.tile(pattern + offset, 128)
        centered = shifted - offset

        shifted_score = crps(shifted, offset)
        centered_score = crps(centered, jnp.float32(0.0))

        assert jnp.array_equal(shifted_score, centered_score)

    @pytest.mark.parametrize(
        ("num_samples", "value"),
        [(10_000, 1e35), (1, 2e38), (100, 1e-37)],
    )
    def test_crps_large_finite_score_does_not_overflow(
        self, num_samples, value
    ):
        from smcx.diagnostics import crps

        predictions = jnp.full((num_samples,), value, dtype=jnp.float32)

        result = crps(predictions, jnp.float32(0.0))

        assert jnp.isfinite(result)
        # Thirty-two eps cover interval weights and the final reduction.
        relative_tolerance = 32.0 * float(jnp.finfo(jnp.float32).eps)
        assert float(result) == pytest.approx(
            float(predictions[0]),
            rel=relative_tolerance,
            abs=0.0,
        )

    @pytest.mark.parametrize(
        "endpoint",
        [jnp.finfo(jnp.float32).min, jnp.finfo(jnp.float32).max],
    )
    def test_crps_dtype_endpoint_cloud_stays_finite(self, endpoint):
        from smcx.diagnostics import crps

        predictions = jnp.full((7,), endpoint, dtype=jnp.float32)
        observation = jnp.asarray(0.0, dtype=jnp.float32)
        expected = jnp.abs(jnp.asarray(endpoint, dtype=jnp.float32))

        assert jnp.array_equal(crps(predictions, observation), expected)
        assert jnp.array_equal(
            jax.jit(crps)(predictions, observation), expected
        )

    def test_crps_mixed_endpoint_cloud_stays_finite(self):
        from smcx.diagnostics import crps

        dtype_max = jnp.asarray(
            jnp.finfo(jnp.float32).max,
            dtype=jnp.float32,
        )
        predictions = jnp.concatenate([
            jnp.full((9,), -dtype_max, dtype=jnp.float32),
            jnp.full((3,), dtype_max, dtype=jnp.float32),
        ])
        observation = jnp.float32(0.75) * dtype_max
        expected = jnp.asarray(dtype_max, dtype=jnp.float32)

        assert jnp.array_equal(crps(predictions, observation), expected)
        assert jnp.array_equal(
            jax.jit(crps)(predictions, observation), expected
        )

    @pytest.mark.parametrize(
        "dtype",
        [jnp.float8_e4m3fn, jnp.float8_e5m2],
    )
    @pytest.mark.skipif(
        jax.default_backend() == "mps",
        reason="jax-mps cannot represent float8 buffers",
    )
    def test_crps_preserves_float8_support(self, dtype):
        from smcx.diagnostics import crps

        predictions = jnp.asarray([0.0, 1.0], dtype=dtype)
        observation = jnp.asarray(0.5, dtype=dtype)
        expected = jnp.asarray(0.25, dtype=dtype)

        result = crps(predictions, observation)
        compiled = jax.jit(crps)(predictions, observation)

        assert jnp.array_equal(result, expected)
        assert jnp.array_equal(compiled, expected)

    def test_crps_float16_large_sample_uses_wide_ranks(self):
        from smcx.diagnostics import crps

        predictions = jnp.concatenate([
            jnp.zeros((32_753,), dtype=jnp.float16),
            jnp.ones((32_753,), dtype=jnp.float16),
        ])
        observation = jnp.asarray(0.5, dtype=jnp.float16)
        expected = jnp.asarray(0.25, dtype=jnp.float16)

        assert jnp.array_equal(crps(predictions, observation), expected)

    def test_crps_bfloat16_rank_precision(self):
        from smcx.diagnostics import crps

        dtype = jnp.bfloat16
        predictions = jnp.concatenate([
            jnp.zeros((179,), dtype=dtype),
            jnp.ones((78,), dtype=dtype),
        ])
        observation = jnp.asarray(0.2, dtype=dtype)
        mass = Fraction(179, 257)
        represented_observation = Fraction(205, 1024)
        exact = (
            represented_observation * mass**2
            + (1 - represented_observation) * (1 - mass) ** 2
        )
        expected = jnp.asarray(float(exact), dtype=dtype)

        assert jnp.array_equal(crps(predictions, observation), expected)
        assert jnp.array_equal(
            jax.jit(crps)(predictions, observation),
            expected,
        )

    @pytest.mark.parametrize("observation", [-3e38, 3e38, 0.0, -0.0])
    def test_crps_extreme_opposite_signs_remain_finite(self, observation):
        from smcx.diagnostics import crps

        predictions = jnp.asarray([-3e38, 3e38], dtype=jnp.float32)

        result = crps(predictions, jnp.float32(observation))

        assert jnp.isfinite(result)
        relative_tolerance = 8.0 * float(jnp.finfo(jnp.float32).eps)
        assert float(result) == pytest.approx(
            float(jnp.float32(1.5e38)),
            rel=relative_tolerance,
        )

    def test_crps_exact_finite_overflow_boundary_stays_finite(self):
        """Rounding below the float32 overflow midpoint stays finite."""
        from smcx.diagnostics import crps

        predictions = np.asarray([-(2.0**105), 1.0], dtype=np.float32)
        observation = np.float32(np.finfo(np.float32).max)
        overflow_midpoint = Fraction(2**128 - 2**103)
        assert _exact_crps_oracle(predictions, observation) < overflow_midpoint

        batched_predictions = jnp.asarray(np.stack([predictions, -predictions]))
        batched_observations = jnp.asarray([observation, -observation])
        expected = jnp.full(
            (2,),
            jnp.finfo(jnp.float32).max,
            dtype=jnp.float32,
        )

        assert jnp.array_equal(
            crps(batched_predictions[0], batched_observations[0]),
            expected[0],
        )
        assert jnp.array_equal(
            jax.jit(crps)(
                batched_predictions[0],
                batched_observations[0],
            ),
            expected[0],
        )
        assert jnp.array_equal(
            jax.jit(jax.vmap(crps))(
                batched_predictions,
                batched_observations,
            ),
            expected,
        )

    def test_crps_exact_infinite_overflow_boundary_returns_infinity(self):
        """Rounding at or above the overflow midpoint returns infinity."""
        from smcx.diagnostics import crps

        value_scale = 2.0**80
        threshold = 2.0**23
        predictions = np.asarray(
            [
                (-threshold - 4.0) * value_scale,
                (-threshold - 2.0) * value_scale,
                (-threshold - 1.0) * value_scale,
                (-threshold + 2.0) * value_scale,
            ],
            dtype=np.float32,
        )
        observation = np.float32(np.finfo(np.float32).max)
        overflow_midpoint = Fraction(2**128 - 2**103)
        assert _exact_crps_oracle(predictions, observation) > overflow_midpoint
        tie_predictions = np.asarray([-(2.0**103)], dtype=np.float32)
        assert (
            _exact_crps_oracle(tie_predictions, observation)
            == overflow_midpoint
        )
        tie_observation = jnp.asarray(observation)
        assert jnp.isinf(crps(jnp.asarray(tie_predictions), tie_observation))
        assert jnp.isinf(
            jax.jit(crps)(jnp.asarray(tie_predictions), tie_observation)
        )

        batched_predictions = jnp.asarray(np.stack([predictions, -predictions]))
        batched_observations = jnp.asarray([observation, -observation])
        expected = jnp.full((2,), jnp.inf, dtype=jnp.float32)

        assert jnp.array_equal(
            crps(batched_predictions[0], batched_observations[0]),
            expected[0],
        )
        assert jnp.array_equal(
            jax.jit(crps)(
                batched_predictions[0],
                batched_observations[0],
            ),
            expected[0],
        )
        assert jnp.array_equal(
            jax.jit(jax.vmap(crps))(
                batched_predictions,
                batched_observations,
            ),
            expected,
        )

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
        # Four f32 eps cover normalized interval arithmetic and rescaling.
        tolerance = 4.0 * float(jnp.finfo(jnp.float32).eps)
        assert float(result) == pytest.approx(0.25, rel=tolerance, abs=0.0)

    def test_crps_matches_pairwise_oracle_with_repeated_values(self):
        """Order-statistic CRPS matches the independent pairwise identity."""
        from smcx.diagnostics import crps

        samples = [-2.0, -2.0, 0.0, 1.0, 1.0, 4.0]
        observation = 0.25
        n = len(samples)
        term1 = sum(abs(x - observation) for x in samples) / n
        term2 = sum(abs(x - y) for x in samples for y in samples) / (n * n)
        expected = term1 - 0.5 * term2

        result = float(
            crps(
                jnp.asarray(samples, dtype=jnp.float32),
                jnp.float32(observation),
            )
        )
        # Five f32 eps at the score's unit scale covers the final reduction.
        tolerance = 5.0 * float(jnp.finfo(jnp.float32).eps)
        assert result == pytest.approx(expected, rel=0.0, abs=tolerance)

    def test_crps_jit_vmap_compatible(self):
        """CRPS batches observations and empirical forecasts under JIT."""
        from smcx.diagnostics import crps

        predictions = jnp.array([[0.0, 1.0], [1.0, 1.0]])
        observations = jnp.array([0.5, 2.0])
        result = jax.jit(jax.vmap(crps))(predictions, observations)
        # Four f32 eps cover normalized interval arithmetic and rescaling.
        tolerance = 4.0 * float(jnp.finfo(jnp.float32).eps)
        assert jnp.allclose(
            result,
            jnp.array([0.25, 1.0]),
            rtol=tolerance,
            atol=0.0,
        )

    @pytest.mark.parametrize("outlier", [-1.0, 1.0])
    def test_crps_large_sample_preserves_reflected_outlier(self, outlier):
        """Large empirical forecasts retain reflected tail contributions."""
        from smcx.diagnostics import crps

        num_samples = 6_556_022
        predictions = (
            jnp.zeros((num_samples,), dtype=jnp.float32).at[-1].set(outlier)
        )
        result = float(crps(predictions, jnp.asarray(0.0, dtype=jnp.float32)))
        expected = 1.0 / (num_samples * num_samples)
        # fl(1/N) contributes at most eps/2 relative error; squaring it
        # and the two final products remain below five float32 eps.
        relative_tolerance = 5.0 * float(jnp.finfo(jnp.float32).eps)
        assert result == pytest.approx(
            expected,
            rel=relative_tolerance,
            abs=0.0,
        )


class TestPosteriorPredictiveSample:
    """Tests for posterior_predictive_sample."""

    @pytest.mark.parametrize(
        ("emission", "message"),
        [
            ([0.0], "must be a JAX array"),
            (jnp.asarray(0.0), "shape \\(emission_dim,\\)"),
            (jnp.empty((0,)), "shape \\(emission_dim,\\)"),
            (jnp.asarray([0], dtype=jnp.int32), "floating dtype"),
        ],
    )
    def test_rejects_invalid_emission(self, emission, message):
        """Predictive emissions are nonempty floating vectors."""
        with pytest.raises(ValueError, match=message):
            posterior_predictive_sample(
                jr.key(0),
                _make_posterior(),
                lambda _key, state: state,
                lambda _key, _state: emission,
                num_samples=2,
            )

    @pytest.mark.parametrize("num_samples", [0, -1])
    def test_num_samples_must_be_positive(self, num_samples):
        """An explicit predictive sample count must be positive."""
        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            posterior_predictive_sample(
                jr.key(0),
                _make_posterior(),
                lambda _key, state: state,
                lambda _key, state: state,
                num_samples=num_samples,
            )

    def test_shape(self, lgssm_params):
        """Output shape is (ntime, num_samples, emission_dim)."""
        pf_post = _make_posterior()
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

    def test_finite(self, lgssm_params):
        """All predictive samples should be finite."""
        pf_post = _make_posterior()
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

    def test_default_num_samples(self, lgssm_params):
        """Default num_samples should equal num_particles."""
        pf_post = _make_posterior()
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
        assert result.shape[1] == pf_post.filtered_log_weights.shape[1]


class TestParetoKDiagnostic:
    """Tests for pareto_k_diagnostic."""

    def test_pareto_k_shape(self):
        """Output shape matches (ntime,)."""
        pf_post = _make_posterior()
        k_hat = pareto_k_diagnostic(pf_post)
        assert k_hat.shape == (pf_post.filtered_log_weights.shape[0],)

    def test_pareto_k_finite(self):
        """All k-hat values should be finite."""
        pf_post = _make_posterior()
        k_hat = pareto_k_diagnostic(pf_post)
        assert jnp.all(jnp.isfinite(k_hat))

    def test_pareto_k_uniform_weights_low(self):
        """Uniform weights should give low k (no heavy tail)."""
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

    def test_pareto_k_jit_compatible(self):
        """pareto_k_diagnostic compiles under jax.jit."""
        pf_post = _make_posterior()
        result = jax.jit(pareto_k_diagnostic)(pf_post)
        assert jnp.all(jnp.isfinite(result))

    def test_single_particle_posterior_reports_undefined_pareto_k(self):
        """Pareto-k is undefined, rather than erroneous, when N is one."""
        posterior = ParticleFilterPosterior(
            marginal_loglik=jnp.asarray(-1.0),
            filtered_particles=jnp.array([[[0.0]], [[1.0]]]),
            filtered_log_weights=jnp.zeros((2, 1)),
            ancestors=jnp.zeros((2, 1), dtype=jnp.int32),
            ess=jnp.ones(2),
            log_evidence_increments=jnp.array([-0.25, -0.75]),
        )

        assert jnp.all(jnp.isnan(pareto_k_diagnostic(posterior)))
        summary = diagnose(posterior)
        assert jnp.isnan(summary["max_pareto_k"])
        assert any(
            "Pareto-k was undefined at 2 step(s)" in warning
            for warning in summary["warnings"]
        )

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
        te = float(tail_ess(posterior, q=Fraction(1, 20))[0])
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
        f64 = jnp.asarray(pf_post.marginal_loglik).dtype == jnp.float64
        if f64:
            assert float(result[-1]) == pytest.approx(
                float(pf_post.marginal_loglik), abs=1e-6
            )
        else:
            # The cumulative f32 reduction may differ by several ulps.
            assert float(result[-1]) == pytest.approx(
                float(pf_post.marginal_loglik), rel=1e-5
            )

    def test_cumulative_log_score_compensates_runner_cancellation(self):
        increments = jnp.array(
            [1e8, 1.0, -1e8],
            dtype=jnp.float32,
        )
        particles = jnp.zeros((1, 1), dtype=increments.dtype)

        def record(increment):
            return ParticleFilterRecord(
                particles,
                jnp.zeros(1, dtype=increments.dtype),
                jnp.zeros(1, dtype=jnp.int32),
                increment,
            )

        def initialize(time_index, emission, key):
            del time_index, key
            return particles, record(emission[0])

        def step(carry, time_index, emission, key):
            del time_index, key
            return carry, record(emission[0])

        posterior = run_particle_filter(
            jr.key(23),
            initialize,
            step,
            increments[:, None],
            store_history=False,
        )
        result = jax.jit(cumulative_log_score)(posterior)
        expected = _neumaier_prefix_oracle(np.asarray(increments))
        assert jnp.array_equal(result, jnp.asarray(expected))
        assert jnp.array_equal(result[-1], posterior.marginal_loglik)

    def test_cumulative_log_score_preserves_long_constant_prefixes(self):
        increments = jnp.full((100_000,), -300.1, dtype=jnp.float32)
        posterior = _posterior_for_increment_contract(increments)
        result = cumulative_log_score(posterior)
        expected = _neumaier_prefix_oracle(np.asarray(increments))
        assert jnp.array_equal(result, jnp.asarray(expected))
        assert jnp.array_equal(result[-1], posterior.marginal_loglik)

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

    def test_diagnose_returns_dict(self):
        """Diagnose returns a dict with expected keys."""
        pf_post = _make_posterior()
        result = diagnose(pf_post)
        assert "min_ess" in result
        assert "min_diversity" in result
        assert "max_pareto_k" in result
        assert "ess_below_threshold" in result
        assert "warnings" in result

    def test_pareto_k_threshold_adds_warning(self):
        pf_post = _make_posterior()
        quiet = diagnose(pf_post, pareto_k_threshold=float("inf"))
        forced = diagnose(pf_post, pareto_k_threshold=-1.0)

        assert len(forced["warnings"]) > len(quiet["warnings"])

    def test_diagnose_uses_finite_pareto_k_when_one_step_is_undefined(self):
        posterior = _make_posterior()
        log_weights = posterior.filtered_log_weights
        log_weights = log_weights.at[0].set(
            jnp.linspace(-1.0, 1.0, log_weights.shape[1])
        )
        log_weights = log_weights.at[1, -1].set(jnp.inf)
        posterior = posterior._replace(filtered_log_weights=log_weights)

        k_hat = pareto_k_diagnostic(posterior)
        summary = diagnose(posterior, pareto_k_threshold=float("inf"))

        assert jnp.array_equal(
            jnp.isfinite(k_hat),
            jnp.array([True, False, True]),
        )
        assert k_hat[2] > k_hat[0]
        assert summary["max_pareto_k"] == pytest.approx(float(k_hat[2]))
        assert any(
            "Pareto-k was undefined at 1 step(s)" in warning
            for warning in summary["warnings"]
        )

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
