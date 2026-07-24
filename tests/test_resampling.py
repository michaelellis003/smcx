# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Contract and distributional tests for the native resamplers.

The variance checks follow Douc, Cappe, and Moulines (2005),
https://doi.org/10.1109/ISPA.2005.195385.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import config as jaxtyping_config

from smcx import multinomial, residual, stratified, systematic
from smcx.types import ResamplingFn

SCHEMES = [systematic, stratified, multinomial, residual]
SCHEME_IDS = ["systematic", "stratified", "multinomial", "residual"]


def _replicated_counts(
    resampler: ResamplingFn,
    weights: np.ndarray,
    num_samples: int,
    num_replicates: int,
) -> np.ndarray:
    """Return one offspring-count vector for each independent JAX key."""
    weights_jax = jnp.asarray(weights, dtype=jnp.float32)
    keys = jr.split(jr.PRNGKey(20260718), num_replicates)
    draw = jax.jit(
        jax.vmap(lambda key: resampler(key, weights_jax, num_samples))
    )
    ancestors = draw(keys)
    counts = jnp.sum(
        jax.nn.one_hot(ancestors, weights.size, dtype=jnp.int32), axis=1
    )
    return np.asarray(counts, dtype=np.float64)


class TestContract:
    """Structural contract shared by all resampling schemes."""

    def test_float32_cdf_is_monotone_with_exact_endpoint(self) -> None:
        from smcx.resampling import _normalized_cdf

        weights = jnp.exp(-jnp.linspace(0.0, 20.0, 512, dtype=jnp.float32))
        cdf = np.asarray(jax.jit(_normalized_cdf)(weights))

        assert np.all(np.diff(cdf) >= 0)
        assert cdf[-1] == np.float32(1.0)

    @pytest.mark.parametrize(
        ("resampler", "expected"),
        [
            (
                systematic,
                [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
            ),
            (
                stratified,
                [1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4],
            ),
            (
                multinomial,
                [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
            ),
            (
                residual,
                [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 0, 2],
            ),
        ],
        ids=SCHEME_IDS,
    )
    def test_monotone_repair_preserves_ordinary_seeded_draws(
        self,
        resampler: ResamplingFn,
        expected: list[int],
    ) -> None:
        """Keep the pre-repair stream where the original CDF was ordered."""
        weights = jnp.array(
            [0.05, 0.35, 0.10, 0.30, 0.20],
            dtype=jnp.float32,
        )

        with jax.enable_x64(False):
            ancestors = resampler(jr.key(155), weights, 12)

        np.testing.assert_array_equal(ancestors, expected)

    @pytest.mark.parametrize(
        ("resampler", "seed"),
        [
            (systematic, 5),
            (stratified, 123),
            (multinomial, 123),
        ],
        ids=SCHEME_IDS[:3],
    )
    def test_adversarial_float32_draw_matches_monotone_cdf_oracle(
        self,
        resampler: ResamplingFn,
        seed: int,
    ) -> None:
        from smcx.resampling import _below_one, _scale_by_max

        num_particles = 65_536
        weights = jnp.exp(
            -jnp.linspace(
                0.0,
                20.0,
                num_particles,
                dtype=jnp.float32,
            )
        )
        with jax.enable_x64(False):
            key = jr.key(seed)
            unrepaired = jnp.cumsum(_scale_by_max(weights))
            unrepaired = unrepaired / unrepaired[-1]
            oracle_cdf = np.maximum.accumulate(
                np.minimum(np.asarray(unrepaired), np.float32(1.0))
            )
            oracle_cdf[-1] = np.float32(1.0)

            if resampler is systematic:
                queries = jax.random.uniform(key) + jnp.arange(num_particles)
                queries = queries / num_particles
            elif resampler is stratified:
                queries = jnp.arange(num_particles) + jax.random.uniform(
                    key, (num_particles,)
                )
                queries = queries / num_particles
            else:
                spacings = -jnp.log1p(
                    -jax.random.uniform(key, (num_particles + 1,))
                )
                partial_sums = jax.lax.associative_scan(
                    jnp.maximum,
                    jnp.cumsum(spacings),
                )
                queries = partial_sums[:-1] / jnp.maximum(
                    partial_sums[-1],
                    1e-30,
                )
            queries = jnp.minimum(queries, _below_one(weights.dtype))
            expected = np.searchsorted(
                oracle_cdf,
                np.asarray(queries),
                side="right",
            )
            expected = np.clip(
                expected,
                0,
                num_particles - 1,
            ).astype(np.int32)

            draw = jax.jit(
                lambda draw_key, draw_weights: resampler(
                    draw_key,
                    draw_weights,
                    num_particles,
                )
            )
            actual = draw(key, weights)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (
                jnp.array([0.0, 0.0, 0.0]),
                "weights must have positive total mass",
            ),
            (
                jnp.array([-1.0, 2.0, 0.0]),
                "weights must be nonnegative",
            ),
            (
                jnp.array([jnp.nan, 1.0, 1.0]),
                "weights must contain only finite values",
            ),
            (
                jnp.array([jnp.inf, 1.0, 1.0]),
                "weights must contain only finite values",
            ),
        ],
    )
    def test_rejects_non_normalizable_weights(
        self,
        resampler: ResamplingFn,
        weights: jax.Array,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            resampler(jr.key(162), weights, 8)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    @pytest.mark.parametrize(
        ("weights", "num_samples", "message"),
        [
            (jnp.ones((2, 2)), 2, r"weights must have shape \(N,\)"),
            (jnp.empty((0,)), 2, "weights must contain at least one value"),
            (
                jnp.ones((2,), dtype=jnp.int32),
                2,
                "weights must have a floating dtype",
            ),
            (jnp.ones((2,)), 0, "num_samples must be >= 1"),
        ],
    )
    def test_rejects_malformed_structural_inputs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        resampler: ResamplingFn,
        weights: jax.Array,
        num_samples: int,
        message: str,
    ) -> None:
        monkeypatch.setattr(jaxtyping_config, "jaxtyping_disable", True)

        with pytest.raises(ValueError, match=message):
            resampler(jr.key(6), weights, num_samples)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    def test_shape_dtype_bounds_and_seeded_determinism(
        self, resampler: ResamplingFn
    ) -> None:
        weights = jnp.array([0.05, 0.35, 0.10, 0.30, 0.20])
        key = jr.PRNGKey(7)

        first = resampler(key, weights, 31)
        second = resampler(key, weights, 31)

        assert first.shape == (31,)
        assert first.dtype == jnp.int32
        assert bool(jnp.all((first >= 0) & (first < weights.size)))
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(np.ones(512), id="uniform"),
            pytest.param(
                np.concatenate(([1.0], np.full(511, 1e-3))),
                id="skewed",
            ),
        ],
    )
    def test_rejects_precision_below_float32(
        self,
        resampler: ResamplingFn,
        dtype,
        values: np.ndarray,
    ) -> None:
        weights = jnp.asarray(values, dtype=dtype)

        with pytest.raises(ValueError, match="at least float32 precision"):
            resampler(jr.key(156), weights, weights.size)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    def test_zero_weight_particles_are_never_selected(
        self, resampler: ResamplingFn
    ) -> None:
        weights = jnp.array([0.5, 0.0, 0.25, 0.0, 0.25])
        ancestors = np.asarray(resampler(jr.PRNGKey(8), weights, 256))

        assert not np.isin(ancestors, [1, 3]).any()

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    def test_weights_accept_any_positive_scale(
        self, resampler: ResamplingFn
    ) -> None:
        weights = jnp.array([0.03, 0.11, 0.17, 0.29, 0.40])
        key = jr.PRNGKey(81)

        normalized = resampler(key, weights, 41)
        scaled = resampler(key, 13.0 * weights, 41)

        np.testing.assert_array_equal(scaled, normalized)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    def test_tiny_positive_scale_preserves_same_key_draw(
        self, resampler: ResamplingFn
    ) -> None:
        """Normalization must not replace a valid sub-1e-30 total."""
        weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
        tiny_weights = jnp.float32(1e-31) * weights
        key = jr.PRNGKey(82)

        ordinary = resampler(key, weights, 257)
        tiny = resampler(key, tiny_weights, 257)

        np.testing.assert_array_equal(tiny, ordinary)

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    def test_large_finite_scale_preserves_same_key_draw(
        self, resampler: ResamplingFn
    ) -> None:
        """Normalization must not overflow a valid finite f32 total."""
        weights = jnp.array([1.0, 1.0], dtype=jnp.float32)
        large_weights = jnp.float32(2e38) * weights
        key = jr.PRNGKey(83)

        ordinary = resampler(key, weights, 257)
        large = resampler(key, large_weights, 257)

        np.testing.assert_array_equal(large, ordinary)

    def test_public_systematic_clamps_rounded_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The public query construction, not just its helper, clamps 1."""

        def endpoint_uniform(key):
            del key
            return jnp.array(1.0, dtype=jnp.float32)

        monkeypatch.setattr(jax.random, "uniform", endpoint_uniform)
        ancestor = systematic(
            jr.PRNGKey(80),
            jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
            1,
        )

        np.testing.assert_array_equal(ancestor, np.array([0]))

    def test_rounded_endpoint_never_selects_trailing_zero_mass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def endpoint_uniform(key):
            del key
            return jnp.array(1.0, dtype=jnp.float32)

        monkeypatch.setattr(jax.random, "uniform", endpoint_uniform)
        positive = jnp.exp(-jnp.linspace(0.0, 20.0, 512, dtype=jnp.float32))
        weights = jnp.concatenate([
            positive,
            jnp.zeros((2,), dtype=jnp.float32),
        ])

        ancestor = systematic(jr.key(155), weights, 1)

        assert float(weights[ancestor[0]]) > 0

    @pytest.mark.skipif(
        not jax.config.read("jax_enable_x64"),
        reason="float64 endpoint contract",
    )
    def test_public_systematic_preserves_float64_tail_mass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The endpoint guard must not hide valid float64 tail mass."""

        def tail_uniform(key):
            del key
            return jnp.array(0.99999999, dtype=jnp.float64)

        monkeypatch.setattr(jax.random, "uniform", tail_uniform)
        ancestor = systematic(
            jr.PRNGKey(84),
            jnp.array([0.99999998, 0.00000002], dtype=jnp.float64),
            1,
        )

        np.testing.assert_array_equal(ancestor, np.array([1]))

    def test_systematic_uniform_weights_select_every_particle_once(
        self,
    ) -> None:
        weights = jnp.full((64,), 1.0 / 64)
        ancestors = systematic(jr.PRNGKey(9), weights, 64)

        np.testing.assert_array_equal(ancestors, np.arange(64))

    def test_multinomial_large_output_remains_nondecreasing(self) -> None:
        """Parallel f32 prefix rounding must not invert ordered queries."""
        num_particles = 100_000
        weights = jnp.exp(
            -jnp.linspace(
                0.0,
                5.0,
                num_particles,
                dtype=jnp.float32,
            )
        )
        weights = weights / jnp.sum(weights)
        # The fifth committed validation key exposed a one-index inversion at
        # N=100,000. This is a deterministic public ordering contract, so the
        # failing key is retained rather than re-rolled.
        key = jr.split(jr.key(20260720), 8)[4]
        with jax.enable_x64(False):
            draw = jax.jit(
                lambda draw_key, draw_weights: multinomial(
                    draw_key,
                    draw_weights,
                    num_particles,
                )
            )
            ancestors = np.asarray(draw(key, weights))

        assert np.all(np.diff(ancestors.astype(np.int64)) >= 0)

    def test_residual_guarantees_the_deterministic_floor(self) -> None:
        # Dyadic weights are exact in f32, so backend-specific reduction
        # rounding cannot move an expected count across an integer boundary.
        weights = np.array([0.53125, 0.28125, 0.1875])
        # floor(4 * weights) is exactly [2, 1, 0].
        counts = _replicated_counts(residual, weights, 4, 512)

        assert np.all(counts >= np.array([2.0, 1.0, 0.0]))
        assert np.all(counts.sum(axis=1) == 4)


class TestOffspringMoments:
    """Distributional identities with five-standard-error gates."""

    @pytest.mark.parametrize("resampler", SCHEMES, ids=SCHEME_IDS)
    def test_expected_counts(self, resampler: ResamplingFn) -> None:
        weights = np.array([0.03, 0.11, 0.17, 0.29, 0.40])
        # E[counts] = M * weights for every unbiased scheme, here M=17.
        expected = np.array([0.51, 1.87, 2.89, 4.93, 6.80])
        counts = _replicated_counts(resampler, weights, 17, 5_000)

        observed = counts.mean(axis=0)
        # For independent committed-seed replicates, the estimator SE is
        # the sample SD / sqrt(K). Five SE is the repository's prescribed
        # Monte-Carlo-error-honest tolerance; 1e-6 covers f32 weights.
        estimator_se = counts.std(axis=0, ddof=1) / np.sqrt(counts.shape[0])
        np.testing.assert_array_less(
            np.abs(observed - expected), 5 * estimator_se + 1e-6
        )

    @pytest.mark.parametrize(
        ("resampler", "expected_covariance"),
        [
            (
                systematic,
                np.array([
                    [0.16, 0.0, -0.16],
                    [0.0, 0.0, 0.0],
                    [-0.16, 0.0, 0.16],
                ]),
            ),
            (
                stratified,
                np.array([
                    [0.16, -0.16, 0.0],
                    [-0.16, 0.32, -0.16],
                    [0.0, -0.16, 0.16],
                ]),
            ),
            (
                multinomial,
                np.array([
                    [0.99, -0.55, -0.44],
                    [-0.55, 0.75, -0.20],
                    [-0.44, -0.20, 0.64],
                ]),
            ),
            (
                residual,
                np.array([
                    [0.109375, -0.015625, -0.09375],
                    [-0.015625, 0.109375, -0.09375],
                    [-0.09375, -0.09375, 0.1875],
                ]),
            ),
        ],
        ids=SCHEME_IDS,
    )
    def test_count_covariance(
        self,
        resampler: ResamplingFn,
        expected_covariance: np.ndarray,
    ) -> None:
        weights = np.array([0.55, 0.25, 0.20])
        expected_mean = np.array([2.20, 1.00, 0.80])
        if resampler is residual:
            # Exact-f32 fixture: floor(4w)=[2,1,0], leaving one categorical
            # remainder draw with probabilities [.125, .125, .75].
            weights = np.array([0.53125, 0.28125, 0.1875])
            expected_mean = np.array([2.125, 1.125, 0.75])
        counts = _replicated_counts(resampler, weights, 4, 10_000)

        # On this fixture, systematic has one Bernoulli(0.2) boundary
        # crossing; stratified has two independent Bernoulli(0.2)
        # crossings; residual has the categorical remainder described above.
        # Multinomial uses M * (diag(w) - outer(w, w)). These identities
        # give the hard-coded matrices above without an outside package.
        centered_products = (counts - expected_mean)[:, :, None] * (
            counts - expected_mean
        )[:, None, :]
        observed = centered_products.mean(axis=0)
        # Each covariance entry is a mean of centered products, so its
        # estimator SE is the product SD / sqrt(K).
        estimator_se = centered_products.std(axis=0, ddof=1) / np.sqrt(
            counts.shape[0]
        )
        np.testing.assert_array_less(
            np.abs(observed - expected_covariance),
            5 * estimator_se + 1e-6,
        )
