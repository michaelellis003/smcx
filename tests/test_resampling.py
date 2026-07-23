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
