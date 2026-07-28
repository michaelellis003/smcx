# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Conjugate DLM filter tests (West & Harrison 1997, ch. 4).

The primary gate is an exact rational-arithmetic reference of the
Normal-Inverse-Gamma recursion computed inside this file: any
formula drift in the implementation fails against exact expected
values, not tolerances.
"""

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smcx


def _rational_reference(m0, c0, w, n0, s0, observations, delta_v=None):
    """Run the scalar NIG recursion exactly (G = F = 1, V-tilde = 1).

    All inputs are Fractions; returns per-step exact
    (m_t, c_t, n_t, s_t, f_t, q_t, big_q_t) where big_q_t is the
    Student-t forecast scale S_{t-1} * q_t. ``delta_v`` applies the
    W&H variance discount, n_t = delta_v * n_{t-1} + 1, at every
    evolution (never before the first observation).
    """
    m, c, n, s = m0, c0, n0, s0
    steps = []
    for t, y in enumerate(observations):
        if t == 0:
            a, r = m, c
            n_minus = n
        else:
            a, r = m, c + w
            n_minus = n if delta_v is None else delta_v * n
        q = r + 1
        f = a
        e = y - f
        gain = r / q
        big_q = s * q
        m = a + gain * e
        c = r - gain * gain * q
        n = n_minus + 1
        s = s * (n_minus + e * e / (q * s)) / n
        steps.append((m, c, n, s, f, q, big_q, n_minus, e))
    return steps


def _student_t_logpdf(e, big_q, dof):
    """Exact Student-t log density of residual e at scale^2 big_q."""
    return (
        math.lgamma((dof + 1.0) / 2.0)
        - math.lgamma(dof / 2.0)
        - 0.5 * math.log(dof * math.pi * big_q)
        - (dof + 1.0) / 2.0 * math.log1p(e * e / (dof * big_q))
    )


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="frozen CPU/x64 arithmetic contract",
)
class TestExactRecursion:
    """The implementation reproduces the exact NIG recursion."""

    @pytest.mark.parametrize(
        "delta_v", [None, Fraction(9, 10)], ids=["constant-V", "discounted-V"]
    )
    def test_matches_rational_reference(self, delta_v):
        m0 = Fraction(1, 2)
        c0 = Fraction(3)
        w = Fraction(1, 4)
        n0 = Fraction(5)
        s0 = Fraction(2)
        observations = [Fraction(1), Fraction(-1, 2), Fraction(3, 4)]
        expected = _rational_reference(
            m0, c0, w, n0, s0, observations, delta_v=delta_v
        )

        posterior = smcx.dlm_filter(
            jnp.asarray([float(m0)]),
            jnp.asarray([[float(c0)]]),
            jnp.eye(1),
            jnp.asarray([1.0]),
            jnp.asarray([float(y) for y in observations]),
            scale_free_transition_covariance=jnp.asarray([[float(w)]]),
            prior_shape=float(n0),
            prior_scale=float(s0),
            variance_discount=1.0 if delta_v is None else float(delta_v),
        )

        for t, step in enumerate(expected):
            m, c, n, s, _f, _q, big_q, n_minus, e = step
            np.testing.assert_allclose(
                float(posterior.filtered_means[t, 0]), float(m), rtol=1e-13
            )
            np.testing.assert_allclose(
                float(posterior.filtered_scale_free_covariances[t, 0, 0]),
                float(c),
                rtol=1e-13,
            )
            # Discounted shapes are non-integer: one rounding of the
            # exact rational versus sequential float arithmetic.
            np.testing.assert_allclose(
                float(posterior.scale_shapes[t]), float(n), rtol=1e-14
            )
            np.testing.assert_allclose(
                float(posterior.scale_estimates[t]), float(s), rtol=1e-13
            )
            np.testing.assert_allclose(
                float(posterior.log_evidence_increments[t]),
                _student_t_logpdf(float(e), float(big_q), float(n_minus)),
                rtol=1e-12,
            )
        np.testing.assert_allclose(
            float(posterior.marginal_loglik),
            sum(
                _student_t_logpdf(float(e), float(big_q), float(n_minus))
                for (_, _, _, _, _, _, big_q, n_minus, e) in expected
            ),
            rtol=1e-12,
        )


class TestReductions:
    """Limiting cases recover established filters and identities."""

    def test_large_prior_shape_reduces_to_kalman(self):
        """As n_0 grows with S_0 = V, the filter approaches kalman_filter.

        The Inverse-Gamma prior concentrates on V, Student-t forecasts
        approach Gaussians, and the scaled moments approach the exact
        known-variance recursion.
        """
        variance = 0.49
        transition = jnp.asarray([[0.9, 0.1], [0.0, 0.8]])
        observation = jnp.asarray([1.0, 0.5])
        scale_free_w = jnp.asarray([[0.3, 0.05], [0.05, 0.2]])
        initial_mean = jnp.asarray([0.2, -0.1])
        scale_free_c0 = jnp.asarray([[1.0, 0.1], [0.1, 0.8]])
        emissions = jnp.asarray([0.3, -0.2, 0.5, 0.1])

        conjugate = smcx.dlm_filter(
            initial_mean,
            scale_free_c0,
            transition,
            observation,
            emissions,
            scale_free_transition_covariance=scale_free_w,
            prior_shape=1e8,
            prior_scale=variance,
        )
        exact = smcx.kalman_filter(
            initial_mean,
            variance * scale_free_c0,
            transition,
            variance * scale_free_w,
            observation[None, :],
            jnp.asarray([[variance]]),
            emissions[:, None],
        )

        np.testing.assert_allclose(
            np.asarray(conjugate.filtered_means),
            np.asarray(exact.filtered_means),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            np.asarray(conjugate.scale_estimates[-1]),
            variance,
            rtol=1e-6,
        )

    @pytest.mark.skipif(
        jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
        reason="frozen CPU/x64 arithmetic contract",
    )
    def test_discount_equals_its_implied_evolution_covariance(self):
        """A discount run and its implied explicit W-tilde must agree."""
        transition = jnp.asarray([[0.95]])
        observation = jnp.asarray([1.0])
        initial_mean = jnp.asarray([0.0])
        scale_free_c0 = jnp.asarray([[2.0]])
        emissions = jnp.asarray([0.4, -0.3, 0.7])
        delta = 0.9

        discounted = smcx.dlm_filter(
            initial_mean,
            scale_free_c0,
            transition,
            observation,
            emissions,
            discount=delta,
            prior_shape=3.0,
            prior_scale=1.5,
        )
        prior_covs = discounted.filtered_scale_free_covariances[:-1]
        implied = transition @ prior_covs @ transition.T * (1.0 - delta) / delta
        explicit = smcx.dlm_filter(
            initial_mean,
            scale_free_c0,
            transition,
            observation,
            emissions,
            scale_free_transition_covariance=implied,
            prior_shape=3.0,
            prior_scale=1.5,
        )

        for discount_field, explicit_field in zip(
            discounted, explicit, strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(discount_field),
                np.asarray(explicit_field),
                rtol=1e-12,
                atol=1e-13,
            )

    def test_kurit_table_first_two_steps(self):
        """West & Harrison's printed first-order polynomial table holds.

        Prior N(130, 400), V = 100, W = 5 (p. 40; the table is
        independently reproduced in Huerta's UNM course notes,
        https://www.math.unm.edu/~ghuerta/tseries/dlmch2.pdf). Known
        V is emulated with a concentrated Inverse-Gamma prior; table
        values are printed to about three figures.
        """
        # The book evolves before its first observation; smcx
        # conditions emissions[0] on the prior. Feeding the pre-evolved
        # W&H prior (C_1-pre = C_0 + W = 405) aligns the conventions.
        posterior = smcx.dlm_filter(
            jnp.asarray([130.0]),
            jnp.asarray([[4.05]]),  # (400 + 5) / V
            jnp.eye(1),
            jnp.asarray([1.0]),
            jnp.asarray([150.0, 136.0]),
            scale_free_transition_covariance=jnp.asarray([[0.05]]),  # 5 / V
            prior_shape=1e9,
            prior_scale=100.0,
        )

        scaled_c = (
            posterior.scale_estimates[:, None, None]
            * posterior.filtered_scale_free_covariances
        )
        np.testing.assert_allclose(
            np.asarray(posterior.filtered_means[:, 0]),
            [146.0, 141.4],
            atol=0.05,
        )
        np.testing.assert_allclose(
            np.asarray(scaled_c[:, 0, 0]), [80.0, 46.0], atol=0.5
        )


class TestBoundaries:
    """Entry validation and transform behavior."""

    def test_requires_exactly_one_evolution_specification(self):
        arrays = self._minimal_arrays()
        with pytest.raises(ValueError, match="exactly one"):
            smcx.dlm_filter(*arrays, prior_shape=2.0, prior_scale=1.0)
        with pytest.raises(ValueError, match="exactly one"):
            smcx.dlm_filter(
                *arrays,
                scale_free_transition_covariance=jnp.eye(1),
                discount=0.9,
                prior_shape=2.0,
                prior_scale=1.0,
            )

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("discount", 0.0, "discount"),
            ("discount", 1.5, "discount"),
            ("prior_shape", 0.0, "prior_shape"),
            ("prior_scale", -1.0, "prior_scale"),
            ("variance_discount", 0.0, "variance_discount"),
            ("variance_discount", 1.2, "variance_discount"),
        ],
    )
    def test_rejects_invalid_scalars(self, name, value, message):
        arrays = self._minimal_arrays()
        config = {"prior_shape": 2.0, "prior_scale": 1.0, "discount": 0.9}
        config[name] = value
        with pytest.raises(ValueError, match=message):
            smcx.dlm_filter(
                *arrays,
                **config,  # ty: ignore[invalid-argument-type]
            )

    def test_rejects_multivariate_emissions(self):
        with pytest.raises(ValueError, match="univariate"):
            smcx.dlm_filter(
                jnp.zeros(1),
                jnp.eye(1),
                jnp.eye(1),
                jnp.ones(1),
                jnp.zeros((3, 2)),
                discount=0.9,
                prior_shape=2.0,
                prior_scale=1.0,
            )

    def test_jit_matches_eager(self):
        arrays = self._minimal_arrays()

        def run(emissions):
            return smcx.dlm_filter(
                arrays[0],
                arrays[1],
                arrays[2],
                arrays[3],
                emissions,
                discount=0.9,
                prior_shape=2.0,
                prior_scale=1.0,
            )

        eager = run(arrays[4])
        compiled = jax.jit(run)(arrays[4])
        # Thirty-two eps of the working dtype covers eager/compiled
        # fusion differences on every backend, including f32 Metal.
        eps = float(np.finfo(np.asarray(eager[0]).dtype).eps)
        for eager_field, compiled_field in zip(eager, compiled, strict=True):
            np.testing.assert_allclose(
                np.asarray(eager_field),
                np.asarray(compiled_field),
                rtol=32 * eps,
                atol=32 * eps,
            )

    def test_gradient_with_respect_to_prior_scale_is_finite(self):
        arrays = self._minimal_arrays()

        def objective(prior_scale):
            return smcx.dlm_filter(
                arrays[0],
                arrays[1],
                arrays[2],
                arrays[3],
                arrays[4],
                discount=0.9,
                prior_shape=2.0,
                prior_scale=prior_scale,
            ).marginal_loglik

        gradient = jax.grad(objective)(1.0)
        assert np.isfinite(float(gradient))

    @staticmethod
    def _minimal_arrays():
        return (
            jnp.zeros(1),
            jnp.eye(1),
            jnp.eye(1),
            jnp.ones(1),
            jnp.asarray([0.2, -0.1, 0.3]),
        )


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="frozen CPU/x64 arithmetic contract",
)
def test_matches_pinned_pybats_reference():
    """dlm_filter reproduces PyBATS 0.0.5 (Apache-2.0, West's group).

    Frozen from pybats==0.0.5 under numpy 2.5.1 with a local-level DLM
    (a0=0.5, R0=3.0, n0=4, s0=1.5, deltrend=0.95, delVar=1): PyBATS
    carries the scaled covariance, so C0-tilde = R0/s0 and the scaled
    comparison is S_t * C_t-tilde. Agreement is asserted only at unit
    variance discount, where PyBATS's n-update ordering coincides with
    West and Harrison's (ADR-0024).
    """
    observations = jnp.asarray([0.8, -0.2, 1.1, 0.4, -0.6, 0.9])
    posterior = smcx.dlm_filter(
        jnp.asarray([0.5]),
        jnp.asarray([[2.0]]),  # R0 / s0
        jnp.eye(1),
        jnp.asarray([1.0]),
        observations,
        discount=0.95,
        prior_shape=4.0,
        prior_scale=1.5,
    )

    frozen = np.array([
        [0.7, 0.8039999999999999, 5.0, 1.206],
        [0.32886597938144324, 0.4471463492400892, 6.0, 1.0843298969072164],
        [0.5622777147181234, 0.2992545410202221, 7.0, 0.9886621898955587],
        [0.5230665841098208, 0.20963215770259658, 8.0, 0.8675757866620523],
        [0.29533952676692504, 0.17902859586406952, 9.0, 0.882903577471228],
        [0.4016992229376064, 0.1450722707351706, 10.0, 0.8247434980912007],
    ])
    scaled_c = (
        posterior.scale_estimates
        * posterior.filtered_scale_free_covariances[:, 0, 0]
    )

    np.testing.assert_allclose(
        np.asarray(posterior.filtered_means[:, 0]), frozen[:, 0], rtol=1e-10
    )
    np.testing.assert_allclose(np.asarray(scaled_c), frozen[:, 1], rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(posterior.scale_shapes), frozen[:, 2], rtol=0
    )
    np.testing.assert_allclose(
        np.asarray(posterior.scale_estimates), frozen[:, 3], rtol=1e-10
    )
