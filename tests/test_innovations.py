# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""innovations diagnostics gates: identity, normality, masks (#435)."""

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

MU0 = jnp.asarray([0.5, -0.25])
P0 = jnp.asarray([[1.0, 0.2], [0.2, 0.8]])
A = jnp.asarray([[0.9, 0.1], [-0.05, 0.85]])
Q = jnp.asarray([[0.3, 0.05], [0.05, 0.4]])
H = jnp.asarray([[1.0, 0.0], [0.5, 1.0]])
R = jnp.asarray([[0.5, 0.1], [0.1, 0.6]])
Y = jnp.asarray([
    [0.3, -0.1],
    [0.6, 0.2],
    [-0.4, 0.9],
    [0.1, 0.05],
])

MODEL = smcx.LinearGaussianModel(
    initial_mean=MU0,
    initial_covariance=P0,
    transition_matrix=A,
    transition_covariance=Q,
    observation_matrix=H,
    observation_covariance=R,
)


def _rtol(array):
    return 1e-10 if array.dtype == jnp.float64 else 2e-4


def _assemble_increments(result):
    """Rebuild the Gaussian log density from the returned pieces."""
    standardized = np.asarray(result.standardized, dtype=np.float64)
    scales = np.asarray(result.scales, dtype=np.float64)
    observed = ~np.isnan(standardized)
    increments = []
    for w_row, s_row, o_row in zip(standardized, scales, observed, strict=True):
        increments.append(
            -0.5
            * (
                o_row.sum() * math.log(2.0 * math.pi)
                + 2.0 * np.log(s_row[o_row]).sum()
                + (w_row[o_row] ** 2).sum()
            )
        )
    return np.asarray(increments)


def test_increment_identity_on_the_covariance_record():
    """The returned pieces reassemble log_evidence_increments."""
    posterior = smcx.kalman_filter(MODEL, Y)
    result = smcx.innovations(posterior, MODEL, Y)
    rtol = _rtol(posterior.filtered_means)
    np.testing.assert_allclose(
        _assemble_increments(result),
        np.asarray(posterior.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )


def test_increment_identity_on_the_sqrt_record():
    """The square-root record supports the same identity."""
    posterior = smcx.sqrt_kalman_filter(MODEL, Y)
    result = smcx.innovations(posterior, MODEL, Y)
    rtol = 1e-8 if result.standardized.dtype == jnp.float64 else 5e-4
    np.testing.assert_allclose(
        _assemble_increments(result),
        np.asarray(posterior.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )


def test_record_path_matches_array_path_bitwise():
    """Model record and loose arrays produce identical output."""
    posterior = smcx.kalman_filter(MODEL, Y)
    from_record = smcx.innovations(posterior, MODEL, Y)
    from_arrays = smcx.innovations(posterior, H, R, Y)
    np.testing.assert_array_equal(
        np.asarray(from_record.standardized),
        np.asarray(from_arrays.standardized),
    )
    np.testing.assert_array_equal(
        np.asarray(from_record.scales), np.asarray(from_arrays.scales)
    )


def test_standardized_innovations_are_iid_standard_normal():
    """On simulated data the whitened innovations pass moment checks."""
    num_timesteps = 4000
    key = jr.key(3)
    _states, observations = smcx.simulate(
        key,
        lambda k: MU0 + jnp.linalg.cholesky(P0) @ jr.normal(k, (2,)),
        lambda k, state: (
            A @ state + jnp.linalg.cholesky(Q) @ jr.normal(k, (2,))
        ),
        lambda k, state: (
            H @ state + jnp.linalg.cholesky(R) @ jr.normal(k, (2,))
        ),
        num_timesteps=num_timesteps,
    )
    posterior = smcx.kalman_filter(MODEL, observations)
    result = smcx.innovations(posterior, MODEL, observations)
    values = np.asarray(result.standardized, dtype=np.float64).reshape(-1)
    count = values.shape[0]
    assert abs(values.mean()) < 6.0 / np.sqrt(count)
    assert abs(values.var(ddof=1) - 1.0) < 6.0 * np.sqrt(2.0 / count)
    lag_one = np.corrcoef(values[:-1], values[1:])[0, 1]
    assert abs(lag_one) < 6.0 / np.sqrt(count)


def test_masked_components_are_nan_and_observed_ones_standard():
    """Partial rows yield NaN at masked positions, valid elsewhere."""
    emissions = Y.at[1, 0].set(jnp.nan).at[2].set(jnp.nan)
    posterior = smcx.kalman_filter(MODEL, emissions)
    result = smcx.innovations(posterior, MODEL, emissions)
    standardized = np.asarray(result.standardized)
    scales = np.asarray(result.scales)
    assert np.isnan(standardized[1, 0]) and np.isnan(scales[1, 0])
    assert np.isfinite(standardized[1, 1]) and np.isfinite(scales[1, 1])
    assert np.all(np.isnan(standardized[2])) and np.all(np.isnan(scales[2]))
    rtol = _rtol(posterior.filtered_means)
    np.testing.assert_allclose(
        _assemble_increments(result),
        np.asarray(posterior.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )


def test_biases_and_inputs_enter_the_forecast_mean():
    """Observation offsets shift the innovations exactly."""
    bias = jnp.asarray([0.05, 0.15])
    inputs = jnp.asarray([[0.5], [-1.0], [0.25], [0.75]])
    input_matrix = jnp.asarray([[0.2], [-0.3]])
    posterior = smcx.kalman_filter(
        MU0,
        P0,
        A,
        Q,
        H,
        R,
        Y,
        observation_bias=bias,
        observation_input_matrix=input_matrix,
        inputs=inputs,
    )
    result = smcx.innovations(
        posterior,
        H,
        R,
        Y,
        observation_bias=bias,
        observation_input_matrix=input_matrix,
        inputs=inputs,
    )
    rtol = _rtol(posterior.filtered_means)
    np.testing.assert_allclose(
        _assemble_increments(result),
        np.asarray(posterior.log_evidence_increments),
        rtol=rtol,
        atol=rtol,
    )


def test_shape_mismatch_is_rejected():
    """Emissions not matching the record raise at the boundary."""
    posterior = smcx.kalman_filter(MODEL, Y)
    with pytest.raises(ValueError, match="emissions"):
        smcx.innovations(posterior, MODEL, Y[:-1])


class TestDLMInnovations:
    """The univariate Student-t analogue."""

    G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
    F = jnp.asarray([1.0, 0.0])
    W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
    M0 = jnp.asarray([0.2, -0.1])
    C0 = jnp.asarray([[1.0, 0.1], [0.1, 0.5]])
    Y1 = jnp.asarray([0.4, 0.9, 1.1, 1.6, 2.2])

    def _filter(self, emissions):
        return smcx.dlm_filter(
            self.M0,
            self.C0,
            self.G,
            self.F,
            emissions,
            scale_free_transition_covariance=self.W,
            prior_shape=5.0,
        )

    def test_increment_identity(self):
        """Student-t densities from the pieces rebuild the increments."""
        from scipy import stats

        posterior = self._filter(self.Y1)
        result = smcx.dlm_innovations(
            posterior,
            self.M0,
            self.C0,
            self.G,
            self.F,
            self.Y1,
            scale_free_transition_covariance=self.W,
            prior_shape=5.0,
        )
        standardized = np.asarray(result.standardized, dtype=np.float64)
        scales = np.asarray(result.scales, dtype=np.float64)
        dofs = np.asarray(result.dofs, dtype=np.float64)
        rebuilt = stats.t.logpdf(standardized, df=dofs) - np.log(scales)
        rtol = 1e-9 if result.standardized.dtype == jnp.float64 else 1e-4
        np.testing.assert_allclose(
            rebuilt,
            np.asarray(posterior.log_evidence_increments),
            rtol=rtol,
        )

    def test_missing_rows_are_nan(self):
        """An all-NaN datum yields NaN innovations, others valid."""
        emissions = self.Y1.at[2].set(jnp.nan)
        posterior = self._filter(emissions)
        result = smcx.dlm_innovations(
            posterior,
            self.M0,
            self.C0,
            self.G,
            self.F,
            emissions,
            scale_free_transition_covariance=self.W,
            prior_shape=5.0,
        )
        assert np.isnan(np.asarray(result.standardized)[2])
        finite = np.delete(np.asarray(result.standardized), 2)
        assert np.all(np.isfinite(finite))


class TestDLMInnovationsBoundaries:
    """Validation matrix for the DLM analogue."""

    G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
    F = jnp.asarray([1.0, 0.0])
    W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
    M0 = jnp.asarray([0.2, -0.1])
    C0 = jnp.asarray([[1.0, 0.1], [0.1, 0.5]])
    Y1 = jnp.asarray([0.4, 0.9, 1.1, 1.6, 2.2])

    def _posterior(self, **kwargs):
        return smcx.dlm_filter(
            self.M0,
            self.C0,
            self.G,
            self.F,
            self.Y1,
            **kwargs,
        )

    def test_discount_path_identity(self):
        """The discount specification supports the same identity."""
        from scipy import stats

        posterior = self._posterior(discount=0.9, prior_shape=5.0)
        result = smcx.dlm_innovations(
            posterior,
            self.M0,
            self.C0,
            self.G,
            self.F,
            self.Y1,
            discount=0.9,
            prior_shape=5.0,
        )
        rebuilt = stats.t.logpdf(
            np.asarray(result.standardized, dtype=np.float64),
            df=np.asarray(result.dofs, dtype=np.float64),
        ) - np.log(np.asarray(result.scales, dtype=np.float64))
        rtol = 1e-9 if result.standardized.dtype == jnp.float64 else 1e-4
        np.testing.assert_allclose(
            rebuilt,
            np.asarray(posterior.log_evidence_increments),
            rtol=rtol,
        )

    def test_boundary_matrix(self):
        """The shared validation raises with the documented messages."""
        posterior = self._posterior(scale_free_transition_covariance=self.W)
        with pytest.raises(ValueError, match="exactly one"):
            smcx.dlm_innovations(
                posterior, self.M0, self.C0, self.G, self.F, self.Y1
            )
        with pytest.raises(ValueError, match="univariate"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                self.C0,
                self.G,
                self.F,
                jnp.zeros((5, 2)),
                scale_free_transition_covariance=self.W,
            )
        with pytest.raises(ValueError, match="one row per stored step"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                self.C0,
                self.G,
                self.F,
                self.Y1[:-1],
                scale_free_transition_covariance=self.W,
            )
        with pytest.raises(ValueError, match="transition_matrix"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                self.C0,
                jnp.eye(3),
                self.F,
                self.Y1,
                scale_free_transition_covariance=self.W,
            )
        with pytest.raises(ValueError, match="discount"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                self.C0,
                self.G,
                self.F,
                self.Y1,
                discount=1.5,
            )
        with pytest.raises(ValueError, match="variance_discount"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                self.C0,
                self.G,
                self.F,
                self.Y1,
                scale_free_transition_covariance=self.W,
                variance_discount=1.5,
            )


def test_gaussian_boundary_matrix():
    """The record-or-arrays resolver raises its documented errors."""
    from typing import Any

    posterior = smcx.kalman_filter(MODEL, Y)
    with pytest.raises(ValueError, match="record only"):
        smcx.innovations(posterior, MODEL, Y, observation_bias=jnp.zeros(2))
    with pytest.raises(ValueError, match="pass them once"):
        smcx.innovations(posterior, MODEL, Y, emissions=Y)
    with pytest.raises(ValueError, match="LinearGaussianModel"):
        smcx.innovations(posterior, H, emissions=Y)
    non_record: Any = object()
    with pytest.raises(ValueError, match="GaussianFilterPosterior"):
        smcx.innovations(non_record, MODEL, Y)
    with pytest.raises(ValueError, match="input matrices require inputs"):
        smcx.innovations(
            posterior,
            H,
            R,
            Y,
            observation_input_matrix=jnp.asarray([[0.2], [-0.3]]),
        )


class TestDLMInnovationsReviewGaps:
    """Boundary gaps from the 2026-08-06 pre-release review (P1-7, P2-1)."""

    G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
    F = jnp.asarray([1.0, 0.0])
    W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
    M0 = jnp.asarray([0.2, -0.1])
    C0 = jnp.asarray([[1.0, 0.1], [0.1, 0.5]])
    Y1 = jnp.asarray([0.4, 0.9, 1.1, 1.6, 2.2])

    def test_large_prior_scale_keeps_the_identity(self):
        """The scale as a product of square roots survives 2e38.

        Forming the product before the root overflowed float32 and
        broke the evidence identity.
        """
        from scipy import stats

        args = [
            jnp.asarray(value, dtype=jnp.float32)
            for value in (self.M0, self.C0, self.G, self.F, self.Y1)
        ]
        kwargs = {
            "scale_free_transition_covariance": jnp.asarray(
                self.W, dtype=jnp.float32
            ),
            "prior_shape": 5.0,
            "prior_scale": 2e38,
        }
        posterior = smcx.dlm_filter(*args[:4], args[4], **kwargs)
        assert np.isfinite(float(posterior.marginal_loglik))
        result = smcx.dlm_innovations(posterior, *args[:4], args[4], **kwargs)
        scales = np.asarray(result.scales, dtype=np.float64)
        assert np.all(np.isfinite(scales))
        rebuilt = stats.t.logpdf(
            np.asarray(result.standardized, dtype=np.float64),
            df=np.asarray(result.dofs, dtype=np.float64),
        ) - np.log(scales)
        np.testing.assert_allclose(
            rebuilt,
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            rtol=1e-4,
        )

    def test_timed_observation_vectors_replay(self):
        """A run filtered with a timed F history replays exactly."""
        from scipy import stats

        timed_f = jnp.asarray([
            [1.0, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        posterior = smcx.dlm_filter(
            self.M0,
            self.C0,
            self.G,
            timed_f,
            self.Y1,
            scale_free_transition_covariance=self.W,
            prior_shape=5.0,
        )
        result = smcx.dlm_innovations(
            posterior,
            self.M0,
            self.C0,
            self.G,
            timed_f,
            self.Y1,
            scale_free_transition_covariance=self.W,
            prior_shape=5.0,
        )
        rebuilt = stats.t.logpdf(
            np.asarray(result.standardized, dtype=np.float64),
            df=np.asarray(result.dofs, dtype=np.float64),
        ) - np.log(np.asarray(result.scales, dtype=np.float64))
        np.testing.assert_allclose(
            rebuilt,
            np.asarray(posterior.log_evidence_increments),
            rtol=1e-9,
        )

    def test_infinite_emissions_are_rejected(self):
        posterior = smcx.dlm_filter(
            self.M0,
            self.C0,
            self.G,
            self.F,
            self.Y1,
            scale_free_transition_covariance=self.W,
        )
        with pytest.raises(ValueError, match="finite"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                self.C0,
                self.G,
                self.F,
                self.Y1.at[1].set(jnp.inf),
                scale_free_transition_covariance=self.W,
            )

    def test_initial_covariance_domain_is_checked(self):
        posterior = smcx.dlm_filter(
            self.M0,
            self.C0,
            self.G,
            self.F,
            self.Y1,
            scale_free_transition_covariance=self.W,
        )
        with pytest.raises(ValueError, match="initial_scale_free_covariance"):
            smcx.dlm_innovations(
                posterior,
                self.M0,
                -self.C0,
                self.G,
                self.F,
                self.Y1,
                scale_free_transition_covariance=self.W,
            )
