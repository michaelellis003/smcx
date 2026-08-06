# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""dlm_forecast_sample gates: joint Student-t coherence (#415)."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
F = jnp.asarray([1.0, 0.0])
W = jnp.asarray([[0.05, 0.01], [0.01, 0.1]])
M0 = jnp.asarray([0.2, -0.1])
C0 = jnp.asarray([[1.0, 0.1], [0.1, 0.5]])
Y = jnp.asarray([0.4, 0.9, 1.1, 1.6, 2.2])
DISCOUNT = 0.9

FILTERED_W = smcx.dlm_filter(
    M0, C0, G, F, Y, scale_free_transition_covariance=W, prior_shape=5.0
)
NUM_STEPS = 3
NUM_DRAWS = 15_000
PATHS = smcx.dlm_forecast_sample(
    jr.key(5),
    FILTERED_W,
    G,
    F,
    num_steps=NUM_STEPS,
    num_draws=NUM_DRAWS,
    scale_free_transition_covariance=W,
)
CLOSED = smcx.dlm_forecast(
    FILTERED_W,
    G,
    F,
    num_steps=NUM_STEPS,
    scale_free_transition_covariance=W,
)


def test_path_shapes():
    """Draw-major state and emission paths with the documented shapes."""
    assert PATHS.state_paths.shape == (NUM_DRAWS, NUM_STEPS, 2)
    assert PATHS.emission_paths.shape == (NUM_DRAWS, NUM_STEPS)


def test_emission_marginals_match_the_student_t_closed_form():
    """Per-horizon emission moments reproduce the dlm_forecast t's."""
    emissions = np.asarray(PATHS.emission_paths, dtype=np.float64)
    dof = np.asarray(CLOSED.scale_shapes, dtype=np.float64)
    for k in range(NUM_STEPS):
        location = float(CLOSED.observation_means[k])
        scale2 = float(CLOSED.observation_scales[k])
        t_var = scale2 * dof[k] / (dof[k] - 2.0)
        se_mean = np.sqrt(t_var / NUM_DRAWS)
        assert se_mean < 0.2 * np.sqrt(t_var)  # non-vacuity ceiling
        assert abs(emissions[:, k].mean() - location) < 5.0 * se_mean
        excess = 6.0 / (dof[k] - 4.0)
        se_var = t_var * np.sqrt((2.0 + excess) / NUM_DRAWS)
        assert se_var < 0.2 * t_var  # non-vacuity ceiling
        assert abs(emissions[:, k].var(ddof=1) - t_var) < 5.0 * se_var


def test_state_marginals_match_the_scaled_t_moments():
    """State variances equal S * R_tilde(k) * dof / (dof - 2)."""
    states = np.asarray(PATHS.state_paths, dtype=np.float64)
    dof = float(CLOSED.scale_shapes[0])
    scale = float(CLOSED.scale_estimates[0])
    for k in range(NUM_STEPS):
        target = (
            scale
            * np.diag(
                np.asarray(
                    CLOSED.state_scale_free_covariances[k], dtype=np.float64
                )
            )
            * dof
            / (dof - 2.0)
        )
        sample_var = states[:, k].var(axis=0, ddof=1)
        excess = 6.0 / (dof - 4.0)
        se_var = target * np.sqrt((2.0 + excess) / NUM_DRAWS)
        assert np.all(se_var < 0.2 * target)  # non-vacuity ceiling
        np.testing.assert_array_less(np.abs(sample_var - target), 5.0 * se_var)


def test_shared_variance_couples_the_horizons():
    """Squared innovations correlate across horizons (shared V)."""
    emissions = np.asarray(PATHS.emission_paths, dtype=np.float64)
    first = (emissions[:, 0] - emissions[:, 0].mean()) ** 2
    second = (emissions[:, 2] - emissions[:, 2].mean()) ** 2
    correlation = np.corrcoef(first, second)[0, 1]
    assert correlation > 5.0 / np.sqrt(NUM_DRAWS)


def test_variance_discount_decays_the_first_horizon_dof():
    """The k=1 emission variance carries the decayed dof."""
    variance_discount = 0.9
    filtered = smcx.dlm_filter(
        M0,
        C0,
        G,
        F,
        Y,
        scale_free_transition_covariance=W,
        prior_shape=40.0,
        variance_discount=variance_discount,
    )
    closed = smcx.dlm_forecast(
        filtered,
        G,
        F,
        num_steps=1,
        scale_free_transition_covariance=W,
        variance_discount=variance_discount,
    )
    paths = smcx.dlm_forecast_sample(
        jr.key(9),
        filtered,
        G,
        F,
        num_steps=1,
        num_draws=NUM_DRAWS,
        scale_free_transition_covariance=W,
        variance_discount=variance_discount,
    )
    dof = float(closed.scale_shapes[0])
    t_var = float(closed.observation_scales[0]) * dof / (dof - 2.0)
    sample_var = float(
        np.asarray(paths.emission_paths[:, 0], dtype=np.float64).var(ddof=1)
    )
    excess = 6.0 / (dof - 4.0)
    se_var = t_var * np.sqrt((2.0 + excess) / NUM_DRAWS)
    assert se_var < 0.2 * t_var  # non-vacuity ceiling
    assert abs(sample_var - t_var) < 5.0 * se_var


def test_supply_exactly_one_evolution_specification():
    """W and discount together, or neither, get the documented error."""
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dlm_forecast_sample(
            jr.key(0),
            FILTERED_W,
            G,
            F,
            num_steps=1,
            num_draws=8,
            scale_free_transition_covariance=W,
            discount=DISCOUNT,
        )
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dlm_forecast_sample(
            jr.key(0), FILTERED_W, G, F, num_steps=1, num_draws=8
        )


def _sample_with(num_steps=1, num_draws=8):
    return smcx.dlm_forecast_sample(
        jr.key(0),
        FILTERED_W,
        G,
        F,
        num_steps=num_steps,
        num_draws=num_draws,
        scale_free_transition_covariance=W,
    )


def test_count_boundary_matrix():
    """Both counts share the positive-integer contract."""
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _sample_with(num_steps=bad)
        with pytest.raises(ValueError, match="num_draws"):
            _sample_with(num_draws=bad)


def test_discount_path_marginals_match_the_closed_form():
    """The frozen-frontier discount path reproduces its Student-t."""
    filtered = smcx.dlm_filter(
        M0, C0, G, F, Y, discount=DISCOUNT, prior_shape=5.0
    )
    closed = smcx.dlm_forecast(filtered, G, F, num_steps=2, discount=DISCOUNT)
    paths = smcx.dlm_forecast_sample(
        jr.key(21),
        filtered,
        G,
        F,
        num_steps=2,
        num_draws=NUM_DRAWS,
        discount=DISCOUNT,
    )
    emissions = np.asarray(paths.emission_paths, dtype=np.float64)
    dof = np.asarray(closed.scale_shapes, dtype=np.float64)
    for k in range(2):
        location = float(closed.observation_means[k])
        t_var = float(closed.observation_scales[k]) * dof[k] / (dof[k] - 2.0)
        se_mean = np.sqrt(t_var / NUM_DRAWS)
        assert se_mean < 0.2 * np.sqrt(t_var)  # non-vacuity ceiling
        assert abs(emissions[:, k].mean() - location) < 5.0 * se_mean


def test_bad_transition_matrix_shape_is_rejected():
    """A mis-shaped transition matrix raises at the boundary."""
    with pytest.raises(ValueError, match="transition_matrix"):
        smcx.dlm_forecast_sample(
            jr.key(0),
            FILTERED_W,
            jnp.eye(3),
            F,
            num_steps=1,
            num_draws=8,
            scale_free_transition_covariance=W,
        )


def test_discount_above_one_is_rejected():
    """The sampler validates the discount like the filter."""
    with pytest.raises(ValueError, match="discount"):
        smcx.dlm_forecast_sample(
            jr.key(0),
            FILTERED_W,
            G,
            F,
            num_steps=1,
            num_draws=8,
            discount=1.5,
        )


def test_variance_discount_above_one_is_rejected():
    """The sampler validates the variance discount like the filter."""
    with pytest.raises(ValueError, match="variance_discount"):
        smcx.dlm_forecast_sample(
            jr.key(0),
            FILTERED_W,
            G,
            F,
            num_steps=1,
            num_draws=8,
            scale_free_transition_covariance=W,
            variance_discount=1.5,
        )


class TestDiscountedVarianceCoherence:
    """Discounted paths must reproduce the dlm_forecast marginals.

    The sampler drew the frontier state under the initial precision
    and never rescaled inherited deviations as the beta-gamma walk
    decayed it, so path variances undershot the closed Student-t
    forms whenever ``variance_discount < 1`` (2026-08-06 review,
    P1-6). Tolerances are five estimator standard errors with a
    non-vacuity ceiling (AGENTS.md D6/D7); the SE of a sample
    variance under a Student-t with ``dof > 4`` uses the exact
    fourth moment ``3 * sigma**4 * (dof - 2) / (dof - 4)``.
    """

    VARIANCE_DISCOUNT = 0.5
    STEPS = 2
    DRAWS = 200_000

    @classmethod
    def _run(cls):
        filtered = smcx.dlm_filter(
            M0,
            C0,
            G,
            F,
            Y,
            scale_free_transition_covariance=W,
            prior_shape=60.0,
        )
        closed = smcx.dlm_forecast(
            filtered,
            G,
            F,
            num_steps=cls.STEPS,
            scale_free_transition_covariance=W,
            variance_discount=cls.VARIANCE_DISCOUNT,
        )
        paths = smcx.dlm_forecast_sample(
            jr.key(11),
            filtered,
            G,
            F,
            num_steps=cls.STEPS,
            num_draws=cls.DRAWS,
            scale_free_transition_covariance=W,
            variance_discount=cls.VARIANCE_DISCOUNT,
        )
        return closed, paths

    def test_state_variances_match_within_five_se(self):
        closed, paths = self._run()
        dof = np.asarray(closed.scale_shapes, dtype=np.float64)
        scale = np.asarray(closed.scale_estimates, dtype=np.float64)
        states = np.asarray(paths.state_paths, dtype=np.float64)
        for k in range(self.STEPS):
            assert dof[k] > 4.0  # exact fourth moment exists
            cov = (
                np.asarray(
                    closed.state_scale_free_covariances[k], dtype=np.float64
                )
                * scale[k]
            )
            t_var = np.diag(cov) * dof[k] / (dof[k] - 2.0)
            fourth = 3.0 * t_var**2 * (dof[k] - 2.0) / (dof[k] - 4.0)
            se_var = np.sqrt((fourth - t_var**2) / self.DRAWS)
            assert np.all(se_var < 0.2 * t_var)  # non-vacuity ceiling
            sample_var = states[:, k].var(axis=0)
            np.testing.assert_array_less(
                np.abs(sample_var - t_var), 5.0 * se_var
            )

    def test_emission_variances_match_within_five_se(self):
        closed, paths = self._run()
        dof = np.asarray(closed.scale_shapes, dtype=np.float64)
        emissions = np.asarray(paths.emission_paths, dtype=np.float64)
        for k in range(self.STEPS):
            scale2 = float(closed.observation_scales[k])
            t_var = scale2 * dof[k] / (dof[k] - 2.0)
            fourth = 3.0 * t_var**2 * (dof[k] - 2.0) / (dof[k] - 4.0)
            se_var = np.sqrt((fourth - t_var**2) / self.DRAWS)
            assert se_var < 0.2 * t_var  # non-vacuity ceiling
            sample_var = emissions[:, k].var()
            assert abs(sample_var - t_var) < 5.0 * se_var
