# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""dglm_forecast_sample gates: honest path approximation (#415)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

G = jnp.asarray([[1.0, 1.0], [0.0, 1.0]])
F = jnp.asarray([1.0, 0.0])
W = jnp.asarray([[0.02, 0.005], [0.005, 0.04]])
M0 = jnp.asarray([0.3, 0.05])
C0 = jnp.asarray([[0.3, 0.05], [0.05, 0.1]])
Y_COUNTS = jnp.asarray([2.0, 1.0, 4.0, 3.0, 5.0])

FILTERED = smcx.dglm_filter(
    M0,
    C0,
    G,
    F,
    Y_COUNTS,
    family=smcx.poisson(),
    transition_covariance=W,
)


def _poisson_sample_emission(key, linear_predictor):
    """The Poisson log-link commitment (the docstring recipe)."""
    draw = jr.poisson(key, jnp.exp(linear_predictor))
    return jnp.asarray(draw, dtype=linear_predictor.dtype)


def _bernoulli_sample_emission(key, linear_predictor):
    """The Bernoulli logit-link commitment."""
    draw = jr.bernoulli(key, jax.nn.sigmoid(linear_predictor))
    return jnp.asarray(draw, dtype=linear_predictor.dtype)


NUM_STEPS = 3
NUM_DRAWS = 40_000
PATHS = smcx.dglm_forecast_sample(
    jr.key(13),
    FILTERED,
    G,
    F,
    sample_emission=_poisson_sample_emission,
    num_steps=NUM_STEPS,
    num_draws=NUM_DRAWS,
    transition_covariance=W,
)
CLOSED = smcx.dglm_forecast(
    FILTERED,
    G,
    F,
    family=smcx.poisson(),
    num_steps=NUM_STEPS,
    transition_covariance=W,
)


def test_path_shapes():
    """Draw-major state and emission paths with the documented shapes."""
    assert PATHS.state_paths.shape == (NUM_DRAWS, NUM_STEPS, 2)
    assert PATHS.emission_paths.shape == (NUM_DRAWS, NUM_STEPS)


def test_state_marginals_match_the_closed_form_within_se():
    """Per-horizon state moments reproduce dglm_forecast's Gaussians."""
    states = np.asarray(PATHS.state_paths, dtype=np.float64)
    for k in range(NUM_STEPS):
        mean = np.asarray(CLOSED.state_means[k], dtype=np.float64)
        cov = np.asarray(CLOSED.state_covariances[k], dtype=np.float64)
        se_mean = np.sqrt(np.diag(cov) / NUM_DRAWS)
        np.testing.assert_array_less(
            np.abs(states[:, k].mean(axis=0) - mean), 6.0 * se_mean
        )
        se_var = np.diag(cov) * np.sqrt(2.0 / NUM_DRAWS)
        np.testing.assert_array_less(
            np.abs(states[:, k].var(axis=0, ddof=1) - np.diag(cov)),
            6.0 * se_var,
        )


def test_poisson_emission_means_match_the_lognormal_mixture():
    """Count means equal exp(f + q/2), the log-link mixture mean."""
    emissions = np.asarray(PATHS.emission_paths, dtype=np.float64)
    for k in range(NUM_STEPS):
        predictor_mean = float(CLOSED.linear_predictor_means[k])
        predictor_var = float(CLOSED.linear_predictor_variances[k])
        mixture_mean = np.exp(predictor_mean + predictor_var / 2.0)
        mixture_var = mixture_mean + (np.exp(predictor_var) - 1.0) * np.exp(
            2.0 * predictor_mean + predictor_var
        )
        se = np.sqrt(mixture_var / NUM_DRAWS)
        assert abs(emissions[:, k].mean() - mixture_mean) < 6.0 * se


def test_dispersion_discount_inflates_the_predictor_variance():
    """Extra-dispersion shocks reproduce the closed-form inflation."""
    rho = 0.8
    filtered = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        Y_COUNTS,
        family=smcx.poisson(),
        transition_covariance=W,
        dispersion_discount=rho,
    )
    closed = smcx.dglm_forecast(
        filtered,
        G,
        F,
        family=smcx.poisson(),
        num_steps=2,
        transition_covariance=W,
        dispersion_discount=rho,
    )
    paths = smcx.dglm_forecast_sample(
        jr.key(17),
        filtered,
        G,
        F,
        sample_emission=_poisson_sample_emission,
        num_steps=2,
        num_draws=NUM_DRAWS,
        transition_covariance=W,
        dispersion_discount=rho,
    )
    emissions = np.asarray(paths.emission_paths, dtype=np.float64)
    for k in range(2):
        predictor_mean = float(closed.linear_predictor_means[k])
        predictor_var = float(closed.linear_predictor_variances[k])
        mixture_mean = np.exp(predictor_mean + predictor_var / 2.0)
        mixture_var = mixture_mean + (np.exp(predictor_var) - 1.0) * np.exp(
            2.0 * predictor_mean + predictor_var
        )
        se = np.sqrt(mixture_var / NUM_DRAWS)
        assert abs(emissions[:, k].mean() - mixture_mean) < 6.0 * se


def test_discount_path_state_marginals_match_the_closed_form():
    """The frozen-frontier discount path reproduces its Gaussians."""
    filtered = smcx.dglm_filter(
        M0, C0, G, F, Y_COUNTS, family=smcx.poisson(), discount=0.9
    )
    closed = smcx.dglm_forecast(
        filtered, G, F, family=smcx.poisson(), num_steps=2, discount=0.9
    )
    paths = smcx.dglm_forecast_sample(
        jr.key(23),
        filtered,
        G,
        F,
        sample_emission=_poisson_sample_emission,
        num_steps=2,
        num_draws=NUM_DRAWS,
        discount=0.9,
    )
    states = np.asarray(paths.state_paths, dtype=np.float64)
    for k in range(2):
        mean = np.asarray(closed.state_means[k], dtype=np.float64)
        cov = np.asarray(closed.state_covariances[k], dtype=np.float64)
        se_mean = np.sqrt(np.diag(cov) / NUM_DRAWS)
        np.testing.assert_array_less(
            np.abs(states[:, k].mean(axis=0) - mean), 6.0 * se_mean
        )


def test_bernoulli_link_paths_sample_binary_emissions():
    """The logit-link recipe yields {0, 1} emissions."""
    emissions = jnp.asarray([1.0, 0.0, 1.0, 1.0, 0.0])
    filtered = smcx.dglm_filter(
        M0,
        C0,
        G,
        F,
        emissions,
        family=smcx.bernoulli(),
        transition_covariance=W,
    )
    paths = smcx.dglm_forecast_sample(
        jr.key(29),
        filtered,
        G,
        F,
        sample_emission=_bernoulli_sample_emission,
        num_steps=2,
        num_draws=256,
        transition_covariance=W,
    )
    values = np.asarray(paths.emission_paths)
    assert set(np.unique(values)).issubset({0.0, 1.0})


def test_supply_exactly_one_evolution_specification():
    """W and discount together, or neither, get the documented error."""
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dglm_forecast_sample(
            jr.key(0),
            FILTERED,
            G,
            F,
            sample_emission=_poisson_sample_emission,
            num_steps=1,
            num_draws=8,
            transition_covariance=W,
            discount=0.9,
        )
    with pytest.raises(ValueError, match="exactly one"):
        smcx.dglm_forecast_sample(
            jr.key(0),
            FILTERED,
            G,
            F,
            sample_emission=_poisson_sample_emission,
            num_steps=1,
            num_draws=8,
        )


def _sample_with(num_steps=1, num_draws=8):
    return smcx.dglm_forecast_sample(
        jr.key(0),
        FILTERED,
        G,
        F,
        sample_emission=_poisson_sample_emission,
        num_steps=num_steps,
        num_draws=num_draws,
        transition_covariance=W,
    )


def test_count_boundary_matrix():
    """Both counts share the positive-integer contract."""
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="num_steps"):
            _sample_with(num_steps=bad)
        with pytest.raises(ValueError, match="num_draws"):
            _sample_with(num_draws=bad)


def test_bad_transition_matrix_shape_is_rejected():
    """A mis-shaped transition matrix raises at the boundary."""
    with pytest.raises(ValueError, match="transition_matrix"):
        smcx.dglm_forecast_sample(
            jr.key(0),
            FILTERED,
            jnp.eye(3),
            F,
            sample_emission=_poisson_sample_emission,
            num_steps=1,
            num_draws=8,
            transition_covariance=W,
        )


def test_discount_above_one_is_rejected():
    """The sampler validates the discount like the filter."""
    with pytest.raises(ValueError, match="discount"):
        smcx.dglm_forecast_sample(
            jr.key(0),
            FILTERED,
            G,
            F,
            sample_emission=_poisson_sample_emission,
            num_steps=1,
            num_draws=8,
            discount=1.5,
        )


def test_dispersion_discount_above_one_is_rejected():
    """The sampler validates the dispersion discount like the filter."""
    with pytest.raises(ValueError, match="dispersion_discount"):
        smcx.dglm_forecast_sample(
            jr.key(0),
            FILTERED,
            G,
            F,
            sample_emission=_poisson_sample_emission,
            num_steps=1,
            num_draws=8,
            transition_covariance=W,
            dispersion_discount=1.5,
        )
