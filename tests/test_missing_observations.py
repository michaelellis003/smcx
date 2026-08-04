# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Missing-observation semantics for the linear Kalman filter (ADR-0034)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx


def _model(dtype=None):
    dtype = dtype or (
        jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    )
    transition = jnp.asarray([[0.9, 0.1], [0.0, 0.8]], dtype=dtype)
    evolution = jnp.asarray([[0.2, 0.05], [0.05, 0.1]], dtype=dtype)
    observation = jnp.asarray([[1.0, 0.0]], dtype=dtype)
    variance = jnp.asarray([[0.3]], dtype=dtype)
    mean0 = jnp.zeros(2, dtype=dtype)
    cov0 = jnp.eye(2, dtype=dtype)
    return mean0, cov0, transition, evolution, observation, variance


def _run(emissions, transition=None, evolution=None):
    mean0, cov0, default_a, default_w, observation, variance = _model(
        emissions.dtype
    )
    return smcx.kalman_filter(
        initial_mean=mean0,
        initial_covariance=cov0,
        transition_matrix=default_a if transition is None else transition,
        transition_covariance=default_w if evolution is None else evolution,
        observation_matrix=observation,
        observation_covariance=variance,
        emissions=emissions,
    )


def _tolerance(dtype, scale=1.0):
    return 64.0 * float(np.finfo(dtype).eps) * scale


def test_gap_step_stores_prediction_and_zero_increment():
    """A missing datum is the identity update with a zero increment."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    posterior = _run(emissions)

    np.testing.assert_array_equal(
        posterior.filtered_means[1], posterior.predicted_means[1]
    )
    np.testing.assert_array_equal(
        posterior.filtered_covariances[1], posterior.predicted_covariances[1]
    )
    np.testing.assert_array_equal(
        posterior.log_evidence_increments[1],
        jnp.zeros_like(posterior.log_evidence_increments[1]),
    )
    assert bool(jnp.isfinite(posterior.marginal_loglik))


def test_gap_matches_composed_two_step_model():
    """Filtering through a gap equals composing the transition across it."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    _, _, transition, evolution, _, _ = _model(emissions.dtype)
    gapped = _run(emissions)

    composed_transition = transition @ transition
    composed_evolution = transition @ evolution @ transition.T + evolution
    two_step = _run(
        jnp.asarray([[0.2], [0.4]]),
        transition=composed_transition[None],
        evolution=composed_evolution[None],
    )

    scale = float(jnp.max(jnp.abs(two_step.filtered_covariances)))
    atol = _tolerance(emissions.dtype, max(1.0, scale))
    np.testing.assert_allclose(
        gapped.filtered_means[2], two_step.filtered_means[1], atol=atol
    )
    np.testing.assert_allclose(
        gapped.filtered_covariances[2],
        two_step.filtered_covariances[1],
        atol=atol,
    )
    np.testing.assert_allclose(
        gapped.log_evidence_increments[2],
        two_step.log_evidence_increments[1],
        atol=atol,
    )
    np.testing.assert_allclose(
        gapped.marginal_loglik, two_step.marginal_loglik, atol=atol
    )


def test_leading_gap_keeps_the_prior():
    """A missing first datum stores the untouched prior."""
    emissions = jnp.asarray([[jnp.nan], [0.3]])
    posterior = _run(emissions)
    mean0, cov0, *_ = _model(emissions.dtype)

    np.testing.assert_array_equal(posterior.filtered_means[0], mean0)
    np.testing.assert_array_equal(posterior.filtered_covariances[0], cov0)
    np.testing.assert_array_equal(
        posterior.log_evidence_increments[0],
        jnp.zeros_like(posterior.marginal_loglik),
    )


def test_all_missing_gives_pure_prediction_and_zero_marginal():
    """An entirely missing series is the prior predictive recursion."""
    emissions = jnp.full((3, 1), jnp.nan)
    posterior = _run(emissions)

    np.testing.assert_array_equal(
        posterior.filtered_means, posterior.predicted_means
    )
    np.testing.assert_array_equal(
        posterior.filtered_covariances, posterior.predicted_covariances
    )
    np.testing.assert_array_equal(
        posterior.marginal_loglik, jnp.zeros_like(posterior.marginal_loglik)
    )


@pytest.mark.parametrize(
    "bad_row",
    [
        [jnp.nan, 0.5],
        [jnp.inf, 0.0],
        [-jnp.inf, jnp.nan],
    ],
)
def test_partial_and_infinite_rows_raise(bad_row):
    """Rows must be fully observed finite values or entirely NaN."""
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    emissions = jnp.asarray([[0.1, 0.2], list(bad_row)], dtype=dtype)
    mean0 = jnp.zeros(2, dtype=dtype)
    with pytest.raises(ValueError, match="fully observed"):
        smcx.kalman_filter(
            initial_mean=mean0,
            initial_covariance=jnp.eye(2, dtype=dtype),
            transition_matrix=0.9 * jnp.eye(2, dtype=dtype),
            transition_covariance=0.1 * jnp.eye(2, dtype=dtype),
            observation_matrix=jnp.eye(2, dtype=dtype),
            observation_covariance=0.3 * jnp.eye(2, dtype=dtype),
            emissions=emissions,
        )


def test_gradient_through_gap_is_finite_and_matches_composed():
    """The gap contributes no NaN to gradients of the marginal."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    dtype = emissions.dtype
    mean0, cov0, transition, evolution, observation, variance = _model(dtype)

    def gapped_marginal(a):
        return smcx.kalman_filter(
            initial_mean=mean0,
            initial_covariance=cov0,
            transition_matrix=a,
            transition_covariance=evolution,
            observation_matrix=observation,
            observation_covariance=variance,
            emissions=emissions,
        ).marginal_loglik

    def composed_marginal(a):
        return smcx.kalman_filter(
            initial_mean=mean0,
            initial_covariance=cov0,
            transition_matrix=(a @ a)[None],
            transition_covariance=(a @ evolution @ a.T + evolution)[None],
            observation_matrix=observation,
            observation_covariance=variance,
            emissions=jnp.asarray([[0.2], [0.4]]),
        ).marginal_loglik

    gapped_grad = jax.grad(gapped_marginal)(transition)
    composed_grad = jax.grad(composed_marginal)(transition)
    assert bool(jnp.all(jnp.isfinite(gapped_grad)))
    scale = max(1.0, float(jnp.max(jnp.abs(composed_grad))))
    np.testing.assert_allclose(
        gapped_grad,
        composed_grad,
        atol=_tolerance(dtype, 8.0 * scale),
    )


def test_smoother_consumes_gapped_record():
    """The RTS smoother accepts a gapped record and matches composition."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    _, _, transition, evolution, _, _ = _model(emissions.dtype)
    gapped = smcx.rts_smoother(_run(emissions), transition)

    composed_transition = transition @ transition
    composed_evolution = transition @ evolution @ transition.T + evolution
    two_step = smcx.rts_smoother(
        _run(
            jnp.asarray([[0.2], [0.4]]),
            transition=composed_transition[None],
            evolution=composed_evolution[None],
        ),
        composed_transition[None],
    )

    assert bool(jnp.all(jnp.isfinite(gapped.smoothed_means)))
    scale = max(1.0, float(jnp.max(jnp.abs(two_step.smoothed_covariances))))
    atol = _tolerance(emissions.dtype, 4.0 * scale)
    np.testing.assert_allclose(
        gapped.smoothed_means[0], two_step.smoothed_means[0], atol=atol
    )
    np.testing.assert_allclose(
        gapped.smoothed_means[2], two_step.smoothed_means[1], atol=atol
    )
    np.testing.assert_allclose(
        gapped.smoothed_covariances[0],
        two_step.smoothed_covariances[0],
        atol=atol,
    )


def test_posterior_sample_consumes_gapped_record():
    """Joint draws from a gapped record are finite with sane shape."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    _, _, transition, _, _, _ = _model(emissions.dtype)
    draws = smcx.posterior_sample(
        jr.key(379),
        _run(emissions),
        transition,
        num_draws=64,
    )
    assert draws.shape == (64, 3, 2)
    assert bool(jnp.all(jnp.isfinite(draws)))


def test_jit_matches_eager_on_gapped_series():
    """Compiled and eager gapped runs agree within a small eps multiple."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    dtype = emissions.dtype
    mean0, cov0, transition, evolution, observation, variance = _model(dtype)

    def run(y):
        return smcx.kalman_filter(
            initial_mean=mean0,
            initial_covariance=cov0,
            transition_matrix=transition,
            transition_covariance=evolution,
            observation_matrix=observation,
            observation_covariance=variance,
            emissions=y,
        )

    eager = run(emissions)
    compiled = jax.jit(run)(emissions)
    atol = 32.0 * float(np.finfo(dtype).eps)
    for eager_field, compiled_field in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(
            np.asarray(compiled_field),
            np.asarray(eager_field),
            rtol=0.0,
            atol=atol * max(1.0, float(jnp.max(jnp.abs(eager_field)))),
        )


def _nonlinear_callbacks(dtype):
    transition = jnp.asarray([[0.9, 0.1], [0.0, 0.8]], dtype=dtype)
    observation = jnp.asarray([[1.0, 0.0]], dtype=dtype)

    def transition_mean(state):
        return transition @ state

    def transition_jacobian(state):
        del state
        return transition

    def observation_mean(state):
        return observation @ state

    def observation_jacobian(state):
        del state
        return observation

    return (
        transition_mean,
        transition_jacobian,
        observation_mean,
        observation_jacobian,
    )


def test_extended_gap_reduces_to_the_linear_gapped_run():
    """A linear EKF model with a gap matches kalman_filter exactly."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    dtype = emissions.dtype
    mean0, cov0, _, evolution, _, variance = _model(dtype)
    means, jacobian, obs_mean, obs_jacobian = _nonlinear_callbacks(dtype)

    extended = smcx.extended_kalman_filter(
        initial_mean=mean0,
        initial_covariance=cov0,
        transition_mean_fn=means,
        transition_jacobian_fn=jacobian,
        transition_covariance=evolution,
        observation_mean_fn=obs_mean,
        observation_jacobian_fn=obs_jacobian,
        observation_covariance=variance,
        emissions=emissions,
    )
    linear = _run(emissions)

    np.testing.assert_array_equal(
        extended.filtered_means[1], extended.predicted_means[1]
    )
    np.testing.assert_array_equal(
        extended.log_evidence_increments[1],
        jnp.zeros_like(extended.marginal_loglik),
    )
    atol = _tolerance(dtype, 4.0)
    np.testing.assert_allclose(
        extended.filtered_means, linear.filtered_means, atol=atol
    )
    np.testing.assert_allclose(
        extended.marginal_loglik, linear.marginal_loglik, atol=atol
    )


def test_unscented_gap_step_identity():
    """The UKF stores the prediction and a zero increment at a gap."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    dtype = emissions.dtype
    mean0, cov0, _, evolution, _, variance = _model(dtype)
    means, _, obs_mean, _ = _nonlinear_callbacks(dtype)

    unscented = smcx.unscented_kalman_filter(
        initial_mean=mean0,
        initial_covariance=cov0,
        transition_mean_fn=means,
        transition_covariance=evolution,
        observation_mean_fn=obs_mean,
        observation_covariance=variance,
        emissions=emissions,
    )
    np.testing.assert_array_equal(
        unscented.filtered_means[1], unscented.predicted_means[1]
    )
    np.testing.assert_array_equal(
        unscented.filtered_covariances[1],
        unscented.predicted_covariances[1],
    )
    np.testing.assert_array_equal(
        unscented.log_evidence_increments[1],
        jnp.zeros_like(unscented.marginal_loglik),
    )
    assert bool(jnp.isfinite(unscented.marginal_loglik))


@pytest.mark.parametrize("method_name", ["taylor", "unscented"])
def test_gaussian_filter_dispatch_matches_named_on_gaps(method_name):
    """The strategy entry point inherits missing-data behavior bitwise."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    dtype = emissions.dtype
    mean0, cov0, _, evolution, _, variance = _model(dtype)
    means, jacobian, obs_mean, obs_jacobian = _nonlinear_callbacks(dtype)

    if method_name == "taylor":
        method = smcx.taylor_order1(jacobian, obs_jacobian)
        named = smcx.extended_kalman_filter(
            initial_mean=mean0,
            initial_covariance=cov0,
            transition_mean_fn=means,
            transition_jacobian_fn=jacobian,
            transition_covariance=evolution,
            observation_mean_fn=obs_mean,
            observation_jacobian_fn=obs_jacobian,
            observation_covariance=variance,
            emissions=emissions,
        )
    else:
        method = smcx.unscented()
        named = smcx.unscented_kalman_filter(
            initial_mean=mean0,
            initial_covariance=cov0,
            transition_mean_fn=means,
            transition_covariance=evolution,
            observation_mean_fn=obs_mean,
            observation_covariance=variance,
            emissions=emissions,
        )
    strategy = smcx.gaussian_filter(
        mean0,
        cov0,
        means,
        evolution,
        obs_mean,
        variance,
        emissions,
        method=method,
    )
    for named_field, strategy_field in zip(named, strategy, strict=True):
        np.testing.assert_array_equal(
            np.asarray(strategy_field), np.asarray(named_field)
        )


@pytest.mark.parametrize("filter_name", ["extended", "unscented"])
def test_nonlinear_partial_rows_raise(filter_name):
    """EKF and UKF reject partially observed rows like the linear filter."""
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    emissions = jnp.asarray([[0.1, jnp.nan]], dtype=dtype)
    mean0 = jnp.zeros(2, dtype=dtype)
    identity = jnp.eye(2, dtype=dtype)

    def state_mean(state):
        return state

    with pytest.raises(ValueError, match="fully observed"):
        if filter_name == "extended":
            smcx.extended_kalman_filter(
                initial_mean=mean0,
                initial_covariance=identity,
                transition_mean_fn=state_mean,
                transition_jacobian_fn=lambda s: identity,
                transition_covariance=0.1 * identity,
                observation_mean_fn=state_mean,
                observation_jacobian_fn=lambda s: identity,
                observation_covariance=0.3 * identity,
                emissions=emissions,
            )
        else:
            smcx.unscented_kalman_filter(
                initial_mean=mean0,
                initial_covariance=identity,
                transition_mean_fn=state_mean,
                transition_covariance=0.1 * identity,
                observation_mean_fn=state_mean,
                observation_covariance=0.3 * identity,
                emissions=emissions,
            )


def test_extended_gradient_through_gap_is_finite():
    """Gaps contribute no NaN to EKF gradients."""
    emissions = jnp.asarray([[0.2], [jnp.nan], [0.4]])
    dtype = emissions.dtype
    mean0, cov0, _, evolution, _, variance = _model(dtype)
    _, _, obs_mean, obs_jacobian = _nonlinear_callbacks(dtype)

    def marginal(scale):
        return smcx.extended_kalman_filter(
            initial_mean=mean0,
            initial_covariance=cov0,
            transition_mean_fn=lambda s: scale * s,
            transition_jacobian_fn=lambda s: scale * jnp.eye(2, dtype=dtype),
            transition_covariance=evolution,
            observation_mean_fn=obs_mean,
            observation_jacobian_fn=obs_jacobian,
            observation_covariance=variance,
            emissions=emissions,
        ).marginal_loglik

    gradient = jax.grad(marginal)(jnp.asarray(0.9, dtype=dtype))
    assert bool(jnp.isfinite(gradient))
