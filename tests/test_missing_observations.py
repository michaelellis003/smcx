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


def _two_dim_kalman(emissions, dtype):
    return smcx.kalman_filter(
        initial_mean=jnp.zeros(2, dtype=dtype),
        initial_covariance=jnp.eye(2, dtype=dtype),
        transition_matrix=0.9 * jnp.eye(2, dtype=dtype),
        transition_covariance=0.1 * jnp.eye(2, dtype=dtype),
        observation_matrix=jnp.eye(2, dtype=dtype),
        observation_covariance=0.3 * jnp.eye(2, dtype=dtype),
        emissions=emissions,
    )


@pytest.mark.parametrize(
    "bad_row",
    [
        [jnp.inf, 0.0],
        [-jnp.inf, jnp.nan],
    ],
)
def test_infinite_rows_raise(bad_row):
    """Infinite entries remain never-meaningful and fail eagerly."""
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    emissions = jnp.asarray([[0.1, 0.2], list(bad_row)], dtype=dtype)
    with pytest.raises(ValueError, match="finite"):
        _two_dim_kalman(emissions, dtype)


def test_partial_rows_now_filter_the_observed_components():
    """A partially NaN row is a partial observation, not an error.

    The #433 widening: the previously rejected input filters on its
    observed component alone.
    """
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    emissions = jnp.asarray([[0.1, 0.2], [jnp.nan, 0.5]], dtype=dtype)
    posterior = _two_dim_kalman(emissions, dtype)
    assert bool(jnp.all(jnp.isfinite(posterior.filtered_means)))
    assert bool(jnp.all(jnp.isfinite(posterior.log_evidence_increments)))


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


def _dlm_run(emissions, **kwargs):
    dtype = emissions.dtype
    defaults = {
        "initial_mean": jnp.zeros(1, dtype=dtype),
        "initial_scale_free_covariance": jnp.eye(1, dtype=dtype),
        "transition_matrix": jnp.eye(1, dtype=dtype),
        "observation_vector": jnp.ones(1, dtype=dtype),
        "scale_free_transition_covariance": 0.2 * jnp.eye(1, dtype=dtype),
        "prior_shape": 4.0,
        "prior_scale": 1.0,
    }
    defaults.update(kwargs)
    return smcx.dlm_filter(emissions=emissions, **defaults)


def test_dlm_gap_preserves_conjugate_state_and_zero_increment():
    """A missing datum leaves (n, S) unobserved and stores the prior."""
    emissions = jnp.asarray([0.2, jnp.nan, 0.4])
    posterior = _dlm_run(emissions)

    np.testing.assert_array_equal(
        posterior.scale_shapes,
        jnp.asarray([5.0, 5.0, 6.0], dtype=emissions.dtype),
    )
    np.testing.assert_array_equal(
        posterior.scale_estimates[1], posterior.scale_estimates[0]
    )
    np.testing.assert_array_equal(
        posterior.log_evidence_increments[1],
        jnp.zeros_like(posterior.marginal_loglik),
    )
    prior_mean = posterior.filtered_means[0]
    prior_cov = posterior.filtered_scale_free_covariances[0] + 0.2 * jnp.eye(
        1, dtype=emissions.dtype
    )
    np.testing.assert_array_equal(posterior.filtered_means[1], prior_mean)
    np.testing.assert_allclose(
        posterior.filtered_scale_free_covariances[1],
        prior_cov,
        atol=_tolerance(emissions.dtype),
    )


def test_dlm_gap_matches_composed_evolution():
    """A gap under identity dynamics equals doubling the evolution noise."""
    emissions = jnp.asarray([0.2, jnp.nan, 0.4])
    dtype = emissions.dtype
    evolution = 0.2 * jnp.eye(1, dtype=dtype)
    gapped = _dlm_run(emissions)
    composed = _dlm_run(
        jnp.asarray([0.2, 0.4]),
        scale_free_transition_covariance=(2.0 * evolution)[None],
    )
    atol = _tolerance(dtype, 4.0)
    np.testing.assert_allclose(
        gapped.filtered_means[2], composed.filtered_means[1], atol=atol
    )
    np.testing.assert_allclose(
        gapped.filtered_scale_free_covariances[2],
        composed.filtered_scale_free_covariances[1],
        atol=atol,
    )
    np.testing.assert_array_equal(
        gapped.scale_shapes[2], composed.scale_shapes[1]
    )
    np.testing.assert_allclose(
        gapped.scale_estimates[2], composed.scale_estimates[1], atol=atol
    )
    np.testing.assert_allclose(
        gapped.marginal_loglik, composed.marginal_loglik, atol=atol
    )


def test_dlm_variance_discount_still_decays_at_a_gap():
    """The variance discount still applies when the datum is missing.

    Discounting is evolution, not update: only the +1 observation
    step is skipped at a gap.
    """
    emissions = jnp.asarray([0.2, jnp.nan, 0.4])
    discount_v = 0.9
    posterior = _dlm_run(emissions, variance_discount=discount_v)

    n0 = 4.0
    n1 = n0 + 1.0
    n_gap = discount_v * n1
    n2 = discount_v * n_gap + 1.0
    np.testing.assert_allclose(
        posterior.scale_shapes,
        jnp.asarray([n1, n_gap, n2], dtype=emissions.dtype),
        atol=_tolerance(emissions.dtype),
    )
    np.testing.assert_array_equal(
        posterior.scale_estimates[1], posterior.scale_estimates[0]
    )


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="the 1e8-dof flat-prior reduction needs float64 headroom",
)
def test_dlm_gapped_kalman_reduction():
    """The flat-prior reduction to kalman_filter holds through gaps."""
    emissions = jnp.asarray([0.2, jnp.nan, 0.4])
    dtype = emissions.dtype
    dlm = _dlm_run(emissions, prior_shape=1e8, prior_scale=0.3)
    kalman = smcx.kalman_filter(
        initial_mean=jnp.zeros(1, dtype=dtype),
        initial_covariance=0.3 * jnp.eye(1, dtype=dtype),
        transition_matrix=jnp.eye(1, dtype=dtype),
        transition_covariance=0.3 * 0.2 * jnp.eye(1, dtype=dtype),
        observation_matrix=jnp.ones((1, 1), dtype=dtype),
        observation_covariance=0.3 * jnp.eye(1, dtype=dtype),
        emissions=emissions[:, None],
    )
    atol = 1e-3 if dtype == jnp.float32 else 1e-6
    np.testing.assert_allclose(
        dlm.filtered_means, kalman.filtered_means, atol=atol
    )
    np.testing.assert_allclose(
        dlm.marginal_loglik, kalman.marginal_loglik, atol=atol
    )


def test_dlm_rejects_infinite_emissions():
    """Infinite entries stay rejected under the shared row rule."""
    with pytest.raises(ValueError, match="fully observed"):
        _dlm_run(jnp.asarray([0.2, jnp.inf]))


def test_dlm_smoother_consumes_gapped_record():
    """Constant-V retrospection accepts a gapped filter record."""
    emissions = jnp.asarray([0.2, jnp.nan, 0.4])
    dtype = emissions.dtype
    posterior = _dlm_run(emissions)
    smoothed = smcx.dlm_smoother(
        posterior,
        jnp.eye(1, dtype=dtype),
        scale_free_transition_covariance=0.2 * jnp.eye(1, dtype=dtype),
    )
    assert bool(jnp.all(jnp.isfinite(smoothed.smoothed_means)))
    assert bool(jnp.all(jnp.isfinite(smoothed.smoothed_scale_free_covariances)))


def _dglm_run(emissions, family=None, **kwargs):
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    defaults = {
        "initial_mean": jnp.zeros(1, dtype=dtype),
        "initial_covariance": jnp.eye(1, dtype=dtype),
        "transition_matrix": jnp.eye(1, dtype=dtype),
        "observation_vector": jnp.ones(1, dtype=dtype),
        "transition_covariance": 0.1 * jnp.eye(1, dtype=dtype),
    }
    defaults.update(kwargs)
    return smcx.dglm_filter(
        emissions=emissions,
        family=family or smcx.poisson(),
        **defaults,
    )


def test_dglm_gap_bypasses_family_validation_and_stores_prior():
    """An all-NaN datum skips the family check and the update."""
    emissions = jnp.asarray([1.0, jnp.nan, 3.0])
    posterior = _dglm_run(emissions)

    prior_mean = posterior.filtered_means[0]
    prior_cov = posterior.filtered_covariances[0] + 0.1 * jnp.eye(
        1, dtype=posterior.filtered_means.dtype
    )
    np.testing.assert_array_equal(posterior.filtered_means[1], prior_mean)
    np.testing.assert_allclose(
        posterior.filtered_covariances[1],
        prior_cov,
        atol=_tolerance(posterior.filtered_means.dtype),
    )
    np.testing.assert_array_equal(
        posterior.log_evidence_increments[1],
        jnp.zeros_like(posterior.marginal_loglik),
    )
    assert bool(jnp.isfinite(posterior.marginal_loglik))
    assert bool(jnp.all(jnp.isfinite(posterior.conjugate_alphas)))
    assert bool(jnp.all(jnp.isfinite(posterior.conjugate_betas)))


def test_dglm_normal_family_gap_reduces_to_gapped_kalman():
    """The normal-family reduction holds through gaps."""
    emissions = jnp.asarray([0.2, jnp.nan, 0.4])
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    variance = 0.3

    def normal_family():
        return smcx.DGLMFamily(
            match_moments=lambda f, q: (f, q + variance),
            log_forecast=lambda y, alpha, beta: (
                -0.5 * (jnp.log(2.0 * jnp.pi * beta) + (y - alpha) ** 2 / beta)
            ),
            update=lambda y, alpha, beta: (
                alpha + (beta - variance) / beta * (y - alpha),
                (beta - variance) - (beta - variance) ** 2 / beta + variance,
            ),
            posterior_moments=lambda alpha, beta: (alpha, beta - variance),
        )

    dglm = _dglm_run(emissions, family=normal_family())
    kalman = smcx.kalman_filter(
        initial_mean=jnp.zeros(1, dtype=dtype),
        initial_covariance=jnp.eye(1, dtype=dtype),
        transition_matrix=jnp.eye(1, dtype=dtype),
        transition_covariance=0.1 * jnp.eye(1, dtype=dtype),
        observation_matrix=jnp.ones((1, 1), dtype=dtype),
        observation_covariance=variance * jnp.eye(1, dtype=dtype),
        emissions=emissions[:, None],
    )
    atol = _tolerance(dtype, 8.0)
    np.testing.assert_allclose(
        dglm.filtered_means, kalman.filtered_means, atol=atol
    )
    np.testing.assert_allclose(
        dglm.marginal_loglik, kalman.marginal_loglik, atol=atol
    )


def test_dglm_rejects_infinite_and_keeps_family_check_for_observed():
    """The row rule precedes the family check for observed entries.

    Infinite entries get the shared rejection; an invalid observed
    value beside a missing one still faces the family.
    """
    with pytest.raises(ValueError, match="fully observed"):
        _dglm_run(jnp.asarray([1.0, jnp.inf]))
    with pytest.raises(ValueError, match="nonnegative integer"):
        _dglm_run(jnp.asarray([1.0, jnp.nan, 2.5]))


def test_dglm_smoother_consumes_gapped_record():
    """Retrospective state moments accept a gapped filter record."""
    emissions = jnp.asarray([1.0, jnp.nan, 3.0])
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    posterior = _dglm_run(emissions)
    smoothed = smcx.dglm_smoother(
        posterior,
        jnp.eye(1, dtype=dtype),
        transition_covariance=0.1 * jnp.eye(1, dtype=dtype),
    )
    assert bool(jnp.all(jnp.isfinite(smoothed.smoothed_means)))
    assert bool(jnp.all(jnp.isfinite(smoothed.smoothed_covariances)))
