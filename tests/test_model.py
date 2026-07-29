# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""StateSpaceModel record and Feynman-Kac derivation tests."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

EMISSIONS = jnp.asarray([[0.2], [-0.1], [0.4], [0.05], [-0.3]])
INPUTS = jnp.asarray([0.5, -0.2, 0.1, 0.4, -0.5])


def _ar1_model():
    def sample_initial(key, params, input_0):
        del input_0
        scale = params["process_scale"] / jnp.sqrt(1.0 - params["rho"] ** 2)
        return scale * jr.normal(key, (1,))

    def sample_transition(key, state, params, input_t):
        del input_t
        noise = params["process_scale"] * jr.normal(key, state.shape)
        return params["rho"] * state + noise

    def log_observation(emission, state, params, input_t):
        del input_t
        residual = (emission[0] - state[0]) / params["observation_scale"]
        return -0.5 * residual**2

    def log_lookahead(emission, state, params, input_t):
        del input_t
        residual = emission[0] - params["rho"] * state[0]
        return -0.5 * (residual / 0.85) ** 2

    def log_transition(state, prev_state, params, input_t):
        del input_t
        residual = (state[0] - params["rho"] * prev_state[0]) / params[
            "process_scale"
        ]
        return -0.5 * residual**2

    def sample_proposal(key, prev_state, emission, params, input_t):
        del input_t
        center = 0.5 * (params["rho"] * prev_state + emission)
        return center + 0.25 * jr.normal(key, prev_state.shape)

    def log_proposal(emission, state, prev_state, params, input_t):
        del input_t
        center = 0.5 * (params["rho"] * prev_state[0] + emission[0])
        return -0.5 * ((state[0] - center) / 0.25) ** 2

    return smcx.StateSpaceModel(
        sample_initial=sample_initial,
        sample_transition=sample_transition,
        log_observation=log_observation,
        log_transition=log_transition,
        sample_proposal=sample_proposal,
        log_proposal=log_proposal,
        log_lookahead=log_lookahead,
    )


PARAMS = {
    "rho": jnp.asarray(0.9),
    "process_scale": jnp.asarray(0.3),
    "observation_scale": jnp.asarray(0.7),
}


def test_bootstrap_fk_matches_callback_filter_bitwise():
    """The model path equals the callback path under one key schedule.

    The callback adapters below reproduce the derivation's per-particle
    initial key split, so both routes trace identical programs and the
    outputs must agree bitwise.
    """
    model = _ar1_model()

    def initial(key, num_particles):
        keys = jr.split(key, num_particles)
        return jax.vmap(lambda k: model.sample_initial(k, PARAMS, None))(keys)

    def transition(key, state):
        return model.sample_transition(key, state, PARAMS, None)

    def log_obs(emission, state):
        return model.log_observation(emission, state, PARAMS, None)

    via_model = smcx.run_smc(
        jr.key(5),
        smcx.bootstrap_fk(model, PARAMS, EMISSIONS),
        8,
    )
    via_callbacks = smcx.bootstrap_filter(
        jr.key(5), initial, transition, log_obs, EMISSIONS, 8
    )

    for model_field, callback_field in zip(
        via_model, via_callbacks, strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(model_field), np.asarray(callback_field)
        )


def test_auxiliary_fk_flat_lookahead_reduces_to_bootstrap():
    """A constant look-ahead reproduces bootstrap up to normalizer noise."""
    model = _ar1_model()
    flat = model._replace(
        log_lookahead=lambda emission, state, params, input_t: jnp.asarray(0.0)
    )

    twisted = smcx.run_smc(
        jr.key(9), smcx.auxiliary_fk(flat, PARAMS, EMISSIONS), 16
    )
    plain = smcx.run_smc(
        jr.key(9), smcx.bootstrap_fk(flat, PARAMS, EMISSIONS), 16
    )

    np.testing.assert_allclose(
        np.asarray(twisted.marginal_loglik),
        np.asarray(plain.marginal_loglik),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(twisted.filtered_log_weights),
        np.asarray(plain.filtered_log_weights),
        rtol=1e-12,
        atol=1e-12,
    )


def test_guided_fk_runs_and_input_threading_reaches_callbacks():
    """Guided derivation runs; inputs arrive as length-one vectors."""
    model = _ar1_model()
    seen = {}

    def probing_observation(emission, state, params, input_t):
        seen["input_shape"] = getattr(input_t, "shape", None)
        residual = (emission[0] - state[0] - 0.05 * input_t[0]) / params[
            "observation_scale"
        ]
        return -0.5 * residual**2

    probing = model._replace(log_observation=probing_observation)
    posterior = smcx.run_smc(
        jr.key(3),
        smcx.guided_fk(probing, PARAMS, EMISSIONS, inputs=INPUTS),
        8,
    )

    assert np.isfinite(float(posterior.marginal_loglik))
    assert seen["input_shape"] == (1,)


def test_guided_fk_requires_proposal_capabilities():
    """A missing capability raises a named error at derivation time."""
    model = _ar1_model()._replace(sample_proposal=None)
    with pytest.raises(ValueError, match=r"model\.sample_proposal"):
        smcx.guided_fk(model, PARAMS, EMISSIONS)


def test_auxiliary_fk_requires_lookahead():
    model = _ar1_model()._replace(log_lookahead=None)
    with pytest.raises(ValueError, match=r"model\.log_lookahead"):
        smcx.auxiliary_fk(model, PARAMS, EMISSIONS)


def test_gradient_flows_through_params():
    """The marginal likelihood differentiates with respect to params.

    Explicit params threading is what makes this possible; the value
    must be finite and nonzero for an informative model.
    """
    model = _ar1_model()

    def objective(params):
        posterior = smcx.run_smc(
            jr.key(11),
            smcx.bootstrap_fk(model, params, EMISSIONS),
            32,
            resampling_threshold=0.0,
        )
        return posterior.marginal_loglik

    gradient = jax.grad(objective)(PARAMS)

    for leaf in jax.tree.leaves(gradient):
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert float(jnp.abs(gradient["observation_scale"])) > 0.0


def test_discrete_emissions_keep_model_dtype():
    """Integer emissions reach the model callbacks unconverted."""
    model = _ar1_model()
    seen = {}

    def counting_observation(emission, state, params, input_t):
        del input_t
        seen["dtype"] = emission.dtype
        rate = jnp.exp(state[0])
        return emission[0] * jnp.log(rate) - rate

    counting = model._replace(log_observation=counting_observation)
    counts = jnp.asarray([[1], [0], [2], [1], [3]], dtype=jnp.int32)
    posterior = smcx.run_smc(
        jr.key(13), smcx.bootstrap_fk(counting, PARAMS, counts), 8
    )

    assert np.isfinite(float(posterior.marginal_loglik))
    assert seen["dtype"] == jnp.int32


def test_gradient_without_resampling_estimates_the_score():
    """With resampling off, the mean gradient over keys is the score.

    Tier-2 stochastic gate for the sequential-inference guide's
    gradient-semantics passage (#284): on a linear-Gaussian model the
    exact score is the gradient of the Kalman marginal log-likelihood,
    and the pathwise SMC gradient with ``resampling_threshold=0.0``
    is an unbiased estimator of it. The tolerance is a z-band on the
    Monte Carlo standard error of the replicate mean.
    """
    emissions = jnp.asarray([[0.5], [-0.3], [0.8]])
    rho, transition_sd, observation_sd = 0.7, 1.0, 0.5

    exact_score = jax.grad(
        lambda value: (
            smcx.kalman_filter(
                jnp.zeros(1),
                jnp.eye(1),
                value * jnp.eye(1),
                transition_sd**2 * jnp.eye(1),
                jnp.eye(1),
                observation_sd**2 * jnp.eye(1),
                emissions,
            ).marginal_loglik
        )
    )(rho)

    model = smcx.StateSpaceModel(
        sample_initial=lambda key, params, input_0: jr.normal(key, (1,)),
        sample_transition=lambda key, state, params, input_t: (
            params["rho"] * state + transition_sd * jr.normal(key, (1,))
        ),
        log_observation=lambda emission, state, params, input_t: (
            -0.5 * ((emission[0] - state[0]) / observation_sd) ** 2
            - jnp.log(observation_sd)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        ),
    )

    def replicate(key):
        def objective(value):
            fk = smcx.bootstrap_fk(model, {"rho": value}, emissions)
            return smcx.run_smc(
                key, fk, 512, resampling_threshold=0.0
            ).marginal_loglik

        return jax.grad(objective)(rho)

    gradients = jax.jit(jax.vmap(replicate))(jr.split(jr.key(7), 200))
    mean = float(jnp.mean(gradients))
    standard_error = float(jnp.std(gradients) / jnp.sqrt(200))

    assert abs(mean - float(exact_score)) < 4.0 * standard_error
