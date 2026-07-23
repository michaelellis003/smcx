# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`smcx.simulate`."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from smcx.simulate import simulate
from tests.conftest import _mvn_sample


def _make_lgssm_samplers(lgssm_params):
    """Build (initial, transition, emission) sampler closures."""
    m0 = lgssm_params["initial_mean"]
    P0 = lgssm_params["initial_cov"]
    F = lgssm_params["dynamics_weights"]
    Q = lgssm_params["dynamics_cov"]
    H = lgssm_params["emissions_weights"]
    R = lgssm_params["emissions_cov"]

    def initial_sampler(key):
        return _mvn_sample(key, m0, P0)

    def transition_sampler(key, state):
        mean = (F @ state[:, None]).squeeze(-1)
        return _mvn_sample(key, mean, Q)

    def emission_sampler(key, state):
        mean = (H @ state[:, None]).squeeze(-1)
        return _mvn_sample(key, mean, R)

    return initial_sampler, transition_sampler, emission_sampler


class TestSimulateOutputShapes:
    """Simulate should produce correct output shapes."""

    def test_simulate_output_shapes(self, lgssm_params):
        """States and emissions have correct shapes."""
        init_fn, trans_fn, emit_fn = _make_lgssm_samplers(lgssm_params)
        states, emissions = simulate(
            key=jr.PRNGKey(0),
            initial_sampler=init_fn,
            transition_sampler=trans_fn,
            emission_sampler=emit_fn,
            num_timesteps=20,
        )
        assert states.shape == (20, 1)
        assert emissions.shape == (20, 1)


def test_simulate_applies_callbacks_in_time_order():
    """The initial state is emitted before transitions begin."""

    def initial_sampler(key):
        del key
        return jnp.array([1.0])

    def transition_sampler(key, state):
        del key
        return state + 1

    def emission_sampler(key, state):
        del key
        return 2 * state

    states, emissions = simulate(
        jr.key(42),
        initial_sampler,
        transition_sampler,
        emission_sampler,
        num_timesteps=4,
    )

    assert jnp.array_equal(states[:, 0], jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert jnp.array_equal(emissions[:, 0], jnp.array([2.0, 4.0, 6.0, 8.0]))


class TestSimulateJIT:
    """Simulate should be JIT-compilable."""

    def test_simulate_jit_compiles(self, lgssm_params):
        """Simulate runs under jax.jit."""
        init_fn, trans_fn, emit_fn = _make_lgssm_samplers(lgssm_params)

        @jax.jit
        def run(key):
            return simulate(
                key=key,
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                emission_sampler=emit_fn,
                num_timesteps=10,
            )

        states, emissions = run(jr.PRNGKey(0))
        assert states.shape == (10, 1)
        assert jnp.all(jnp.isfinite(states))
        assert jnp.all(jnp.isfinite(emissions))


class TestSimulateInputs:
    """Simulation aligns inputs with initialization and each transition."""

    def test_inputs_reach_initial_transition_and_emission_samplers(self):
        def initial_sampler(key, input_t):
            del key
            return input_t

        def transition_sampler(key, state, input_t):
            del key
            return state + input_t

        def emission_sampler(key, state, input_t):
            del key
            return state + 10.0 * input_t

        states, emissions = simulate(
            key=jr.key(0),
            initial_sampler=initial_sampler,
            transition_sampler=transition_sampler,
            emission_sampler=emission_sampler,
            num_timesteps=3,
            inputs=jnp.array([1.0, 2.0, 4.0]),
        )

        assert jnp.array_equal(states[:, 0], jnp.array([1.0, 3.0, 7.0]))
        assert jnp.array_equal(emissions[:, 0], jnp.array([11.0, 23.0, 47.0]))

    def test_inputs_must_match_num_timesteps(self):
        def initial_sampler(key, input_t):
            del key
            return input_t

        def transition_sampler(key, state, input_t):
            del key, input_t
            return state

        def emission_sampler(key, state, input_t):
            del key, input_t
            return state

        with pytest.raises(ValueError, match="leading dimension T=3"):
            simulate(
                jr.key(0),
                initial_sampler,
                transition_sampler,
                emission_sampler,
                num_timesteps=3,
                inputs=jnp.ones(2),
            )
