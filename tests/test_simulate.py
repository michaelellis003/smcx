# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for smcjax.simulate.

Validates output shapes, statistical properties against analytical
moments of a linear Gaussian SSM, and JIT compatibility.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from smcjax.simulate import simulate
from tests.conftest import _mvn_sample


def _make_lgssm_samplers(lgssm_params):
    """Build (initial, transition, emission) sampler closures."""
    m0 = lgssm_params['initial_mean']
    P0 = lgssm_params['initial_cov']
    F = lgssm_params['dynamics_weights']
    Q = lgssm_params['dynamics_cov']
    H = lgssm_params['emissions_weights']
    R = lgssm_params['emissions_cov']

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


class TestSimulateLGSSMStatistics:
    """Simulated samples should match analytical LGSSM moments."""

    def test_simulate_lgssm_statistics(self, lgssm_params):
        """Mean and variance of states near analytical steady-state."""
        init_fn, trans_fn, emit_fn = _make_lgssm_samplers(lgssm_params)

        # Simulate many trajectories to estimate moments
        keys = jr.split(jr.PRNGKey(42), 5_000)

        def run_one(key):
            states, _ = simulate(
                key=key,
                initial_sampler=init_fn,
                transition_sampler=trans_fn,
                emission_sampler=emit_fn,
                num_timesteps=100,
            )
            return states[-1, 0]  # final state, scalar

        final_states = jax.vmap(run_one)(keys)

        # Steady-state variance of AR(1) with rho=0.9, sigma^2=0.25:
        # V_ss = 0.25 / (1 - 0.81) = 0.25/0.19 ≈ 1.316
        # Steady-state mean = 0
        sample_mean = float(jnp.mean(final_states))
        sample_var = float(jnp.var(final_states))

        assert abs(sample_mean) < 0.1  # near zero
        assert sample_var > 0.5  # substantially positive
        assert sample_var < 3.0  # not unreasonably large


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
