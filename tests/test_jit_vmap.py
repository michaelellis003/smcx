# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for JIT and vmap compatibility of all public functions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from smcjax import ess, log_ess, multinomial, residual, stratified, systematic
from smcjax.weights import log_normalize, normalize
from tests.conftest import _mvn_logpdf, _mvn_sample


class TestWeightsJIT:
    """Weights functions compile under jit."""

    def test_log_normalize_jit(self):
        lw = jnp.array([1.0, 2.0, 3.0])
        jitted = jax.jit(log_normalize)
        log_norm, log_ev = jitted(lw)
        assert jnp.all(jnp.isfinite(log_norm))
        assert jnp.isfinite(log_ev)

    def test_normalize_jit(self):
        lw = jnp.array([1.0, 2.0, 3.0])
        w = jax.jit(normalize)(lw)
        assert jnp.allclose(jnp.sum(w), 1.0, atol=1e-6)


class TestESSJIT:
    """ESS functions compile under jit."""

    def test_ess_jit(self):
        lw = jnp.zeros(10)
        result = jax.jit(ess)(lw)
        assert jnp.allclose(result, 10.0, atol=1e-5)

    def test_log_ess_jit(self):
        lw = jnp.zeros(10)
        result = jax.jit(log_ess)(lw)
        assert jnp.allclose(jnp.exp(result), 10.0, atol=1e-5)


class TestWeightsVmap:
    """Weights functions work under vmap (batch of weight vectors)."""

    def test_log_normalize_vmap(self):
        lw_batch = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        vmapped = jax.vmap(log_normalize)
        log_norms, log_evs = vmapped(lw_batch)
        assert log_norms.shape == (2, 3)
        assert log_evs.shape == (2,)

    def test_ess_vmap(self):
        lw_batch = jnp.array([[0.0, 0.0, 0.0], [0.0, -1e10, -1e10]])
        result = jax.vmap(ess)(lw_batch)
        assert jnp.allclose(result[0], 3.0, atol=1e-4)
        assert jnp.allclose(result[1], 1.0, atol=1e-4)


class TestResamplingJIT:
    """Resampling functions compile under jit with static num_samples."""

    @pytest.mark.parametrize(
        'resample_fn', [systematic, stratified, multinomial, residual]
    )
    def test_jit_compiles(self, resample_fn):
        key = jr.PRNGKey(0)
        w = jnp.array([0.25, 0.25, 0.25, 0.25])
        jitted = jax.jit(resample_fn, static_argnums=(2,))
        idx = jitted(key, w, 4)
        assert idx.shape == (4,)
        assert jnp.all(idx >= 0)
        assert jnp.all(idx < 4)


class TestBootstrapJIT:
    """Bootstrap filter compiles under jit."""

    def test_jit_compiles(self):
        from smcjax.bootstrap import bootstrap_filter

        m0 = jnp.array([0.0])
        P0 = jnp.array([[1.0]])
        F = jnp.array([[0.9]])
        Q = jnp.array([[0.25]])
        H = jnp.array([[1.0]])
        R = jnp.array([[1.0]])

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        emissions = jnp.ones((10, 1))

        @jax.jit
        def run(key):
            return bootstrap_filter(
                key=key,
                initial_sampler=init,
                transition_sampler=trans,
                log_observation_fn=obs,
                emissions=emissions,
                num_particles=50,
            )

        result = run(jr.PRNGKey(0))
        assert result.filtered_particles.shape == (10, 50, 1)
        assert jnp.isfinite(result.marginal_loglik)


class TestAuxiliaryJIT:
    """Auxiliary particle filter compiles under jit."""

    def test_jit_compiles(self):
        from smcjax.auxiliary import auxiliary_filter

        m0 = jnp.array([0.0])
        P0 = jnp.array([[1.0]])
        F = jnp.array([[0.9]])
        Q = jnp.array([[0.25]])
        H = jnp.array([[1.0]])
        R = jnp.array([[1.0]])

        def init(key, n):
            return _mvn_sample(key, m0, P0, shape=(n,))

        def trans(key, state):
            mean = (F @ state[:, None]).squeeze(-1)
            return _mvn_sample(key, mean, Q)

        def obs(emission, state):
            mean = (H @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, mean, R)

        def aux(emission, state):
            pred = (H @ F @ state[:, None]).squeeze(-1)
            return _mvn_logpdf(emission, pred, R)

        emissions = jnp.ones((10, 1))

        @jax.jit
        def run(key):
            return auxiliary_filter(
                key=key,
                initial_sampler=init,
                transition_sampler=trans,
                log_observation_fn=obs,
                log_auxiliary_fn=aux,
                emissions=emissions,
                num_particles=50,
            )

        result = run(jr.PRNGKey(0))
        assert result.filtered_particles.shape == (10, 50, 1)
        assert jnp.isfinite(result.marginal_loglik)
