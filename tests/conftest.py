# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for smcx."""

import os

# The suite runs on CPU by default so results are deterministic across
# machines and unaffected by an installed `metal` extra (jax-mps
# registers at higher priority than CPU). Set SMCX_TEST_PLATFORM=mps to
# run the suite on the Apple-GPU backend explicitly. Must
# happen before any JAX import triggers initialization.
os.environ.setdefault(
    "JAX_PLATFORMS", os.environ.get("SMCX_TEST_PLATFORM", "cpu")
)

# Configure JAX to use 64-bit floats for higher precision in tests —
# on CPU only: the Metal backend has no float64 (jax-mps/MLX limit),
# so the SMCX_TEST_PLATFORM=mps run stays in float32.
import jax

if os.environ["JAX_PLATFORMS"] == "cpu":
    jax.config.update("jax_enable_x64", True)

# Install the jaxtyping import hook BEFORE importing smcx so that all
# jaxtyped annotations are validated at runtime during tests.
from jaxtyping import install_import_hook

install_import_hook("smcx", typechecker="beartype.beartype")

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import pytest

import smcx
from tests._lgssm_reference import EMISSIONS, STATES


def _mvn_sample(key, mean, cov, shape=()):
    """Sample from a multivariate normal using pure JAX."""
    chol = jnp.linalg.cholesky(cov)
    d = mean.shape[-1]
    z = jr.normal(key, (*shape, d))
    return mean + z @ chol.T


def _mvn_logpdf(x, mean, cov):
    """Log-pdf of a multivariate normal using jax.scipy."""
    return jstats.multivariate_normal.logpdf(x, mean, cov)


@pytest.fixture
def package():
    """Return the top-level package module for introspection."""
    return smcx


@pytest.fixture
def key():
    """Fixed JAX PRNG key for reproducibility."""
    return jr.PRNGKey(42)


@pytest.fixture
def lgssm_params():
    """Simple 1-D linear Gaussian SSM parameters.

    Model:
        z_0  ~ N(0, 1)
        z_t  = 0.9 * z_{t-1} + eps,  eps ~ N(0, 0.5^2)
        y_t  = z_t + eta,             eta ~ N(0, 1.0^2)

    Returns a dict with keys matching Dynamax ``make_lgssm_params``.
    """
    return dict(
        initial_mean=jnp.array([0.0]),
        initial_cov=jnp.array([[1.0]]),
        dynamics_weights=jnp.array([[0.9]]),
        dynamics_cov=jnp.array([[0.25]]),  # 0.5^2
        emissions_weights=jnp.array([[1.0]]),
        emissions_cov=jnp.array([[1.0]]),
    )


@pytest.fixture
def lgssm_data():
    """Return frozen externally generated 1-D LGSSM data.

    Returns (states, emissions) each of shape (50, 1).
    """
    return jnp.asarray(STATES), jnp.asarray(EMISSIONS)
