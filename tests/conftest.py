# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for smcx."""

import os

# The suite runs on CPU by default so results are deterministic across
# machines and unaffected by an installed `metal` extra (jax-mps
# registers at higher priority than CPU). Set SMCX_TEST_PLATFORM=mps to
# run the suite on the Apple-GPU backend explicitly. Validate and set
# both backend and precision before any JAX import triggers
# initialization.
_PLATFORM_X64 = {"cpu": True, "mps": False}
_TRUE_VALUES = {"1", "on", "t", "true", "y", "yes"}
_FALSE_VALUES = {"0", "f", "false", "n", "no", "off"}


def _parse_jax_boolean(name: str, value: str) -> bool:
    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise RuntimeError(
        f"Invalid {name}={value!r}; expected a JAX boolean value"
    )


_selected_platform = os.environ.get("SMCX_TEST_PLATFORM", "cpu")
if _selected_platform not in _PLATFORM_X64:
    raise RuntimeError(
        "Invalid SMCX_TEST_PLATFORM="
        f'{_selected_platform!r}; expected "cpu" or "mps"'
    )

_inherited_platform = os.environ.get("JAX_PLATFORMS")
if (
    _inherited_platform is not None
    and _inherited_platform != _selected_platform
):
    raise RuntimeError(
        "JAX_PLATFORMS conflicts with selected test platform: "
        f"expected {_selected_platform!r}, got {_inherited_platform!r}"
    )

_expected_x64 = _PLATFORM_X64[_selected_platform]
_inherited_x64 = os.environ.get("JAX_ENABLE_X64")
if (
    _inherited_x64 is not None
    and _parse_jax_boolean("JAX_ENABLE_X64", _inherited_x64) != _expected_x64
):
    raise RuntimeError(
        "JAX_ENABLE_X64 conflicts with selected test platform: "
        f"{_selected_platform!r} requires "
        f"{str(_expected_x64).lower()}"
    )

os.environ["JAX_PLATFORMS"] = _selected_platform
os.environ["JAX_ENABLE_X64"] = str(_expected_x64).lower()

import jax

_actual_backend = jax.default_backend()
_actual_devices = jax.devices()
_actual_x64 = bool(jax.config.read("jax_enable_x64"))

if _actual_backend != _selected_platform:
    raise RuntimeError(
        f"JAX selected {_actual_backend!r}, expected {_selected_platform!r}"
    )
if not _actual_devices or any(
    device.platform != _selected_platform for device in _actual_devices
):
    raise RuntimeError(
        f"JAX devices {_actual_devices!r} do not match {_selected_platform!r}"
    )
if _actual_x64 != _expected_x64:
    raise RuntimeError(
        f"JAX x64 is {_actual_x64}, expected {_expected_x64} on "
        f"{_selected_platform!r}"
    )

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


def pytest_report_header(config: pytest.Config) -> str:
    """Report the backend contract exercised by this test process."""
    del config
    return (
        f"JAX backend: {_actual_backend}; "
        f"JAX devices: {_actual_devices}; "
        f"JAX x64: {_actual_x64}"
    )


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
