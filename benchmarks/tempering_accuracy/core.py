# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen targets and campaign plan for the tempering accuracy study."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from smcx.types import DenseInitialSampler, StaticLogDensity

ACCURACY_ROOT = 20_260_720
_EXTENSION_TAG = 0x54414343
_SIGMA2 = 0.49
_RHO = 0.9


class GaussianTarget(NamedTuple):
    """One lane-rounded Gaussian target and its float64 oracle."""

    geometry: str
    dimension: int
    dtype: str
    sigma2: float
    rho: float
    observation: np.ndarray
    likelihood_covariance: np.ndarray
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    log_evidence: float


class Callbacks(NamedTuple):
    """JAX callbacks evaluated in the target lane's declared dtype."""

    initial_sampler: DenseInitialSampler
    log_prior: StaticLogDensity
    log_likelihood: StaticLogDensity


def _chol_solve(matrix: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Solve a positive-definite system through its Cholesky factor."""
    lower = np.linalg.cholesky(matrix)
    return np.linalg.solve(lower.T, np.linalg.solve(lower, right))


def build_target(
    geometry: str, dimension: int, dtype: type[np.floating]
) -> GaussianTarget:
    """Construct G0/G1 after rounding defining values to one lane dtype."""
    if geometry not in {"G0", "G1"}:
        raise ValueError(f"unknown geometry: {geometry}")
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    lane_dtype = np.dtype(dtype)
    sigma2 = float(lane_dtype.type(_SIGMA2))
    rho = float(lane_dtype.type(_RHO))
    observation = np.linspace(-1, 1, dimension, dtype=lane_dtype).astype(
        np.float64
    )
    if geometry == "G0":
        covariance = sigma2 * np.eye(dimension)
    else:
        indices = np.arange(dimension)
        lags = np.abs(np.subtract.outer(indices, indices))
        covariance = sigma2 * rho**lags
    identity = np.eye(dimension)
    inverse_r = _chol_solve(covariance, identity)
    posterior_covariance = _chol_solve(identity + inverse_r, identity)
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    posterior_mean = posterior_covariance @ _chol_solve(covariance, observation)
    marginal = identity + covariance
    marginal_chol = np.linalg.cholesky(marginal)
    log_evidence = -0.5 * (
        dimension * math.log(2 * math.pi)
        + 2 * np.log(np.diag(marginal_chol)).sum()
        + observation @ _chol_solve(marginal, observation)
    )
    return GaussianTarget(
        geometry,
        dimension,
        lane_dtype.name,
        sigma2,
        rho,
        observation,
        covariance,
        posterior_mean,
        posterior_covariance,
        float(log_evidence),
    )


def make_callbacks(target: GaussianTarget) -> Callbacks:
    """Build O(d) prior and Gaussian-likelihood callbacks for one target."""
    dtype = jnp.dtype(target.dtype)
    dimension = target.dimension
    observation = jnp.asarray(target.observation, dtype=dtype)
    sigma2 = jnp.asarray(target.sigma2, dtype=dtype)
    rho = jnp.asarray(target.rho, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    log_two_pi = jnp.log(jnp.asarray(2 * math.pi, dtype=dtype))

    def initial_sampler(key: jax.Array, count: int) -> jax.Array:
        return jr.normal(key, (count, dimension), dtype=dtype)

    def log_prior(value: jax.Array) -> jax.Array:
        return -half * (dimension * log_two_pi + jnp.sum(value**2))

    def log_likelihood(value: jax.Array) -> jax.Array:
        residual = observation - value
        if target.geometry == "G0":
            quadratic = jnp.sum(residual**2) / sigma2
            logdet = dimension * jnp.log(sigma2)
        else:
            endpoints = residual[0] ** 2 + residual[-1] ** 2
            interior = (1 + rho**2) * jnp.sum(residual[1:-1] ** 2)
            adjacent = -2 * rho * jnp.sum(residual[:-1] * residual[1:])
            quadratic = (endpoints + interior + adjacent) / (
                sigma2 * (1 - rho**2)
            )
            logdet = dimension * jnp.log(sigma2) + (dimension - 1) * jnp.log(
                1 - rho**2
            )
        return -half * (dimension * log_two_pi + logdet + quadratic)

    return Callbacks(initial_sampler, log_prior, log_likelihood)


def accuracy_keys() -> jax.Array:
    """Return the exact 12-key prefix and tagged 20-key extension."""
    root = jr.key(ACCURACY_ROOT)
    prefix = jr.split(root, 12)
    extension_root = jr.fold_in(root, _EXTENSION_TAG)
    extension = jnp.stack([jr.fold_in(extension_root, i) for i in range(20)])
    return jnp.concatenate((prefix, extension))
