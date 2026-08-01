# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Derivation-independent dense Gaussian smoothing oracle."""

import numpy as np


def dense_joint_moments(
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    transition_matrices: np.ndarray,
    transition_covariances: np.ndarray,
    transition_offsets: np.ndarray,
    observation_matrices: np.ndarray,
    observation_covariances: np.ndarray,
    observation_offsets: np.ndarray,
    emissions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve marginal and adjacent-state moments from joint precision.

    The full posterior has information form
    ``exp(-0.5 * x.T @ precision @ x + information.T @ x)``. This
    construction works directly from the model densities and does not use
    filtering or backward-recursion identities.
    """
    initial_mean = np.asarray(initial_mean, dtype=np.float64)
    initial_covariance = np.asarray(initial_covariance, dtype=np.float64)
    transition_matrices = np.asarray(transition_matrices, dtype=np.float64)
    transition_covariances = np.asarray(
        transition_covariances,
        dtype=np.float64,
    )
    transition_offsets = np.asarray(transition_offsets, dtype=np.float64)
    observation_matrices = np.asarray(observation_matrices, dtype=np.float64)
    observation_covariances = np.asarray(
        observation_covariances,
        dtype=np.float64,
    )
    observation_offsets = np.asarray(observation_offsets, dtype=np.float64)
    emissions = np.asarray(emissions, dtype=np.float64)

    num_timesteps = emissions.shape[0]
    state_dim = initial_mean.shape[0]
    total_dim = num_timesteps * state_dim
    state_identity = np.eye(state_dim, dtype=np.float64)
    precision = np.zeros((total_dim, total_dim), dtype=np.float64)
    information = np.zeros(total_dim, dtype=np.float64)

    def _state_slice(time: int) -> slice:
        start = time * state_dim
        return slice(start, start + state_dim)

    first = _state_slice(0)
    prior_solve = np.linalg.solve(initial_covariance, state_identity)
    precision[first, first] += prior_solve
    information[first] += np.linalg.solve(initial_covariance, initial_mean)

    for time, (transition, covariance, offset) in enumerate(
        zip(
            transition_matrices,
            transition_covariances,
            transition_offsets,
            strict=True,
        ),
        start=1,
    ):
        pair = slice((time - 1) * state_dim, (time + 1) * state_dim)
        design = np.concatenate((-transition, state_identity), axis=1)
        weighted_design = np.linalg.solve(covariance, design)
        precision[pair, pair] += design.T @ weighted_design
        information[pair] += design.T @ np.linalg.solve(covariance, offset)

    for time, (operator, covariance, emission, offset) in enumerate(
        zip(
            observation_matrices,
            observation_covariances,
            emissions,
            observation_offsets,
            strict=True,
        )
    ):
        current = _state_slice(time)
        weighted_operator = np.linalg.solve(covariance, operator)
        precision[current, current] += operator.T @ weighted_operator
        information[current] += operator.T @ np.linalg.solve(
            covariance,
            emission - offset,
        )

    joint_mean = np.linalg.solve(precision, information)
    joint_covariance = np.linalg.solve(
        precision,
        np.eye(total_dim, dtype=np.float64),
    )
    marginal_covariances = np.stack([
        joint_covariance[_state_slice(time), _state_slice(time)]
        for time in range(num_timesteps)
    ])
    cross_covariances = np.stack([
        joint_covariance[_state_slice(time), _state_slice(time + 1)]
        for time in range(num_timesteps - 1)
    ])
    return (
        joint_mean.reshape(num_timesteps, state_dim),
        marginal_covariances,
        cross_covariances,
    )
