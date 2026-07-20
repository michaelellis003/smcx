# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Equivalent dense and semantic-PyTree tracking workloads.

Both callback bundles below implement exactly the same controlled
constant-velocity linear Gaussian state-space model.  Only the state
representation differs: a dense four-vector versus two two-vector leaves.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray
from numpy.typing import NDArray

from smcx.types import (
    InitialSamplerWithInput,
    LogObservationFnWithInput,
    TransitionSamplerWithInput,
)


@dataclass(frozen=True)
class TrackingLGSSM:
    """Controlled two-dimensional constant-velocity tracking model."""

    timesteps: int = 200
    time_delta: float = 1.0
    process_intensity: float = 0.1
    observation_variance: float = 0.5
    observation_correlation: float = 0.3
    covariance_regime: str = "correlated"
    control_scale: float = 0.05

    def __post_init__(self) -> None:
        """Reject invalid model parameters before profiling starts."""
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if self.time_delta <= 0.0:
            raise ValueError("time_delta must be positive")
        if self.process_intensity <= 0.0:
            raise ValueError("process_intensity must be positive")
        if self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        if not -1.0 < self.observation_correlation < 1.0:
            raise ValueError(
                "observation_correlation must be strictly between -1 and 1"
            )
        if self.covariance_regime not in {"correlated", "diagonal"}:
            raise ValueError(
                "covariance_regime must be 'correlated' or 'diagonal'"
            )
        if self.control_scale < 0.0:
            raise ValueError("control_scale must be non-negative")

    def transition_matrix(self) -> NDArray[np.float64]:
        """Return the dense constant-velocity transition matrix."""
        delta = self.time_delta
        return np.array(
            [
                [1.0, 0.0, delta, 0.0],
                [0.0, 1.0, 0.0, delta],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def input_matrix(self) -> NDArray[np.float64]:
        """Return the known-acceleration input matrix."""
        delta = self.time_delta
        half_delta_sq = 0.5 * delta**2
        return np.array(
            [
                [half_delta_sq, 0.0],
                [0.0, half_delta_sq],
                [delta, 0.0],
                [0.0, delta],
            ],
            dtype=np.float64,
        )

    def transition_covariance(self) -> NDArray[np.float64]:
        """Return integrated-white-acceleration process covariance."""
        delta = self.time_delta
        covariance = self.process_intensity * np.array(
            [
                [delta**3 / 3.0, 0.0, delta**2 / 2.0, 0.0],
                [0.0, delta**3 / 3.0, 0.0, delta**2 / 2.0],
                [delta**2 / 2.0, 0.0, delta, 0.0],
                [0.0, delta**2 / 2.0, 0.0, delta],
            ],
            dtype=np.float64,
        )
        if self.covariance_regime == "diagonal":
            return np.diag(np.diag(covariance))
        return covariance

    def observation_matrix(self) -> NDArray[np.float64]:
        """Return the position-only observation matrix."""
        return np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )

    def observation_covariance(self) -> NDArray[np.float64]:
        """Return the correlated two-coordinate observation covariance."""
        correlation = (
            self.observation_correlation
            if self.covariance_regime == "correlated"
            else 0.0
        )
        return self.observation_variance * np.array(
            [[1.0, correlation], [correlation, 1.0]],
            dtype=np.float64,
        )

    def initial_mean(self) -> NDArray[np.float64]:
        """Return the initial state mean."""
        return np.zeros(4, dtype=np.float64)

    def initial_covariance(self) -> NDArray[np.float64]:
        """Return the initial position/velocity covariance."""
        return np.diag(np.array([1.0, 1.0, 0.25, 0.25], dtype=np.float64))


class TrackingState(NamedTuple):
    """Semantic two-leaf representation of one or more tracking states."""

    position: Float[Array, "*batch 2"]
    velocity: Float[Array, "*batch 2"]


class TrackingData(NamedTuple):
    """Frozen dense states, observations, and known controls."""

    states: NDArray[np.float64]
    emissions: NDArray[np.float64]
    inputs: NDArray[np.float64]


class TrackingKalmanOracle(NamedTuple):
    """Exact float64 filtering target for the tracking workload."""

    log_evidence: float
    filtered_means: NDArray[np.float64]
    filtered_covariances: NDArray[np.float64]


class TrackingCallbacks(NamedTuple):
    """Bootstrap callbacks for one state representation."""

    initial_sampler: InitialSamplerWithInput
    transition_sampler: TransitionSamplerWithInput
    log_observation_fn: LogObservationFnWithInput


def flatten_tracking_state(
    state: TrackingState,
) -> Float[Array, "*batch 4"]:
    """Concatenate semantic position and velocity leaves."""
    return jnp.concatenate((state.position, state.velocity), axis=-1)


def _tracking_inputs(model: TrackingLGSSM) -> NDArray[np.float64]:
    """Return deterministic accelerations shared by both representations."""
    times = np.arange(model.timesteps, dtype=np.float64)
    return model.control_scale * np.stack(
        (np.sin(0.11 * times), np.cos(0.07 * times)),
        axis=-1,
    )


def make_tracking_data(
    model: TrackingLGSSM,
    *,
    seed: int,
) -> TrackingData:
    """Generate an interleaved, deterministic float64 tracking path."""
    transition = model.transition_matrix()
    input_matrix = model.input_matrix()
    observation = model.observation_matrix()
    transition_chol = np.linalg.cholesky(model.transition_covariance())
    observation_chol = np.linalg.cholesky(model.observation_covariance())
    initial_chol = np.linalg.cholesky(model.initial_covariance())
    inputs = _tracking_inputs(model)
    rng = np.random.default_rng(seed)

    states = np.empty((model.timesteps, 4), dtype=np.float64)
    emissions = np.empty((model.timesteps, 2), dtype=np.float64)
    states[0] = model.initial_mean() + initial_chol @ rng.normal(size=4)
    emissions[0] = observation @ states[0] + observation_chol @ rng.normal(
        size=2
    )
    for time_index in range(1, model.timesteps):
        states[time_index] = (
            transition @ states[time_index - 1]
            + input_matrix @ inputs[time_index]
            + transition_chol @ rng.normal(size=4)
        )
        emissions[time_index] = observation @ states[
            time_index
        ] + observation_chol @ rng.normal(size=2)
    return TrackingData(states=states, emissions=emissions, inputs=inputs)


def _as_matrix_series(
    values: NDArray[np.float64] | Float[Array, "..."],
    *,
    width: int,
    timesteps: int,
    name: str,
) -> NDArray[np.float64]:
    """Canonicalize one dense time series for the float64 oracle."""
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (timesteps, width):
        raise ValueError(f"{name} must have shape ({timesteps}, {width})")
    return result


def tracking_kalman_oracle(
    model: TrackingLGSSM,
    emissions: NDArray[np.float64] | Float[Array, "..."],
    inputs: NDArray[np.float64] | Float[Array, "..."],
) -> TrackingKalmanOracle:
    """Evaluate the analytic multivariate Kalman recurrence in float64."""
    observations = _as_matrix_series(
        emissions,
        width=2,
        timesteps=model.timesteps,
        name="emissions",
    )
    controls = _as_matrix_series(
        inputs,
        width=2,
        timesteps=model.timesteps,
        name="inputs",
    )
    transition = model.transition_matrix()
    input_matrix = model.input_matrix()
    process_covariance = model.transition_covariance()
    observation = model.observation_matrix()
    observation_covariance = model.observation_covariance()
    mean = model.initial_mean()
    covariance = model.initial_covariance()
    means = np.empty((model.timesteps, 4), dtype=np.float64)
    covariances = np.empty((model.timesteps, 4, 4), dtype=np.float64)
    log_evidence = np.float64(0.0)

    for time_index, emission in enumerate(observations):
        if time_index > 0:
            mean = transition @ mean + input_matrix @ controls[time_index]
            covariance = (
                transition @ covariance @ transition.T + process_covariance
            )
        innovation = emission - observation @ mean
        innovation_covariance = (
            observation @ covariance @ observation.T + observation_covariance
        )
        sign, log_determinant = np.linalg.slogdet(innovation_covariance)
        if sign <= 0.0:
            raise ValueError("innovation covariance is not positive definite")
        solved_innovation = np.linalg.solve(innovation_covariance, innovation)
        log_evidence -= 0.5 * (
            2.0 * math.log(2.0 * math.pi)
            + log_determinant
            + innovation @ solved_innovation
        )
        covariance_observation = covariance @ observation.T
        gain = np.linalg.solve(
            innovation_covariance, covariance_observation.T
        ).T
        mean = mean + gain @ innovation
        covariance = covariance - gain @ innovation_covariance @ gain.T
        covariance = 0.5 * (covariance + covariance.T)
        means[time_index] = mean
        covariances[time_index] = covariance

    return TrackingKalmanOracle(
        log_evidence=float(log_evidence),
        filtered_means=means,
        filtered_covariances=covariances,
    )


def _callback_primitives(model: TrackingLGSSM):
    """Build shared dense primitives used by both callback bundles."""
    initial_mean = jnp.asarray(model.initial_mean(), dtype=jnp.float32)
    initial_chol = jnp.asarray(
        np.linalg.cholesky(model.initial_covariance()),
        dtype=jnp.float32,
    )
    transition = jnp.asarray(model.transition_matrix(), dtype=jnp.float32)
    input_matrix = jnp.asarray(model.input_matrix(), dtype=jnp.float32)
    transition_chol = jnp.asarray(
        np.linalg.cholesky(model.transition_covariance()),
        dtype=jnp.float32,
    )
    observation = jnp.asarray(
        model.observation_matrix(),
        dtype=jnp.float32,
    )
    observation_covariance = model.observation_covariance()
    observation_precision = jnp.asarray(
        np.linalg.inv(observation_covariance),
        dtype=jnp.float32,
    )
    _, observation_logdet = np.linalg.slogdet(observation_covariance)
    observation_constant = jnp.asarray(
        -0.5 * (2.0 * math.log(2.0 * math.pi) + observation_logdet),
        dtype=jnp.float32,
    )

    def sample_initial(
        key: PRNGKeyArray,
        num_particles: int,
    ) -> Float[Array, "num_particles 4"]:
        noise = jr.normal(key, (num_particles, 4), dtype=jnp.float32)
        return initial_mean + noise @ initial_chol.T

    def sample_transition(
        key: PRNGKeyArray,
        state: Float[Array, " 4"],
        input_t: Float[Array, " 2"],
    ) -> Float[Array, " 4"]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        input_array = jnp.asarray(input_t, dtype=jnp.float32)
        mean = transition @ state_array + input_matrix @ input_array
        noise = transition_chol @ jr.normal(
            key,
            (4,),
            dtype=jnp.float32,
        )
        return mean + noise

    def log_observation(
        emission: Float[Array, " 2"],
        state: Float[Array, " 4"],
    ) -> Float[Array, ""]:
        emission_array = jnp.asarray(emission, dtype=jnp.float32)
        state_array = jnp.asarray(state, dtype=jnp.float32)
        error = emission_array - observation @ state_array
        return observation_constant - 0.5 * (
            error @ observation_precision @ error
        )

    return sample_initial, sample_transition, log_observation


def make_dense_tracking_callbacks(model: TrackingLGSSM) -> TrackingCallbacks:
    """Build input-aware callbacks using one dense four-coordinate state."""
    sample_initial, sample_transition, log_observation = _callback_primitives(
        model
    )

    def initial_sampler(
        key: PRNGKeyArray,
        num_particles: int,
        input_t: Float[Array, " 2"],
        /,
    ) -> Float[Array, "num_particles 4"]:
        del input_t
        return sample_initial(key, num_particles)

    def transition_sampler(
        key: PRNGKeyArray,
        state: Float[Array, " 4"],
        input_t: Float[Array, " 2"],
        /,
    ) -> Float[Array, " 4"]:
        return sample_transition(key, state, input_t)

    def log_observation_fn(
        emission: Float[Array, " 2"],
        state: Float[Array, " 4"],
        input_t: Float[Array, " 2"],
        /,
    ) -> Float[Array, ""]:
        del input_t
        return log_observation(emission, state)

    return TrackingCallbacks(
        initial_sampler=initial_sampler,
        transition_sampler=transition_sampler,
        log_observation_fn=log_observation_fn,
    )


def make_tree_tracking_callbacks(model: TrackingLGSSM) -> TrackingCallbacks:
    """Build equivalent callbacks using semantic position/velocity leaves."""
    sample_initial, sample_transition, log_observation = _callback_primitives(
        model
    )

    def initial_sampler(
        key: PRNGKeyArray,
        num_particles: int,
        input_t: Float[Array, " 2"],
        /,
    ) -> TrackingState:
        del input_t
        dense = sample_initial(key, num_particles)
        return TrackingState(position=dense[:, :2], velocity=dense[:, 2:])

    def transition_sampler(
        key: PRNGKeyArray,
        state: TrackingState,
        input_t: Float[Array, " 2"],
        /,
    ) -> TrackingState:
        dense = sample_transition(key, flatten_tracking_state(state), input_t)
        return TrackingState(position=dense[:2], velocity=dense[2:])

    def log_observation_fn(
        emission: Float[Array, " 2"],
        state: TrackingState,
        input_t: Float[Array, " 2"],
        /,
    ) -> Float[Array, ""]:
        del input_t
        return log_observation(emission, flatten_tracking_state(state))

    return TrackingCallbacks(
        initial_sampler=initial_sampler,
        transition_sampler=transition_sampler,
        log_observation_fn=log_observation_fn,
    )
