# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free model fixtures for the profiling campaign.

Model construction and float64 correctness oracles live together here so
every worker consumes the same mathematical workload.  Model setup and oracle
evaluation are outside every timed region.
"""

import math
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from smcx.types import (
    DenseInitialSampler,
    InitialSampler,
    InitialSamplerWithInput,
    LogObservationFn,
    LogObservationFnWithInput,
    LogProposalFnWithInput,
    LogTransitionFnWithInput,
    ParamInitialSampler,
    ParamLogObservationFn,
    ParamTransitionSampler,
    ProposalSamplerWithInput,
    TransitionSampler,
    TransitionSamplerWithInput,
)


@dataclass(frozen=True)
class LGSSM:
    """Scalar controlled linear Gaussian state-space model."""

    timesteps: int = 100
    autoregressive_coefficient: float = 0.9
    input_coefficient: float = 0.25
    initial_mean: float = 0.0
    initial_variance: float = 1.0
    transition_variance: float = 0.2
    observation_variance: float = 0.3
    input_frequency: float = 0.17
    outlier_standard_deviations: float = 0.0

    def __post_init__(self) -> None:
        """Reject malformed workloads before a benchmark process starts."""
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        if self.transition_variance <= 0.0:
            raise ValueError("transition_variance must be positive")
        if self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")

    @property
    def a(self) -> float:
        """Short mathematical alias for the AR coefficient."""
        return self.autoregressive_coefficient

    @property
    def b(self) -> float:
        """Short mathematical alias for the input coefficient."""
        return self.input_coefficient

    @property
    def m0(self) -> float:
        """Short mathematical alias for the initial mean."""
        return self.initial_mean

    @property
    def p0(self) -> float:
        """Short mathematical alias for the initial variance."""
        return self.initial_variance

    @property
    def q(self) -> float:
        """Short mathematical alias for the transition variance."""
        return self.transition_variance

    @property
    def r(self) -> float:
        """Short mathematical alias for the observation variance."""
        return self.observation_variance


class LGSSMData(NamedTuple):
    """Frozen data consumed by an LGSSM profiling cell."""

    states: np.ndarray
    emissions: np.ndarray
    inputs: np.ndarray


class KalmanOracle(NamedTuple):
    """Analytic scalar-LGSSM filtering results evaluated in float64."""

    log_evidence: float
    filtered_means: np.ndarray
    filtered_variances: np.ndarray


class LGSSMCallbacks(NamedTuple):
    """Input-aware callbacks for all three standard particle filters."""

    initial_sampler: InitialSamplerWithInput
    transition_sampler: TransitionSamplerWithInput
    log_observation_fn: LogObservationFnWithInput
    log_auxiliary_fn: LogObservationFnWithInput
    proposal_sampler: ProposalSamplerWithInput
    log_proposal_fn: LogProposalFnWithInput
    log_transition_fn: LogTransitionFnWithInput


def make_lgssm_data(model: LGSSM, *, seed: int) -> LGSSMData:
    """Generate a deterministic controlled LGSSM path in NumPy float64.

    State and observation draws are deliberately interleaved at each time
    point.  This draw order is part of the frozen workload contract.
    """
    rng = np.random.default_rng(seed)
    times = np.arange(model.timesteps, dtype=np.float64)
    inputs = np.sin(model.input_frequency * times)
    states = np.empty(model.timesteps, dtype=np.float64)
    emissions = np.empty(model.timesteps, dtype=np.float64)

    states[0] = rng.normal(model.m0, math.sqrt(model.p0))
    emissions[0] = states[0] + rng.normal(0.0, math.sqrt(model.r))
    for time_index in range(1, model.timesteps):
        transition_mean = (
            model.a * states[time_index - 1] + model.b * inputs[time_index]
        )
        states[time_index] = transition_mean + rng.normal(
            0.0, math.sqrt(model.q)
        )
        emissions[time_index] = states[time_index] + rng.normal(
            0.0, math.sqrt(model.r)
        )

    emissions[model.timesteps // 2] += (
        model.outlier_standard_deviations * math.sqrt(model.r)
    )

    return LGSSMData(
        states=states[:, None],
        emissions=emissions[:, None],
        inputs=inputs[:, None],
    )


def _as_scalar_series(
    values: np.ndarray | Float[Array, "..."],
    *,
    name: str,
) -> np.ndarray:
    """Canonicalize a scalar time series to a NumPy float64 vector."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 2 and result.shape[1] == 1:
        result = result[:, 0]
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape (T,) or (T, 1)")
    return result


def kalman_oracle(
    model: LGSSM,
    emissions: np.ndarray | Float[Array, "..."],
    inputs: np.ndarray | Float[Array, "..."] | None = None,
) -> KalmanOracle:
    """Evaluate the analytic scalar Kalman filter in NumPy float64."""
    observations = _as_scalar_series(emissions, name="emissions")
    if observations.shape[0] != model.timesteps:
        raise ValueError("emissions length must match model.timesteps")
    if inputs is None:
        controls = np.zeros(model.timesteps, dtype=np.float64)
    else:
        controls = _as_scalar_series(inputs, name="inputs")
        if controls.shape[0] != model.timesteps:
            raise ValueError("inputs length must match model.timesteps")

    mean = np.float64(model.m0)
    variance = np.float64(model.p0)
    log_evidence = np.float64(0.0)
    filtered_means = np.empty(model.timesteps, dtype=np.float64)
    filtered_variances = np.empty(model.timesteps, dtype=np.float64)

    for time_index, observation in enumerate(observations):
        if time_index > 0:
            mean = model.a * mean + model.b * controls[time_index]
            variance = model.a**2 * variance + model.q
        innovation = observation - mean
        innovation_variance = variance + model.r
        log_evidence -= 0.5 * (
            math.log(2.0 * math.pi * innovation_variance)
            + innovation**2 / innovation_variance
        )
        gain = variance / innovation_variance
        mean += gain * innovation
        variance *= 1.0 - gain
        filtered_means[time_index] = mean
        filtered_variances[time_index] = variance

    return KalmanOracle(
        log_evidence=float(log_evidence),
        filtered_means=filtered_means[:, None],
        filtered_variances=filtered_variances[:, None],
    )


def _jax_normal_logpdf(
    value: Float[Array, "..."],
    mean: Float[Array, "..."] | float,
    variance: Float[Array, "..."] | float,
) -> Float[Array, "..."]:
    """Elementwise scalar-Normal log density in campaign f32 arithmetic."""
    value_array = jnp.asarray(value, dtype=jnp.float32)
    mean_array = jnp.asarray(mean, dtype=jnp.float32)
    variance_array = jnp.asarray(variance, dtype=jnp.float32)
    normalizer = jnp.asarray(2.0 * math.pi, dtype=jnp.float32)
    return -0.5 * (
        jnp.log(normalizer * variance_array)
        + (value_array - mean_array) ** 2 / variance_array
    )


def guided_log_weight_terms(
    model: LGSSM,
    emission: Float[Array, "..."],
    propagated: Float[Array, "..."],
    previous: Float[Array, "..."],
    input_t: Float[Array, " input_dim"],
) -> tuple[
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, "..."],
    Float[Array, ""],
]:
    """Return the exact guided correction and predictive potential.

    For the locally optimal proposal, ``log_g + log_f - log_q`` is the
    predictive log likelihood and therefore does not depend on the sampled
    propagated state.  Keeping all four terms visible makes that identity a
    unit-testable workload invariant.
    """
    emission_value = jnp.ravel(jnp.asarray(emission, dtype=jnp.float32))[0]
    previous_value = jnp.ravel(jnp.asarray(previous, dtype=jnp.float32))[0]
    input_value = jnp.ravel(jnp.asarray(input_t, dtype=jnp.float32))[0]
    coefficient = jnp.asarray(model.a, dtype=jnp.float32)
    input_coefficient = jnp.asarray(model.b, dtype=jnp.float32)
    transition_variance = jnp.asarray(model.q, dtype=jnp.float32)
    observation_variance = jnp.asarray(model.r, dtype=jnp.float32)
    transition_mean = (
        coefficient * previous_value + input_coefficient * input_value
    )
    proposal_variance = (
        transition_variance
        * observation_variance
        / (transition_variance + observation_variance)
    )
    proposal_mean = (
        observation_variance * transition_mean
        + transition_variance * emission_value
    ) / (transition_variance + observation_variance)

    propagated_array = jnp.asarray(propagated, dtype=jnp.float32)
    log_g = _jax_normal_logpdf(
        emission_value,
        propagated_array,
        observation_variance,
    )
    log_f = _jax_normal_logpdf(
        propagated_array,
        transition_mean,
        transition_variance,
    )
    log_q = _jax_normal_logpdf(
        propagated_array, proposal_mean, proposal_variance
    )
    log_predictive = _jax_normal_logpdf(
        emission_value,
        transition_mean,
        transition_variance + observation_variance,
    )
    return log_g, log_f, log_q, log_predictive


def make_lgssm_callbacks(model: LGSSM) -> LGSSMCallbacks:
    """Build local JAX callbacks for the L1 filter workloads."""
    initial_mean = jnp.asarray(model.m0, dtype=jnp.float32)
    initial_scale = jnp.asarray(math.sqrt(model.p0), dtype=jnp.float32)
    coefficient = jnp.asarray(model.a, dtype=jnp.float32)
    input_coefficient = jnp.asarray(model.b, dtype=jnp.float32)
    transition_variance = jnp.asarray(model.q, dtype=jnp.float32)
    observation_variance = jnp.asarray(model.r, dtype=jnp.float32)
    transition_scale = jnp.sqrt(transition_variance)
    proposal_variance = (
        transition_variance
        * observation_variance
        / (transition_variance + observation_variance)
    )
    proposal_scale = jnp.sqrt(proposal_variance)

    def initial_sampler(
        key: PRNGKeyArray,
        num_particles: int,
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, "num_particles 1"]:
        del input_t
        return initial_mean + initial_scale * jr.normal(
            key,
            (num_particles, 1),
            dtype=jnp.float32,
        )

    def transition_sampler(
        key: PRNGKeyArray,
        state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " 1"]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        input_array = jnp.asarray(input_t, dtype=jnp.float32)
        mean = coefficient * state_array + input_coefficient * input_array[0]
        return mean + transition_scale * jr.normal(
            key,
            state_array.shape,
            dtype=jnp.float32,
        )

    def log_observation_fn(
        emission: Float[Array, " 1"],
        state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, ""]:
        del input_t
        return _jax_normal_logpdf(
            emission[0],
            state[0],
            observation_variance,
        )

    def log_auxiliary_fn(
        emission: Float[Array, " 1"],
        state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, ""]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        input_array = jnp.asarray(input_t, dtype=jnp.float32)
        mean = coefficient * state_array[0] + input_coefficient * input_array[0]
        return _jax_normal_logpdf(
            emission[0],
            mean,
            transition_variance + observation_variance,
        )

    def proposal_sampler(
        key: PRNGKeyArray,
        state: Float[Array, " 1"],
        emission: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " 1"]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        emission_array = jnp.asarray(emission, dtype=jnp.float32)
        input_array = jnp.asarray(input_t, dtype=jnp.float32)
        transition_mean = (
            coefficient * state_array + input_coefficient * input_array[0]
        )
        mean = (
            observation_variance * transition_mean
            + transition_variance * emission_array[0]
        ) / (transition_variance + observation_variance)
        return mean + proposal_scale * jr.normal(
            key,
            state_array.shape,
            dtype=jnp.float32,
        )

    def log_proposal_fn(
        emission: Float[Array, " 1"],
        new_state: Float[Array, " 1"],
        old_state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, ""]:
        emission_array = jnp.asarray(emission, dtype=jnp.float32)
        old_state_array = jnp.asarray(old_state, dtype=jnp.float32)
        input_array = jnp.asarray(input_t, dtype=jnp.float32)
        transition_mean = (
            coefficient * old_state_array[0]
            + input_coefficient * input_array[0]
        )
        mean = (
            observation_variance * transition_mean
            + transition_variance * emission_array[0]
        ) / (transition_variance + observation_variance)
        return _jax_normal_logpdf(new_state[0], mean, proposal_variance)

    def log_transition_fn(
        new_state: Float[Array, " 1"],
        old_state: Float[Array, " 1"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, ""]:
        old_state_array = jnp.asarray(old_state, dtype=jnp.float32)
        input_array = jnp.asarray(input_t, dtype=jnp.float32)
        mean = (
            coefficient * old_state_array[0]
            + input_coefficient * input_array[0]
        )
        return _jax_normal_logpdf(new_state[0], mean, transition_variance)

    return LGSSMCallbacks(
        initial_sampler=initial_sampler,
        transition_sampler=transition_sampler,
        log_observation_fn=log_observation_fn,
        log_auxiliary_fn=log_auxiliary_fn,
        proposal_sampler=proposal_sampler,
        log_proposal_fn=log_proposal_fn,
        log_transition_fn=log_transition_fn,
    )


@dataclass(frozen=True)
class StochasticVolatility:
    """Scalar stochastic-volatility workload."""

    timesteps: int = 500
    mean: float = -0.5
    persistence: float = 0.97
    innovation_scale: float = 0.2
    outlier_multiplier: float = 8.0

    def __post_init__(self) -> None:
        """Validate stationarity and scale assumptions."""
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")
        if not -1.0 < self.persistence < 1.0:
            raise ValueError("persistence must be strictly between -1 and 1")
        if self.innovation_scale <= 0.0:
            raise ValueError("innovation_scale must be positive")


class StochasticVolatilityData(NamedTuple):
    """Frozen latent log volatility and observed returns."""

    states: np.ndarray
    emissions: np.ndarray


class StochasticVolatilityCallbacks(NamedTuple):
    """Callbacks for the bootstrap stochastic-volatility workload."""

    initial_sampler: InitialSampler
    transition_sampler: TransitionSampler
    log_observation_fn: LogObservationFn


def make_stochastic_volatility_data(
    model: StochasticVolatility,
    *,
    seed: int,
) -> StochasticVolatilityData:
    """Generate the committed stochastic-volatility path in float64."""
    rng = np.random.default_rng(seed)
    stationary_scale = model.innovation_scale / math.sqrt(
        1.0 - model.persistence**2
    )
    states = np.empty(model.timesteps, dtype=np.float64)
    emissions = np.empty(model.timesteps, dtype=np.float64)
    states[0] = model.mean + stationary_scale * rng.normal()
    emissions[0] = math.exp(states[0] / 2.0) * rng.normal()
    for time_index in range(1, model.timesteps):
        states[time_index] = (
            model.mean
            + model.persistence * (states[time_index - 1] - model.mean)
            + model.innovation_scale * rng.normal()
        )
        emissions[time_index] = (
            math.exp(states[time_index] / 2.0) * rng.normal()
        )
    emissions[model.timesteps // 2] *= model.outlier_multiplier
    return StochasticVolatilityData(
        states=states[:, None],
        emissions=emissions[:, None],
    )


def make_stochastic_volatility_callbacks(
    model: StochasticVolatility,
) -> StochasticVolatilityCallbacks:
    """Build local JAX callbacks for the N1 bootstrap workload."""
    mean_value = jnp.asarray(model.mean, dtype=jnp.float32)
    persistence = jnp.asarray(model.persistence, dtype=jnp.float32)
    innovation_scale = jnp.asarray(
        model.innovation_scale,
        dtype=jnp.float32,
    )
    stationary_scale = innovation_scale / jnp.sqrt(
        jnp.asarray(1.0, dtype=jnp.float32) - persistence**2
    )
    log_two_pi = jnp.asarray(math.log(2.0 * math.pi), dtype=jnp.float32)

    def initial_sampler(
        key: PRNGKeyArray, num_particles: int, /
    ) -> Float[Array, "num_particles 1"]:
        return mean_value + stationary_scale * jr.normal(
            key,
            (num_particles, 1),
            dtype=jnp.float32,
        )

    def transition_sampler(
        key: PRNGKeyArray, state: Float[Array, " 1"], /
    ) -> Float[Array, " 1"]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        mean = mean_value + persistence * (state_array - mean_value)
        return mean + innovation_scale * jr.normal(
            key,
            state_array.shape,
            dtype=jnp.float32,
        )

    def log_observation_fn(
        emission: Float[Array, " 1"],
        state: Float[Array, " 1"],
        /,
    ) -> Float[Array, ""]:
        emission_array = jnp.asarray(emission, dtype=jnp.float32)
        state_array = jnp.asarray(state, dtype=jnp.float32)
        return -0.5 * (
            log_two_pi
            + state_array[0]
            + emission_array[0] ** 2 * jnp.exp(-state_array[0])
        )

    return StochasticVolatilityCallbacks(
        initial_sampler=initial_sampler,
        transition_sampler=transition_sampler,
        log_observation_fn=log_observation_fn,
    )


class StaticInitialSampler(Protocol):
    """Draw a dense cloud for a static target."""

    def __call__(
        self, key: PRNGKeyArray, num_particles: int, /
    ) -> Float[Array, "num_particles dimension"]:
        """Draw ``num_particles`` prior states."""
        ...


class StaticLogDensity(Protocol):
    """Evaluate a single static-target particle."""

    def __call__(
        self, state: Float[Array, " dimension"], /
    ) -> Float[Array, ""]:
        """Evaluate one state's log density."""
        ...


class GaussianTargetOracle(NamedTuple):
    """Closed-form evidence and posterior moments for G1."""

    log_evidence: float
    posterior_mean: np.ndarray
    posterior_variance: float


class GaussianTargetCallbacks(NamedTuple):
    """Tempering callbacks and fixed observation for G1."""

    initial_sampler: StaticInitialSampler
    log_prior_fn: StaticLogDensity
    log_likelihood_fn: StaticLogDensity
    observation: Float[Array, " dimension"]
    observation_variance: Float[Array, ""]


def _gaussian_observation(dimension: int) -> np.ndarray:
    """Return the deterministic observation vector for G1."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return np.linspace(-1.0, 1.0, dimension, dtype=np.float64)


def gaussian_target_oracle(
    *,
    dimension: int,
    observation: np.ndarray | None = None,
    observation_scale: float = 0.7,
    observation_variance: float | None = None,
) -> GaussianTargetOracle:
    """Return the conjugate Normal evidence and posterior moments."""
    if observation_variance is None:
        if observation_scale <= 0.0:
            raise ValueError("observation_scale must be positive")
        effective_observation_variance = observation_scale**2
    else:
        if observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        effective_observation_variance = observation_variance
    observation_array = (
        _gaussian_observation(dimension)
        if observation is None
        else np.asarray(observation, dtype=np.float64)
    )
    if observation_array.shape != (dimension,):
        raise ValueError("observation must have shape (dimension,)")
    marginal_variance = 1.0 + effective_observation_variance
    posterior_variance = effective_observation_variance / marginal_variance
    posterior_mean = observation_array / marginal_variance
    log_evidence = float(
        np.sum(
            -0.5
            * (
                np.log(2.0 * np.pi * marginal_variance)
                + observation_array**2 / marginal_variance
            )
        )
    )
    return GaussianTargetOracle(
        log_evidence=log_evidence,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
    )


def make_gaussian_target_callbacks(
    *,
    dimension: int,
    observation_scale: float = 0.7,
) -> GaussianTargetCallbacks:
    """Build JAX callbacks for the conjugate G1 tempering workload."""
    if observation_scale <= 0.0:
        raise ValueError("observation_scale must be positive")
    observation = jnp.asarray(
        _gaussian_observation(dimension),
        dtype=jnp.float32,
    )
    observation_variance = jnp.asarray(
        observation_scale**2,
        dtype=jnp.float32,
    )

    def initial_sampler(
        key: PRNGKeyArray, num_particles: int, /
    ) -> Float[Array, "num_particles dimension"]:
        return jr.normal(
            key,
            (num_particles, dimension),
            dtype=jnp.float32,
        )

    def log_prior_fn(state: Float[Array, " dimension"], /) -> Float[Array, ""]:
        return jnp.sum(_jax_normal_logpdf(state, 0.0, 1.0))

    def log_likelihood_fn(
        state: Float[Array, " dimension"], /
    ) -> Float[Array, ""]:
        return jnp.sum(
            _jax_normal_logpdf(
                observation,
                state,
                observation_variance,
            )
        )

    return GaussianTargetCallbacks(
        initial_sampler=initial_sampler,
        log_prior_fn=log_prior_fn,
        log_likelihood_fn=log_likelihood_fn,
        observation=observation,
        observation_variance=observation_variance,
    )


class SMC2InitialSampler(Protocol):
    """Draw an inner state cloud conditional on one parameter."""

    def __call__(
        self,
        key: PRNGKeyArray,
        num_particles: int,
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, "num_particles state_dim"]:
        """Draw the initial inner cloud."""
        ...


class UnknownARCallbacks(NamedTuple):
    """Shared P1 callbacks for Liu--West and SMC2."""

    param_initial_sampler: ParamInitialSampler
    log_prior_fn: StaticLogDensity
    liu_west_initial_sampler: DenseInitialSampler
    liu_west_transition_sampler: ParamTransitionSampler
    liu_west_log_observation_fn: ParamLogObservationFn
    liu_west_log_auxiliary_fn: ParamLogObservationFn
    smc2_initial_sampler: SMC2InitialSampler
    smc2_transition_sampler: ParamTransitionSampler
    smc2_log_observation_fn: ParamLogObservationFn


class UnknownAROracle(NamedTuple):
    """Float64 parameter evidence and moments for the P1 grid."""

    log_evidence: float
    posterior_mean: float
    posterior_variance: float


class ExchangeableUnknownAROracle(NamedTuple):
    """Float64 projections for the exchangeable-mean P1 parameterization."""

    log_evidence: float
    aggregate_mean: float
    aggregate_second_moment: float
    orthogonal_spread: float | None
    parameter_mean: np.ndarray
    parameter_covariance: np.ndarray


def _validate_unknown_ar_model(
    model: LGSSM,
    prior_scale: float,
) -> None:
    """Validate assumptions shared by P1 callbacks and its oracle."""
    if model.b:
        raise ValueError("unknown-AR workloads require input_coefficient=0")
    if prior_scale <= 0.0:
        raise ValueError("prior_scale must be positive")


def make_unknown_ar_callbacks(
    model: LGSSM,
    *,
    prior_mean: float = 0.9,
    prior_scale: float = 0.15,
    parameter_dimension: int = 1,
) -> UnknownARCallbacks:
    """Build the shared input-free unknown-AR model for P1.

    At dimension one this is the scalar P1 model used by Liu--West and SMC2.
    Higher dimensions are benchmark-only: exchangeable coordinates have prior
    variance ``parameter_dimension * prior_scale**2`` and their mean is the AR
    coefficient.  The effective coefficient therefore retains the scalar P1
    prior while exercising Liu--West's full parameter covariance operations.
    """
    _validate_unknown_ar_model(model, prior_scale)
    if parameter_dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    prior_mean_value = jnp.asarray(prior_mean, dtype=jnp.float32)
    coordinate_prior_scale = jnp.asarray(
        math.sqrt(parameter_dimension) * prior_scale,
        dtype=jnp.float32,
    )
    initial_mean = jnp.asarray(model.m0, dtype=jnp.float32)
    initial_scale = jnp.asarray(math.sqrt(model.p0), dtype=jnp.float32)
    transition_scale = jnp.asarray(math.sqrt(model.q), dtype=jnp.float32)
    transition_variance = jnp.asarray(model.q, dtype=jnp.float32)
    observation_variance = jnp.asarray(model.r, dtype=jnp.float32)

    def param_initial_sampler(
        key: PRNGKeyArray, num_particles: int, /
    ) -> Float[Array, "num_particles param_dim"]:
        return prior_mean_value + coordinate_prior_scale * jr.normal(
            key,
            (num_particles, parameter_dimension),
            dtype=jnp.float32,
        )

    def log_prior_fn(params: Float[Array, " param_dim"], /) -> Float[Array, ""]:
        values = jnp.asarray(params, dtype=jnp.float32)
        if parameter_dimension == 1:
            return _jax_normal_logpdf(
                values[0],
                prior_mean_value,
                coordinate_prior_scale**2,
            )
        return jnp.sum(
            _jax_normal_logpdf(
                values,
                prior_mean_value,
                coordinate_prior_scale**2,
            )
        )

    def effective_coefficient(
        params: Float[Array, " param_dim"],
    ) -> Float[Array, ""]:
        values = jnp.asarray(params, dtype=jnp.float32)
        if parameter_dimension == 1:
            return values[0]
        return jnp.mean(values)

    def liu_west_initial_sampler(
        key: PRNGKeyArray, num_particles: int, /
    ) -> Float[Array, "num_particles 1"]:
        return initial_mean + initial_scale * jr.normal(
            key,
            (num_particles, 1),
            dtype=jnp.float32,
        )

    def transition_sampler(
        key: PRNGKeyArray,
        state: Float[Array, " 1"],
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, " 1"]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        mean = effective_coefficient(params) * state_array
        return mean + transition_scale * jr.normal(
            key,
            state_array.shape,
            dtype=jnp.float32,
        )

    def log_observation_fn(
        emission: Float[Array, " 1"],
        state: Float[Array, " 1"],
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, ""]:
        del params
        return _jax_normal_logpdf(
            emission[0],
            state[0],
            observation_variance,
        )

    def log_auxiliary_fn(
        emission: Float[Array, " 1"],
        state: Float[Array, " 1"],
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, ""]:
        state_array = jnp.asarray(state, dtype=jnp.float32)
        predicted_mean = effective_coefficient(params) * state_array[0]
        return _jax_normal_logpdf(
            emission[0],
            predicted_mean,
            transition_variance + observation_variance,
        )

    def smc2_initial_sampler(
        key: PRNGKeyArray,
        num_particles: int,
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, "num_particles 1"]:
        del params
        return initial_mean + initial_scale * jr.normal(
            key,
            (num_particles, 1),
            dtype=jnp.float32,
        )

    return UnknownARCallbacks(
        param_initial_sampler=param_initial_sampler,
        log_prior_fn=log_prior_fn,
        liu_west_initial_sampler=liu_west_initial_sampler,
        liu_west_transition_sampler=transition_sampler,
        liu_west_log_observation_fn=log_observation_fn,
        liu_west_log_auxiliary_fn=log_auxiliary_fn,
        smc2_initial_sampler=smc2_initial_sampler,
        smc2_transition_sampler=transition_sampler,
        smc2_log_observation_fn=log_observation_fn,
    )


def unknown_ar_grid_oracle(
    model: LGSSM,
    emissions: np.ndarray | Float[Array, "..."],
    *,
    prior_mean: float = 0.9,
    prior_scale: float = 0.15,
    num_points: int = 20_001,
) -> UnknownAROracle:
    """Numerically integrate a float64 Kalman likelihood over the AR prior."""
    _validate_unknown_ar_model(model, prior_scale)
    if num_points < 3 or num_points % 2 == 0:
        raise ValueError("num_points must be an odd integer of at least 3")
    observations = _as_scalar_series(emissions, name="emissions")
    if observations.shape[0] != model.timesteps:
        raise ValueError("emissions length must match model.timesteps")

    grid = np.linspace(
        prior_mean - 8.0 * prior_scale,
        prior_mean + 8.0 * prior_scale,
        num_points,
        dtype=np.float64,
    )
    log_joint = np.empty(num_points, dtype=np.float64)
    prior_variance = prior_scale**2
    prior_constant = math.log(2.0 * math.pi * prior_variance)
    for index, coefficient in enumerate(grid):
        candidate = replace(
            model,
            autoregressive_coefficient=float(coefficient),
        )
        log_likelihood = kalman_oracle(candidate, observations).log_evidence
        log_prior = -0.5 * (
            prior_constant + (coefficient - prior_mean) ** 2 / prior_variance
        )
        log_joint[index] = log_likelihood + log_prior

    offset = float(np.max(log_joint))
    shifted = np.exp(log_joint - offset)
    shifted_evidence = float(np.trapezoid(shifted, grid))
    density = shifted / shifted_evidence
    posterior_mean = float(np.trapezoid(density * grid, grid))
    posterior_second = float(np.trapezoid(density * grid**2, grid))
    return UnknownAROracle(
        log_evidence=offset + math.log(shifted_evidence),
        posterior_mean=posterior_mean,
        posterior_variance=posterior_second - posterior_mean**2,
    )


def exchangeable_unknown_ar_oracle(
    model: LGSSM,
    emissions: np.ndarray | Float[Array, "..."],
    *,
    parameter_dimension: int,
    prior_mean: float = 0.9,
    prior_scale: float = 0.15,
    num_points: int = 20_001,
) -> ExchangeableUnknownAROracle:
    """Lift the scalar grid oracle to exchangeable mean coordinates.

    Independent coordinates have variance ``d * prior_scale**2`` and the
    scalar AR coefficient is their mean.  Conditional Gaussian identities
    therefore recover the full parameter covariance without multidimensional
    quadrature while leaving the scalar evidence unchanged.
    """
    if parameter_dimension < 1:
        raise ValueError("parameter_dimension must be positive")
    scalar = unknown_ar_grid_oracle(
        model,
        emissions,
        prior_mean=prior_mean,
        prior_scale=prior_scale,
        num_points=num_points,
    )
    dimension = parameter_dimension
    prior_variance = prior_scale**2
    parameter_mean = np.full(
        dimension,
        scalar.posterior_mean,
        dtype=np.float64,
    )
    parameter_covariance = dimension * prior_variance * np.eye(
        dimension, dtype=np.float64
    ) + (scalar.posterior_variance - prior_variance) * np.ones(
        (dimension, dimension), dtype=np.float64
    )
    return ExchangeableUnknownAROracle(
        log_evidence=scalar.log_evidence,
        aggregate_mean=scalar.posterior_mean,
        aggregate_second_moment=(
            scalar.posterior_variance + scalar.posterior_mean**2
        ),
        orthogonal_spread=(
            None if dimension == 1 else dimension * prior_variance
        ),
        parameter_mean=parameter_mean,
        parameter_covariance=parameter_covariance,
    )
