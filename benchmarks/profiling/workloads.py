# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Adapters from profiling cells to current smcx production entry points."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from benchmarks.profiling.common import (
    DEFAULT_SEED,
    WORKLOADS,
    replicated_evidence_ratio_gate,
)
from benchmarks.profiling.models import (
    LGSSM,
    StochasticVolatility,
    exchangeable_unknown_ar_oracle,
    gaussian_target_oracle,
    kalman_oracle,
    make_gaussian_target_callbacks,
    make_lgssm_callbacks,
    make_lgssm_data,
    make_stochastic_volatility_callbacks,
    make_stochastic_volatility_data,
    make_unknown_ar_callbacks,
    unknown_ar_grid_oracle,
)
from benchmarks.profiling.representation import (
    TrackingLGSSM,
    flatten_tracking_state,
    make_dense_tracking_callbacks,
    make_tracking_data,
    make_tree_tracking_callbacks,
    tracking_kalman_oracle,
)
from smcx import (
    auxiliary_filter,
    bootstrap_filter,
    guided_filter,
    liu_west_filter,
    multinomial,
    residual,
    smc2,
    stratified,
    systematic,
    temper,
)


@dataclass(frozen=True)
class PreparedWorkload:
    """One fully constructed operation and its untimed evaluators."""

    algorithm: str
    model: str
    execution_mode: str
    operation: Any
    arguments: tuple[Any, ...]
    check: Any
    check_replicates: Any
    measure_work: Any


_RESAMPLERS = {
    "systematic": systematic,
    "stratified": stratified,
    "multinomial": multinomial,
    "residual": residual,
}


def _validated_parameters(
    workload: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject accidental changes to a preregistered parameter envelope."""
    spec = WORKLOADS[workload]
    names = set(parameters)
    accepted = (
        set(spec.smoke_parameters),
        set(spec.baseline_parameters),
    )
    if names not in accepted:
        raise ValueError(
            f"parameters for {workload} do not match a registered profile"
        )
    return dict(parameters)


def _logsumexp_numpy(values: np.ndarray) -> float:
    """Compute a stable scalar log-sum-exp in NumPy float64."""
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _all_finite(value: Any) -> bool:
    """Return whether every array leaf contains finite values."""
    leaves = jax.tree.leaves(jax.device_get(value))
    return bool(
        leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    )


def _unregistered_replicated_gate(outputs: list[Any]) -> dict[str, Any]:
    """Return a visible no-op for workloads without a statistical gate."""
    return {
        "gate": "not_registered",
        "passed": True,
        "replicates": len(outputs),
    }


def _replicated_scalar_mean_gate(
    estimates: list[float],
    *,
    oracle: float,
) -> dict[str, Any]:
    """Compare a replicated scalar estimator mean at five standard errors."""
    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("estimates must contain at least two replicates")
    if not np.all(np.isfinite(values)) or not math.isfinite(oracle):
        raise ValueError("estimates and oracle must be finite")
    standard_deviation = float(np.std(values, ddof=1))
    estimator_se = standard_deviation / math.sqrt(values.size)
    tolerance = max(5.0 * estimator_se, 5e-5)
    error = float(np.mean(values) - oracle)
    return {
        "error": error,
        "estimates": values.tolist(),
        "estimator_se": estimator_se,
        "mean": float(np.mean(values)),
        "oracle": float(oracle),
        "passed": bool(abs(error) <= tolerance),
        "replicates": int(values.size),
        "standard_deviation": standard_deviation,
        "tolerance": tolerance,
    }


def _replicated_vector_mean_gate(
    estimates: list[np.ndarray],
    *,
    absolute_floor: float = 5e-5,
    oracle: np.ndarray,
) -> dict[str, Any]:
    """Compare replicated vector means coordinatewise at five SEs."""
    values = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(oracle, dtype=np.float64)
    if values.ndim != reference.ndim + 1 or values.shape[0] < 2:
        raise ValueError(
            "estimates must contain at least two vectors matching oracle"
        )
    if values.shape[1:] != reference.shape:
        raise ValueError("estimate and oracle shapes must match")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(reference)):
        raise ValueError("estimates and oracle must be finite")
    standard_deviation = np.std(values, axis=0, ddof=1)
    estimator_se = standard_deviation / math.sqrt(values.shape[0])
    tolerance = np.maximum(5.0 * estimator_se, absolute_floor)
    mean = np.mean(values, axis=0)
    error = mean - reference
    coordinate_passes = np.abs(error) <= tolerance
    return {
        "coordinate_passes": coordinate_passes.tolist(),
        "error": error.tolist(),
        "estimator_se": estimator_se.tolist(),
        "mean": mean.tolist(),
        "oracle": reference.tolist(),
        "passed": bool(np.all(coordinate_passes)),
        "replicates": int(values.shape[0]),
        "standard_deviation": standard_deviation.tolist(),
        "tolerance": tolerance.tolist(),
    }


def _weighted_state_moments(posterior: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return final weighted coordinate means and raw second moments."""
    leaves = jax.tree.leaves(posterior.filtered_particles)
    if not leaves:
        raise ValueError("filtered_particles must contain at least one leaf")
    particle_blocks = []
    for leaf in leaves:
        if leaf.ndim < 2:
            raise ValueError(
                "particle leaves must have history and particle axes"
            )
        final = np.asarray(jax.device_get(leaf[-1]), dtype=np.float64)
        particle_blocks.append(final.reshape(final.shape[0], -1))
    particles = np.concatenate(particle_blocks, axis=-1)
    return _weighted_array_moments(
        particles,
        posterior.filtered_log_weights[-1],
    )


def _weighted_tracking_state_moments(
    posterior: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return moments in the tracking oracle's position/velocity order."""
    final_state = jax.tree.map(
        lambda leaf: leaf[-1],
        posterior.filtered_particles,
    )
    particles = flatten_tracking_state(final_state)
    return _weighted_array_moments(
        particles,
        posterior.filtered_log_weights[-1],
    )


def _weighted_array_moments(
    particles: Any,
    log_weights: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted coordinate moments for one final particle cloud."""
    particle_values = np.asarray(jax.device_get(particles), dtype=np.float64)
    if particle_values.ndim != 2:
        raise ValueError("final particles must be a matrix")
    log_weights = np.asarray(
        jax.device_get(log_weights),
        dtype=np.float64,
    )
    if log_weights.shape != (particle_values.shape[0],):
        raise ValueError("final particles and log weights must align")
    log_weights -= _logsumexp_numpy(log_weights)
    weights = np.exp(log_weights)[:, None]
    return (
        np.sum(weights * particle_values, axis=0),
        np.sum(weights * particle_values**2, axis=0),
    )


def _filter_oracle_accuracy_gate(
    posteriors: list[Any],
    *,
    final_mean: np.ndarray,
    final_variance: np.ndarray,
    log_evidence: float,
    state_moment_fn: Any = _weighted_state_moments,
) -> dict[str, Any]:
    """Gate replicated filter evidence and final weighted state moments."""
    reference_mean = np.asarray(final_mean, dtype=np.float64).reshape(-1)
    reference_variance = np.asarray(
        final_variance,
        dtype=np.float64,
    ).reshape(-1)
    if reference_mean.shape != reference_variance.shape:
        raise ValueError("oracle mean and marginal variance shapes must match")
    moments = [state_moment_fn(posterior) for posterior in posteriors]
    evidence = replicated_evidence_ratio_gate(
        [
            float(jax.device_get(posterior.marginal_loglik))
            for posterior in posteriors
        ],
        oracle=log_evidence,
    )
    state_mean = _replicated_vector_mean_gate(
        [mean for mean, _ in moments],
        oracle=reference_mean,
    )
    state_second_moment = _replicated_vector_mean_gate(
        [second_moment for _, second_moment in moments],
        oracle=reference_variance + reference_mean**2,
    )
    return {
        "evidence_ratio": evidence,
        "passed": bool(
            evidence["passed"]
            and state_mean["passed"]
            and state_second_moment["passed"]
        ),
        "replicates": len(posteriors),
        "state_mean": state_mean,
        "state_second_moment": state_second_moment,
    }


def _weighted_parameter_mean(posterior: Any) -> float:
    """Return the final weighted mean of the parameter-coordinate average."""
    parameters = np.asarray(
        jax.device_get(posterior.filtered_params[-1]),
        dtype=np.float64,
    )
    log_weights = np.asarray(
        jax.device_get(posterior.filtered_log_weights[-1]),
        dtype=np.float64,
    )
    log_weights -= _logsumexp_numpy(log_weights)
    aggregate = np.mean(parameters, axis=-1)
    return float(np.sum(np.exp(log_weights) * aggregate))


def _weighted_parameter_second_moment(posterior: Any) -> float:
    """Return the weighted raw second moment of the coordinate average."""
    parameters = np.asarray(
        jax.device_get(posterior.filtered_params[-1]),
        dtype=np.float64,
    )
    log_weights = np.asarray(
        jax.device_get(posterior.filtered_log_weights[-1]),
        dtype=np.float64,
    )
    log_weights -= _logsumexp_numpy(log_weights)
    aggregate = np.mean(parameters, axis=-1)
    return float(np.sum(np.exp(log_weights) * aggregate**2))


def _weighted_parameter_orthogonal_spread(posterior: Any) -> float:
    """Return mean squared spread orthogonal to the coordinate average."""
    parameters = np.asarray(
        jax.device_get(posterior.filtered_params[-1]),
        dtype=np.float64,
    )
    dimension = parameters.shape[-1]
    if dimension < 2:
        raise ValueError("orthogonal spread requires at least two coordinates")
    log_weights = np.asarray(
        jax.device_get(posterior.filtered_log_weights[-1]),
        dtype=np.float64,
    )
    log_weights -= _logsumexp_numpy(log_weights)
    centered = parameters - np.mean(parameters, axis=-1, keepdims=True)
    particle_spread = np.sum(centered**2, axis=-1) / (dimension - 1)
    return float(np.sum(np.exp(log_weights) * particle_spread))


def _f32_rounded_model(model: Any) -> Any:
    """Round floating dataclass fields to the worker's exact f32 values."""
    updates = {
        field.name: float(np.float32(value))
        for field in fields(model)
        if isinstance((value := getattr(model, field.name)), float)
    }
    return replace(model, **updates)


def _f32_oracle_array(value: Any) -> np.ndarray:
    """Return the exact worker input values represented in float64."""
    return np.asarray(value, dtype=np.float32).astype(np.float64)


def _filter_correctness(
    posterior: Any,
    *,
    num_particles: int,
    state_shapes: tuple[tuple[int, ...], ...],
    timesteps: int,
    store_history: bool,
) -> dict[str, Any]:
    """Check structural and log-weight invariants shared by filters."""
    history_length = timesteps if store_history else 1
    log_weights_raw = np.asarray(jax.device_get(posterior.filtered_log_weights))
    ancestors_raw = np.asarray(jax.device_get(posterior.ancestors))
    ess_raw = np.asarray(jax.device_get(posterior.ess))
    increments_raw = np.asarray(
        jax.device_get(posterior.log_evidence_increments)
    )
    marginal_raw = np.asarray(jax.device_get(posterior.marginal_loglik))
    leaves = jax.tree.leaves(jax.device_get(posterior.filtered_particles))
    particle_shapes_ok = len(leaves) == len(state_shapes) and all(
        np.asarray(leaf).shape
        == (history_length, num_particles, *expected_shape)
        for leaf, expected_shape in zip(leaves, state_shapes, strict=True)
    )
    particle_dtypes_ok = bool(leaves) and all(
        np.asarray(leaf).dtype == np.dtype(np.float32) for leaf in leaves
    )
    weight_shape_ok = log_weights_raw.shape == (
        history_length,
        num_particles,
    )
    ancestor_shape_ok = ancestors_raw.shape == (
        history_length,
        num_particles,
    )
    summary_shapes_ok = bool(
        marginal_raw.shape == ()
        and weight_shape_ok
        and ancestor_shape_ok
        and ess_raw.shape == (timesteps,)
        and increments_raw.shape == (timesteps,)
    )
    float32 = np.dtype(np.float32)
    summary_dtypes_ok = bool(
        marginal_raw.dtype == float32
        and log_weights_raw.dtype == float32
        and ess_raw.dtype == float32
        and increments_raw.dtype == float32
        and ancestors_raw.dtype == np.dtype(np.int32)
    )
    dtypes_ok = bool(particle_dtypes_ok and summary_dtypes_ok)
    shapes_ok = bool(particle_shapes_ok and summary_shapes_ok)
    ancestor_range_ok = bool(
        ancestor_shape_ok
        and np.all(ancestors_raw >= 0)
        and np.all(ancestors_raw < num_particles)
    )

    log_weights = log_weights_raw.astype(np.float64, copy=False)
    ess = ess_raw.astype(np.float64, copy=False)
    increments = increments_raw.astype(np.float64, copy=False)
    marginal = (
        float(marginal_raw)
        if marginal_raw.shape == () and np.isfinite(marginal_raw)
        else math.nan
    )
    normalized_error = (
        abs(_logsumexp_numpy(log_weights[-1]))
        if weight_shape_ok and np.all(np.isfinite(log_weights[-1]))
        else math.inf
    )
    evidence_error = (
        abs(float(np.sum(increments)) - marginal)
        if increments.shape == (timesteps,)
        and np.all(np.isfinite(increments))
        and math.isfinite(marginal)
        else math.inf
    )
    evidence_tolerance = (
        max(1e-3, 2e-5 * abs(marginal)) if math.isfinite(marginal) else math.inf
    )
    ess_bounds_ok = bool(
        ess.shape == (timesteps,)
        and np.all(np.isfinite(ess))
        and np.all(ess > 0.0)
        and np.all(ess <= num_particles * (1.0 + 5e-6))
    )
    finite = _all_finite(posterior)
    passed = all((
        finite,
        ancestor_range_ok,
        dtypes_ok,
        ess_bounds_ok,
        shapes_ok,
        normalized_error <= 2e-5,
        evidence_error <= evidence_tolerance,
    ))
    return {
        "ancestor_range_ok": ancestor_range_ok,
        "ancestor_shape_ok": ancestor_shape_ok,
        "dtypes_ok": dtypes_ok,
        "evidence_identity_error": evidence_error,
        "evidence_identity_tolerance": evidence_tolerance,
        "ess_bounds_ok": ess_bounds_ok,
        "final_log_weight_lse_error": normalized_error,
        "finite": finite,
        "passed": bool(passed),
        "particle_dtypes_ok": particle_dtypes_ok,
        "particle_shapes_ok": particle_shapes_ok,
        "shapes_ok": shapes_ok,
        "summary_dtypes_ok": summary_dtypes_ok,
        "summary_shapes_ok": summary_shapes_ok,
        "weight_shape_ok": weight_shape_ok,
    }


def _filter_work_metrics(
    posterior: Any,
    *,
    num_particles: int,
    resampling_observable: bool,
    resampling_threshold: float,
    timesteps: int,
    store_history: bool,
) -> dict[str, Any]:
    """Summarize actual filter work without instrumenting compiled code."""
    ess = np.asarray(jax.device_get(posterior.ess), dtype=np.float64)
    leaves = jax.tree.leaves(posterior.filtered_particles)
    scalar_dimension = sum(
        int(np.prod(leaf.shape[2:], dtype=np.int64)) for leaf in leaves
    )
    if resampling_observable:
        resampling_event_count = int(
            np.sum(ess[:-1] < resampling_threshold * num_particles)
        )
    elif resampling_threshold <= 0.0:
        resampling_event_count = 0
    elif resampling_threshold > 1.0:
        resampling_event_count = timesteps - 1
    else:
        resampling_event_count = None
    return {
        "history_entries": (
            timesteps * num_particles if store_history else num_particles
        ),
        "mean_ess": float(np.mean(ess)),
        "minimum_ess": float(np.min(ess)),
        "resampling_event_count": resampling_event_count,
        "state_leaf_count": len(leaves),
        "state_scalar_dimension": scalar_dimension,
    }


def _observation_variance(regime: str) -> tuple[float, float]:
    """Return L1 observation variance and committed outlier magnitude."""
    values = {
        "diffuse": (2.0, 0.0),
        "calibrated": (0.3, 0.0),
        "sharp": (0.03, 6.0),
    }
    try:
        return values[regime]
    except KeyError as error:
        raise ValueError(f"unknown observation regime: {regime}") from error


def _resampling_weights(
    num_particles: int,
    regime: str,
) -> jax.Array:
    """Construct one preregistered normalized resampling-weight vector."""
    if num_particles < 2:
        raise ValueError("resampler workloads require at least two particles")
    if regime == "uniform":
        weights = jnp.ones(num_particles, dtype=jnp.float32)
    elif regime == "moderately_uneven":
        weights = jnp.exp(
            -jnp.linspace(0.0, 5.0, num_particles, dtype=jnp.float32)
        )
    elif regime == "one_dominant":
        tail = 0.1 / (num_particles - 1)
        weights = jnp.full(num_particles, tail, dtype=jnp.float32)
        weights = weights.at[0].set(0.9)
    elif regime == "zero_tail":
        positive_count = num_particles - num_particles // 4
        positive = jnp.exp(
            -jnp.linspace(0.0, 5.0, positive_count, dtype=jnp.float32)
        )
        weights = jnp.pad(positive, (0, num_particles - positive_count))
    else:
        raise ValueError(f"unknown weight regime: {regime}")
    return weights / jnp.sum(weights)


_RESAMPLER_CONTIGUOUS_PARTITION_COUNT = 16
_RESAMPLER_HASH_PARTITION_COUNT = 8
_RESAMPLER_HASH_MULTIPLIER = 104_729
_RESAMPLER_HASH_OFFSET = 12_345
_RESAMPLER_HASH_MODULUS = 1_000_003


def _contiguous_resampler_partition(
    indices: np.ndarray,
    num_particles: int,
) -> np.ndarray:
    """Map ancestor indices into sixteen contiguous probability probes."""
    return np.minimum(
        _RESAMPLER_CONTIGUOUS_PARTITION_COUNT * indices // num_particles,
        _RESAMPLER_CONTIGUOUS_PARTITION_COUNT - 1,
    )


def _hashed_resampler_partition(indices: np.ndarray) -> np.ndarray:
    """Map ancestor indices through a fixed independent affine partition."""
    mixed = (
        indices.astype(np.int64) * _RESAMPLER_HASH_MULTIPLIER
        + _RESAMPLER_HASH_OFFSET
    ) % _RESAMPLER_HASH_MODULUS
    return mixed % _RESAMPLER_HASH_PARTITION_COUNT


def _prepare_lgssm(
    workload: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare one of the three exact L1 standard-filter workloads."""
    num_particles = int(parameters["num_particles"])
    timesteps = int(parameters["timesteps"])
    threshold = float(parameters["resampling_threshold"])
    store_history = bool(parameters["store_history"])
    variance, outlier = _observation_variance(
        str(parameters["observation_regime"])
    )
    model = LGSSM(
        timesteps=timesteps,
        observation_variance=variance,
        outlier_standard_deviations=outlier,
    )
    data = make_lgssm_data(model, seed=DEFAULT_SEED)
    callbacks = make_lgssm_callbacks(model)
    emissions = jnp.asarray(data.emissions, dtype=jnp.float32)
    inputs = jnp.asarray(data.inputs, dtype=jnp.float32)
    key = jr.key(seed)

    if workload == "bootstrap_lgssm":

        def operation(run_key, observations, controls):
            return bootstrap_filter(
                run_key,
                callbacks.initial_sampler,
                callbacks.transition_sampler,
                callbacks.log_observation_fn,
                observations,
                num_particles,
                resampling_threshold=threshold,
                inputs=controls,
                store_history=store_history,
            )

    elif workload == "auxiliary_lgssm":

        def operation(run_key, observations, controls):
            return auxiliary_filter(
                run_key,
                callbacks.initial_sampler,
                callbacks.transition_sampler,
                callbacks.log_observation_fn,
                callbacks.log_auxiliary_fn,
                observations,
                num_particles,
                resampling_threshold=threshold,
                inputs=controls,
                store_history=store_history,
            )

    elif workload == "guided_lgssm":

        def operation(run_key, observations, controls):
            return guided_filter(
                run_key,
                callbacks.initial_sampler,
                callbacks.proposal_sampler,
                callbacks.log_proposal_fn,
                callbacks.log_transition_fn,
                callbacks.log_observation_fn,
                observations,
                num_particles,
                resampling_threshold=threshold,
                inputs=controls,
                store_history=store_history,
            )

    else:  # pragma: no cover - private dispatcher contract
        raise ValueError(f"unknown L1 workload: {workload}")

    def check(posterior):
        return _filter_correctness(
            posterior,
            num_particles=num_particles,
            state_shapes=((1,),),
            timesteps=timesteps,
            store_history=store_history,
        )

    def measure_work(posterior):
        return _filter_work_metrics(
            posterior,
            num_particles=num_particles,
            resampling_observable=workload != "auxiliary_lgssm",
            resampling_threshold=threshold,
            timesteps=timesteps,
            store_history=store_history,
        )

    oracle = kalman_oracle(
        _f32_rounded_model(model),
        _f32_oracle_array(data.emissions),
        _f32_oracle_array(data.inputs),
    )

    def check_replicates(posteriors):
        return _filter_oracle_accuracy_gate(
            posteriors,
            final_mean=np.asarray([oracle.filtered_means[-1]]),
            final_variance=np.asarray([oracle.filtered_variances[-1]]),
            log_evidence=oracle.log_evidence,
        )

    spec = WORKLOADS[workload]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, emissions, inputs),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def _prepare_sv(
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare the nonlinear stochastic-volatility bootstrap workload."""
    num_particles = int(parameters["num_particles"])
    timesteps = int(parameters["timesteps"])
    threshold = float(parameters["resampling_threshold"])
    store_history = bool(parameters["store_history"])
    model = StochasticVolatility(timesteps=timesteps)
    data = make_stochastic_volatility_data(model, seed=DEFAULT_SEED + 1)
    callbacks = make_stochastic_volatility_callbacks(model)
    emissions = jnp.asarray(data.emissions, dtype=jnp.float32)
    key = jr.key(seed)

    def operation(run_key, observations):
        return bootstrap_filter(
            run_key,
            callbacks.initial_sampler,
            callbacks.transition_sampler,
            callbacks.log_observation_fn,
            observations,
            num_particles,
            resampling_threshold=threshold,
            store_history=store_history,
        )

    def check(posterior):
        return _filter_correctness(
            posterior,
            num_particles=num_particles,
            state_shapes=((1,),),
            timesteps=timesteps,
            store_history=store_history,
        )

    def measure_work(posterior):
        return _filter_work_metrics(
            posterior,
            num_particles=num_particles,
            resampling_observable=True,
            resampling_threshold=threshold,
            timesteps=timesteps,
            store_history=store_history,
        )

    spec = WORKLOADS["bootstrap_sv"]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, emissions),
        check=check,
        check_replicates=_unregistered_replicated_gate,
        measure_work=measure_work,
    )


def _prepare_liu_west(
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare P1 for Liu--West parameter learning."""
    num_particles = int(parameters["num_particles"])
    parameter_dimension = int(parameters["parameter_dimension"])
    timesteps = int(parameters["timesteps"])
    threshold = float(parameters["resampling_threshold"])
    shrinkage = float(parameters["shrinkage"])
    store_history = bool(parameters["store_history"])
    model = LGSSM(timesteps=timesteps, input_coefficient=0.0)
    data = make_lgssm_data(model, seed=DEFAULT_SEED + 2)
    callbacks = make_unknown_ar_callbacks(
        model,
        parameter_dimension=parameter_dimension,
    )
    emissions = jnp.asarray(data.emissions, dtype=jnp.float32)
    key = jr.key(seed)

    def operation(run_key, observations):
        # Profiling workers disable x64.  Mirror that campaign contract while
        # tracing under the CPU test suite, where x64 is intentionally on.
        with jax.enable_x64(False):
            return liu_west_filter(
                run_key,
                callbacks.liu_west_initial_sampler,
                callbacks.liu_west_transition_sampler,
                callbacks.liu_west_log_observation_fn,
                callbacks.liu_west_log_auxiliary_fn,
                callbacks.param_initial_sampler,
                observations,
                num_particles,
                shrinkage=shrinkage,
                resampling_threshold=threshold,
                store_history=store_history,
            )

    def check(posterior):
        result = _filter_correctness(
            posterior,
            num_particles=num_particles,
            state_shapes=((1,),),
            timesteps=timesteps,
            store_history=store_history,
        )
        history_length = timesteps if store_history else 1
        parameter_array = np.asarray(jax.device_get(posterior.filtered_params))
        parameter_shapes_ok = parameter_array.shape == (
            history_length,
            num_particles,
            parameter_dimension,
        )
        parameter_dtypes_ok = parameter_array.dtype == np.dtype(np.float32)
        parameters_finite = bool(np.all(np.isfinite(parameter_array)))
        result["parameters_finite"] = parameters_finite
        result["parameter_dtypes_ok"] = parameter_dtypes_ok
        result["parameter_shapes_ok"] = parameter_shapes_ok
        result["passed"] = bool(
            result["passed"]
            and parameter_dtypes_ok
            and parameters_finite
            and parameter_shapes_ok
        )
        return result

    def measure_work(posterior):
        result = _filter_work_metrics(
            posterior,
            num_particles=num_particles,
            resampling_observable=threshold > 1.0,
            resampling_threshold=threshold,
            timesteps=timesteps,
            store_history=store_history,
        )
        result["parameter_dimension"] = int(posterior.filtered_params.shape[-1])
        return result

    def check_replicates(posteriors):
        oracle = exchangeable_unknown_ar_oracle(
            _f32_rounded_model(model),
            _f32_oracle_array(data.emissions),
            parameter_dimension=parameter_dimension,
            prior_mean=float(np.float32(0.9)),
            prior_scale=float(np.float32(0.15)),
        )
        evidence_ratios = [
            math.exp(
                float(jax.device_get(posterior.marginal_loglik))
                - oracle.log_evidence
            )
            for posterior in posteriors
        ]
        evidence = _replicated_scalar_mean_gate(
            evidence_ratios,
            oracle=1.0,
        )
        parameter_mean = _replicated_scalar_mean_gate(
            [_weighted_parameter_mean(posterior) for posterior in posteriors],
            oracle=oracle.aggregate_mean,
        )
        parameter_second_moment = _replicated_scalar_mean_gate(
            [
                _weighted_parameter_second_moment(posterior)
                for posterior in posteriors
            ],
            oracle=oracle.aggregate_second_moment,
        )
        orthogonal_spread = (
            {
                "gate": "not_applicable",
                "passed": True,
                "reason": "parameter_dimension=1",
                "replicates": len(posteriors),
            }
            if oracle.orthogonal_spread is None
            else _replicated_scalar_mean_gate(
                [
                    _weighted_parameter_orthogonal_spread(posterior)
                    for posterior in posteriors
                ],
                oracle=oracle.orthogonal_spread,
            )
        )
        return {
            "evidence_ratio": evidence,
            "orthogonal_spread": orthogonal_spread,
            "parameter_mean": parameter_mean,
            "parameter_second_moment": parameter_second_moment,
            "passed": bool(
                evidence["passed"]
                and orthogonal_spread["passed"]
                and parameter_mean["passed"]
                and parameter_second_moment["passed"]
            ),
            "replicates": len(posteriors),
        }

    spec = WORKLOADS["liu_west_unknown_ar"]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, emissions),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def _prepare_temper(
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare the conjugate Gaussian tempered-SMC workload."""
    dimension = int(parameters["dimension"])
    num_particles = int(parameters["num_particles"])
    num_mcmc_steps = int(parameters["num_mcmc_steps"])
    target_ess = float(parameters["target_ess"])
    callbacks = make_gaussian_target_callbacks(dimension=dimension)
    oracle = gaussian_target_oracle(
        dimension=dimension,
        observation=_f32_oracle_array(callbacks.observation),
        observation_variance=float(
            jax.device_get(callbacks.observation_variance)
        ),
    )
    key = jr.key(seed)

    def operation(run_key):
        # Profiling workers disable x64.  Mirror that campaign contract under
        # the CPU test suite, where x64 is intentionally enabled.
        with jax.enable_x64(False):
            return temper(
                run_key,
                callbacks.initial_sampler,
                callbacks.log_prior_fn,
                callbacks.log_likelihood_fn,
                num_particles,
                num_mcmc_steps=num_mcmc_steps,
                target_ess=target_ess,
            )

    def check(posterior):
        particles = np.asarray(jax.device_get(posterior.particles))
        log_weights_raw = np.asarray(jax.device_get(posterior.log_weights))
        marginal_raw = np.asarray(jax.device_get(posterior.marginal_loglik))
        temperatures_raw = np.asarray(jax.device_get(posterior.temperatures))
        ess_raw = np.asarray(jax.device_get(posterior.ess))
        acceptance_raw = np.asarray(jax.device_get(posterior.acceptance_rates))
        trace_shapes_ok = bool(
            temperatures_raw.ndim == 1
            and temperatures_raw.size > 0
            and ess_raw.shape == temperatures_raw.shape
            and acceptance_raw.shape == temperatures_raw.shape
        )
        shapes_ok = bool(
            particles.shape == (num_particles, dimension)
            and log_weights_raw.shape == (num_particles,)
            and marginal_raw.shape == ()
            and trace_shapes_ok
        )
        f32 = np.dtype(np.float32)
        dtypes_ok = all(
            value.dtype == f32
            for value in (
                particles,
                log_weights_raw,
                marginal_raw,
                temperatures_raw,
                ess_raw,
                acceptance_raw,
            )
        )
        log_weights = log_weights_raw.astype(np.float64, copy=False)
        temperatures = temperatures_raw.astype(np.float64, copy=False)
        ess = ess_raw.astype(np.float64, copy=False)
        acceptance = acceptance_raw.astype(np.float64, copy=False)
        normalized_error = (
            abs(_logsumexp_numpy(log_weights))
            if log_weights.shape == (num_particles,)
            and np.all(np.isfinite(log_weights))
            else math.inf
        )
        uniform_log_weight_error = (
            float(np.max(np.abs(log_weights + math.log(num_particles))))
            if log_weights.shape == (num_particles,)
            and np.all(np.isfinite(log_weights))
            else math.inf
        )
        uniform_log_weights_ok = uniform_log_weight_error <= 2e-5
        finite = _all_finite(posterior)
        temperature_trace_ok = bool(
            trace_shapes_ok
            and np.all(np.isfinite(temperatures))
            and np.all(temperatures > 0.0)
            and np.all(temperatures <= 1.0 + 1e-6)
            and np.all(np.diff(temperatures) > 0.0)
            and abs(float(temperatures[-1]) - 1.0) <= 1e-6
        )
        ess_bounds_ok = bool(
            trace_shapes_ok
            and np.all(np.isfinite(ess))
            and np.all(ess > 0.0)
            and np.all(ess <= num_particles * (1.0 + 5e-6))
        )
        acceptance_bounds_ok = bool(
            trace_shapes_ok
            and np.all(np.isfinite(acceptance))
            and np.all(acceptance >= 0.0)
            and np.all(acceptance <= 1.0)
        )
        marginal = (
            float(marginal_raw)
            if marginal_raw.shape == () and np.isfinite(marginal_raw)
            else math.nan
        )
        return {
            "absolute_log_evidence_error": abs(marginal - oracle.log_evidence),
            "acceptance_bounds_ok": acceptance_bounds_ok,
            "dtypes_ok": dtypes_ok,
            "ess_bounds_ok": ess_bounds_ok,
            "final_log_weight_lse_error": normalized_error,
            "finite": finite,
            "passed": bool(
                finite
                and shapes_ok
                and dtypes_ok
                and ess_bounds_ok
                and acceptance_bounds_ok
                and temperature_trace_ok
                and normalized_error <= 2e-5
                and uniform_log_weights_ok
            ),
            "reached_temperature_one": bool(
                temperatures.ndim == 1
                and temperatures.size
                and abs(float(temperatures[-1]) - 1.0) <= 1e-6
            ),
            "shapes_ok": shapes_ok,
            "temperature_trace_ok": temperature_trace_ok,
            "trace_shapes_ok": trace_shapes_ok,
            "uniform_log_weight_error": uniform_log_weight_error,
            "uniform_log_weights_ok": uniform_log_weights_ok,
        }

    def measure_work(posterior):
        acceptance = np.asarray(
            jax.device_get(posterior.acceptance_rates), dtype=np.float64
        )
        return {
            "dimension": dimension,
            "mean_acceptance_rate": float(np.mean(acceptance)),
            "num_mcmc_steps": num_mcmc_steps,
            "temperature_stages": int(posterior.temperatures.shape[0]),
        }

    def check_replicates(posteriors):
        evidence_ratios = [
            math.exp(
                float(jax.device_get(posterior.marginal_loglik))
                - oracle.log_evidence
            )
            for posterior in posteriors
        ]
        evidence = _replicated_scalar_mean_gate(
            evidence_ratios,
            oracle=1.0,
        )
        posterior_mean = _replicated_vector_mean_gate(
            [
                np.mean(
                    np.asarray(
                        jax.device_get(posterior.particles),
                        dtype=np.float64,
                    ),
                    axis=0,
                )
                for posterior in posteriors
            ],
            oracle=oracle.posterior_mean,
        )
        posterior_second_moment = _replicated_vector_mean_gate(
            [
                np.mean(
                    np.asarray(
                        jax.device_get(posterior.particles),
                        dtype=np.float64,
                    )
                    ** 2,
                    axis=0,
                )
                for posterior in posteriors
            ],
            oracle=oracle.posterior_variance + oracle.posterior_mean**2,
        )
        return {
            "evidence_ratio": evidence,
            "passed": bool(
                evidence["passed"]
                and posterior_mean["passed"]
                and posterior_second_moment["passed"]
            ),
            "posterior_mean": posterior_mean,
            "posterior_second_moment": posterior_second_moment,
            "replicates": len(posteriors),
        }

    spec = WORKLOADS["temper_gaussian"]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key,),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def _prepare_smc2(
    workload: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare the forward or forced-rejuvenation P1 SMC2 workload."""
    num_theta = int(parameters["num_theta"])
    num_x = int(parameters["num_x"])
    timesteps = int(parameters["timesteps"])
    ess_threshold = float(parameters["ess_threshold"])
    num_pmmh_steps = int(parameters["num_pmmh_steps"])
    store_history = bool(parameters["store_history"])
    model = LGSSM(timesteps=timesteps, input_coefficient=0.0)
    data = make_lgssm_data(model, seed=DEFAULT_SEED + 2)
    callbacks = make_unknown_ar_callbacks(model)
    emissions = jnp.asarray(data.emissions, dtype=jnp.float32)
    key = jr.key(seed)

    def operation(run_key, observations):
        # Profiling workers disable x64.  Mirror that campaign contract under
        # the CPU test suite, where x64 is intentionally enabled.
        with jax.enable_x64(False):
            return smc2(
                run_key,
                callbacks.param_initial_sampler,
                callbacks.log_prior_fn,
                callbacks.smc2_initial_sampler,
                callbacks.smc2_transition_sampler,
                callbacks.smc2_log_observation_fn,
                observations,
                num_theta,
                num_x,
                ess_threshold=ess_threshold,
                num_pmmh_steps=num_pmmh_steps,
                store_history=store_history,
            )

    def check(posterior):
        history_length = timesteps if store_history else 1
        marginal_raw = np.asarray(jax.device_get(posterior.marginal_loglik))
        parameters_raw = np.asarray(jax.device_get(posterior.filtered_params))
        log_weights_raw = np.asarray(
            jax.device_get(posterior.filtered_log_weights)
        )
        ess_raw = np.asarray(jax.device_get(posterior.ess))
        increments_raw = np.asarray(
            jax.device_get(posterior.log_evidence_increments)
        )
        acceptance_raw = np.asarray(jax.device_get(posterior.acceptance_rates))
        trace_shapes_ok = bool(
            ess_raw.shape == (timesteps,)
            and increments_raw.shape == (timesteps,)
            and acceptance_raw.shape == (timesteps,)
        )
        shapes_ok = bool(
            marginal_raw.shape == ()
            and parameters_raw.shape == (history_length, num_theta, 1)
            and log_weights_raw.shape == (history_length, num_theta)
            and trace_shapes_ok
        )
        f32 = np.dtype(np.float32)
        dtypes_ok = all(
            value.dtype == f32
            for value in (
                marginal_raw,
                parameters_raw,
                log_weights_raw,
                ess_raw,
                increments_raw,
                acceptance_raw,
            )
        )
        log_weights = log_weights_raw.astype(np.float64, copy=False)
        ess = ess_raw.astype(np.float64, copy=False)
        increments = increments_raw.astype(np.float64, copy=False)
        acceptance = acceptance_raw.astype(np.float64, copy=False)
        marginal = (
            float(marginal_raw)
            if marginal_raw.shape == () and np.isfinite(marginal_raw)
            else math.nan
        )
        normalized_error = (
            abs(_logsumexp_numpy(log_weights[-1]))
            if log_weights.shape == (history_length, num_theta)
            and np.all(np.isfinite(log_weights[-1]))
            else math.inf
        )
        evidence_error = (
            abs(float(np.sum(increments)) - marginal)
            if increments.shape == (timesteps,)
            and np.all(np.isfinite(increments))
            and math.isfinite(marginal)
            else math.inf
        )
        tolerance = (
            max(1e-3, 2e-5 * abs(marginal))
            if math.isfinite(marginal)
            else math.inf
        )
        ess_bounds_ok = bool(
            ess.shape == (timesteps,)
            and np.all(np.isfinite(ess))
            and np.all(ess > 0.0)
            and np.all(ess <= num_theta * (1.0 + 5e-6))
        )
        acceptance_bounds_ok = bool(
            acceptance.shape == (timesteps,)
            and np.all(np.isfinite(acceptance))
            and np.all(acceptance >= 0.0)
            and np.all(acceptance <= 1.0)
        )
        finite = _all_finite(posterior)
        return {
            "acceptance_bounds_ok": acceptance_bounds_ok,
            "dtypes_ok": dtypes_ok,
            "evidence_identity_error": evidence_error,
            "evidence_identity_tolerance": tolerance,
            "ess_bounds_ok": ess_bounds_ok,
            "final_log_weight_lse_error": normalized_error,
            "finite": finite,
            "passed": bool(
                finite
                and shapes_ok
                and dtypes_ok
                and ess_bounds_ok
                and acceptance_bounds_ok
                and normalized_error <= 2e-5
                and evidence_error <= tolerance
            ),
            "shapes_ok": shapes_ok,
            "trace_shapes_ok": trace_shapes_ok,
        }

    def measure_work(posterior):
        acceptance = np.asarray(
            jax.device_get(posterior.acceptance_rates), dtype=np.float64
        )
        return {
            "expected_rejuvenation_opportunities": (
                timesteps if ess_threshold > 1.0 else 0
            ),
            "mean_acceptance_rate": float(np.mean(acceptance)),
            "num_pmmh_steps": num_pmmh_steps,
            "num_theta": num_theta,
            "num_x": num_x,
            "rejuvenation_event_count": (
                timesteps if ess_threshold > 1.0 else 0
            ),
        }

    def check_replicates(posteriors):
        oracle = unknown_ar_grid_oracle(
            _f32_rounded_model(model),
            _f32_oracle_array(data.emissions),
        )
        evidence_ratios = [
            math.exp(
                float(jax.device_get(posterior.marginal_loglik))
                - oracle.log_evidence
            )
            for posterior in posteriors
        ]
        evidence = _replicated_scalar_mean_gate(
            evidence_ratios,
            oracle=1.0,
        )
        parameter_mean = _replicated_scalar_mean_gate(
            [_weighted_parameter_mean(posterior) for posterior in posteriors],
            oracle=oracle.posterior_mean,
        )
        parameter_second_moment = _replicated_scalar_mean_gate(
            [
                _weighted_parameter_second_moment(posterior)
                for posterior in posteriors
            ],
            oracle=oracle.posterior_variance + oracle.posterior_mean**2,
        )
        return {
            "evidence_ratio": evidence,
            "parameter_mean": parameter_mean,
            "parameter_second_moment": parameter_second_moment,
            "passed": bool(
                evidence["passed"]
                and parameter_mean["passed"]
                and parameter_second_moment["passed"]
            ),
            "replicates": len(posteriors),
        }

    spec = WORKLOADS[workload]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, emissions),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def _prepare_resampler(
    workload: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare one public resampling kernel under a fixed weight regime."""
    num_particles = int(parameters["num_particles"])
    regime = str(parameters["weight_regime"])
    weights = _resampling_weights(num_particles, regime)
    key = jr.key(seed)
    algorithm = WORKLOADS[workload].algorithm
    resampler = _RESAMPLERS[algorithm]

    def operation(run_key, probability_weights):
        return resampler(run_key, probability_weights, num_particles)

    def check(ancestors):
        values = np.asarray(jax.device_get(ancestors))
        structural_ok = bool(
            values.shape == (num_particles,)
            and values.dtype == np.int32
            and np.all(values >= 0)
            and np.all(values < num_particles)
        )
        ordered_output_required = algorithm in {
            "multinomial",
            "stratified",
            "systematic",
        }
        ordered_output_ok = bool(
            not ordered_output_required
            or (structural_ok and np.all(np.diff(values.astype(np.int64)) >= 0))
        )
        passed = bool(structural_ok and ordered_output_ok)
        return {
            "index_range_ok": structural_ok,
            "ordered_output_ok": ordered_output_ok,
            "ordered_output_required": ordered_output_required,
            "passed": passed,
        }

    def measure_work(_ancestors):
        return {
            "num_particles": num_particles,
            "weight_regime": regime,
        }

    # Reproduce the public kernels' f32 normalization outside every timed
    # region.  The CDF-based kernels sample the represented CDF increments;
    # residual resampling targets its separately normalized probability vector.
    _, weight_exponent = jnp.frexp(jnp.max(weights))
    scaled_weights = jnp.ldexp(weights, -weight_exponent)
    normalized_weights = scaled_weights / jnp.sum(scaled_weights)
    represented_cdf = jnp.cumsum(scaled_weights)
    represented_cdf /= represented_cdf[-1]
    represented_probabilities = jnp.diff(
        jnp.concatenate((
            jnp.zeros(1, dtype=weights.dtype),
            represented_cdf,
        ))
    )
    expected_weights = np.asarray(
        jax.device_get(
            normalized_weights
            if algorithm == "residual"
            else represented_probabilities
        ),
        dtype=np.float64,
    )
    expected_weights /= np.sum(expected_weights)
    index_values = np.arange(num_particles, dtype=np.int64)
    contiguous_partition = _contiguous_resampler_partition(
        index_values,
        num_particles,
    )
    hashed_partition = _hashed_resampler_partition(index_values)
    expected_contiguous = np.bincount(
        contiguous_partition,
        weights=expected_weights,
        minlength=_RESAMPLER_CONTIGUOUS_PARTITION_COUNT,
    )
    expected_hashed = np.bincount(
        hashed_partition,
        weights=expected_weights,
        minlength=_RESAMPLER_HASH_PARTITION_COUNT,
    )
    expected_cdf = np.asarray(
        jax.device_get(represented_cdf),
        dtype=np.float64,
    )
    expected_residual_floor = np.asarray(
        jax.device_get(jnp.floor(num_particles * normalized_weights)),
        dtype=np.int64,
    )

    def check_replicates(outputs):
        structural_results = [check(output) for output in outputs]
        structural_outputs_ok = bool(
            structural_results
            and all(result["passed"] for result in structural_results)
        )
        observed_contiguous = []
        observed_hashed = []
        output_values = []
        for output in outputs:
            values = np.asarray(jax.device_get(output), dtype=np.int64)
            output_values.append(values)
            observed_contiguous.append(
                np.bincount(
                    _contiguous_resampler_partition(values, num_particles),
                    minlength=_RESAMPLER_CONTIGUOUS_PARTITION_COUNT,
                )
                / num_particles
            )
            observed_hashed.append(
                np.bincount(
                    _hashed_resampler_partition(values),
                    minlength=_RESAMPLER_HASH_PARTITION_COUNT,
                )
                / num_particles
            )
        contiguous_gate = _replicated_vector_mean_gate(
            observed_contiguous,
            absolute_floor=max(5e-5, 2.0 / num_particles),
            oracle=expected_contiguous,
        )
        hashed_gate = _replicated_vector_mean_gate(
            observed_hashed,
            absolute_floor=max(5e-5, 2.0 / num_particles),
            oracle=expected_hashed,
        )

        cdf_discrepancies: list[float] | None = None
        cdf_discrepancy_tolerance: float | None = None
        cdf_discrepancy_ok = True
        if algorithm in {"stratified", "systematic"}:
            cdf_discrepancies = []
            for values in output_values:
                counts = np.bincount(values, minlength=num_particles)
                empirical_cdf = np.cumsum(counts, dtype=np.float64)
                empirical_cdf /= num_particles
                cdf_discrepancies.append(
                    float(np.max(np.abs(empirical_cdf - expected_cdf)))
                )
            # One query per stratum bounds CDF discrepancy by 1/N in exact
            # arithmetic.  One extra 1/N plus an f32 accumulation floor covers
            # represented-CDF endpoint rounding.
            cdf_discrepancy_tolerance = 2.0 / num_particles + 5e-6
            cdf_discrepancy_ok = bool(
                all(
                    discrepancy <= cdf_discrepancy_tolerance
                    for discrepancy in cdf_discrepancies
                )
            )

        residual_floor_counts_ok: bool | None = None
        if algorithm == "residual":
            residual_floor_counts_ok = bool(
                all(
                    np.all(
                        np.bincount(values, minlength=num_particles)
                        >= expected_residual_floor
                    )
                    for values in output_values
                )
            )

        joint_invariants_passed = bool(
            structural_outputs_ok
            and cdf_discrepancy_ok
            and residual_floor_counts_ok is not False
        )
        joint_invariants = {
            "cdf_discrepancies": cdf_discrepancies,
            "cdf_discrepancy_ok": cdf_discrepancy_ok,
            "cdf_discrepancy_tolerance": cdf_discrepancy_tolerance,
            "passed": joint_invariants_passed,
            "residual_floor_counts_ok": residual_floor_counts_ok,
            "structural_outputs_ok": structural_outputs_ok,
        }
        return {
            # Keep the original name as an explicit compatibility alias for
            # report readers while exposing the stronger partition names.
            "bin_probabilities": contiguous_gate,
            "contiguous_probabilities": contiguous_gate,
            "hashed_probabilities": hashed_gate,
            "joint_invariants": joint_invariants,
            "passed": bool(
                contiguous_gate["passed"]
                and hashed_gate["passed"]
                and joint_invariants_passed
            ),
            "replicates": len(outputs),
        }

    spec = WORKLOADS[workload]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, weights),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def _prepare_tracking(
    workload: str,
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare one of the mathematically identical L2 state encodings."""
    num_particles = int(parameters["num_particles"])
    timesteps = int(parameters["timesteps"])
    covariance_regime = str(parameters["covariance_regime"])
    threshold = float(parameters["resampling_threshold"])
    store_history = bool(parameters["store_history"])
    model = TrackingLGSSM(
        timesteps=timesteps,
        covariance_regime=covariance_regime,
    )
    data = make_tracking_data(model, seed=DEFAULT_SEED + 3)
    if workload == "bootstrap_tracking_dense":
        callbacks = make_dense_tracking_callbacks(model)
    elif workload == "bootstrap_tracking_pytree":
        callbacks = make_tree_tracking_callbacks(model)
    else:  # pragma: no cover - private dispatcher contract
        raise ValueError(f"unknown tracking workload: {workload}")
    emissions = jnp.asarray(data.emissions, dtype=jnp.float32)
    inputs = jnp.asarray(data.inputs, dtype=jnp.float32)
    key = jr.key(seed)

    def operation(run_key, observations, controls):
        return bootstrap_filter(
            run_key,
            callbacks.initial_sampler,
            callbacks.transition_sampler,
            callbacks.log_observation_fn,
            observations,
            num_particles,
            resampling_threshold=threshold,
            inputs=controls,
            store_history=store_history,
        )

    def check(posterior):
        state_shapes = (
            ((4,),) if workload == "bootstrap_tracking_dense" else ((2,), (2,))
        )
        return _filter_correctness(
            posterior,
            num_particles=num_particles,
            state_shapes=state_shapes,
            timesteps=timesteps,
            store_history=store_history,
        )

    def check_replicates(posteriors):
        oracle = tracking_kalman_oracle(
            _f32_rounded_model(model),
            _f32_oracle_array(data.emissions),
            _f32_oracle_array(data.inputs),
        )
        return _filter_oracle_accuracy_gate(
            posteriors,
            final_mean=oracle.filtered_means[-1],
            final_variance=np.diag(oracle.filtered_covariances[-1]),
            log_evidence=oracle.log_evidence,
            state_moment_fn=(
                _weighted_state_moments
                if workload == "bootstrap_tracking_dense"
                else _weighted_tracking_state_moments
            ),
        )

    def measure_work(posterior):
        return _filter_work_metrics(
            posterior,
            num_particles=num_particles,
            resampling_observable=True,
            resampling_threshold=threshold,
            timesteps=timesteps,
            store_history=store_history,
        )

    spec = WORKLOADS[workload]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, emissions, inputs),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def _prepare_dynamax_lgssm(
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Prepare the optional public-Dynamax-callback integration arm."""
    from benchmarks.profiling.dynamax_adapter import (
        make_dynamax_lgssm_adapter,
    )

    num_particles = int(parameters["num_particles"])
    timesteps = int(parameters["timesteps"])
    threshold = float(parameters["resampling_threshold"])
    store_history = bool(parameters["store_history"])
    variance, outlier = _observation_variance(
        str(parameters["observation_regime"])
    )
    model = LGSSM(
        timesteps=timesteps,
        observation_variance=variance,
        outlier_standard_deviations=outlier,
    )
    data = make_lgssm_data(model, seed=DEFAULT_SEED)
    callbacks = make_dynamax_lgssm_adapter(model)
    emissions = jnp.asarray(data.emissions, dtype=jnp.float32)
    inputs = jnp.asarray(data.inputs, dtype=jnp.float32)
    key = jr.key(seed)
    oracle = kalman_oracle(
        _f32_rounded_model(model),
        _f32_oracle_array(data.emissions),
        _f32_oracle_array(data.inputs),
    )

    # Dynamax applies input u[t] while predicting x[t + 1], whereas smcx's L1
    # callback contract supplies u[t] while propagating x[t - 1] -> x[t].  Shift
    # only this independent exact-filter check; the timed adapter receives the
    # original controls and therefore implements L1 directly.
    dynamax_inputs = jnp.concatenate((inputs[1:], inputs[-1:]), axis=0)
    dynamax_log_evidence_array = callbacks.model.marginal_log_prob(
        callbacks.params,
        emissions,
        inputs=dynamax_inputs,
    )
    jax.block_until_ready(dynamax_log_evidence_array)
    dynamax_log_evidence = float(jax.device_get(dynamax_log_evidence_array))
    dynamax_oracle_error = abs(dynamax_log_evidence - oracle.log_evidence)
    # Both analytic recurrences accumulate T linear-algebra updates.  This
    # relative allowance plus an absolute floor covers their rounding-order
    # difference while remaining far below a one-step input-alignment defect.
    dynamax_oracle_tolerance = max(
        1e-4,
        2e-6 * abs(oracle.log_evidence),
    )
    dynamax_oracle_ok = bool(
        math.isfinite(dynamax_log_evidence)
        and dynamax_oracle_error <= dynamax_oracle_tolerance
    )
    if not dynamax_oracle_ok:
        raise RuntimeError(
            "Dynamax marginal_log_prob does not match the independent "
            "float64 Kalman oracle: "
            f"error={dynamax_oracle_error}, "
            f"tolerance={dynamax_oracle_tolerance}"
        )

    def operation(run_key, observations, controls):
        return bootstrap_filter(
            run_key,
            callbacks.initial_sampler,
            callbacks.transition_sampler,
            callbacks.log_observation_fn,
            observations,
            num_particles,
            resampling_threshold=threshold,
            inputs=controls,
            store_history=store_history,
        )

    def check(posterior):
        result = _filter_correctness(
            posterior,
            num_particles=num_particles,
            state_shapes=((1,),),
            timesteps=timesteps,
            store_history=store_history,
        )
        result["dynamax_kalman_log_evidence_error"] = dynamax_oracle_error
        result["dynamax_kalman_log_evidence_tolerance"] = (
            dynamax_oracle_tolerance
        )
        result["dynamax_kalman_oracle_ok"] = dynamax_oracle_ok
        result["passed"] = bool(result["passed"] and dynamax_oracle_ok)
        return result

    def check_replicates(posteriors):
        return _filter_oracle_accuracy_gate(
            posteriors,
            final_mean=np.asarray([oracle.filtered_means[-1]]),
            final_variance=np.asarray([oracle.filtered_variances[-1]]),
            log_evidence=oracle.log_evidence,
        )

    def measure_work(posterior):
        return _filter_work_metrics(
            posterior,
            num_particles=num_particles,
            resampling_observable=True,
            resampling_threshold=threshold,
            timesteps=timesteps,
            store_history=store_history,
        )

    spec = WORKLOADS["bootstrap_lgssm_dynamax"]
    return PreparedWorkload(
        algorithm=spec.algorithm,
        model=spec.model,
        execution_mode=spec.execution_mode,
        operation=operation,
        arguments=(key, emissions, inputs),
        check=check,
        check_replicates=check_replicates,
        measure_work=measure_work,
    )


def prepare_workload(
    workload: str,
    *,
    parameters: Mapping[str, Any],
    seed: int,
) -> PreparedWorkload:
    """Construct one registered profiling workload outside timed regions."""
    if workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {workload}")
    validated = _validated_parameters(workload, parameters)
    if workload in {
        "bootstrap_lgssm",
        "auxiliary_lgssm",
        "guided_lgssm",
    }:
        return _prepare_lgssm(workload, validated, seed)
    if workload == "bootstrap_sv":
        return _prepare_sv(validated, seed)
    if workload == "liu_west_unknown_ar":
        return _prepare_liu_west(validated, seed)
    if workload == "temper_gaussian":
        return _prepare_temper(validated, seed)
    if workload in {"smc2_forward", "smc2_forced"}:
        return _prepare_smc2(workload, validated, seed)
    if workload.startswith("resample_"):
        return _prepare_resampler(workload, validated, seed)
    if workload in {
        "bootstrap_tracking_dense",
        "bootstrap_tracking_pytree",
    }:
        return _prepare_tracking(workload, validated, seed)
    if workload == "bootstrap_lgssm_dynamax":
        return _prepare_dynamax_lgssm(validated, seed)
    raise ValueError(f"unknown workload: {workload}")
