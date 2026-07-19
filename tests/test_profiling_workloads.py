# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Execution contracts for profiling workload adapters."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from benchmarks.profiling import workloads as profiling_workloads
from benchmarks.profiling.common import WORKLOADS, plan_cells
from benchmarks.profiling.models import (
    gaussian_target_oracle,
    make_gaussian_target_callbacks,
)
from benchmarks.profiling.workloads import prepare_workload


def _run_replicates(prepared, *, count: int = 2):
    """Execute independent replicas of one outer-jittable workload."""
    operation = jax.jit(prepared.operation)
    outputs = [
        operation(key, *prepared.arguments[1:])
        for key in jr.split(jr.key(20260720), count)
    ]
    jax.block_until_ready(outputs)
    return outputs


@pytest.fixture(scope="module")
def smoke_cells():
    """Return one small CPU cell for every registered workload."""
    return plan_cells("smoke", platforms=("cpu",), seed=20260719)


def test_every_registered_workload_has_one_smoke_cell(smoke_cells) -> None:
    expected = {
        workload
        for workload, spec in WORKLOADS.items()
        if "smoke" in spec.profiles
    }
    assert {cell.workload for cell in smoke_cells} == expected


@pytest.mark.parametrize(
    "workload",
    sorted(
        workload
        for workload, spec in WORKLOADS.items()
        if "smoke" in spec.profiles
    ),
)
def test_smoke_workload_executes_and_passes_invariants(
    workload: str,
    smoke_cells,
) -> None:
    cell = next(cell for cell in smoke_cells if cell.workload == workload)
    prepared = prepare_workload(
        workload,
        parameters=cell.parameters,
        seed=20260719,
    )
    spec = WORKLOADS[workload]
    assert prepared.algorithm == spec.algorithm
    assert prepared.model == spec.model
    assert prepared.execution_mode == spec.execution_mode
    assert callable(prepared.check_replicates)

    operation = (
        jax.jit(prepared.operation)
        if prepared.execution_mode == "whole_program_jit"
        else prepared.operation
    )
    output = operation(*prepared.arguments)
    jax.block_until_ready(output)
    correctness = prepared.check(output)
    metrics = prepared.measure_work(output)
    assert correctness["passed"], correctness
    if workload in {
        "auxiliary_lgssm",
        "bootstrap_lgssm",
        "bootstrap_sv",
        "guided_lgssm",
        "liu_west_unknown_ar",
    }:
        assert correctness["particle_shapes_ok"]
        assert correctness["particle_dtypes_ok"]
    if workload in {
        "bootstrap_lgssm",
        "bootstrap_sv",
        "guided_lgssm",
    }:
        assert isinstance(metrics["resampling_event_count"], int)
    if workload in {"auxiliary_lgssm", "liu_west_unknown_ar"}:
        assert metrics["resampling_event_count"] is None
    if workload.startswith("smc2_"):
        assert isinstance(metrics["rejuvenation_event_count"], int)
    assert metrics


def test_prepare_workload_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown workload"):
        prepare_workload("missing", parameters={}, seed=1)


def test_prepare_workload_rejects_parameter_drift(smoke_cells) -> None:
    cell = next(
        cell for cell in smoke_cells if cell.workload == "bootstrap_lgssm"
    )
    with pytest.raises(ValueError, match="parameters"):
        prepare_workload(
            cell.workload,
            parameters={**cell.parameters, "unregistered": True},
            seed=1,
        )


@pytest.mark.parametrize(
    "workload",
    ["bootstrap_lgssm", "auxiliary_lgssm", "guided_lgssm"],
)
def test_l1_replicated_gate_includes_final_state_moments(
    workload: str,
) -> None:
    parameters = {
        **WORKLOADS[workload].smoke_parameters,
        "num_particles": 64,
        "timesteps": 8,
    }
    prepared = prepare_workload(
        workload,
        parameters=parameters,
        seed=20260719,
    )

    replicated = prepared.check_replicates(_run_replicates(prepared))

    assert set(replicated) == {
        "evidence_ratio",
        "passed",
        "replicates",
        "state_mean",
        "state_second_moment",
    }
    assert replicated["replicates"] == 2
    assert np.asarray(replicated["state_mean"]["oracle"]).shape == (1,)
    assert np.asarray(replicated["state_second_moment"]["oracle"]).shape == (1,)


def test_liu_west_parameter_dimension_axis_executes_and_gates() -> None:
    parameters = {
        **WORKLOADS["liu_west_unknown_ar"].baseline_parameters,
        "num_particles": 128,
        "parameter_dimension": 4,
        "resampling_threshold": 1.1,
        "timesteps": 8,
    }
    prepared = prepare_workload(
        "liu_west_unknown_ar",
        parameters=parameters,
        seed=20260719,
    )
    operation = jax.jit(prepared.operation)
    outputs = [
        operation(key, prepared.arguments[1])
        for key in jr.split(jr.key(20260720), 2)
    ]
    jax.block_until_ready(outputs)

    correctness = prepared.check(outputs[0])
    assert correctness["passed"], correctness
    assert correctness["parameter_shapes_ok"]
    assert correctness["parameter_dtypes_ok"]
    assert outputs[0].filtered_params.shape == (1, 128, 4)

    metrics = prepared.measure_work(outputs[0])
    assert metrics["parameter_dimension"] == 4
    assert metrics["resampling_event_count"] == 7

    replicated = prepared.check_replicates(outputs)
    assert replicated["replicates"] == 2
    assert set(replicated) == {
        "evidence_ratio",
        "orthogonal_spread",
        "parameter_mean",
        "parameter_second_moment",
        "passed",
        "replicates",
    }
    assert replicated["orthogonal_spread"]["oracle"] == pytest.approx(0.09)


def test_forced_liu_west_dimension_cell_records_99_resampling_events() -> None:
    parameters = {
        **WORKLOADS["liu_west_unknown_ar"].baseline_parameters,
        "num_particles": 32,
        "parameter_dimension": 1,
        "resampling_threshold": 1.1,
    }
    prepared = prepare_workload(
        "liu_west_unknown_ar",
        parameters=parameters,
        seed=20260719,
    )
    output = jax.jit(prepared.operation)(*prepared.arguments)
    jax.block_until_ready(output)
    assert prepared.measure_work(output)["resampling_event_count"] == 99


@pytest.mark.parametrize(
    ("threshold", "expected_events"),
    [(0.0, 0), (1.1, 7)],
)
def test_auxiliary_forced_regimes_have_exact_resampling_work(
    threshold: float,
    expected_events: int,
) -> None:
    """Controlled thresholds expose exact work despite hidden lookahead ESS."""
    parameters = {
        **WORKLOADS["auxiliary_lgssm"].smoke_parameters,
        "resampling_threshold": threshold,
        "timesteps": 8,
    }
    prepared = prepare_workload(
        "auxiliary_lgssm",
        parameters=parameters,
        seed=20260719,
    )
    output = jax.jit(prepared.operation)(*prepared.arguments)
    jax.block_until_ready(output)

    assert (
        prepared.measure_work(output)["resampling_event_count"]
        == expected_events
    )


@pytest.mark.parametrize(
    "regime",
    ["uniform", "moderately_uneven", "one_dominant", "zero_tail"],
)
def test_resampler_scaling_weight_regimes_execute(regime: str) -> None:
    parameters = {
        **WORKLOADS["resample_systematic"].smoke_parameters,
        "weight_regime": regime,
    }
    prepared = prepare_workload(
        "resample_systematic",
        parameters=parameters,
        seed=20260719,
    )
    output = jax.jit(prepared.operation)(*prepared.arguments)
    jax.block_until_ready(output)
    assert prepared.check(output)["passed"]
    assert prepared.measure_work(output) == {
        "num_particles": parameters["num_particles"],
        "weight_regime": regime,
    }


def test_resampler_replicates_reject_shape_correct_wrong_distribution() -> None:
    prepared = prepare_workload(
        "resample_multinomial",
        parameters={
            **WORKLOADS["resample_multinomial"].smoke_parameters,
            "weight_regime": "moderately_uneven",
        },
        seed=20260719,
    )
    operation = jax.jit(prepared.operation)
    keys = jr.split(jr.key(20260720), 8)
    outputs = [operation(key, prepared.arguments[1]) for key in keys]
    jax.block_until_ready(outputs)
    valid = prepared.check_replicates(outputs)
    assert valid["passed"], valid
    assert valid["replicates"] == 8

    # Preserve the old four-bin counts exactly while destroying the target
    # distribution inside every bin.  A four-bin-only gate cannot detect this.
    num_particles = outputs[0].shape[0]
    old_bin_width = num_particles // 4
    bin_first = [
        (output // old_bin_width * old_bin_width).astype(jnp.int32)
        for output in outputs
    ]
    assert all(prepared.check(output)["passed"] for output in bin_first)
    invalid = prepared.check_replicates(bin_first)
    assert not invalid["passed"]
    assert not invalid["contiguous_probabilities"]["passed"]
    assert not invalid["hashed_probabilities"]["passed"]


@pytest.mark.parametrize(
    "workload",
    [
        "resample_multinomial",
        "resample_residual",
        "resample_stratified",
        "resample_systematic",
    ],
)
def test_resampler_replicated_gate_checks_complementary_partitions(
    workload: str,
) -> None:
    prepared = prepare_workload(
        workload,
        parameters=WORKLOADS[workload].smoke_parameters,
        seed=20260719,
    )
    operation = jax.jit(prepared.operation)
    outputs = [
        operation(key, prepared.arguments[1])
        for key in jr.split(jr.key(20260720), 8)
    ]
    jax.block_until_ready(outputs)

    result = prepared.check_replicates(outputs)

    assert result["passed"], result
    assert result["contiguous_probabilities"]["passed"]
    assert result["hashed_probabilities"]["passed"]
    assert result["joint_invariants"]["passed"]


def test_ordered_resampler_gate_rejects_reversed_output() -> None:
    prepared = prepare_workload(
        "resample_systematic",
        parameters=WORKLOADS["resample_systematic"].smoke_parameters,
        seed=20260719,
    )
    output = jax.jit(prepared.operation)(*prepared.arguments)
    jax.block_until_ready(output)

    invalid = prepared.check(output[::-1])

    assert not invalid["ordered_output_ok"]
    assert not invalid["passed"]


def test_residual_gate_requires_deterministic_floor_copies() -> None:
    parameters = WORKLOADS["resample_residual"].smoke_parameters
    residual_prepared = prepare_workload(
        "resample_residual",
        parameters=parameters,
        seed=20260719,
    )
    multinomial_prepared = prepare_workload(
        "resample_multinomial",
        parameters=parameters,
        seed=20260719,
    )
    operation = jax.jit(multinomial_prepared.operation)
    outputs = [
        operation(key, multinomial_prepared.arguments[1])
        for key in jr.split(jr.key(20260720), 8)
    ]
    jax.block_until_ready(outputs)

    invalid = residual_prepared.check_replicates(outputs)

    assert not invalid["joint_invariants"]["residual_floor_counts_ok"]
    assert not invalid["passed"]


@pytest.mark.parametrize(
    "workload",
    ["bootstrap_tracking_dense", "bootstrap_tracking_pytree"],
)
def test_tracking_representation_workloads_execute(workload: str) -> None:
    prepared = prepare_workload(
        workload,
        parameters={
            "covariance_regime": "correlated",
            "num_particles": 128,
            "resampling_threshold": 0.5,
            "store_history": False,
            "timesteps": 12,
        },
        seed=20260719,
    )
    outputs = _run_replicates(prepared)
    output = outputs[0]
    correctness = prepared.check(output)
    assert correctness["passed"], correctness
    assert correctness["particle_shapes_ok"]
    assert correctness["particle_dtypes_ok"]
    assert prepared.measure_work(output)["state_scalar_dimension"] == 4

    replicated = prepared.check_replicates(outputs)
    assert set(replicated) == {
        "evidence_ratio",
        "passed",
        "replicates",
        "state_mean",
        "state_second_moment",
    }
    assert np.asarray(replicated["state_mean"]["oracle"]).shape == (4,)
    assert np.asarray(replicated["state_second_moment"]["oracle"]).shape == (4,)
    expected_means = []
    expected_second_moments = []
    for posterior in outputs:
        state = posterior.filtered_particles
        particles = (
            np.asarray(state[-1], dtype=np.float64)
            if workload == "bootstrap_tracking_dense"
            else np.concatenate(
                (
                    np.asarray(state.position[-1], dtype=np.float64),
                    np.asarray(state.velocity[-1], dtype=np.float64),
                ),
                axis=-1,
            )
        )
        log_weights = np.asarray(
            posterior.filtered_log_weights[-1],
            dtype=np.float64,
        )
        weights = np.exp(log_weights - float(np.max(log_weights)))
        weights /= np.sum(weights)
        expected_means.append(np.sum(weights[:, None] * particles, axis=0))
        expected_second_moments.append(
            np.sum(weights[:, None] * particles**2, axis=0)
        )
    np.testing.assert_allclose(
        replicated["state_mean"]["mean"],
        np.mean(expected_means, axis=0),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        replicated["state_second_moment"]["mean"],
        np.mean(expected_second_moments, axis=0),
        rtol=1e-12,
        atol=1e-12,
    )


def test_temper_gate_enforces_full_contract() -> None:
    prepared = prepare_workload(
        "temper_gaussian",
        parameters=WORKLOADS["temper_gaussian"].smoke_parameters,
        seed=20260719,
    )
    output = prepared.operation(*prepared.arguments)
    jax.block_until_ready(output)

    valid = prepared.check(output)
    assert valid["passed"], valid
    assert valid["shapes_ok"]
    assert valid["dtypes_ok"]
    assert valid["ess_bounds_ok"]
    assert valid["acceptance_bounds_ok"]
    assert valid["temperature_trace_ok"]

    wrong_shapes = {
        "particles": output.particles[:, :-1],
        "log_weights": output.log_weights[:-1],
        "marginal_loglik": output.marginal_loglik[None],
        "temperatures": output.temperatures[:-1],
        "ess": output.ess[:-1],
        "acceptance_rates": output.acceptance_rates[:-1],
    }
    for field, malformed in wrong_shapes.items():
        result = prepared.check(output._replace(**{field: malformed}))
        assert not result["passed"], field
        assert not result["shapes_ok"], field

    for field in output._fields:
        malformed = getattr(output, field).astype(jnp.int32)
        result = prepared.check(output._replace(**{field: malformed}))
        assert not result["passed"], field
        assert not result["dtypes_ok"], field

    bad_ess = output._replace(ess=jnp.zeros_like(output.ess))
    assert not prepared.check(bad_ess)["ess_bounds_ok"]
    bad_acceptance = output._replace(
        acceptance_rates=jnp.full_like(output.acceptance_rates, 1.1)
    )
    assert not prepared.check(bad_acceptance)["acceptance_bounds_ok"]
    bad_temperatures = output._replace(
        temperatures=jnp.zeros_like(output.temperatures)
    )
    assert not prepared.check(bad_temperatures)["temperature_trace_ok"]
    duplicate_temperature = output._replace(
        temperatures=jnp.concatenate((
            output.temperatures,
            output.temperatures[-1:],
        )),
        ess=jnp.concatenate((output.ess, output.ess[-1:])),
        acceptance_rates=jnp.concatenate((
            output.acceptance_rates,
            output.acceptance_rates[-1:],
        )),
    )
    duplicate_result = prepared.check(duplicate_temperature)
    assert duplicate_result["shapes_ok"]
    assert not duplicate_result["temperature_trace_ok"]
    assert not duplicate_result["passed"]

    nonuniform = jnp.linspace(
        -1.0,
        1.0,
        output.log_weights.shape[0],
        dtype=jnp.float32,
    )
    nonuniform -= jax.nn.logsumexp(nonuniform)
    nonuniform_result = prepared.check(output._replace(log_weights=nonuniform))
    assert not nonuniform_result["uniform_log_weights_ok"]
    assert not nonuniform_result["passed"]


def test_temper_replicates_reject_within_cloud_collapse() -> None:
    parameters = WORKLOADS["temper_gaussian"].smoke_parameters
    prepared = prepare_workload(
        "temper_gaussian",
        parameters=parameters,
        seed=20260719,
    )
    template = prepared.operation(*prepared.arguments)
    callbacks = make_gaussian_target_callbacks(
        dimension=parameters["dimension"],
    )
    oracle = gaussian_target_oracle(
        dimension=parameters["dimension"],
        observation=np.asarray(callbacks.observation, dtype=np.float32),
        observation_variance=float(callbacks.observation_variance),
    )
    offset = np.sqrt(oracle.posterior_variance)
    outputs = []
    for sign in (-1.0, 1.0):
        point = jnp.asarray(
            oracle.posterior_mean + sign * offset,
            dtype=jnp.float32,
        )
        collapsed = jnp.broadcast_to(point, template.particles.shape)
        outputs.extend([
            template._replace(
                particles=collapsed,
                marginal_loglik=jnp.asarray(
                    oracle.log_evidence,
                    dtype=jnp.float32,
                ),
            )
            for _ in range(6)
        ])

    replicated = prepared.check_replicates(outputs)

    assert replicated["posterior_mean"]["passed"]
    assert replicated["posterior_second_moment"]["passed"]
    assert not replicated["posterior_within_variance"]["passed"]
    assert not replicated["passed"]


def test_temper_oracle_uses_callback_f32_observation_variance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def recording_oracle(**kwargs):
        captured.update(kwargs)
        return gaussian_target_oracle(**kwargs)

    monkeypatch.setattr(
        profiling_workloads,
        "gaussian_target_oracle",
        recording_oracle,
    )

    prepare_workload(
        "temper_gaussian",
        parameters=WORKLOADS["temper_gaussian"].smoke_parameters,
        seed=20260719,
    )

    operative_variance = float(np.float32(0.7**2))
    assert captured["observation_variance"] == operative_variance
    assert "observation_scale" not in captured


def test_standard_filter_gate_enforces_all_summary_contracts() -> None:
    prepared = prepare_workload(
        "bootstrap_lgssm",
        parameters=WORKLOADS["bootstrap_lgssm"].smoke_parameters,
        seed=20260719,
    )
    output = jax.jit(prepared.operation)(*prepared.arguments)
    jax.block_until_ready(output)

    valid = prepared.check(output)
    assert valid["passed"], valid
    assert valid["shapes_ok"]
    assert valid["dtypes_ok"]
    assert valid["ancestor_range_ok"]

    wrong_shapes = {
        "marginal_loglik": output.marginal_loglik[None],
        "filtered_log_weights": output.filtered_log_weights[:, :-1],
        "ancestors": output.ancestors[:, :-1],
        "ess": output.ess[:-1],
        "log_evidence_increments": output.log_evidence_increments[:-1],
    }
    for field, malformed in wrong_shapes.items():
        result = prepared.check(output._replace(**{field: malformed}))
        assert not result["shapes_ok"], field
        assert not result["passed"], field

    float_fields = (
        "marginal_loglik",
        "filtered_log_weights",
        "ess",
        "log_evidence_increments",
    )
    for field in float_fields:
        malformed = np.asarray(
            jax.device_get(getattr(output, field)),
            dtype=np.float64,
        )
        result = prepared.check(output._replace(**{field: malformed}))
        assert not result["dtypes_ok"], field
        assert not result["passed"], field

    float_ancestors = output.ancestors.astype(jnp.float32)
    ancestor_dtype_result = prepared.check(
        output._replace(ancestors=float_ancestors)
    )
    assert not ancestor_dtype_result["dtypes_ok"]
    assert not ancestor_dtype_result["passed"]

    out_of_range = output.ancestors.at[-1, 0].set(
        output.filtered_log_weights.shape[-1]
    )
    ancestor_range_result = prepared.check(
        output._replace(ancestors=out_of_range)
    )
    assert not ancestor_range_result["ancestor_range_ok"]
    assert not ancestor_range_result["passed"]


@pytest.mark.parametrize("workload", ["smc2_forward", "smc2_forced"])
def test_smc2_gate_enforces_full_contract(workload: str) -> None:
    parameters = {
        **WORKLOADS[workload].smoke_parameters,
        "store_history": workload == "smc2_forced",
    }
    prepared = prepare_workload(
        workload,
        parameters=parameters,
        seed=20260719,
    )
    output = prepared.operation(*prepared.arguments)
    jax.block_until_ready(output)

    valid = prepared.check(output)
    assert valid["passed"], valid
    assert valid["shapes_ok"]
    assert valid["dtypes_ok"]
    assert valid["ess_bounds_ok"]
    assert valid["acceptance_bounds_ok"]
    assert valid["trace_shapes_ok"]

    wrong_shapes = {
        "marginal_loglik": output.marginal_loglik[None],
        "filtered_params": output.filtered_params[..., :0],
        "filtered_log_weights": output.filtered_log_weights[:, :-1],
        "ess": output.ess[:-1],
        "log_evidence_increments": output.log_evidence_increments[:-1],
        "acceptance_rates": output.acceptance_rates[:-1],
    }
    for field, malformed in wrong_shapes.items():
        result = prepared.check(output._replace(**{field: malformed}))
        assert not result["passed"], field
        assert not result["shapes_ok"], field

    for field in output._fields:
        malformed = getattr(output, field).astype(jnp.int32)
        result = prepared.check(output._replace(**{field: malformed}))
        assert not result["passed"], field
        assert not result["dtypes_ok"], field

    bad_ess = output._replace(ess=jnp.zeros_like(output.ess))
    assert not prepared.check(bad_ess)["ess_bounds_ok"]
    bad_acceptance = output._replace(
        acceptance_rates=jnp.full_like(output.acceptance_rates, -0.1)
    )
    assert not prepared.check(bad_acceptance)["acceptance_bounds_ok"]
