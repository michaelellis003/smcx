# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the current-JAX all-algorithm profiling harness."""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import benchmarks.profiling.run as run_module
import benchmarks.profiling.worker as worker_module
from benchmarks.profiling.common import (
    ALGORITHMS,
    PROFILES,
    SCHEMA_VERSION,
    SEED_CONTRACT,
    WORKLOADS,
    build_manifest,
    campaign_identity,
    plan_cells,
    profiling_runtime_flags,
    replicated_evidence_ratio_gate,
    summarize,
    validate_result,
    worker_environment,
)
from benchmarks.profiling.models import (
    LGSSM,
    StochasticVolatility,
    exchangeable_unknown_ar_oracle,
    gaussian_target_oracle,
    guided_log_weight_terms,
    kalman_oracle,
    make_gaussian_target_callbacks,
    make_lgssm_callbacks,
    make_lgssm_data,
    make_stochastic_volatility_callbacks,
    make_unknown_ar_callbacks,
    unknown_ar_grid_oracle,
)
from benchmarks.profiling.run import (
    CampaignIdentityError,
    build_worker_command,
    raw_filename,
    supervise,
)
from benchmarks.profiling.run import main as profiling_main
from benchmarks.profiling.worker import run_cell, run_validation


class _RecordingLock:
    """Minimal context manager used to prove campaign lock scope."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self):
        self._events.append("enter")
        return self

    def __exit__(self, *args) -> None:
        del args
        self._events.append("exit")


def _successful_result() -> dict:
    times = [0.4, 0.2, 0.3]
    return {
        "algorithm": "bootstrap",
        "backend": "cpu",
        "block": 0,
        "correctness": {
            "passed": True,
            "replicated": {
                "gate": "not_requested",
                "passed": True,
                "replicates": 0,
            },
        },
        "correctness_replicates": 0,
        "correctness_level": "structural",
        "dispatch_mode": "asynchronous",
        "environment": {
            "device_kind": "test-cpu",
            "machine": "arm64",
            "macos": "15.0",
            "os": "Darwin",
        },
        "execution_mode": "whole_program_jit",
        "failure": None,
        "first_execution_s": 0.5,
        "lifecycle": {
            "backend_compile_s": 0.2,
            "lowering_s": 0.1,
            "unavailable_reason": None,
        },
        "memory": {"process_max_rss_bytes": 1024},
        "model": "lgssm",
        "parameters": {"num_particles": 128, "timesteps": 20},
        "platform_requested": "cpu",
        "repeats": 3,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_sha256": "b" * 64,
        },
        "steady_summary": summarize(times),
        "steady_times_s": times,
        "versions": {"jax": "0.10.2"},
        "work_metrics": {"minimum_ess": 64.0},
        "workload": "bootstrap_lgssm",
        "warmups": 1,
    }


def _result_for_cell(cell, identity) -> dict:
    """Return a campaign-matching injected timing or validation result."""
    spec = WORKLOADS[cell.workload]
    result = _successful_result()
    result.update({
        "algorithm": spec.algorithm,
        "block": cell.block,
        "correctness_replicates": cell.correctness_replicates,
        "correctness_level": (
            spec.replicated_correctness_level
            if cell.correctness_replicates
            else "structural"
        ),
        "backend": cell.platform,
        "dispatch_mode": ("asynchronous" if cell.platform == "cpu" else "safe"),
        "environment": {
            **identity["host"],
            "device_id": 0,
            "device_kind": "cpu" if cell.platform == "cpu" else "gpu",
            "runtime_flags": profiling_runtime_flags(
                worker_environment(cell.platform, base={})
            ),
        },
        "execution_mode": cell.execution_mode,
        "model": spec.model,
        "parameters": dict(cell.parameters),
        "platform_requested": cell.platform,
        "repeats": cell.repeats,
        "source": identity["source"],
        "versions": identity["packages"],
        "workload": cell.workload,
        "warmups": cell.warmups,
    })
    times = [0.3] * cell.repeats
    result["steady_times_s"] = times
    result["steady_summary"] = summarize(times)
    if cell.correctness_replicates:
        result["correctness"]["replicated"] = {
            "gate": "injected_oracle",
            "passed": True,
            "replicates": cell.correctness_replicates,
        }
    if cell.execution_mode == "host_shell":
        result["lifecycle"] = {
            "backend_compile_s": None,
            "lowering_s": None,
            "unavailable_reason": "host_controlled",
        }
    return result


def _validation_for_cell(cell, identity) -> dict:
    """Return a campaign-matching injected validation sidecar."""
    return {
        "backend": cell.platform,
        "block": cell.block,
        "correctness_level": (
            WORKLOADS[cell.workload].replicated_correctness_level
        ),
        "correctness_replicates": cell.correctness_replicates,
        "dispatch_mode": ("asynchronous" if cell.platform == "cpu" else "safe"),
        "environment": {
            **identity["host"],
            "device_id": 0,
            "device_kind": "cpu" if cell.platform == "cpu" else "gpu",
            "runtime_flags": profiling_runtime_flags(
                worker_environment(cell.platform, base={})
            ),
        },
        "execution_mode": cell.execution_mode,
        "parameters": dict(cell.parameters),
        "platform_requested": cell.platform,
        "replicated": {
            "gate": "injected_oracle",
            "passed": True,
            "replicates": cell.correctness_replicates,
        },
        "schema_version": SCHEMA_VERSION,
        "source": identity["source"],
        "versions": identity["packages"],
        "workload": cell.workload,
    }


def _set_attested_worker_environment(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    """Install the exact flags used by a fresh profiling subprocess."""
    monkeypatch.delenv("PYTHONHOME", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    for name, value in worker_environment(platform, base={}).items():
        monkeypatch.setenv(name, value)


def test_smoke_profile_covers_every_algorithm() -> None:
    cells = plan_cells("smoke", platforms=("cpu",), seed=20260719)
    covered = {WORKLOADS[cell.workload].algorithm for cell in cells}
    assert covered == set(ALGORITHMS)
    assert {cell.execution_mode for cell in cells} == {
        "host_shell",
        "whole_program_jit",
    }


def test_plan_is_deterministic_and_platform_complete() -> None:
    first = plan_cells("smoke", platforms=("cpu", "mps"), seed=19)
    second = plan_cells("smoke", platforms=("cpu", "mps"), seed=19)
    assert first == second
    assert {cell.platform for cell in first} == {"cpu", "mps"}
    assert len(first) == 2 * len(
        plan_cells("smoke", platforms=("cpu",), seed=19)
    )


def test_inferential_plan_interleaves_mathematical_cells_by_block() -> None:
    cells = plan_cells("baseline", platforms=("cpu",), seed=19)
    orders = []
    for block in range(PROFILES["baseline"].blocks):
        block_cells = [cell for cell in cells if cell.block == block]
        orders.append([
            (cell.workload, tuple(sorted(cell.parameters.items())))
            for cell in block_cells
        ])
    expected = set(orders[0])
    assert all(set(order) == expected for order in orders)
    assert len({tuple(order) for order in orders}) == len(orders)
    for mathematical_cell in expected:
        positions = [order.index(mathematical_cell) for order in orders]
        assert len(set(positions)) == len(positions)


def test_manifest_records_frozen_cell_order() -> None:
    cells = plan_cells("smoke", platforms=("cpu",), order_seed=7)
    manifest = build_manifest(
        "smoke",
        cells,
        order_seed=7,
        platforms=("cpu",),
    )
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["profile"] == "smoke"
    assert manifest["order_seed"] == 7
    assert "seed" not in manifest
    assert manifest["seed_contract"] == SEED_CONTRACT
    assert manifest["platforms"] == ["cpu"]
    assert manifest["cells"] == [cell._asdict() for cell in cells]
    identity = manifest["campaign_identity"]
    assert len(identity["source"]["source_sha256"]) == 64
    assert len(identity["source"]["lock_sha256"]) == 64
    assert "tfp-nightly" in identity["packages"]
    assert identity["host"]["machine"]


def test_one_time_dynamax_profile_is_not_permanent() -> None:
    assert "integration" not in PROFILES
    assert not any("dynamax" in workload for workload in WORKLOADS)
    data_seed_offsets = SEED_CONTRACT["data_seed_offsets"]
    assert isinstance(data_seed_offsets, dict)
    assert "dynamax_lgssm" not in data_seed_offsets
    with pytest.raises(ValueError, match="unknown profile: integration"):
        plan_cells("integration", platforms=("cpu",), order_seed=7)


def test_profiles_have_protocol_minimums() -> None:
    assert PROFILES["smoke"].blocks == 1
    assert PROFILES["smoke"].repeats == 1
    assert PROFILES["baseline"].blocks >= 5
    assert PROFILES["baseline"].repeats >= 7


def test_filter_regimes_profile_is_the_preregistered_cross_product() -> None:
    cells = plan_cells("filter-regimes", platforms=("cpu",), seed=11)
    mathematical_cells = {
        (
            cell.workload,
            tuple(sorted(cell.parameters.items())),
        )
        for cell in cells
    }
    assert len(mathematical_cells) == 54
    assert {cell.workload for cell in cells} == {
        "auxiliary_lgssm",
        "bootstrap_lgssm",
        "guided_lgssm",
    }
    assert {cell.parameters["observation_regime"] for cell in cells} == {
        "calibrated",
        "diffuse",
        "sharp",
    }
    assert {cell.parameters["resampling_threshold"] for cell in cells} == {
        0.0,
        0.5,
        1.1,
    }
    assert {cell.parameters["store_history"] for cell in cells} == {
        False,
        True,
    }


def test_scaling_profile_covers_every_preregistered_axis() -> None:
    cells = plan_cells("scaling", platforms=("cpu",), seed=12)
    mathematical_cells = {
        (
            cell.workload,
            tuple(sorted(cell.parameters.items())),
        )
        for cell in cells
    }
    assert len(mathematical_cells) == 75

    standard = [
        cell
        for cell in cells
        if cell.block == 0
        and cell.workload
        in {"bootstrap_lgssm", "auxiliary_lgssm", "guided_lgssm"}
    ]
    assert {cell.parameters["num_particles"] for cell in standard} == {
        1_000,
        10_000,
        100_000,
    }

    parameter_dimension_cells = [
        cell
        for cell in cells
        if cell.block == 0
        and cell.workload == "liu_west_unknown_ar"
        and cell.parameters["num_particles"] == 1_000
        and math.isclose(cell.parameters["resampling_threshold"], 1.1)
    ]
    assert len(parameter_dimension_cells) == 4
    assert {
        cell.parameters["parameter_dimension"]
        for cell in parameter_dimension_cells
    } == {1, 4, 16, 64}
    assert {
        cell.parameters["num_particles"] for cell in parameter_dimension_cells
    } == {1_000}
    assert {
        cell.parameters["timesteps"] for cell in parameter_dimension_cells
    } == {100}
    assert all(
        cell.correctness_replicates == 12 for cell in parameter_dimension_cells
    )

    resampler_cells = [
        cell
        for cell in cells
        if cell.block == 0 and cell.workload.startswith("resample_")
    ]
    assert {cell.parameters["weight_regime"] for cell in resampler_cells} == {
        "moderately_uneven",
        "one_dominant",
        "uniform",
        "zero_tail",
    }
    assert {cell.parameters["num_particles"] for cell in resampler_cells} == {
        10_000,
        100_000,
        1_000_000,
    }
    assert all(cell.correctness_replicates == 128 for cell in resampler_cells)


def test_resampler_validation_keys_preserve_committed_prefix() -> None:
    validation_seed = SEED_CONTRACT["validation_seed"]
    assert isinstance(validation_seed, int)
    root = jr.key(validation_seed)
    committed = jr.split(root, 8)

    keys_64 = worker_module._correctness_keys(
        jax,
        workload="resample_systematic",
        count=64,
    )
    keys_128 = worker_module._correctness_keys(
        jax,
        workload="resample_systematic",
        count=128,
    )

    np.testing.assert_array_equal(
        jr.key_data(keys_64[:8]),
        jr.key_data(committed),
    )
    np.testing.assert_array_equal(
        jr.key_data(keys_128[:64]),
        jr.key_data(keys_64),
    )
    key_rows = np.asarray(jr.key_data(keys_128))
    assert np.unique(key_rows, axis=0).shape[0] == 128

    unchanged = worker_module._correctness_keys(
        jax,
        workload="bootstrap_lgssm",
        count=3,
    )
    np.testing.assert_array_equal(
        jr.key_data(unchanged),
        jr.key_data(jr.split(root, 3)),
    )


def test_oracle_backed_inferential_variants_validate_in_block_zero() -> None:
    for profile in ("filter-regimes", "scaling"):
        cells = plan_cells(profile, platforms=("cpu",), seed=17)
        oracle_cells = [
            cell
            for cell in cells
            if WORKLOADS[cell.workload].replicated_correctness_level
            == "oracle_accuracy"
        ]
        assert oracle_cells
        assert all(
            (cell.correctness_replicates > 0)
            is (
                cell.block == 0
                and not (
                    profile == "filter-regimes"
                    and cell.workload in {"bootstrap_lgssm", "auxiliary_lgssm"}
                    and not cell.parameters["resampling_threshold"]
                )
            )
            for cell in oracle_cells
        )


def test_collapsed_no_resampling_cells_are_structural_only() -> None:
    cells = plan_cells("filter-regimes", platforms=("cpu",), seed=17)
    no_resampling = [
        cell for cell in cells if not cell.parameters["resampling_threshold"]
    ]
    collapsed = [
        cell
        for cell in no_resampling
        if cell.workload in {"bootstrap_lgssm", "auxiliary_lgssm"}
    ]
    stable = [
        cell
        for cell in no_resampling
        if cell.workload == "guided_lgssm" and cell.block == 0
    ]

    assert collapsed
    assert all(cell.correctness_replicates == 0 for cell in collapsed)
    assert stable
    assert all(cell.correctness_replicates == 20 for cell in stable)


def test_representation_history_validation_is_structural_only() -> None:
    cells = plan_cells("representation", platforms=("cpu",), seed=17)
    history_cells = [cell for cell in cells if cell.parameters["store_history"]]
    assert history_cells
    assert all(cell.correctness_replicates == 0 for cell in history_cells)


def test_representation_profile_pairs_dense_and_pytree_states() -> None:
    cells = plan_cells("representation", platforms=("cpu",), seed=13)
    tracking_cells = [
        cell
        for cell in cells
        if cell.workload.startswith("bootstrap_tracking_")
    ]
    mathematical_cells = {
        (
            cell.workload,
            tuple(sorted(cell.parameters.items())),
        )
        for cell in tracking_cells
    }
    assert len(mathematical_cells) == 8
    assert {cell.workload for cell in tracking_cells} == {
        "bootstrap_tracking_dense",
        "bootstrap_tracking_pytree",
    }
    assert {cell.parameters["store_history"] for cell in tracking_cells} == {
        False,
        True,
    }
    assert {
        cell.parameters["covariance_regime"] for cell in tracking_cells
    } == {
        "correlated",
        "diagonal",
    }
    assert {cell.parameters["num_particles"] for cell in tracking_cells} == {
        10_000
    }
    assert {cell.parameters["timesteps"] for cell in tracking_cells} == {200}


def test_representation_profile_covers_liu_west_history() -> None:
    cells = plan_cells("representation", platforms=("cpu",), seed=13)
    liu_west_cells = [
        cell for cell in cells if cell.workload == "liu_west_unknown_ar"
    ]
    mathematical_cells = {
        tuple(sorted(cell.parameters.items())) for cell in liu_west_cells
    }

    assert len(mathematical_cells) == 2
    assert {cell.parameters["store_history"] for cell in liu_west_cells} == {
        False,
        True,
    }
    assert {cell.parameters["num_particles"] for cell in liu_west_cells} == {
        10_000
    }
    assert {
        cell.parameters["parameter_dimension"] for cell in liu_west_cells
    } == {1}
    assert {cell.parameters["timesteps"] for cell in liu_west_cells} == {100}
    assert {
        cell.parameters["resampling_threshold"] for cell in liu_west_cells
    } == {1.1}


def test_scaling_forces_exact_work_for_hidden_resampling_decisions() -> None:
    cells = plan_cells("scaling", platforms=("cpu",), seed=14)
    hidden_decision_cells = [
        cell
        for cell in cells
        if cell.workload in {"auxiliary_lgssm", "liu_west_unknown_ar"}
    ]

    assert hidden_decision_cells
    assert {
        cell.parameters["resampling_threshold"]
        for cell in hidden_decision_cells
    } == {1.1}


def test_summary_retains_robust_statistics() -> None:
    assert summarize([0.4, 0.2, 0.3]) == {
        "iqr_s": pytest.approx(0.1),
        "mad_s": pytest.approx(0.1),
        "median_s": pytest.approx(0.3),
        "min_s": pytest.approx(0.2),
        "q1_s": pytest.approx(0.25),
        "q3_s": pytest.approx(0.35),
    }
    with pytest.raises(ValueError, match="non-empty"):
        summarize([])


def test_replicated_evidence_ratio_gate_is_mc_error_honest() -> None:
    passing = replicated_evidence_ratio_gate(
        [-4.3, -3.8, -4.1, -3.9, -4.0],
        oracle=-4.0,
    )
    assert passing["passed"]
    assert passing["replicates"] == 5

    failing = replicated_evidence_ratio_gate(
        [-2.01, -2.00, -1.99, -2.02, -1.98],
        oracle=-4.0,
    )
    assert not failing["passed"]
    assert passing["tolerance"] >= 5.0 * passing["estimator_se"]

    with pytest.raises(ValueError, match="at least two"):
        replicated_evidence_ratio_gate([-4.0], oracle=-4.0)


def test_result_schema_distinguishes_execution_modes() -> None:
    result = _successful_result()
    validate_result(result)

    host_result = {
        **result,
        "algorithm": "temper",
        "execution_mode": "host_shell",
        "lifecycle": {
            "backend_compile_s": None,
            "lowering_s": None,
            "unavailable_reason": "host_controlled",
        },
        "model": "gaussian_target",
        "workload": "temper_gaussian",
    }
    validate_result(host_result)

    malformed = {**host_result, "lifecycle": result["lifecycle"]}
    with pytest.raises(ValueError, match="host_shell"):
        validate_result(malformed)


def test_result_schema_recomputes_summary() -> None:
    result = _successful_result()
    result["steady_summary"] = {"median_s": 9.0}
    with pytest.raises(ValueError, match="summary"):
        validate_result(result)

    schedule_mismatch = _successful_result()
    schedule_mismatch["repeats"] = 2
    with pytest.raises(ValueError, match="scheduled repeats"):
        validate_result(schedule_mismatch)


@pytest.mark.parametrize("block", [True, -1, 0.5])
def test_result_schema_requires_a_nonnegative_integer_block(block) -> None:
    result = _successful_result()
    result["block"] = block

    with pytest.raises(ValueError, match="block"):
        validate_result(result)


def test_result_schema_cannot_hide_a_failed_replicated_gate() -> None:
    result = _successful_result()
    result["correctness_replicates"] = 2
    result["correctness_level"] = "oracle_accuracy"
    result["correctness"]["replicated"] = {
        "passed": False,
        "replicates": 2,
    }

    with pytest.raises(ValueError, match="replicated gate"):
        validate_result(result)


def test_worker_environment_selects_one_backend() -> None:
    base = {
        "JAX_COMPILATION_CACHE_DIR": "/stale",
        "JAX_ENABLE_X64": "true",
        "JAX_MPS_ASYNC_DISPATCH": "1",
        "JAX_PLATFORM_NAME": "stale",
        "JAX_PLATFORMS": "stale",
        "OMP_NUM_THREADS": "99",
        "PYTHONHOME": "/host/python",
        "PYTHONNOUSERSITE": "0",
        "PYTHONPATH": "/host/imports",
        "XLA_FLAGS": "--stale",
        "KEEP": "yes",
    }
    cpu = worker_environment("cpu", base=base)
    assert cpu["JAX_PLATFORMS"] == "cpu"
    assert "JAX_MPS_ASYNC_DISPATCH" not in cpu
    assert "JAX_PLATFORM_NAME" not in cpu
    assert "JAX_COMPILATION_CACHE_DIR" not in cpu
    assert "OMP_NUM_THREADS" not in cpu
    assert "PYTHONHOME" not in cpu
    assert "PYTHONPATH" not in cpu
    assert "XLA_FLAGS" not in cpu
    assert cpu["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert cpu["JAX_ENABLE_X64"] == "false"
    assert cpu["PYTHONNOUSERSITE"] == "1"
    assert cpu["KEEP"] == "yes"
    assert profiling_runtime_flags(cpu) == {
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_ENABLE_X64": "false",
        "JAX_PLATFORMS": "cpu",
        "PYTHONNOUSERSITE": "1",
    }

    mps = worker_environment("mps", base=base)
    assert mps["JAX_PLATFORMS"] == "mps"
    assert "JAX_MPS_ASYNC_DISPATCH" not in mps


def test_lgssm_data_and_kalman_oracle_are_frozen() -> None:
    model = LGSSM(timesteps=8, observation_variance=0.3)
    data = make_lgssm_data(model, seed=20260719)
    np.testing.assert_allclose(
        data.emissions[:, 0],
        [
            -0.68312900,
            -1.12611748,
            -1.16440476,
            -1.16444222,
            -0.38781474,
            -0.49185037,
            -0.45876055,
            0.01225798,
        ],
        rtol=0.0,
        atol=5e-8,
    )
    oracle = kalman_oracle(model, data.emissions, data.inputs)
    assert oracle.log_evidence == pytest.approx(-6.988928197837548)
    np.testing.assert_allclose(
        oracle.filtered_means[:, 0],
        [
            -0.52548384,
            -0.82238146,
            -0.92528395,
            -0.94787617,
            -0.53521556,
            -0.39708474,
            -0.30824840,
            -0.01531108,
        ],
        rtol=0.0,
        atol=5e-8,
    )


def test_locally_optimal_guided_weight_is_predictive_likelihood() -> None:
    model = LGSSM(timesteps=8, observation_variance=0.3)
    previous = jnp.asarray(-0.4)
    emission = jnp.asarray(-0.8)
    input_t = jnp.asarray([0.2])
    propagated = jnp.asarray([-1.1, -0.5, 0.3])
    log_g, log_f, log_q, log_predictive = guided_log_weight_terms(
        model,
        emission,
        propagated,
        previous,
        input_t,
    )
    np.testing.assert_allclose(
        np.asarray(log_g + log_f - log_q),
        np.broadcast_to(np.asarray(log_predictive), (3,)),
        rtol=2e-6,
        atol=2e-6,
    )


def test_model_callbacks_emit_float32_from_float64_inputs() -> None:
    """Profiling callbacks retain the worker's f32 arithmetic under x64."""

    def assert_f32(value) -> None:
        leaves = jax.tree.leaves(value)
        assert leaves
        assert all(np.asarray(leaf).dtype == np.float32 for leaf in leaves)

    key = jr.key(3)
    state = jnp.asarray([0.2], dtype=jnp.float64)
    other_state = jnp.asarray([-0.1], dtype=jnp.float64)
    emission = jnp.asarray([0.4], dtype=jnp.float64)
    input_t = jnp.asarray([0.3], dtype=jnp.float64)

    lgssm = make_lgssm_callbacks(LGSSM(timesteps=4))
    assert_f32(lgssm.initial_sampler(key, 4, input_t))
    assert_f32(lgssm.transition_sampler(key, state, input_t))
    assert_f32(lgssm.log_observation_fn(emission, state, input_t))
    assert_f32(lgssm.log_auxiliary_fn(emission, state, input_t))
    assert_f32(lgssm.proposal_sampler(key, state, emission, input_t))
    assert_f32(lgssm.log_proposal_fn(emission, other_state, state, input_t))
    assert_f32(lgssm.log_transition_fn(other_state, state, input_t))

    volatility = make_stochastic_volatility_callbacks(
        StochasticVolatility(timesteps=4)
    )
    assert_f32(volatility.initial_sampler(key, 4))
    assert_f32(volatility.transition_sampler(key, state))
    assert_f32(volatility.log_observation_fn(emission, state))

    gaussian = make_gaussian_target_callbacks(dimension=2)
    gaussian_state = jnp.asarray([0.2, -0.3], dtype=jnp.float64)
    assert_f32(gaussian.observation)
    assert_f32(gaussian.initial_sampler(key, 4))
    assert_f32(gaussian.log_prior_fn(gaussian_state))
    assert_f32(gaussian.log_likelihood_fn(gaussian_state))

    unknown_ar = make_unknown_ar_callbacks(
        LGSSM(timesteps=4, input_coefficient=0.0)
    )
    params = jnp.asarray([0.8], dtype=jnp.float64)
    assert_f32(unknown_ar.param_initial_sampler(key, 4))
    assert_f32(unknown_ar.log_prior_fn(params))
    assert_f32(unknown_ar.liu_west_initial_sampler(key, 4))
    assert_f32(unknown_ar.liu_west_transition_sampler(key, state, params))
    assert_f32(unknown_ar.liu_west_log_observation_fn(emission, state, params))
    assert_f32(unknown_ar.liu_west_log_auxiliary_fn(emission, state, params))
    assert_f32(unknown_ar.smc2_initial_sampler(key, 4, params))
    assert_f32(unknown_ar.smc2_transition_sampler(key, state, params))
    assert_f32(unknown_ar.smc2_log_observation_fn(emission, state, params))

    exchangeable = make_unknown_ar_callbacks(
        LGSSM(timesteps=4, input_coefficient=0.0),
        parameter_dimension=4,
    )
    exchangeable_params = jnp.asarray(
        [0.5, 0.7, 1.1, 0.9],
        dtype=jnp.float64,
    )
    parameter_cloud = exchangeable.param_initial_sampler(key, 6)
    assert parameter_cloud.shape == (6, 4)
    assert_f32(parameter_cloud)
    assert_f32(exchangeable.log_prior_fn(exchangeable_params))
    assert_f32(
        exchangeable.liu_west_transition_sampler(
            key,
            state,
            exchangeable_params,
        )
    )
    assert_f32(
        exchangeable.liu_west_log_auxiliary_fn(
            emission,
            state,
            exchangeable_params,
        )
    )


def test_gaussian_target_oracle_is_closed_form() -> None:
    oracle = gaussian_target_oracle(dimension=4, observation_scale=0.7)
    assert oracle.log_evidence == pytest.approx(-5.219018527841555)
    np.testing.assert_allclose(
        oracle.posterior_mean,
        [-0.67114094, -0.22371365, 0.22371365, 0.67114094],
        rtol=0.0,
        atol=5e-8,
    )
    assert oracle.posterior_variance == pytest.approx(0.49 / 1.49)
    assert math.isfinite(oracle.log_evidence)


def test_unknown_ar_grid_oracle_is_frozen_and_converged() -> None:
    model = LGSSM(timesteps=12, input_coefficient=0.0)
    data = make_lgssm_data(model, seed=20260721)
    fine = unknown_ar_grid_oracle(model, data.emissions, num_points=20_001)
    coarse = unknown_ar_grid_oracle(model, data.emissions, num_points=2_001)
    assert fine.log_evidence == pytest.approx(-15.81377712849659, abs=2e-13)
    assert fine.posterior_mean == pytest.approx(0.8637005361751118, abs=2e-13)
    assert fine.posterior_variance == pytest.approx(
        0.013912354380661518,
        abs=2e-13,
    )
    assert coarse.log_evidence == pytest.approx(fine.log_evidence, abs=2e-13)
    assert coarse.posterior_mean == pytest.approx(
        fine.posterior_mean,
        abs=2e-13,
    )
    assert coarse.posterior_variance == pytest.approx(
        fine.posterior_variance,
        abs=2e-13,
    )


def test_exchangeable_unknown_ar_oracle_has_exact_projection_moments() -> None:
    model = LGSSM(timesteps=12, input_coefficient=0.0)
    data = make_lgssm_data(model, seed=20260721)
    scalar = unknown_ar_grid_oracle(model, data.emissions)
    dimension = 4
    prior_scale = 0.15
    oracle = exchangeable_unknown_ar_oracle(
        model,
        data.emissions,
        parameter_dimension=dimension,
        prior_scale=prior_scale,
    )

    assert oracle.log_evidence == pytest.approx(scalar.log_evidence)
    assert oracle.aggregate_mean == pytest.approx(scalar.posterior_mean)
    assert oracle.aggregate_second_moment == pytest.approx(
        scalar.posterior_variance + scalar.posterior_mean**2
    )
    assert oracle.orthogonal_spread == pytest.approx(dimension * prior_scale**2)
    np.testing.assert_allclose(
        oracle.parameter_mean,
        np.full(dimension, scalar.posterior_mean),
        rtol=0.0,
        atol=2e-13,
    )
    expected_covariance = dimension * prior_scale**2 * np.eye(dimension) + (
        scalar.posterior_variance - prior_scale**2
    ) * np.ones((dimension, dimension))
    np.testing.assert_allclose(
        oracle.parameter_covariance,
        expected_covariance,
        rtol=0.0,
        atol=2e-13,
    )

    scalar_projection = exchangeable_unknown_ar_oracle(
        model,
        data.emissions,
        parameter_dimension=1,
    )
    assert scalar_projection.orthogonal_spread is None
    np.testing.assert_allclose(
        scalar_projection.parameter_covariance,
        [[scalar.posterior_variance]],
        rtol=0.0,
        atol=2e-13,
    )


def test_exchangeable_dimension_one_preserves_scalar_callbacks() -> None:
    model = LGSSM(timesteps=4, input_coefficient=0.0)
    default = make_unknown_ar_callbacks(model)
    explicit = make_unknown_ar_callbacks(model, parameter_dimension=1)
    key = jr.key(29)
    state = jnp.asarray([0.25], dtype=jnp.float32)
    params = jnp.asarray([0.85], dtype=jnp.float32)
    emission = jnp.asarray([-0.2], dtype=jnp.float32)

    np.testing.assert_array_equal(
        default.param_initial_sampler(key, 8),
        explicit.param_initial_sampler(key, 8),
    )
    np.testing.assert_array_equal(
        default.liu_west_transition_sampler(key, state, params),
        explicit.liu_west_transition_sampler(key, state, params),
    )
    np.testing.assert_array_equal(
        default.liu_west_log_auxiliary_fn(emission, state, params),
        explicit.liu_west_log_auxiliary_fn(emission, state, params),
    )

    with pytest.raises(ValueError, match="parameter_dimension"):
        make_unknown_ar_callbacks(model, parameter_dimension=0)


def test_worker_command_serializes_the_complete_cell(tmp_path: Path) -> None:
    cell = plan_cells("smoke", platforms=("cpu",), seed=3)[0]
    command = build_worker_command(root=tmp_path, cell=cell)
    assert command[0].endswith("python")
    assert str(tmp_path / "benchmarks/profiling/worker.py") in command
    assert "--cell-json" in command
    assert command[command.index("--phase") + 1] == "timing"
    assert cell.workload in " ".join(command)

    validation = build_worker_command(
        root=tmp_path,
        cell=cell,
        phase="validation",
    )
    assert validation[validation.index("--phase") + 1] == "validation"


def test_raw_filenames_are_unique_and_stable() -> None:
    cells = plan_cells("smoke", platforms=("cpu", "mps"), seed=3)
    names = [raw_filename(cell) for cell in cells]
    assert names == [raw_filename(cell) for cell in cells]
    assert len(names) == len(set(names))
    assert all(name.endswith(".json") for name in names)


def test_dry_run_cli_names_the_order_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "dry-run"
    assert (
        profiling_main([
            "--profile",
            "smoke",
            "--platforms",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--order-seed",
            "37",
            "--dry-run",
        ])
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert summary["order_seed"] == 37
    assert manifest["order_seed"] == 37
    assert summary["preflight"] == manifest["preflight"]
    assert summary["preflight"]["timing_worker_processes"] == summary["cells"]
    assert summary["preflight"]["validation_worker_processes"] == 0
    assert "seed" not in summary
    assert "seed" not in manifest


@pytest.mark.parametrize(
    "relative",
    (Path("src/smcx/profile-output"), Path("benchmarks/profiling/output")),
)
def test_supervisor_rejects_output_inside_attested_source(
    tmp_path: Path,
    relative: Path,
) -> None:
    output_dir = tmp_path / relative
    with pytest.raises(ValueError, match="attested source"):
        supervise(
            "smoke",
            platforms=("cpu",),
            root=tmp_path,
            output_dir=output_dir,
            order_seed=13,
            runner=lambda cell: pytest.fail("worker must not launch"),
        )
    assert not output_dir.exists()


def test_supervisor_writes_manifest_first_and_resumes(
    tmp_path: Path,
) -> None:
    calls = []
    identity = campaign_identity()

    def runner(cell):
        calls.append(cell)
        return _result_for_cell(cell, identity)

    first = supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=tmp_path / "out",
        seed=13,
        runner=runner,
    )
    assert first["failed"] == 0
    assert first["completed"] == first["cells"]
    assert first["preflight"]["timing_worker_processes"] == first["cells"]
    assert (tmp_path / "out/manifest.json").exists()
    first_call_count = len(calls)

    second = supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=tmp_path / "out",
        seed=13,
        runner=runner,
    )
    assert second == first
    assert len(calls) == first_call_count


def test_supervisor_holds_host_lock_for_every_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    identity = campaign_identity()
    planned = plan_cells("smoke", platforms=("cpu",), order_seed=59)[:1]
    monkeypatch.setattr(
        run_module,
        "HostCampaignLock",
        lambda: _RecordingLock(events),
    )
    monkeypatch.setattr(
        run_module,
        "plan_cells",
        lambda *args, **kwargs: planned,
    )

    def runner(cell):
        assert events == ["enter"]
        return _result_for_cell(cell, identity)

    supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=tmp_path / "locked",
        order_seed=59,
        runner=runner,
    )

    assert events == ["enter", "exit"]


def test_supervisor_rechecks_campaign_identity_before_every_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = campaign_identity()
    planned = plan_cells(
        "smoke",
        platforms=("cpu",),
        order_seed=41,
    )[:2]
    monkeypatch.setattr(
        "benchmarks.profiling.run.plan_cells",
        lambda *args, **kwargs: planned,
    )
    identity_checks = 0
    launches = []

    def current_identity():
        nonlocal identity_checks
        identity_checks += 1
        if identity_checks == 1:
            return frozen
        changed = {
            **frozen,
            "source": {**frozen["source"], "source_sha256": "0" * 64},
        }
        return changed

    monkeypatch.setattr(
        "benchmarks.profiling.run.campaign_identity",
        current_identity,
    )

    def runner(cell):
        launches.append(cell)
        return _result_for_cell(cell, frozen)

    with pytest.raises(CampaignIdentityError, match="changed"):
        supervise(
            "smoke",
            platforms=("cpu",),
            root=tmp_path,
            output_dir=tmp_path / "identity-change",
            order_seed=41,
            runner=runner,
        )
    assert len(launches) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda result: result["environment"].update(
                runtime_flags={"JAX_ENABLE_X64": "true"}
            ),
            "runtime flags",
        ),
        (
            lambda result: result["environment"].update(device_kind=""),
            "device_kind",
        ),
        (
            lambda result: result["environment"].update(device_id=1),
            "device_id",
        ),
        (
            lambda result: result.update(dispatch_mode="synchronous"),
            "dispatch",
        ),
    ),
)
def test_supervisor_rejects_unattested_runtime_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    identity = campaign_identity()
    planned = plan_cells(
        "smoke",
        platforms=("cpu",),
        order_seed=43,
    )[:1]
    monkeypatch.setattr(
        "benchmarks.profiling.run.plan_cells",
        lambda *args, **kwargs: planned,
    )
    monkeypatch.setattr(
        "benchmarks.profiling.common.campaign_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        "benchmarks.profiling.run.campaign_identity",
        lambda: identity,
    )

    def runner(cell):
        result = _result_for_cell(cell, identity)
        mutate(result)
        return result

    with pytest.raises(CampaignIdentityError, match=message):
        supervise(
            "smoke",
            platforms=("cpu",),
            root=tmp_path,
            output_dir=tmp_path / message.replace(" ", "-"),
            order_seed=43,
            runner=runner,
        )


def test_supervisor_requires_type_strict_parameter_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = campaign_identity()
    cell = next(
        cell
        for cell in plan_cells(
            "smoke",
            platforms=("cpu",),
            order_seed=53,
        )
        if cell.workload == "bootstrap_lgssm"
    )
    monkeypatch.setattr(
        "benchmarks.profiling.run.plan_cells",
        lambda *args, **kwargs: [cell],
    )
    monkeypatch.setattr(
        "benchmarks.profiling.common.campaign_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        "benchmarks.profiling.run.campaign_identity",
        lambda: identity,
    )

    def runner(scheduled):
        result = _result_for_cell(scheduled, identity)
        result["parameters"]["num_particles"] = float(
            result["parameters"]["num_particles"]
        )
        return result

    output_dir = tmp_path / "type-strict-parameters"
    summary = supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=output_dir,
        order_seed=53,
        runner=runner,
    )

    assert summary["failed"] == 1
    record = json.loads((output_dir / "raw" / raw_filename(cell)).read_text())
    assert record["failure"]["kind"] == "supervisor_error"
    assert "does not match" in record["failure"]["message"]


def test_supervisor_finishes_all_timings_before_oracle_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = campaign_identity()
    planned = [
        cell._replace(correctness_replicates=2)
        for cell in plan_cells("smoke", platforms=("cpu",), seed=23)
        if cell.workload in {"auxiliary_lgssm", "bootstrap_lgssm"}
    ]
    monkeypatch.setattr(
        "benchmarks.profiling.run.plan_cells",
        lambda *args, **kwargs: planned,
    )
    events = []

    def timing_runner(cell):
        assert cell.correctness_replicates == 0
        events.append(("timing", cell.workload))
        return _result_for_cell(cell, identity)

    def validation_runner(cell):
        events.append(("validation", cell.workload))
        return _validation_for_cell(cell, identity)

    output_dir = tmp_path / "two-phase"
    first = supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=output_dir,
        seed=23,
        runner=timing_runner,
        validation_runner=validation_runner,
    )
    assert first["failed"] == 0
    assert [kind for kind, _ in events] == [
        "timing",
        "timing",
        "validation",
        "validation",
    ]

    for cell in planned:
        name = raw_filename(cell)
        timing = json.loads((output_dir / "timing" / name).read_text())
        validation = json.loads((output_dir / "validation" / name).read_text())
        final = json.loads((output_dir / "raw" / name).read_text())
        assert timing["correctness_replicates"] == 0
        assert validation["correctness_replicates"] == 2
        assert final["steady_times_s"] == timing["steady_times_s"]
        assert final["correctness_replicates"] == 2
        assert final["correctness"]["replicated"]["replicates"] == 2
        assert final["correctness_level"] == "oracle_accuracy"
        provenance = final["correctness"]["validation_provenance"]
        assert provenance["backend"] == validation["backend"]
        assert provenance["dispatch_mode"] == validation["dispatch_mode"]
        assert provenance["environment"] == validation["environment"]
        assert provenance["source"] == validation["source"]
        assert provenance["versions"] == validation["versions"]

    # Simulate interruption after an immutable sidecar was written but before
    # the final combined raw record. Resume must rerun neither phase.
    interrupted_name = raw_filename(planned[0])
    (output_dir / "raw" / interrupted_name).unlink()
    events.clear()
    second = supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=output_dir,
        seed=23,
        runner=timing_runner,
        validation_runner=validation_runner,
    )
    assert second == first
    assert events == []

    tampered_path = output_dir / "raw" / interrupted_name
    tampered = json.loads(tampered_path.read_text())
    tampered["correctness"]["validation_provenance"]["environment"][
        "runtime_flags"
    ] = {"JAX_ENABLE_X64": "true"}
    tampered_path.write_text(json.dumps(tampered))
    with pytest.raises(CampaignIdentityError, match="runtime flags"):
        supervise(
            "smoke",
            platforms=("cpu",),
            root=tmp_path,
            output_dir=output_dir,
            order_seed=23,
            runner=timing_runner,
            validation_runner=validation_runner,
        )


def test_supervisor_rejects_timing_validation_device_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = campaign_identity()
    planned = [
        next(
            cell
            for cell in plan_cells(
                "smoke",
                platforms=("cpu",),
                order_seed=47,
            )
            if cell.workload == "bootstrap_lgssm"
        )._replace(correctness_replicates=2)
    ]
    monkeypatch.setattr(
        "benchmarks.profiling.run.plan_cells",
        lambda *args, **kwargs: planned,
    )
    monkeypatch.setattr(
        "benchmarks.profiling.common.campaign_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        "benchmarks.profiling.run.campaign_identity",
        lambda: identity,
    )

    def validation_runner(cell):
        result = _validation_for_cell(cell, identity)
        result["environment"]["device_id"] = 1
        return result

    with pytest.raises(CampaignIdentityError, match="device_id"):
        supervise(
            "smoke",
            platforms=("cpu",),
            root=tmp_path,
            output_dir=tmp_path / "device-mismatch",
            order_seed=47,
            runner=lambda cell: _result_for_cell(cell, identity),
            validation_runner=validation_runner,
        )


def test_validation_failure_preserves_timing_and_continues_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = campaign_identity()
    planned = [
        cell._replace(correctness_replicates=2)
        for cell in plan_cells("smoke", platforms=("cpu",), seed=29)
        if cell.workload in {"auxiliary_lgssm", "bootstrap_lgssm"}
    ]
    monkeypatch.setattr(
        "benchmarks.profiling.run.plan_cells",
        lambda *args, **kwargs: planned,
    )

    def validation_runner(cell):
        if cell == planned[0]:
            raise RuntimeError("injected oracle failure")
        return _validation_for_cell(cell, identity)

    output_dir = tmp_path / "validation-failure"
    summary = supervise(
        "smoke",
        platforms=("cpu",),
        root=tmp_path,
        output_dir=output_dir,
        seed=29,
        runner=lambda cell: _result_for_cell(cell, identity),
        validation_runner=validation_runner,
    )
    assert summary["completed"] == 2
    assert summary["failed"] == 1

    failed_name = raw_filename(planned[0])
    failed = json.loads((output_dir / "raw" / failed_name).read_text())
    validate_result(failed)
    assert failed["failure"]["kind"] == "validation_error"
    assert failed["steady_times_s"] == [0.3] * planned[0].repeats
    assert failed["correctness"]["replicated"] == {
        "completed_replicates": 0,
        "gate": "validation_failed",
        "passed": False,
        "replicates": 2,
    }
    assert not (output_dir / "validation" / failed_name).exists()

    passed_name = raw_filename(planned[1])
    passed = json.loads((output_dir / "raw" / passed_name).read_text())
    assert passed["failure"] is None
    assert passed["correctness"]["passed"]
    assert (output_dir / "validation" / passed_name).exists()


def test_worker_runs_registered_untimed_correctness_replicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        cell
        for cell in plan_cells("smoke", platforms=("cpu",), seed=3)
        if cell.workload == "auxiliary_lgssm"
    )._replace(correctness_replicates=3)
    _set_attested_worker_environment(monkeypatch, cell.platform)
    x64_before = jax.config.read("jax_enable_x64")
    result = run_cell(cell)
    assert jax.config.read("jax_enable_x64") is x64_before
    replicated = result["correctness"]["replicated"]
    assert replicated["replicates"] == 3
    assert result["correctness"]["passed"] is replicated["passed"]
    assert result["environment"]["device_kind"]
    assert set(result["environment"]["pre_timing"]) == {
        "power_status",
        "thermal_status",
    }
    assert set(result["environment"]["post_timing"]) == {
        "power_status",
        "thermal_status",
    }
    assert set(result["environment"]["post_cell"]) == {
        "power_status",
        "thermal_status",
    }
    assert isinstance(result["environment"]["runtime_flags"], dict)
    assert len(result["source"]["source_sha256"]) == 64


def test_validation_worker_returns_only_replicated_oracle_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        cell
        for cell in plan_cells("smoke", platforms=("cpu",), seed=3)
        if cell.workload == "auxiliary_lgssm"
    )._replace(correctness_replicates=3)
    _set_attested_worker_environment(monkeypatch, cell.platform)
    result = run_validation(cell)
    assert result["correctness_level"] == "oracle_accuracy"
    assert result["dispatch_mode"] == "asynchronous"
    assert result["replicated"]["replicates"] == 3
    assert "steady_times_s" not in result
    assert "pre_timing" not in result["environment"]
    assert "post_timing" not in result["environment"]
