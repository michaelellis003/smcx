# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiling plan resource preflight estimates."""

import json

import pytest

from benchmarks.profiling.common import Cell, plan_cells
from benchmarks.profiling.preflight import estimate_plan, main


def _cell(
    workload: str,
    platform: str,
    *,
    warmups: int,
    repeats: int,
    correctness_replicates: int,
    parameters: dict[str, object],
) -> Cell:
    return Cell(
        workload=workload,
        platform=platform,
        block=0,
        warmups=warmups,
        repeats=repeats,
        execution_mode="whole_program_jit",
        parameters=parameters,
        correctness_replicates=correctness_replicates,
    )


def test_estimate_plan_counts_processes_executions_and_timeout() -> None:
    cells = [
        _cell(
            "alpha",
            "cpu",
            warmups=3,
            repeats=4,
            correctness_replicates=2,
            parameters={
                "dimension": 4,
                "num_particles": 10,
                "store_history": True,
            },
        ),
        _cell(
            "alpha",
            "mps",
            warmups=0,
            repeats=2,
            correctness_replicates=0,
            parameters={"dimension": 4, "num_particles": 100},
        ),
        _cell(
            "beta",
            "cpu",
            warmups=1,
            repeats=1,
            correctness_replicates=3,
            parameters={"dimension": 16, "num_particles": 5},
        ),
    ]

    estimate = estimate_plan(cells, timeout_s=60.0)

    assert estimate["scheduled_cells"] == 3
    assert estimate["timing_worker_processes"] == 3
    assert estimate["validation_worker_processes"] == 2
    assert estimate["total_worker_processes"] == 5
    # Each timing worker executes first + remaining warmups + repeats.
    assert estimate["timing_workload_executions"] == 12
    assert estimate["validation_replicate_executions"] == 5
    assert estimate["total_scheduled_workload_executions"] == 17
    assert estimate["configured_timeout_upper_bound_s"] == pytest.approx(300.0)
    assert estimate["timeout_s_per_worker"] == pytest.approx(60.0)


def test_estimate_plan_breaks_counts_down_by_workload_and_platform() -> None:
    cells = [
        _cell(
            "alpha",
            "cpu",
            warmups=3,
            repeats=4,
            correctness_replicates=2,
            parameters={"num_particles": 10},
        ),
        _cell(
            "alpha",
            "mps",
            warmups=0,
            repeats=2,
            correctness_replicates=0,
            parameters={"num_particles": 100},
        ),
        _cell(
            "beta",
            "cpu",
            warmups=1,
            repeats=1,
            correctness_replicates=3,
            parameters={"num_particles": 5},
        ),
    ]

    estimate = estimate_plan(cells, timeout_s=60.0)

    assert estimate["by_workload"] == {
        "alpha": {
            "configured_timeout_upper_bound_s": 180.0,
            "scheduled_cells": 2,
            "timing_worker_processes": 2,
            "timing_workload_executions": 10,
            "total_scheduled_workload_executions": 12,
            "total_worker_processes": 3,
            "validation_replicate_executions": 2,
            "validation_worker_processes": 1,
        },
        "beta": {
            "configured_timeout_upper_bound_s": 120.0,
            "scheduled_cells": 1,
            "timing_worker_processes": 1,
            "timing_workload_executions": 2,
            "total_scheduled_workload_executions": 5,
            "total_worker_processes": 2,
            "validation_replicate_executions": 3,
            "validation_worker_processes": 1,
        },
    }
    assert estimate["by_platform"]["cpu"] == {
        "configured_timeout_upper_bound_s": 240.0,
        "scheduled_cells": 2,
        "timing_worker_processes": 2,
        "timing_workload_executions": 9,
        "total_scheduled_workload_executions": 14,
        "total_worker_processes": 4,
        "validation_replicate_executions": 5,
        "validation_worker_processes": 2,
    }
    assert estimate["by_platform"]["mps"] == {
        "configured_timeout_upper_bound_s": 60.0,
        "scheduled_cells": 1,
        "timing_worker_processes": 1,
        "timing_workload_executions": 3,
        "total_scheduled_workload_executions": 3,
        "total_worker_processes": 1,
        "validation_replicate_executions": 0,
        "validation_worker_processes": 0,
    }


def test_estimate_plan_reports_largest_numeric_parameters_json_safely() -> None:
    cells = [
        _cell(
            "alpha",
            "cpu",
            warmups=1,
            repeats=1,
            correctness_replicates=0,
            parameters={
                "dimension": 4,
                "resampling_threshold": 0.5,
                "store_history": True,
            },
        ),
        _cell(
            "beta",
            "mps",
            warmups=1,
            repeats=1,
            correctness_replicates=0,
            parameters={
                "dimension": 16,
                "resampling_threshold": 0.25,
                "store_history": False,
            },
        ),
    ]

    estimate = estimate_plan(cells, timeout_s=1.0)

    assert estimate["largest_numeric_parameters"] == {
        "dimension": {
            "maximum": 16,
            "platforms": ["mps"],
            "scheduled_cells_at_maximum": 1,
            "workloads": ["beta"],
        },
        "resampling_threshold": {
            "maximum": 0.5,
            "platforms": ["cpu"],
            "scheduled_cells_at_maximum": 1,
            "workloads": ["alpha"],
        },
    }
    assert "adaptive" in estimate["work_estimate_scope"].lower()
    json.dumps(estimate, allow_nan=False)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf")])
def test_estimate_plan_rejects_invalid_timeout(timeout_s: float) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_s must be finite and positive",
    ):
        estimate_plan([], timeout_s=timeout_s)


def test_estimate_plan_rejects_negative_schedule_counts() -> None:
    cell = _cell(
        "alpha",
        "cpu",
        warmups=-1,
        repeats=1,
        correctness_replicates=0,
        parameters={},
    )

    with pytest.raises(ValueError, match="warmups must be non-negative"):
        estimate_plan([cell], timeout_s=1.0)


def test_cli_plans_a_registered_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cells = plan_cells("smoke", platforms=("cpu",), order_seed=17)

    return_code = main([
        "--profile",
        "smoke",
        "--platforms",
        "cpu",
        "--order-seed",
        "17",
        "--timeout-s",
        "12.5",
    ])

    output = json.loads(capsys.readouterr().out)
    assert return_code == 0
    assert output["profile"] == "smoke"
    assert output["platforms"] == ["cpu"]
    assert output["order_seed"] == 17
    assert output["scheduled_cells"] == len(cells)
    assert output["configured_timeout_upper_bound_s"] == pytest.approx(
        len(cells) * 12.5,
    )
