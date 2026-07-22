# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Sanitized raw-measurement contracts for the tempering report."""

import json

import jax.random as jr
import numpy as np
import pytest

from benchmarks.tempering_accuracy import artifacts, report_measurements
from benchmarks.tempering_accuracy.core import accuracy_keys
from benchmarks.tempering_accuracy.plan import current_cells, work_count
from benchmarks.tempering_accuracy.report_data import (
    CampaignData,
    InventoryEntry,
)

_DIGEST = "a" * 64
_CELL = current_cells()[0]
_CHECKS = [
    "backend_ok",
    "shapes_ok",
    "dtypes_ok",
    "finite_ok",
    "trace_shapes_ok",
    "normalized_log_weights_ok",
    "equal_log_weights_ok",
    "temperature_trace_ok",
    "ess_bounds_ok",
    "acceptance_bounds_ok",
    "passed",
]


def _structural():
    return {
        **dict.fromkeys(_CHECKS, True),
        "final_log_weight_lse_error": 0.0,
        "uniform_log_weight_error": 0.0,
    }


def _run(index):
    key = accuracy_keys()[index]
    return {
        "key_index": index,
        "key_words": [int(word) for word in np.asarray(jr.key_data(key))],
        "posterior_mean": [0.0] * _CELL.dimension,
        "posterior_covariance": np.eye(_CELL.dimension).tolist(),
        "log_evidence": -1.25,
        "temperatures": [0.25, 1.0],
        "reweighting_ess": [800.0, 700.0],
        "acceptance_rates": [0.5, 0.4],
        "work": work_count(_CELL, 2)._asdict(),
        "structural": _structural(),
    }


def _timing():
    boundary = {"power_status": "AC", "thermal_status": "nominal"}
    return {
        "execution_mode": "host_shell",
        "backend_startup_burns": 1,
        "warmups": 0,
        "repeats": 7,
        "first_execution_s": 8.0,
        "steady_times_s": [float(value) for value in range(1, 8)],
        "backend": "cpu",
        "dispatch_mode": "asynchronous",
        "environment": {
            "device_id": 0,
            "device_kind": "cpu",
            "runtime_flags": {
                "JAX_PLATFORMS": "cpu",
                "JAX_ENABLE_X64": "true",
                "JAX_DISABLE_JIT": "false",
                "JAX_ENABLE_COMPILATION_CACHE": "false",
            },
            "runtime_state": {
                "backend": "cpu",
                "x64": True,
                "disable_jit": False,
                "cache_enabled": False,
                "cache_dir": None,
                "async": None,
            },
            "pre_timing": boundary,
            "post_timing": boundary,
            "post_cell": boundary,
        },
        "memory": {
            "device_stats": {"peak_bytes_in_use": 123, "private": 999},
            "executable_analysis": None,
            "process_max_rss_before_measurement_bytes": 456,
            "process_max_rss_bytes": 789,
        },
    }


def _payload(request, *, timing=None, runs=None, failure=None):
    return {
        "schema_version": 1,
        "request": artifacts.request_dict(
            artifacts.bind_request(request, _DIGEST)
        ),
        "failure": failure,
        "timing": timing,
        "runs": [] if runs is None else runs,
    }


def _campaign(results):
    inventory = tuple(
        InventoryEntry(index, f"raw-{index}.json", f"{index:064x}")
        for index in range(len(results))
    )
    return CampaignData(
        {}, _DIGEST, "b" * 64, False, None, None, inventory, tuple(results)
    )


def test_measurements_preserve_values_without_private_payloads(monkeypatch):
    timing_request = artifacts.CampaignRequest("timing", _CELL, 0)
    accuracy_request = artifacts.CampaignRequest("accuracy", _CELL, None)
    monkeypatch.setattr(
        report_measurements,
        "campaign_requests",
        lambda: (timing_request, accuracy_request),
    )
    timing = _timing()
    accuracy = _run(0)
    measurements = report_measurements.build_measurements(
        _campaign([
            _payload(timing_request, timing=timing, runs=[accuracy]),
            _payload(accuracy_request, runs=[accuracy]),
        ])
    )

    assert measurements["observed_requests"] == 2
    assert measurements["requests"][0]["timing"]["steady_times_s"] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    assert measurements["requests"][0]["timing"]["memory"] == {
        "device_peak_bytes_in_use": 123,
        "executable_peak_memory_bytes": None,
        "process_max_rss_before_measurement_bytes": 456,
        "process_max_rss_bytes": 789,
    }
    run = measurements["requests"][1]["accuracy_runs"][0]
    assert run["key_index"] == 0 and run["temperatures"] == [0.25, 1.0]
    encoded = json.dumps(measurements, allow_nan=False)
    assert "posterior_mean" not in encoded
    assert "posterior_covariance" not in encoded
    assert "private" not in encoded and "/python" not in encoded


def test_failure_boundary_retained_without_free_form_text(monkeypatch):
    request = artifacts.CampaignRequest("timing", _CELL, 0)
    monkeypatch.setattr(
        report_measurements, "campaign_requests", lambda: (request,)
    )
    failure = {
        "kind": "execution_failure",
        "exception_type": "RuntimeError",
        "message": "secret /private/path",
        "stdout_tail": "secret stdout",
        "stderr_tail": "secret stderr",
        "failed_call": {"role": "steady", "index": 2},
        "timing_prefix": {
            "eligible": False,
            "first_execution_s": 4.0,
            "steady_times_s": [1.0, 2.0],
        },
    }
    value = report_measurements.build_measurements(
        _campaign([_payload(request, failure=failure)])
    )["requests"][0]["failure"]

    assert value == {
        "kind": "execution_failure",
        "failed_call": {"role": "steady", "index": 2},
        "timing_prefix": {
            "eligible": False,
            "first_execution_s": 4.0,
            "steady_times_s": [1.0, 2.0],
        },
    }


@pytest.mark.parametrize("case", ("environment", "run", "schedule", "prefix"))
def test_measurements_fail_closed_on_malformed_nested_schema(monkeypatch, case):
    phase = "accuracy" if case in {"run", "schedule"} else "timing"
    block = None if phase == "accuracy" else 0
    request = artifacts.CampaignRequest(phase, _CELL, block)
    monkeypatch.setattr(
        report_measurements, "campaign_requests", lambda: (request,)
    )
    if phase == "accuracy":
        run = _run(0)
        if case == "run":
            run["unexpected"] = "hidden"
        else:
            run["temperatures"][0] = "0.25"
        payload = _payload(request, runs=[run])
    else:
        timing = _timing()
        if case == "environment":
            timing["environment"]["HOME"] = "/private/home"
            payload = _payload(request, timing=timing, runs=[_run(0)])
        else:
            payload = _payload(
                request,
                failure={
                    "kind": "execution_failure",
                    "failed_call": {"role": "steady", "index": 2},
                    "timing_prefix": {
                        "eligible": False,
                        "first_execution_s": 4.0,
                        "steady_times_s": [1.0],
                    },
                },
            )
    with pytest.raises(ValueError, match="measurement"):
        report_measurements.build_measurements(_campaign([payload]))


def test_incomplete_prefix_is_not_imputed():
    measurements = report_measurements.build_measurements(_campaign([]))
    assert measurements == {
        "schema_version": 1,
        "observed_requests": 0,
        "requests": [],
    }
