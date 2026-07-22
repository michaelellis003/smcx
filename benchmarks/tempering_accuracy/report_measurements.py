# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Sanitized raw measurements for the public tempering report."""

import math
from typing import Any, cast

import numpy as np

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import bind_request, request_dict
from benchmarks.tempering_accuracy.core import build_target
from benchmarks.tempering_accuracy.plan import cell_id
from benchmarks.tempering_accuracy.report_accuracy import (
    _parse_run,
    _structural,
)
from benchmarks.tempering_accuracy.report_data import (
    CampaignData,
    campaign_requests,
)

_RESULT = {"schema_version", "request", "failure", "timing", "runs"}
_RUN = frozenset(
    "key_index key_words posterior_mean posterior_covariance log_evidence "  # noqa: SIM905
    "temperatures reweighting_ess acceptance_rates work structural".split()
)
_TIMING = frozenset(
    "execution_mode backend_startup_burns warmups repeats first_execution_s "  # noqa: SIM905
    "steady_times_s backend dispatch_mode environment memory".split()
)
_ENV = frozenset(
    "device_id device_kind runtime_flags runtime_state pre_timing "  # noqa: SIM905
    "post_timing post_cell".split()
)
_FLAGS = {
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "JAX_DISABLE_JIT",
    "JAX_ENABLE_COMPILATION_CACHE",
}
_STATE = {
    "backend",
    "x64",
    "disable_jit",
    "cache_enabled",
    "cache_dir",
    "async",
}
_MEMORY = {
    "device_stats",
    "executable_analysis",
    "process_max_rss_before_measurement_bytes",
    "process_max_rss_bytes",
}
_FAILURE = frozenset(
    "kind exception_type message key_index key_words key_indices failed_call "  # noqa: SIM905
    "timing_prefix timeout_s stdout_tail stderr_tail returncode prelaunch "
    "expected_source_sha256 observed_source_sha256".split()
)


def _map(
    value: object, fields: set[str] | frozenset[str], name: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"measurement {name} schema is invalid")
    return cast(dict[str, Any], value)


def _numbers(
    value: object, name: str, *, length: int | None = None
) -> list[Any]:
    if type(value) is not list or (length is not None and len(value) != length):
        raise ValueError(f"measurement {name} is invalid")
    for item in value:
        if item is not None and (
            type(item) not in {int, float}
            or not math.isfinite(cast(float, item))
        ):
            raise ValueError(f"measurement {name} is invalid")
    return list(value)


def _boundary(value: object) -> dict[str, str | None]:
    result = _map(value, {"power_status", "thermal_status"}, "boundary")
    if any(
        item is not None and type(item) is not str for item in result.values()
    ):
        raise ValueError("measurement boundary value is invalid")
    return cast(dict[str, str | None], result)


def _failure(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict or not set(value) <= _FAILURE:
        raise ValueError("measurement failure schema is invalid")
    raw = cast(dict[str, Any], value)
    kind = raw.get("kind")
    if type(kind) is not str or not kind.isidentifier():
        raise ValueError("measurement failure kind is invalid")
    result: dict[str, Any] = {"kind": kind}
    if "failed_call" in raw:
        call = _map(raw["failed_call"], {"role", "index"}, "failed call")
        if (
            call["role"] not in {"first", "steady"}
            or type(call["index"]) is not int
        ):
            raise ValueError("measurement failed call is invalid")
        result["failed_call"] = dict(call)
    if "timing_prefix" in raw:
        prefix = _map(
            raw["timing_prefix"],
            {"eligible", "first_execution_s", "steady_times_s"},
            "timing prefix",
        )
        first = prefix["first_execution_s"]
        steady = _numbers(prefix["steady_times_s"], "timing prefix")
        call = result.get("failed_call")
        valid = (
            prefix["eligible"] is False
            and (first is None or type(first) in {int, float})
            and call is not None
            and len(steady) == call["index"]
            and (
                (call["role"] == "first" and first is None) or first is not None
            )
        )
        if not valid:
            raise ValueError("measurement timing prefix is invalid")
        result["timing_prefix"] = {**prefix, "steady_times_s": steady}
    if "prelaunch" in raw:
        result["prelaunch"] = _boundary(raw["prelaunch"])
    return result


def _memory(value: object) -> dict[str, int | None]:
    raw = _map(value, _MEMORY, "memory")
    result = {
        "process_max_rss_before_measurement_bytes": raw[
            "process_max_rss_before_measurement_bytes"
        ],
        "process_max_rss_bytes": raw["process_max_rss_bytes"],
    }
    for item in result.values():
        if type(item) is not int or item < 0:
            raise ValueError("measurement memory value is invalid")
    device = raw["device_stats"]
    executable = raw["executable_analysis"]
    if device is not None and type(device) is not dict:
        raise ValueError("measurement device memory is invalid")
    if executable is not None and type(executable) is not dict:
        raise ValueError("measurement executable memory is invalid")
    result["device_peak_bytes_in_use"] = (
        None if device is None else device.get("peak_bytes_in_use")
    )
    result["executable_peak_memory_bytes"] = (
        None if executable is None else executable.get("peak_memory_in_bytes")
    )
    if any(
        value is not None and type(value) is not int
        for value in result.values()
    ):
        raise ValueError("measurement memory value is invalid")
    return result


def _timing(value: object, lane: str) -> dict[str, Any]:
    raw = _map(value, _TIMING, "timing")
    environment = cast(dict[str, Any], raw["environment"])
    allowed = _ENV | ({"supervisor_prelaunch"} if lane == "mps_f32" else set())
    environment = _map(environment, allowed, "environment")
    backend, x64 = ("cpu", True) if lane == "cpu_f64" else ("mps", False)
    flags = _map(environment["runtime_flags"], _FLAGS, "runtime flags")
    state = _map(environment["runtime_state"], _STATE, "runtime state")
    expected_flags = {
        "JAX_PLATFORMS": backend,
        "JAX_ENABLE_X64": str(x64).lower(),
        "JAX_DISABLE_JIT": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }
    expected_state = {
        "backend": backend,
        "x64": x64,
        "disable_jit": False,
        "cache_enabled": False,
        "cache_dir": None,
        "async": None,
    }
    metadata = (
        raw["execution_mode"] == "host_shell"
        and raw["backend_startup_burns"] == 1
        and raw["warmups"] == 0
        and raw["repeats"] == 7
        and raw["backend"] == backend
        and raw["dispatch_mode"] == ("asynchronous" if x64 else "safe")
        and flags == expected_flags
        and state == expected_state
    )
    first = raw["first_execution_s"]
    steady = _numbers(raw["steady_times_s"], "steady times", length=7)
    if not metadata or type(first) not in {int, float} or first <= 0:
        raise ValueError("measurement timing metadata is invalid")
    boundaries = {
        name: _boundary(environment[name])
        for name in ("pre_timing", "post_timing", "post_cell")
    }
    if "supervisor_prelaunch" in environment:
        boundaries["supervisor_prelaunch"] = _boundary(
            environment["supervisor_prelaunch"]
        )
    return {
        **{name: raw[name] for name in _TIMING - {"environment", "memory"}},
        "steady_times_s": steady,
        "environment": {
            "device_id": environment["device_id"],
            "device_kind": environment["device_kind"],
            "runtime_flags": flags,
            "runtime_state": state,
            **boundaries,
        },
        "memory": _memory(raw["memory"]),
    }


def _accuracy_runs(request: Any, values: object) -> list[dict[str, Any]]:
    if type(values) is not list or len(values) > 32:
        raise ValueError("measurement accuracy runs are invalid")
    target = build_target(
        request.cell.geometry,
        request.cell.dimension,
        np.float64 if request.cell.lane == "cpu_f64" else np.float32,
    )
    result = []
    for index, value in enumerate(values):
        run = _map(value, _RUN, "accuracy run")
        _parse_run(request.cell, run, index, target)
        schedules = {
            name: _numbers(run[name], name)
            for name in ("temperatures", "reweighting_ess", "acceptance_rates")
        }
        structural = dict(run["structural"])
        _structural(structural)
        result.append({
            "key_index": run["key_index"],
            "key_words": list(run["key_words"]),
            "log_evidence": run["log_evidence"],
            **schedules,
            "work": dict(run["work"]),
            "structural": structural,
        })
    return result


def build_measurements(campaign: CampaignData) -> dict[str, Any]:
    """Return strict path-free measurements for the observed raw prefix."""
    requests = campaign_requests()
    if len(campaign.results) != len(campaign.inventory) or len(
        campaign.results
    ) > len(requests):
        raise ValueError("measurement inventory does not match raw results")
    output = []
    for ordinal, (request, raw) in enumerate(
        zip(requests, campaign.results, strict=False)
    ):
        result = _map(raw, _RESULT, "result")
        expected = request_dict(bind_request(request, campaign.manifest_sha256))
        if result["schema_version"] != 1 or canonical_json(
            result["request"]
        ) != canonical_json(expected):
            raise ValueError("measurement request identity is invalid")
        failure = _failure(result["failure"])
        entry: dict[str, Any] = {
            "ordinal": ordinal,
            "phase": request.phase,
            "cell_id": cell_id(request.cell),
            "block": request.block,
            "failure": failure,
        }
        if request.phase == "timing" and result["timing"] is not None:
            entry["timing"] = _timing(result["timing"], request.cell.lane)
        elif request.phase == "accuracy":
            if result["timing"] is not None:
                raise ValueError("measurement accuracy result contains timing")
            entry["accuracy_runs"] = _accuracy_runs(request, result["runs"])
        elif result["timing"] is not None:
            raise ValueError("measurement untimed result contains timing")
        output.append(entry)
    return {
        "schema_version": 1,
        "observed_requests": len(output),
        "requests": output,
    }
