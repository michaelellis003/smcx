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


def _names(value: str) -> frozenset[str]:
    return frozenset(value.split())


_RESULT = {"schema_version", "request", "failure", "timing", "runs"}
_RUN = _names(
    "key_index key_words posterior_mean posterior_covariance log_evidence "
    "temperatures reweighting_ess acceptance_rates work structural"
)
_TIMING = _names(
    "execution_mode backend_startup_burns warmups repeats first_execution_s "
    "steady_times_s backend dispatch_mode environment memory"
)
_ENV = _names(
    "device_id device_kind runtime_flags runtime_state pre_timing post_timing "
    "post_cell"
)
_FLAGS = _names(
    "JAX_PLATFORMS JAX_ENABLE_X64 JAX_DISABLE_JIT JAX_ENABLE_COMPILATION_CACHE"
)
_STATE = _names("backend x64 disable_jit cache_enabled cache_dir async")
_MEMORY = _names(
    "device_stats executable_analysis process_max_rss_before_measurement_bytes "
    "process_max_rss_bytes"
)
_FAILURE = _names(
    "kind exception_type message key_index key_words key_indices failed_call "
    "timing_prefix timeout_s stdout_tail stderr_tail returncode prelaunch "
    "expected_source_sha256 observed_source_sha256 boundaries changed_domains "
    "worker_failure"
)
_IDENTITY_DOMAINS = _names("source lock packages python host")
_METADATA = tuple(
    "execution_mode backend_startup_burns warmups repeats backend "  # noqa: SIM905
    "dispatch_mode".split()
)


def _map(value: object, fields: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"measurement {name} schema is invalid")
    return cast(dict[str, Any], value)


def _numbers(value: object, name: str, length: int | None = None) -> list[Any]:
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
    if not all(item is None or type(item) is str for item in result.values()):
        raise ValueError("measurement boundary value is invalid")
    return cast(dict[str, str | None], result)


def _positive(value: object) -> bool:
    return bool(
        type(value) in {int, float}
        and math.isfinite(cast(float, value))
        and cast(float, value) > 0
    )


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
        valid_call = call == {"role": "first", "index": 0} and (
            first is None and not steady
        )
        valid_call |= bool(
            call is not None
            and call["role"] == "steady"
            and 0 <= call["index"] < 7
            and len(steady) == call["index"]
            and _positive(first)
            and all(_positive(item) for item in steady)
        )
        if prefix["eligible"] is not False or not valid_call:
            raise ValueError("measurement timing prefix is invalid")
        result["timing_prefix"] = {**prefix, "steady_times_s": steady}
    if "prelaunch" in raw:
        result["prelaunch"] = _boundary(raw["prelaunch"])
    if "boundaries" in raw:
        boundaries = _map(
            raw["boundaries"],
            {"pre_timing", "post_timing", "post_cell"},
            "failure boundaries",
        )
        for boundary in boundaries.values():
            _boundary(boundary)
    if "changed_domains" in raw and (
        type(raw["changed_domains"]) is not list
        or not set(raw["changed_domains"]) <= _IDENTITY_DOMAINS
    ):
        raise ValueError("measurement identity domains are invalid")
    if "worker_failure" in raw:
        _failure(raw["worker_failure"])
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
    if any(
        item is not None and type(item) is not dict
        for item in (device, executable)
    ):
        raise ValueError("measurement nested memory is invalid")
    result["device_peak_bytes_in_use"] = (
        None if device is None else device.get("peak_bytes_in_use")
    )
    result["executable_peak_memory_bytes"] = (
        None if executable is None else executable.get("peak_memory_in_bytes")
    )
    if any(
        value is not None and (type(value) is not int or value < 0)
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
    expected = [
        "host_shell",
        1,
        0,
        7,
        backend,
        "asynchronous" if x64 else "safe",
    ]
    metadata = canonical_json([
        raw[name] for name in _METADATA
    ]) == canonical_json(expected)
    metadata &= flags == expected_flags and state == expected_state
    metadata &= type(environment["device_id"]) is int
    metadata &= type(environment["device_kind"]) is str
    first = raw["first_execution_s"]
    steady = _numbers(raw["steady_times_s"], "steady times", 7)
    valid_times = _positive(first) and all(_positive(value) for value in steady)
    if not metadata or not valid_times:
        raise ValueError("measurement timing metadata is invalid")
    names = ("pre_timing", "post_timing", "post_cell")
    if "supervisor_prelaunch" in environment:
        names += ("supervisor_prelaunch",)
    boundaries = {name: _boundary(environment[name]) for name in names}
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
    count = len(campaign.results)
    if count != len(campaign.inventory) or count > len(requests):
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
        runs = result["runs"]
        if type(runs) is not list:
            raise ValueError("measurement run list is invalid")
        entry: dict[str, Any] = {
            "ordinal": ordinal,
            "phase": request.phase,
            "cell_id": cell_id(request.cell),
            "block": request.block,
            "failure": failure,
        }
        if request.phase == "timing" and result["timing"] is not None:
            if len(runs) != 1:
                raise ValueError("measurement timing run is missing")
            run = _map(runs[0], _RUN, "timing run")
            _structural(run["structural"])
            entry["timing"] = _timing(result["timing"], request.cell.lane)
        elif request.phase == "accuracy":
            if result["timing"] is not None:
                raise ValueError("measurement accuracy result contains timing")
            entry["accuracy_runs"] = _accuracy_runs(request, runs)
            complete = (
                failure is None or failure["kind"] == "structural_failure"
            )
            if complete and len(runs) != 32:
                raise ValueError("measurement accuracy result is incomplete")
        elif result["timing"] is not None:
            raise ValueError("measurement untimed result contains timing")
        elif failure is None:
            if request.phase == "timing" or len(runs) != 1:
                raise ValueError("measurement successful result is incomplete")
        output.append(entry)
    return dict(
        schema_version=1, observed_requests=len(output), requests=output
    )
