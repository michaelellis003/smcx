# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Timing eligibility and resource summaries for campaign reports."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

import numpy as np

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.plan import CampaignCell

_PACKAGES = {"jax", "jax-mps", "jaxlib", "ml-dtypes", "numpy", "scipy", "smcx"}
_IDENTITY_FIELDS = {"source", "lock", "packages", "python", "host"}
_IDENTITY_DRIFT = "source_identity_changed_after_launch"


class Summary(NamedTuple):
    """Five-block values and their linear-quartile summary."""

    values: tuple[float, ...]
    median: float
    q1: float
    q3: float
    iqr: float
    minimum: float
    maximum: float


class TimingReport(NamedTuple):
    """Timing status and non-comparable memory scopes."""

    status: str
    first: Summary | None
    steady: Summary | None
    process_rss: Summary | None
    mps_peak: Summary | None


def _summary(values: Sequence[float]) -> Summary:
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(array, (25, 50, 75))
    return Summary(
        tuple(float(value) for value in array),
        float(median),
        float(q1),
        float(q3),
        float(q3 - q1),
        float(np.min(array)),
        float(np.max(array)),
    )


def _positive(value: object) -> bool:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    return math.isfinite(value) and value > 0


def _identity_ok(identity: Mapping[str, Any]) -> bool:
    try:
        source, lock = identity["source"], identity["lock"]
        packages, python, host = (
            identity["packages"],
            identity["python"],
            identity["host"],
        )
        return bool(
            set(identity) == _IDENTITY_FIELDS
            and source["git_dirty"] is False
            and len(source["git_commit"]) == 40
            and len(source["sha256"]) == len(lock["sha256"]) == 64
            and source["files"]
            and lock["path"] == "uv.lock"
            and set(packages) == _PACKAGES
            and all(
                isinstance(packages[name], str) and packages[name]
                for name in _PACKAGES
            )
            and all(
                python.get(name)
                for name in ("implementation", "version", "executable")
            )
            and isinstance(host, dict)
            and host
        )
    except (AttributeError, KeyError, TypeError):
        return False


def _runtime_ok(cell: CampaignCell, timing: Mapping[str, Any]) -> bool:
    backend, x64 = ("cpu", True) if cell.lane == "cpu_f64" else ("mps", False)
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
    try:
        environment, memory = timing["environment"], timing["memory"]
        times = (timing["first_execution_s"], *timing["steady_times_s"])
        return bool(
            canonical_json(timing["backend_startup_burns"]) == "1"
            and canonical_json(timing["warmups"]) == "0"
            and canonical_json(timing["repeats"]) == "7"
            and timing["execution_mode"] == "host_shell"
            and timing["backend"] == backend
            and timing["dispatch_mode"] == ("asynchronous" if x64 else "safe")
            and environment["runtime_flags"] == expected_flags
            and canonical_json(environment["runtime_state"])
            == canonical_json(expected_state)
            and len(timing["steady_times_s"]) == 7
            and all(_positive(value) for value in times)
            and _positive(memory["process_max_rss_bytes"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _environment_ok(
    identity: Mapping[str, Any], timing: Mapping[str, Any]
) -> bool:
    if timing.get("backend") == "cpu":
        return True
    try:
        host, environment = identity["host"], timing["environment"]
        boundaries = ("pre_timing", "post_timing", "post_cell")
        return bool(
            host.get("os") == "Darwin"
            and host.get("machine") == "arm64"
            and str(host.get("cpu_model", "")).startswith("Apple ")
            and str(host.get("hardware_model", "")).startswith("Mac")
            and type(environment["device_id"]) is int
            and environment["device_id"] == 0
            and environment["device_kind"].lower() == "gpu"
            and all(
                "Now drawing from 'AC Power'"
                in (environment[name].get("power_status") or "")
                and "No thermal warning level"
                in (environment[name].get("thermal_status") or "")
                and "No performance warning level"
                in (environment[name].get("thermal_status") or "")
                for name in boundaries
            )
        )
    except (AttributeError, KeyError, TypeError):
        return False


def _empty(status: str) -> TimingReport:
    return TimingReport(status, None, None, None, None)


def analyze_timing(
    cell: CampaignCell,
    results: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> TimingReport:
    """Apply timing precedence and summarize five eligible blocks."""
    kinds = [
        result["failure"].get("kind")
        for result in results
        if isinstance(result.get("failure"), dict)
    ]
    timings = []
    structural_failed = False
    for result in results:
        if result.get("failure") is None:
            runs, timing = result.get("runs"), result.get("timing")
            if (
                not isinstance(runs, list)
                or len(runs) != 1
                or not isinstance(timing, dict)
            ):
                kinds.append("malformed_result")
                continue
            passed = runs[0].get("structural", {}).get("passed")
            if type(passed) is not bool:
                kinds.append("malformed_result")
            else:
                structural_failed |= not passed
                timings.append(timing)
    if any(
        kind not in {"structural_failure", _IDENTITY_DRIFT} for kind in kinds
    ):
        return _empty("failed_execution")
    if "structural_failure" in kinds or structural_failed:
        return _empty("failed_structural")
    if _IDENTITY_DRIFT in kinds:
        return _empty("ineligible_identity")
    if len(results) != 5:
        return _empty("not_run_after_stop")
    if not _identity_ok(identity):
        return _empty("ineligible_identity")
    if len(timings) != 5 or not all(
        _runtime_ok(cell, item) for item in timings
    ):
        return _empty("ineligible_runtime")
    if not all(_environment_ok(identity, item) for item in timings):
        return _empty("ineligible_environment")
    first = [item["first_execution_s"] for item in timings]
    steady = [float(np.median(item["steady_times_s"])) for item in timings]
    rss = [item["memory"]["process_max_rss_bytes"] for item in timings]
    peaks = [
        (item["memory"].get("device_stats") or {}).get("peak_bytes_in_use")
        for item in timings
    ]
    mps_peak = (
        _summary(cast(list[float], peaks))
        if cell.lane == "mps_f32" and all(_positive(value) for value in peaks)
        else None
    )
    return TimingReport(
        "eligible", _summary(first), _summary(steady), _summary(rss), mps_peak
    )
