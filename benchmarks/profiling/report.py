# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Aggregate and render correctness-first profiling campaign reports."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

if __package__ in (None, ""):  # Allow direct ``python .../report.py`` use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.profiling.common import (
    PLATFORMS,
    SCHEMA_VERSION,
    SEED_CONTRACT,
    WORKLOADS,
    Cell,
    canonical_json,
    expected_device_identity,
    plan_cells,
    profiling_runtime_flags,
    record_matches_cell,
    summarize,
    validate_result,
    worker_environment,
)
from benchmarks.profiling.preflight import estimate_plan
from benchmarks.profiling.run import raw_filename

REPORT_SCHEMA_VERSION = 3

_ADAPTIVE_WORK_COUNTERS = {
    "auxiliary": ("resampling_event_count",),
    "bootstrap": ("resampling_event_count",),
    "guided": ("resampling_event_count",),
    "liu_west": ("resampling_event_count",),
    "smc2": ("rejuvenation_event_count",),
    "temper": ("temperature_stages",),
}


def _read_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from a campaign artifact."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _canonical(value: Any) -> str:
    """Return a stable JSON identity for nested manifest values."""
    return canonical_json(value)


def _cell_key(cell: Cell) -> tuple[Any, ...]:
    """Return the exact identity shared by a manifest cell and raw result."""
    return (
        cell.workload,
        cell.platform,
        cell.block,
        cell.warmups,
        cell.repeats,
        cell.correctness_replicates,
        cell.execution_mode,
        _canonical(cell.parameters),
    )


def _record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the manifest identity encoded in a worker record."""
    return (
        str(record["workload"]),
        str(record["platform_requested"]),
        int(record["block"]),
        int(record["warmups"]),
        int(record["repeats"]),
        int(record["correctness_replicates"]),
        str(record["execution_mode"]),
        _canonical(record["parameters"]),
    )


def _group_key(cell: Cell) -> tuple[Any, ...]:
    """Return a workload/backend identity with the process block removed."""
    return (
        cell.workload,
        cell.platform,
        cell.execution_mode,
        cell.warmups,
        cell.repeats,
        _canonical(cell.parameters),
    )


def _comparison_key(aggregate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return an exact mathematical-cell identity with backend removed."""
    return (
        str(aggregate["workload"]),
        str(aggregate["execution_mode"]),
        int(aggregate["warmups"]),
        int(aggregate["repeats"]),
        _canonical(aggregate["parameters"]),
    )


def _parse_manifest(path: Path) -> tuple[dict[str, Any], list[Cell]]:
    """Validate a frozen campaign manifest and reconstruct its cells."""
    manifest = _read_object(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema_version")
    if not isinstance(manifest.get("profile"), str):
        raise ValueError("manifest profile must be a string")
    identity = manifest.get("campaign_identity")
    if not isinstance(identity, dict) or not all(
        isinstance(identity.get(name), dict)
        for name in ("host", "packages", "source")
    ):
        raise ValueError("manifest campaign_identity is malformed")
    raw_cells = manifest.get("cells")
    if not isinstance(raw_cells, list):
        raise ValueError("manifest cells must be a list")
    plan_sha256 = manifest.get("plan_sha256")
    observed_plan_sha256 = hashlib.sha256(
        _canonical(raw_cells).encode()
    ).hexdigest()
    if plan_sha256 != observed_plan_sha256:
        raise ValueError("manifest plan_sha256 does not match ordered cells")
    platforms = manifest.get("platforms")
    if (
        not isinstance(platforms, list)
        or not platforms
        or not all(isinstance(item, str) for item in platforms)
    ):
        raise ValueError("manifest platforms must be a non-empty string list")
    order_seed = manifest.get("order_seed")
    if not isinstance(order_seed, int) or isinstance(order_seed, bool):
        raise ValueError("manifest order_seed must be an integer")
    if manifest.get("seed_contract") != SEED_CONTRACT:
        raise ValueError("manifest seed_contract disagrees with registry")
    registered_cells = plan_cells(
        manifest["profile"],
        platforms=platforms,
        order_seed=order_seed,
    )
    registered_plan = [cell._asdict() for cell in registered_cells]
    if raw_cells != registered_plan:
        raise ValueError("manifest cells do not match the registered plan")

    cells = []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest cell {index} must be an object")
        try:
            cell = Cell(**cast(dict[str, Any], raw))
        except TypeError as error:
            raise ValueError(
                f"manifest cell {index} does not match the Cell contract"
            ) from error
        if cell.workload not in WORKLOADS:
            raise ValueError(f"manifest has unknown workload: {cell.workload}")
        if cell.platform not in PLATFORMS:
            raise ValueError(f"manifest has unknown platform: {cell.platform}")
        if cell.execution_mode != WORKLOADS[cell.workload].execution_mode:
            raise ValueError("manifest execution_mode disagrees with registry")
        if not isinstance(cell.parameters, dict):
            raise ValueError("manifest cell parameters must be an object")
        cells.append(cell)

    identities = [_cell_key(cell) for cell in cells]
    if len(identities) != len(set(identities)):
        raise ValueError("manifest contains duplicate cells")
    preflight = manifest.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("manifest preflight estimate is unavailable")
    timeout_s = preflight.get("timeout_s_per_worker")
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise ValueError("manifest preflight timeout is malformed")
    expected_preflight = estimate_plan(cells, timeout_s=float(timeout_s))
    if preflight != expected_preflight:
        raise ValueError("manifest preflight disagrees with scheduled cells")
    return manifest, cells


def _identity(value: Cell | Mapping[str, Any]) -> dict[str, Any]:
    """Return compact, JSON-ready identity fields for report diagnostics."""
    if isinstance(value, Cell):
        return {
            "block": value.block,
            "parameters": dict(value.parameters),
            "platform": value.platform,
            "workload": value.workload,
        }
    return {
        "block": value.get("block"),
        "parameters": value.get("parameters"),
        "platform": value.get("platform_requested"),
        "workload": value.get("workload"),
    }


def _numeric_summary(values: Sequence[int | float]) -> dict[str, Any] | None:
    """Summarize a numeric field across fresh-process records."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    summary = summarize(finite)
    return {
        "count": len(finite),
        "iqr": summary["iqr_s"],
        "mad": summary["mad_s"],
        "median": summary["median_s"],
        "min": summary["min_s"],
        "q1": summary["q1_s"],
        "q3": summary["q3_s"],
    }


def _dig(record: Mapping[str, Any], *names: str) -> Any:
    """Read a nested result field, returning None when it is unavailable."""
    value: Any = record
    for name in names:
        if not isinstance(value, Mapping) or name not in value:
            return None
        value = value[name]
    return value


def _summarize_path(
    records: Sequence[Mapping[str, Any]],
    *names: str,
) -> dict[str, Any] | None:
    """Summarize finite numeric values found at one nested path."""
    values = []
    for record in records:
        value = _dig(record, *names)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            values.append(value)
    return _numeric_summary(values)


def _summarize_work(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate adaptive-work counters without erasing categorical values."""
    names = sorted({
        str(name)
        for record in records
        for name in record.get("work_metrics", {})
    })
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        values = [
            record["work_metrics"][name]
            for record in records
            if name in record.get("work_metrics", {})
        ]
        numeric = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        )
        if numeric:
            summary = _numeric_summary(values)
            if summary is not None:
                result[name] = summary
            continue

        unique = {_canonical(value): value for value in values}
        if len(unique) == 1:
            result[name] = {
                "count": len(values),
                "value": next(iter(unique.values())),
            }
        else:
            result[name] = {
                "count": len(values),
                "values": [unique[key] for key in sorted(unique)],
            }
    return result


def _sequence_numbers(value: Sequence[Any]) -> list[float] | None:
    """Return all finite numeric leaves, or None for a mixed sequence."""
    numbers: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, str | bytes):
            nested = _sequence_numbers(item)
            if nested is None:
                return None
            numbers.extend(nested)
        elif (
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
        ):
            numbers.append(float(item))
        else:
            return None
    return numbers


def _compact_accuracy_value(value: Any) -> Any:
    """Compact arrays in correctness evidence while retaining diagnostics."""
    if isinstance(value, Mapping):
        return {
            str(name): _compact_accuracy_value(item)
            for name, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if all(isinstance(item, bool) for item in value):
            return {
                "count": len(value),
                "passed": sum(bool(item) for item in value),
            }
        numbers = _sequence_numbers(value)
        if numbers:
            return {
                "count": len(numbers),
                "maximum": max(numbers),
                "maximum_absolute": max(abs(item) for item in numbers),
                "mean": sum(numbers) / len(numbers),
                "minimum": min(numbers),
            }
        return [_compact_accuracy_value(item) for item in value]
    return value


def _correctness_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain validation strength and compact untimed accuracy evidence."""
    completed = [record for record in records if record["failure"] is None]
    levels = sorted({str(record["correctness_level"]) for record in completed})
    accuracy = []
    for record in completed:
        replicated = _dig(record, "correctness", "replicated")
        replicates = int(record["correctness_replicates"])
        if not replicates or not isinstance(replicated, Mapping):
            continue
        metrics = {
            str(name): _compact_accuracy_value(value)
            for name, value in sorted(replicated.items())
            if name not in {"passed", "replicates"}
        }
        accuracy.append({
            "block": int(record["block"]),
            "metrics": metrics,
            "passed": bool(replicated.get("passed")),
            "replicates": replicates,
        })
    return {
        "accuracy": accuracy,
        "completed_blocks": len(completed),
        "levels": levels,
        "passed_blocks": sum(
            bool(record["correctness"]["passed"]) for record in completed
        ),
    }


def _environment_metadata(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate stable environment metadata without timing snapshots."""
    unique = {}
    for record in records:
        environment = record.get("environment")
        if record["failure"] is not None or not isinstance(
            environment, Mapping
        ):
            continue
        stable = dict(environment)
        stable.pop("post_cell", None)
        stable.pop("post_timing", None)
        stable.pop("pre_timing", None)
        unique[_canonical(stable)] = stable
    return [unique[key] for key in sorted(unique)]


def _timing_state_error(record: Mapping[str, Any]) -> str | None:
    """Return why one inferential timing record lacks a stable host state."""
    environment = record.get("environment")
    if not isinstance(environment, Mapping):
        return "environment metadata is unavailable"
    for boundary in ("pre_timing", "post_timing", "post_cell"):
        state = environment.get(boundary)
        if not isinstance(state, Mapping):
            return f"{boundary} metadata is unavailable"
        power_status = state.get("power_status")
        if (
            not isinstance(power_status, str)
            or "Now drawing from 'AC Power'" not in power_status
        ):
            return f"{boundary} is not on AC power"
        thermal_status = state.get("thermal_status")
        if not isinstance(thermal_status, str) or not all(
            marker in thermal_status
            for marker in (
                "No thermal warning level",
                "No performance warning level",
            )
        ):
            return f"{boundary} reports a thermal/performance warning"
    return None


def _timing_states(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group exact timing-boundary snapshots by requested platform."""
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        environment = record.get("environment")
        if record["failure"] is not None or not isinstance(
            environment, Mapping
        ):
            continue
        pre_timing = environment.get("pre_timing")
        post_timing = environment.get("post_timing")
        post_cell = environment.get("post_cell")
        if (
            not isinstance(pre_timing, Mapping)
            or not isinstance(post_timing, Mapping)
            or not isinstance(post_cell, Mapping)
        ):
            continue
        platform = str(record["platform_requested"])
        key = (
            platform,
            _canonical(pre_timing),
            _canonical(post_timing),
            _canonical(post_cell),
        )
        if key not in grouped:
            grouped[key] = {
                "cells": 0,
                "platform": platform,
                "post_cell": dict(post_cell),
                "post_timing": dict(post_timing),
                "pre_timing": dict(pre_timing),
            }
        grouped[key]["cells"] += 1
    return [grouped[key] for key in sorted(grouped)]


def _unique_metadata(
    records: Sequence[Mapping[str, Any]],
    name: str,
) -> list[dict[str, Any]]:
    """Retain each distinct metadata mapping in canonical order."""
    unique = {
        _canonical(record[name]): dict(record[name])
        for record in records
        if record["failure"] is None and isinstance(record.get(name), Mapping)
    }
    return [unique[key] for key in sorted(unique)]


def _campaign_identity_error(
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> str | None:
    """Return why a successful raw record differs from its manifest."""
    if record["failure"] is not None:
        return None
    if record["source"] != identity["source"]:
        return "source identity differs from manifest"
    if record["versions"] != identity["packages"]:
        return "package identity differs from manifest"
    environment = record["environment"]
    if any(
        environment.get(name) != value
        for name, value in identity["host"].items()
    ):
        return "host identity differs from manifest"
    platform = str(record["platform_requested"])
    expected_flags = profiling_runtime_flags(
        worker_environment(platform, base={})
    )
    if environment.get("runtime_flags") != expected_flags:
        return "runtime flags differ from the sanitized contract"
    device_id = environment.get("device_id")
    device_kind = environment.get("device_kind")
    expected_device_kind, expected_device_id = expected_device_identity(
        platform
    )
    if (device_kind, device_id) != (
        expected_device_kind,
        expected_device_id,
    ) or isinstance(device_id, bool):
        return "timing device differs from the scheduled platform"
    expected_dispatch = "asynchronous" if platform == "cpu" else "safe"
    if record.get("dispatch_mode") != expected_dispatch:
        return "dispatch mode differs from the sanitized contract"
    if int(record["correctness_replicates"]) == 0:
        return None

    provenance = _dig(record, "correctness", "validation_provenance")
    if not isinstance(provenance, Mapping):
        return "validation provenance is unavailable"
    if provenance.get("source") != identity["source"]:
        return "validation source identity differs from manifest"
    if provenance.get("versions") != identity["packages"]:
        return "validation package identity differs from manifest"
    validation_environment = provenance.get("environment")
    if not isinstance(validation_environment, Mapping) or any(
        validation_environment.get(name) != value
        for name, value in identity["host"].items()
    ):
        return "validation host identity differs from manifest"
    if validation_environment.get("runtime_flags") != expected_flags:
        return "validation runtime flags differ from the sanitized contract"
    if provenance.get("backend") != platform:
        return "validation backend differs from the scheduled platform"
    if provenance.get("dispatch_mode") != expected_dispatch:
        return "validation dispatch differs from the sanitized contract"
    if (
        validation_environment.get("device_id"),
        validation_environment.get("device_kind"),
    ) != (device_id, device_kind):
        return "validation device differs from the timing device"
    return None


def _aggregate_group(
    expected: Sequence[Cell],
    records: Sequence[Mapping[str, Any]],
    *,
    campaign_eligible: bool,
    require_stable_timing_state: bool,
) -> dict[str, Any]:
    """Aggregate one workload/parameters/backend over fresh-process blocks."""
    first = expected[0]
    usable = [
        record
        for record in records
        if record["failure"] is None and record["correctness"]["passed"]
    ]
    complete = len(records) == len(expected)
    correct = complete and len(usable) == len(expected)
    timing_state_failures = []
    if require_stable_timing_state:
        timing_state_failures = [
            {"block": int(record["block"]), "reason": reason}
            for record in usable
            if (reason := _timing_state_error(record)) is not None
        ]
    timing_eligible = bool(
        complete and correct and campaign_eligible and not timing_state_failures
    )
    process_medians = [
        float(record["steady_summary"]["median_s"]) for record in usable
    ]
    steady = summarize(process_medians) if timing_eligible else None

    unavailable = {
        _dig(record, "lifecycle", "unavailable_reason")
        for record in usable
        if _dig(record, "lifecycle", "unavailable_reason") is not None
    }
    unavailable_reason = (
        next(iter(unavailable)) if len(unavailable) == 1 else None
    )
    dispatch_modes = sorted({str(record["dispatch_mode"]) for record in usable})
    spec = WORKLOADS[first.workload]
    return {
        "algorithm": spec.algorithm,
        "complete": complete,
        "correct": correct,
        "correctness": _correctness_summary(records),
        "dispatch_modes": dispatch_modes,
        "execution_mode": first.execution_mode,
        "expected_blocks": len(expected),
        "lifecycle": {
            "backend_compile_s": _summarize_path(
                usable, "lifecycle", "backend_compile_s"
            ),
            "first_execution_s": _summarize_path(usable, "first_execution_s"),
            "lowering_s": _summarize_path(usable, "lifecycle", "lowering_s"),
            "unavailable_reason": unavailable_reason,
        },
        "memory": {
            "device_peak_bytes_in_use": _summarize_path(
                usable, "memory", "device_stats", "peak_bytes_in_use"
            ),
            "executable_peak_memory_bytes": _summarize_path(
                usable,
                "memory",
                "executable_analysis",
                "peak_memory_in_bytes",
            ),
            "process_max_rss_bytes": _summarize_path(
                usable, "memory", "process_max_rss_bytes"
            ),
        },
        "model": spec.model,
        "observed_blocks": len(records),
        "parameters": dict(first.parameters),
        "platform": first.platform,
        "process_medians_s": process_medians,
        "repeats": first.repeats,
        "steady": steady,
        "timing_eligible": timing_eligible,
        "timing_state_failures": timing_state_failures,
        "work": _summarize_work(usable),
        "work_evidence": [
            {
                "block": int(record["block"]),
                "metrics": dict(record["work_metrics"]),
            }
            for record in usable
        ],
        "workload": first.workload,
        "warmups": first.warmups,
    }


def _adaptive_work_evidence(
    aggregate: Mapping[str, Any],
) -> tuple[dict[int, dict[str, int]] | None, str | None]:
    """Extract discrete counters that attest one adaptive workload's work."""
    required = _ADAPTIVE_WORK_COUNTERS.get(str(aggregate["algorithm"]), ())
    if not required:
        return {}, None
    evidence: dict[int, dict[str, int]] = {}
    for item in aggregate["work_evidence"]:
        metrics = item["metrics"]
        missing = [name for name in required if name not in metrics]
        if missing:
            names = ", ".join(missing)
            return None, f"missing discrete adaptive-work counter(s): {names}"
        counters = {}
        for name in required:
            value = metrics[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                return None, f"invalid discrete adaptive-work counter: {name}"
            counters[name] = value
        evidence[int(item["block"])] = counters
    if len(evidence) != int(aggregate["expected_blocks"]):
        return None, "incomplete discrete adaptive-work evidence"
    return evidence, None


def _matched_adaptive_work(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    mismatch_reason: str,
) -> tuple[dict[int, dict[str, int]] | None, str | None]:
    """Require exact per-block discrete work before emitting a ratio."""
    first_evidence, first_error = _adaptive_work_evidence(first)
    if first_error is not None:
        return None, first_error
    second_evidence, second_error = _adaptive_work_evidence(second)
    if second_error is not None:
        return None, second_error
    if first_evidence != second_evidence:
        return None, mismatch_reason
    return first_evidence, None


def _comparisons(
    aggregates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare matched CPU/MPS cells with attested adaptive work."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for aggregate in aggregates:
        groups[_comparison_key(aggregate)].append(aggregate)

    comparisons = []
    exclusions = []
    for members in groups.values():
        by_platform = {str(item["platform"]): item for item in members}
        if set(by_platform) != {"cpu", "mps"}:
            continue
        cpu = by_platform["cpu"]
        mps = by_platform["mps"]
        if not cpu["timing_eligible"] or not mps["timing_eligible"]:
            continue
        work_evidence, work_error = _matched_adaptive_work(
            cpu,
            mps,
            mismatch_reason=(
                "discrete adaptive work differs between CPU and MPS"
            ),
        )
        if work_error is not None:
            exclusions.append({
                "parameters": cpu["parameters"],
                "reason": work_error,
                "workload": cpu["workload"],
            })
            continue
        cpu_median = float(cpu["steady"]["median_s"])
        mps_median = float(mps["steady"]["median_s"])
        if cpu_median <= 0.0:
            continue
        comparisons.append({
            "algorithm": cpu["algorithm"],
            "adaptive_work": work_evidence,
            "cpu_median_s": cpu_median,
            "model": cpu["model"],
            "mps_median_s": mps_median,
            "mps_over_cpu": mps_median / cpu_median,
            "parameters": cpu["parameters"],
            "workload": cpu["workload"],
        })

    def ordering(item: Mapping[str, Any]) -> tuple[str, str]:
        return str(item["workload"]), _canonical(item["parameters"])

    return sorted(comparisons, key=ordering), sorted(exclusions, key=ordering)


def _arm_comparisons(
    aggregates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare preregistered equivalent representation arms."""
    pairs = (
        (
            "representation",
            "bootstrap_tracking_dense",
            "bootstrap_tracking_pytree",
        ),
    )
    comparisons = []
    exclusions = []
    for kind, denominator_name, numerator_name in pairs:
        denominator = {
            (item["platform"], _canonical(item["parameters"])): item
            for item in aggregates
            if item["workload"] == denominator_name
        }
        numerator = {
            (item["platform"], _canonical(item["parameters"])): item
            for item in aggregates
            if item["workload"] == numerator_name
        }
        for key in sorted(denominator.keys() & numerator.keys()):
            baseline = denominator[key]
            candidate = numerator[key]
            if (
                not baseline["timing_eligible"]
                or not candidate["timing_eligible"]
            ):
                continue
            work_evidence, work_error = _matched_adaptive_work(
                baseline,
                candidate,
                mismatch_reason=(
                    "discrete adaptive work differs between comparison arms"
                ),
            )
            if work_error is not None:
                exclusions.append({
                    "denominator_workload": denominator_name,
                    "kind": kind,
                    "numerator_workload": numerator_name,
                    "parameters": baseline["parameters"],
                    "platform": baseline["platform"],
                    "reason": work_error,
                })
                continue
            baseline_median = float(baseline["steady"]["median_s"])
            candidate_median = float(candidate["steady"]["median_s"])
            if baseline_median <= 0.0:
                continue
            comparisons.append({
                "denominator_median_s": baseline_median,
                "denominator_workload": denominator_name,
                "adaptive_work": work_evidence,
                "kind": kind,
                "numerator_median_s": candidate_median,
                "numerator_workload": numerator_name,
                "parameters": baseline["parameters"],
                "platform": baseline["platform"],
                "ratio": candidate_median / baseline_median,
            })
    return comparisons, exclusions


def _history_comparisons(
    aggregates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare registered history-on/off cells with otherwise equal work."""
    groups: dict[tuple[str, str, str], dict[bool, Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    parameters_by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    for aggregate in aggregates:
        parameters = dict(aggregate["parameters"])
        if "store_history" not in parameters:
            continue
        store_history = bool(parameters.pop("store_history"))
        key = (
            str(aggregate["platform"]),
            str(aggregate["workload"]),
            _canonical(parameters),
        )
        groups[key][store_history] = aggregate
        parameters_by_group[key] = parameters

    comparisons = []
    exclusions = []
    for key in sorted(groups):
        members = groups[key]
        if set(members) != {False, True}:
            continue
        history_off = members[False]
        history_on = members[True]
        if (
            not history_off["timing_eligible"]
            or not history_on["timing_eligible"]
        ):
            continue
        work_evidence, work_error = _matched_adaptive_work(
            history_off,
            history_on,
            mismatch_reason=(
                "discrete adaptive work differs between history arms"
            ),
        )
        if work_error is not None:
            exclusions.append({
                "parameters": parameters_by_group[key],
                "platform": history_off["platform"],
                "reason": work_error,
                "workload": history_off["workload"],
            })
            continue
        off_median = float(history_off["steady"]["median_s"])
        on_median = float(history_on["steady"]["median_s"])
        if off_median <= 0.0:
            continue
        comparisons.append({
            "adaptive_work": work_evidence,
            "history_off_median_s": off_median,
            "history_on_median_s": on_median,
            "history_on_over_off": on_median / off_median,
            "parameters": parameters_by_group[key],
            "platform": history_off["platform"],
            "workload": history_off["workload"],
        })
    return comparisons, exclusions


def build_report(output_dir: Path) -> dict[str, Any]:
    """Load, validate, and aggregate one profiling output directory."""
    output_dir = Path(output_dir)
    manifest, cells = _parse_manifest(output_dir / "manifest.json")
    identity = manifest["campaign_identity"]
    source_clean = bool(
        manifest["profile"] == "smoke"
        or identity["source"].get("git_dirty") is False
    )
    expected_by_key = {_cell_key(cell): cell for cell in cells}
    raw_dir = output_dir / "raw"
    records_by_key: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = (
        defaultdict(list)
    )
    invalid_records = []
    identity_mismatches = []
    unexpected_records = []

    for path in sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []:
        try:
            record = _read_object(path)
            validate_result(record)
            key = _record_key(record)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            invalid_records.append({"error": str(error), "path": str(path)})
            continue
        identity_error = _campaign_identity_error(record, identity)
        if identity_error is not None:
            identity_mismatches.append({
                **_identity(record),
                "error": identity_error,
                "path": str(path),
            })
            continue
        if key not in expected_by_key:
            unexpected_records.append({
                **_identity(record),
                "path": str(path),
            })
            continue
        expected_cell = expected_by_key[key]
        if not record_matches_cell(record, expected_cell):
            invalid_records.append({
                "error": "raw record does not match its manifest cell",
                "path": str(path),
            })
            continue
        expected_name = raw_filename(expected_cell)
        if path.name != expected_name:
            invalid_records.append({
                "error": (
                    "raw record does not use its canonical raw filename "
                    f"{expected_name}"
                ),
                "path": str(path),
            })
            continue
        records_by_key[key].append((path, record))

    duplicate_records = []
    accepted: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, entries in records_by_key.items():
        if len(entries) == 1:
            accepted[key] = entries[0][1]
            continue
        duplicate_records.append({
            **_identity(expected_by_key[key]),
            "paths": [str(path) for path, _record in entries],
        })

    missing_cells = [
        _identity(cell)
        for key, cell in expected_by_key.items()
        if key not in accepted
    ]
    worker_failures = [
        {
            **_identity(record),
            "failure": record["failure"],
        }
        for record in accepted.values()
        if record["failure"] is not None
    ]
    correctness_failures = [
        {
            **_identity(record),
            "correctness": record["correctness"],
            "correctness_level": record["correctness_level"],
        }
        for record in accepted.values()
        if record["failure"] is None and not record["correctness"]["passed"]
    ]

    expected_groups: dict[tuple[Any, ...], list[Cell]] = defaultdict(list)
    accepted_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for cell in cells:
        expected_groups[_group_key(cell)].append(cell)
        record = accepted.get(_cell_key(cell))
        if record is not None:
            accepted_groups[_group_key(cell)].append(record)

    aggregates = [
        _aggregate_group(
            sorted(expected, key=lambda cell: cell.block),
            sorted(
                accepted_groups[key],
                key=lambda record: int(record["block"]),
            ),
            campaign_eligible=source_clean and not identity_mismatches,
            require_stable_timing_state=manifest["profile"] != "smoke",
        )
        for key, expected in sorted(expected_groups.items())
    ]
    complete = bool(
        len(accepted) == len(cells)
        and not invalid_records
        and not identity_mismatches
        and not unexpected_records
        and not duplicate_records
    )
    reproducible = bool(source_clean and not identity_mismatches)
    correct = bool(
        complete
        and reproducible
        and not worker_failures
        and not correctness_failures
    )
    performance_eligible = bool(
        correct
        and aggregates
        and all(aggregate["timing_eligible"] for aggregate in aggregates)
    )
    accepted_records = list(accepted.values())
    comparisons, comparison_exclusions = _comparisons(aggregates)
    arm_comparisons, arm_comparison_exclusions = _arm_comparisons(aggregates)
    history_comparisons, history_comparison_exclusions = _history_comparisons(
        aggregates
    )
    timing_state_failures = [
        {
            "parameters": aggregate["parameters"],
            "platform": aggregate["platform"],
            "workload": aggregate["workload"],
            **failure,
        }
        for aggregate in aggregates
        for failure in aggregate["timing_state_failures"]
    ]
    schedules = sorted({(cell.warmups, cell.repeats) for cell in cells})
    return {
        "aggregates": aggregates,
        "arm_comparison_exclusions": arm_comparison_exclusions,
        "arm_comparisons": arm_comparisons,
        "campaign_schema_version": manifest["schema_version"],
        "comparison_exclusions": comparison_exclusions,
        "comparisons": comparisons,
        "complete": complete,
        "correct": correct,
        "correctness_failures": correctness_failures,
        "duplicate_records": duplicate_records,
        "environments": _environment_metadata(accepted_records),
        "expected_cells": len(cells),
        "exclusions": manifest.get("exclusions", []),
        "history_comparison_exclusions": history_comparison_exclusions,
        "history_comparisons": history_comparisons,
        "identity_mismatches": identity_mismatches,
        "invalid_records": invalid_records,
        "matched_cells": len(accepted),
        "missing_cells": missing_cells,
        "profile": manifest["profile"],
        "preflight": manifest["preflight"],
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "reproducible": reproducible,
        "schedules": [
            {"repeats": repeats, "warmups": warmups}
            for warmups, repeats in schedules
        ],
        "order_seed": manifest.get("order_seed"),
        "performance_eligible": performance_eligible,
        "seed_contract": manifest.get("seed_contract"),
        "sources": _unique_metadata(accepted_records, "source"),
        "timing_states": _timing_states(accepted_records),
        "timing_state_failures": timing_state_failures,
        "unexpected_records": unexpected_records,
        "versions": _unique_metadata(accepted_records, "versions"),
        "worker_failures": worker_failures,
    }


def _format_ms(summary: Mapping[str, Any] | None) -> str:
    """Render one median in milliseconds or a clear unavailable marker."""
    if summary is None:
        return "—"
    return f"{1_000.0 * float(summary['median']):.3f}"


def _format_steady(aggregate: Mapping[str, Any]) -> str:
    """Render the eligible steady median with its block IQR."""
    summary = aggregate["steady"]
    if summary is None:
        return "ineligible"
    median = 1_000.0 * float(summary["median_s"])
    iqr = 1_000.0 * float(summary["iqr_s"])
    return f"{median:.3f} (IQR {iqr:.3f})"


def _format_mib(summary: Mapping[str, Any] | None) -> str:
    """Render one byte-count median in mebibytes."""
    if summary is None:
        return "—"
    return f"{float(summary['median']) / 2**20:.2f}"


def _status(aggregate: Mapping[str, Any]) -> str:
    """Return a compact correctness/completeness label for one aggregate."""
    if not aggregate["complete"]:
        return "incomplete"
    if not aggregate["correct"]:
        return "failed correctness"
    if not aggregate["timing_eligible"]:
        return "timing state ineligible"
    return "eligible"


def _work_text(work: Mapping[str, Mapping[str, Any]]) -> str:
    """Render adaptive-work summaries compactly without hiding their names."""
    values = []
    for name, summary in sorted(work.items()):
        if "median" in summary:
            value = f"{float(summary['median']):.4g}"
        elif "value" in summary:
            value = str(summary["value"])
        else:
            value = "/".join(str(item) for item in summary.get("values", []))
        values.append(f"{name}={value}")
    return ", ".join(values) or "—"


def _accuracy_text(correctness: Mapping[str, Any]) -> str:
    """Render compact untimed accuracy evidence for one aggregate."""
    summaries = correctness["accuracy"]
    if not summaries:
        return "not scheduled"
    rendered = []
    for summary in summaries:
        status = "passed" if summary["passed"] else "failed"
        metrics = summary["metrics"]
        details = _accuracy_metric_details(metrics)
        detail_text = f"; {details}" if details else ""
        rendered.append(
            f"block {summary['block']}: {status} "
            f"(R={summary['replicates']}{detail_text})"
        )
    return "<br>".join(rendered)


def _accuracy_number(value: Any) -> str | None:
    """Format one finite scalar accuracy diagnostic."""
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return f"{float(value):.4g}"
    return None


def _accuracy_gate_text(name: str, value: Mapping[str, Any]) -> str:
    """Render one named replicated gate without embedding raw arrays."""
    passed = value.get("passed")
    status = (
        "passed"
        if passed is True
        else "failed"
        if passed is False
        else "reported"
    )
    details = []
    for field in ("mean_ratio", "mean", "oracle", "error", "tolerance"):
        rendered = _accuracy_number(value.get(field))
        if rendered is not None:
            details.append(f"{field}={rendered}")
    error = value.get("error")
    if isinstance(error, Mapping):
        rendered = _accuracy_number(error.get("maximum_absolute"))
        if rendered is not None:
            details.append(f"max|error|={rendered}")
    tolerance = value.get("tolerance")
    if isinstance(tolerance, Mapping):
        rendered = _accuracy_number(tolerance.get("maximum"))
        if rendered is not None:
            details.append(f"max_tolerance={rendered}")
    suffix = f" [{', '.join(details)}]" if details else ""
    return f"{name}: {status}{suffix}"


def _accuracy_metric_details(metrics: Mapping[str, Any]) -> str:
    """Render the salient scalar diagnostics from replicated evidence."""
    nested = [
        _accuracy_gate_text(str(name), value)
        for name, value in sorted(metrics.items())
        if isinstance(value, Mapping)
        and "passed" in value
        and not (
            name == "bin_probabilities"
            and "contiguous_probabilities" in metrics
        )
    ]
    if nested:
        return "; ".join(nested)

    details = []
    for field in ("mean_ratio", "mean", "oracle", "error", "tolerance"):
        rendered = _accuracy_number(metrics.get(field))
        if rendered is not None:
            details.append(f"{field}={rendered}")
    return ", ".join(details)


def _timing_state_text(state: Mapping[str, Any]) -> str:
    """Render one exact power/thermal snapshot."""
    return (
        f"power=`{_canonical(state.get('power_status'))}`; "
        f"thermal=`{_canonical(state.get('thermal_status'))}`"
    )


def _diagnostic_line(item: Mapping[str, Any]) -> str:
    """Render one failed/missing cell without an unwieldy parameter dump."""
    return (
        f"`{item.get('workload')}` / {item.get('platform')} / "
        f"block {item.get('block')}"
    )


def render_markdown(
    report: Mapping[str, Any],
    *,
    report_date: date | None = None,
) -> str:
    """Render a dated Markdown report from a validated aggregate."""
    rendered_date = date.today() if report_date is None else report_date
    lines = [
        f"# smcx profiling report — {rendered_date.isoformat()}",
        "",
    ]
    if report["profile"] == "smoke":
        lines.extend([
            "> **NON-INFERENTIAL SMOKE RUN:** These measurements only verify ",
            "> the harness and must not be used for performance rankings or ",
            "> optimization claims.",
            "",
        ])

    if not report["complete"]:
        lines.extend([
            "## Incomplete campaign",
            "",
            "Missing, invalid, duplicate, or unexpected records prevent a ",
            "complete timing result.",
            "",
        ])
    elif not report["reproducible"]:
        lines.extend([
            "## Campaign failed reproducibility",
            "",
            "Source, package, or host identity is not eligible for timing ",
            "claims.",
            "",
        ])
    elif not report["correct"]:
        lines.extend([
            "## Campaign failed correctness",
            "",
            "No failed mathematical cell is eligible for timing claims.",
            "",
        ])
    elif report["profile"] != "smoke" and not report["performance_eligible"]:
        lines.extend([
            "## Campaign timing ineligible",
            "",
            "The campaign is complete and correct, but at least one timing ",
            "aggregate failed its registered performance-state gate.",
            "",
        ])
    else:
        lines.extend([
            "Campaign status: **complete and correct**.",
            "",
        ])

    lines.extend([
        "## Campaign",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Order seed: `{report['order_seed']}`",
        f"- Fixed seed contract: `{_canonical(report['seed_contract'])}`",
        (
            "- Scheduled worker processes: "
            f"{report['preflight']['total_worker_processes']} "
            f"({report['preflight']['timing_worker_processes']} timing + "
            f"{report['preflight']['validation_worker_processes']} "
            "validation)."
        ),
        (
            "- Scheduled workload executions: "
            f"{report['preflight']['total_scheduled_workload_executions']} "
            f"({report['preflight']['timing_workload_executions']} timing + "
            f"{report['preflight']['validation_replicate_executions']} "
            "validation replicates)."
        ),
        f"- Preflight scope: {report['preflight']['work_estimate_scope']}",
        (
            f"- Matched cells: {report['matched_cells']} / "
            f"{report['expected_cells']}"
        ),
        "",
        "## Environment and source",
        "",
    ])
    if report["environments"]:
        for environment in report["environments"]:
            lines.append(
                "- Device: "
                f"`{environment.get('device_kind')}`; machine "
                f"`{environment.get('machine')}`; macOS "
                f"`{environment.get('macos')}`; CPU "
                f"`{environment.get('cpu_model')}`; memory "
                f"`{environment.get('physical_memory_bytes')}` bytes"
            )
    else:
        lines.append("- Environment metadata unavailable.")
    for source in report["sources"]:
        lines.append(
            "- Source: commit "
            f"`{source.get('git_commit')}`; dirty "
            f"`{source.get('git_dirty')}`; SHA-256 "
            f"`{source.get('source_sha256')}`"
        )
    for versions in report["versions"]:
        lines.append(
            "- Versions: "
            + ", ".join(
                f"{name}={value}"
                for name, value in sorted(versions.items())
                if value is not None
            )
        )
    for schedule in report["schedules"]:
        lines.append(
            "- Schedule per process: "
            f"{schedule['warmups']} warm-up(s), "
            f"{schedule['repeats']} fenced repeat(s)."
        )
    if report["exclusions"]:
        lines.append("- Preregistered exclusions:")
        lines.extend(
            f"  - `{item['workload']}`: {item['reason']}"
            for item in report["exclusions"]
        )
    lines.extend(["", "### Power and thermal state", ""])
    if report["timing_states"]:
        lines.extend([
            (
                "| Platform | Cells | Before timing | After timing | "
                "After cell |"
            ),
            "|---|---:|---|---|---|",
        ])
        for state in report["timing_states"]:
            lines.append(
                f"| {state['platform']} | {state['cells']} | "
                f"{_timing_state_text(state['pre_timing'])} | "
                f"{_timing_state_text(state['post_timing'])} | "
                f"{_timing_state_text(state['post_cell'])} |"
            )
    else:
        lines.append("No paired pre/post timing-state metadata is available.")

    lines.extend([
        "",
        "## Configuration, correctness, and accuracy",
        "",
        (
            "Correctness levels describe the registered validation evidence. "
            "`statistical` and `oracle_accuracy` results summarize estimator "
            "accuracy; they are not proofs of implementation correctness."
        ),
        "",
        (
            "| Workload | Platform | Parameters | Execution | Dispatch | "
            "Correctness levels | Passed blocks | Accuracy summary |"
        ),
        "|---|---:|---|---|---|---|---:|---|",
    ])
    for aggregate in report["aggregates"]:
        correctness = aggregate["correctness"]
        levels = ", ".join(correctness["levels"]) or "unavailable"
        dispatch = ", ".join(aggregate["dispatch_modes"]) or "unavailable"
        lines.append(
            f"| `{aggregate['workload']}` | {aggregate['platform']} | "
            f"`{_canonical(aggregate['parameters'])}` | "
            f"`{aggregate['execution_mode']}` | `{dispatch}` | "
            f"`{levels}` | {correctness['passed_blocks']} / "
            f"{correctness['completed_blocks']} | "
            f"{_accuracy_text(correctness)} |"
        )
    lines.extend([
        "",
        "## Aggregates",
        "",
        (
            "| Workload | Platform | Status | Blocks | Steady median "
            "(block IQR), ms | Lower, ms | Compile, ms | First, ms | "
            "RSS, MiB | Executable peak, MiB | Device peak, MiB | Work |"
        ),
        ("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"),
    ])
    for aggregate in report["aggregates"]:
        lifecycle = aggregate["lifecycle"]
        lower = _format_ms(lifecycle["lowering_s"])
        compile_time = _format_ms(lifecycle["backend_compile_s"])
        if lifecycle["unavailable_reason"] == "host_controlled":
            lower = compile_time = "host-controlled"
        lines.append(
            "| "
            f"`{aggregate['workload']}` | {aggregate['platform']} | "
            f"{_status(aggregate)} | {aggregate['observed_blocks']} / "
            f"{aggregate['expected_blocks']} | {_format_steady(aggregate)} | "
            f"{lower} | {compile_time} | "
            f"{_format_ms(lifecycle['first_execution_s'])} | "
            f"{_format_mib(aggregate['memory']['process_max_rss_bytes'])} | "
            f"{_format_mib(aggregate['memory']['executable_peak_memory_bytes'])}"
            " "
            "| "
            f"{_format_mib(aggregate['memory']['device_peak_bytes_in_use'])} | "
            f"{_work_text(aggregate['work'])} |"
        )

    lines.extend(["", "## Matched CPU/MPS comparisons", ""])
    if report["comparisons"]:
        lines.extend([
            "| Workload | CPU median, ms | MPS median, ms | MPS / CPU |",
            "|---|---:|---:|---:|",
        ])
        for comparison in report["comparisons"]:
            lines.append(
                f"| `{comparison['workload']}` | "
                f"{1_000.0 * comparison['cpu_median_s']:.3f} | "
                f"{1_000.0 * comparison['mps_median_s']:.3f} | "
                f"{comparison['mps_over_cpu']:.3f} |"
            )
    else:
        lines.append(
            "No exact matched, correctness-eligible pair is available."
        )
    if report["comparison_exclusions"]:
        lines.extend(["", "Ratios withheld for unmatched adaptive work:"])
        lines.extend(
            f"- `{item['workload']}` with "
            f"`{_canonical(item['parameters'])}`: {item['reason']}"
            for item in report["comparison_exclusions"]
        )

    lines.extend([
        "",
        "## Matched representation comparisons",
        "",
    ])
    if report["arm_comparisons"]:
        lines.extend([
            "| Kind | Platform | Numerator / denominator | Ratio |",
            "|---|---:|---|---:|",
        ])
        for comparison in report["arm_comparisons"]:
            lines.append(
                f"| {comparison['kind']} | {comparison['platform']} | "
                f"`{comparison['numerator_workload']}` / "
                f"`{comparison['denominator_workload']}` | "
                f"{comparison['ratio']:.3f} |"
            )
    else:
        lines.append(
            "No matched, correctness-eligible representation pair is available."
        )
    if report["arm_comparison_exclusions"]:
        lines.extend(["", "Ratios withheld for unmatched adaptive work:"])
        lines.extend(
            f"- {item['kind']} `{item['numerator_workload']}` / "
            f"`{item['denominator_workload']}` on {item['platform']} with "
            f"`{_canonical(item['parameters'])}`: {item['reason']}"
            for item in report["arm_comparison_exclusions"]
        )

    lines.extend(["", "## Matched history comparisons", ""])
    if report["history_comparisons"]:
        lines.extend([
            "| Workload | Platform | Off, ms | On, ms | On / off |",
            "|---|---:|---:|---:|---:|",
        ])
        for comparison in report["history_comparisons"]:
            lines.append(
                f"| `{comparison['workload']}` | "
                f"{comparison['platform']} | "
                f"{1_000.0 * comparison['history_off_median_s']:.3f} | "
                f"{1_000.0 * comparison['history_on_median_s']:.3f} | "
                f"{comparison['history_on_over_off']:.3f} |"
            )
    else:
        lines.append(
            "No exact matched, correctness-eligible history pair is available."
        )
    if report["history_comparison_exclusions"]:
        lines.extend(["", "Ratios withheld for unmatched adaptive work:"])
        lines.extend(
            f"- `{item['workload']}` on {item['platform']} with "
            f"`{_canonical(item['parameters'])}`: {item['reason']}"
            for item in report["history_comparison_exclusions"]
        )

    lines.extend(["", "## Failures and completeness", ""])
    if report["missing_cells"]:
        lines.extend(["### Missing cells", ""])
        lines.extend(
            f"- {_diagnostic_line(item)}" for item in report["missing_cells"]
        )
        lines.append("")
    if report["invalid_records"]:
        lines.extend(["### Invalid raw records", ""])
        lines.extend(
            f"- `{item['path']}`: {item['error']}"
            for item in report["invalid_records"]
        )
        lines.append("")
    if report["identity_mismatches"]:
        lines.extend(["### Campaign identity mismatches", ""])
        lines.extend(
            f"- {_diagnostic_line(item)}: {item['error']} (`{item['path']}`)"
            for item in report["identity_mismatches"]
        )
        lines.append("")
    if report["duplicate_records"]:
        lines.extend(["### Duplicate raw records", ""])
        lines.extend(
            f"- {_diagnostic_line(item)}: {len(item['paths'])} files"
            for item in report["duplicate_records"]
        )
        lines.append("")
    if report["unexpected_records"]:
        lines.extend(["### Unexpected raw records", ""])
        lines.extend(
            f"- {_diagnostic_line(item)}: `{item['path']}`"
            for item in report["unexpected_records"]
        )
        lines.append("")
    if report["worker_failures"]:
        lines.extend(["### Worker failures", ""])
        lines.extend(
            f"- {_diagnostic_line(item)}: `{_canonical(item['failure'])}`"
            for item in report["worker_failures"]
        )
        lines.append("")
    if report["timing_state_failures"]:
        lines.extend(["### Timing-state exclusions", ""])
        lines.extend(
            f"- {_diagnostic_line(item)}: {item['reason']}"
            for item in report["timing_state_failures"]
        )
        lines.append("")

    lines.extend(["### Correctness failures", ""])
    if report["correctness_failures"]:
        lines.extend(
            f"- {_diagnostic_line(item)}: "
            f"`{item['correctness_level']}` validation failure; "
            f"`{_canonical(item['correctness'])}`"
            for item in report["correctness_failures"]
        )
    else:
        lines.append("None among valid completed worker records.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Render one profiling output directory as dated Markdown."""
    parser = argparse.ArgumentParser(
        description="Render an smcx profiling campaign report."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--date", dest="date_text")
    args = parser.parse_args(argv)

    report = build_report(args.input_dir)
    report_date = (
        None if args.date_text is None else date.fromisoformat(args.date_text)
    )
    markdown = render_markdown(report, report_date=report_date)
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
        print(args.output)
    return int(
        not report["complete"]
        or not report["correct"]
        or (report["profile"] != "smoke" and not report["performance_eligible"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
