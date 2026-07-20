# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Estimate the process and execution budget of a profiling plan."""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # Allow direct ``python .../preflight.py`` use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.profiling.common import (
    DEFAULT_ORDER_SEED,
    PLATFORMS,
    PROFILES,
    Cell,
    plan_cells,
)

_WORK_ESTIMATE_SCOPE = (
    "Counts scheduled public-workload calls only; adaptive inner steps and "
    "FLOPs are intentionally not estimated."
)


def _schedule_count(cell: Cell, field: str) -> int:
    """Return one validated non-negative integral schedule count."""
    value = getattr(cell, field)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return int(value)


def _count_cells(
    cells: Sequence[Cell],
    *,
    timeout_s: float,
) -> dict[str, int | float]:
    """Return exact schedule counts for a set of cells."""
    timing_executions = 0
    validation_executions = 0
    validation_processes = 0
    for cell in cells:
        warmups = _schedule_count(cell, "warmups")
        repeats = _schedule_count(cell, "repeats")
        replicates = _schedule_count(cell, "correctness_replicates")
        timing_executions += 1 + max(warmups - 1, 0) + repeats
        validation_executions += replicates
        validation_processes += int(replicates > 0)

    timing_processes = len(cells)
    total_processes = timing_processes + validation_processes
    return {
        "configured_timeout_upper_bound_s": total_processes * timeout_s,
        "scheduled_cells": timing_processes,
        "timing_worker_processes": timing_processes,
        "timing_workload_executions": timing_executions,
        "total_scheduled_workload_executions": (
            timing_executions + validation_executions
        ),
        "total_worker_processes": total_processes,
        "validation_replicate_executions": validation_executions,
        "validation_worker_processes": validation_processes,
    }


def _breakdown(
    cells: Sequence[Cell],
    *,
    field: str,
    timeout_s: float,
) -> dict[str, dict[str, int | float]]:
    """Group cells by a string field and count every resulting schedule."""
    grouped: dict[str, list[Cell]] = {}
    for cell in cells:
        key = getattr(cell, field)
        if not isinstance(key, str):
            raise ValueError(f"{field} must be a string")
        grouped.setdefault(key, []).append(cell)
    return {
        key: _count_cells(grouped[key], timeout_s=timeout_s)
        for key in sorted(grouped)
    }


def _native_number(name: str, value: object) -> int | float | None:
    """Convert a finite real parameter to a JSON-native number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, Real):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"numeric parameter {name!r} must be finite")
    return converted


def _largest_numeric_parameters(
    cells: Sequence[Cell],
) -> dict[str, dict[str, Any]]:
    """Describe maxima of explicit numeric axes without inferring work."""
    scheduled: dict[str, list[tuple[int | float, Cell]]] = {}
    for cell in cells:
        for name, value in cell.parameters.items():
            converted = _native_number(name, value)
            if converted is not None:
                scheduled.setdefault(name, []).append((converted, cell))

    largest: dict[str, dict[str, Any]] = {}
    for name in sorted(scheduled):
        maximum = max(value for value, _ in scheduled[name])
        cells_at_maximum = [
            cell for value, cell in scheduled[name] if value == maximum
        ]
        largest[name] = {
            "maximum": maximum,
            "platforms": sorted({cell.platform for cell in cells_at_maximum}),
            "scheduled_cells_at_maximum": len(cells_at_maximum),
            "workloads": sorted({cell.workload for cell in cells_at_maximum}),
        }
    return largest


def estimate_plan(
    cells: Sequence[Cell],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """Return a JSON-ready resource estimate for an exact cell sequence.

    The timeout bound sums the timeout configured for each sequential worker
    process. It bounds configured worker time, not supervisor I/O or teardown.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    scheduled = tuple(cells)
    counts = _count_cells(scheduled, timeout_s=timeout_s)
    return {
        **counts,
        "by_platform": _breakdown(
            scheduled,
            field="platform",
            timeout_s=timeout_s,
        ),
        "by_workload": _breakdown(
            scheduled,
            field="workload",
            timeout_s=timeout_s,
        ),
        "largest_numeric_parameters": _largest_numeric_parameters(scheduled),
        "timeout_s_per_worker": float(timeout_s),
        "work_estimate_scope": _WORK_ESTIMATE_SCOPE,
    }


def main(argv: list[str] | None = None) -> int:
    """Print the preflight estimate for one registered profiling plan."""
    parser = argparse.ArgumentParser(
        description="Estimate a registered profiling plan before launch."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument(
        "--platforms",
        choices=PLATFORMS,
        default=list(PLATFORMS),
        nargs="+",
    )
    parser.add_argument(
        "--order-seed",
        default=DEFAULT_ORDER_SEED,
        type=int,
    )
    parser.add_argument("--timeout-s", default=1_800.0, type=float)
    args = parser.parse_args(argv)

    cells = plan_cells(
        args.profile,
        platforms=args.platforms,
        order_seed=args.order_seed,
    )
    estimate = {
        "order_seed": args.order_seed,
        "platforms": list(args.platforms),
        "profile": args.profile,
        **estimate_plan(cells, timeout_s=args.timeout_s),
    }
    print(json.dumps(estimate, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
