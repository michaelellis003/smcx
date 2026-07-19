# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Count StableHLO operations for outer-jittable profiling workloads."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # Allow direct ``python .../census.py`` use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.profiling.common import (
    DEFAULT_SEED,
    PLATFORMS,
    Cell,
    campaign_identity,
    canonical_json,
    expected_device_identity,
    profiling_runtime_flags,
    record_matches_cell,
    validate_result,
    worker_environment,
)
from benchmarks.profiling.locking import HostCampaignLock
from benchmarks.profiling.report import build_report
from benchmarks.profiling.run import raw_filename

_OPERATION_PATTERN = re.compile(r"\bstablehlo\.([a-zA-Z0-9_]+)\b")


def build_census(cell: Cell) -> dict[str, Any]:
    """Lower one cell and count operations in its StableHLO module."""
    if cell.execution_mode != "whole_program_jit":
        raise ValueError("StableHLO census requires whole_program_jit")

    import jax

    from benchmarks.profiling.workloads import prepare_workload

    actual_backend = jax.default_backend()
    if actual_backend != cell.platform:
        raise RuntimeError(
            f"requested {cell.platform} backend, got {actual_backend}"
        )
    device = jax.devices()[0]
    expected_kind, expected_id = expected_device_identity(cell.platform)
    if (
        str(getattr(device, "device_kind", "unknown")),
        int(getattr(device, "id", -1)),
    ) != (expected_kind, expected_id):
        raise RuntimeError("census device differs from the scheduled platform")
    prepared = prepare_workload(
        cell.workload,
        parameters=cell.parameters,
        seed=DEFAULT_SEED,
    )
    arguments = jax.device_put(tuple(prepared.arguments), device)
    module = (
        jax
        .jit(prepared.operation)
        .lower(*arguments)
        .compiler_ir(dialect="stablehlo")
    )
    text = str(module)
    encoded = text.encode()
    counts = Counter(_OPERATION_PATTERN.findall(text))
    return {
        "algorithm": prepared.algorithm,
        "environment": {
            "device_id": int(getattr(device, "id", 0)),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
            "runtime_flags": {
                **profiling_runtime_flags(),
                "JAX_ENABLE_X64": str(
                    bool(jax.config.read("jax_enable_x64"))
                ).lower(),
                "JAX_PLATFORMS": actual_backend,
            },
        },
        "model": prepared.model,
        "operation_counts": dict(sorted(counts.items())),
        "parameters": dict(cell.parameters),
        "platform": actual_backend,
        "stablehlo_bytes": len(encoded),
        "stablehlo_sha256": hashlib.sha256(encoded).hexdigest(),
        "total_operations": sum(counts.values()),
        "workload": cell.workload,
    }


def _aggregate_matches_cell(
    aggregate: dict[str, Any],
    cell: Cell,
) -> bool:
    """Return whether a report aggregate contains one manifest cell."""
    observed = {
        "execution_mode": aggregate.get("execution_mode"),
        "parameters": aggregate.get("parameters"),
        "platform": aggregate.get("platform"),
        "repeats": aggregate.get("repeats"),
        "warmups": aggregate.get("warmups"),
        "workload": aggregate.get("workload"),
    }
    expected = {
        "execution_mode": cell.execution_mode,
        "parameters": cell.parameters,
        "platform": cell.platform,
        "repeats": cell.repeats,
        "warmups": cell.warmups,
        "workload": cell.workload,
    }
    return canonical_json(observed) == canonical_json(expected)


def load_census_targets(
    campaign_dir: Path,
    *,
    platform: str,
) -> dict[str, Any]:
    """Load exact block-zero cells from a timing-eligible campaign."""
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform: {platform}")
    campaign_dir = Path(campaign_dir).resolve()
    report = build_report(campaign_dir)
    if report["profile"] == "smoke":
        raise ValueError("StableHLO census requires a non-smoke campaign")

    manifest_path = campaign_dir / "manifest.json"
    manifest_encoded = manifest_path.read_bytes()
    manifest = json.loads(manifest_encoded)
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must be a JSON object")
    cells = [
        Cell(**raw)
        for raw in manifest["cells"]
        if raw["block"] == 0
        and raw["platform"] == platform
        and raw["execution_mode"] == "whole_program_jit"
    ]
    if not cells:
        raise ValueError("campaign has no outer-jittable census targets")

    raw_sha256 = {}
    for cell in cells:
        aggregate = next(
            (
                item
                for item in report["aggregates"]
                if _aggregate_matches_cell(item, cell)
            ),
            None,
        )
        if aggregate is None or not aggregate.get("timing_eligible"):
            raise ValueError(
                f"census target is timing-ineligible: {cell.workload}"
            )
        path = campaign_dir / "raw" / raw_filename(cell)
        encoded = path.read_bytes()
        record = json.loads(encoded)
        if not isinstance(record, dict):
            raise ValueError("raw census target must be a JSON object")
        validate_result(record)
        if record["failure"] is not None or not record["correctness"]["passed"]:
            raise ValueError("census target raw result is not correct")
        if not record_matches_cell(record, cell):
            raise ValueError(
                "census target raw result does not match its manifest cell"
            )
        raw_sha256[path.name] = hashlib.sha256(encoded).hexdigest()

    return {
        "campaign_identity": manifest["campaign_identity"],
        "cells": cells,
        "exclusions": manifest.get("exclusions", []),
        "manifest_sha256": hashlib.sha256(manifest_encoded).hexdigest(),
        "plan_sha256": manifest["plan_sha256"],
        "profile": report["profile"],
        "raw_sha256": raw_sha256,
    }


def _activate_environment(platform: str) -> None:
    """Apply the same sanitized environment as the timing campaign."""
    sanitized = worker_environment(platform)
    for name in tuple(os.environ):
        if name not in sanitized:
            os.environ.pop(name)
    os.environ.update(sanitized)


def build_census_campaign(
    campaign_dir: Path,
    *,
    platform: str,
) -> dict[str, Any]:
    """Build one census anchored to an immutable successful campaign."""
    target = load_census_targets(campaign_dir, platform=platform)
    frozen_identity = target["campaign_identity"]
    if campaign_identity() != frozen_identity:
        raise ValueError("census environment differs from campaign identity")
    _activate_environment(platform)
    records = [build_census(cell) for cell in target["cells"]]
    if campaign_identity() != frozen_identity:
        raise ValueError("campaign identity changed while building census")
    return {
        "campaign_identity": frozen_identity,
        "exclusions": target["exclusions"],
        "manifest_sha256": target["manifest_sha256"],
        "plan_sha256": target["plan_sha256"],
        "platform": platform,
        "profile": target["profile"],
        "raw_sha256": target["raw_sha256"],
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    """Write a JSON StableHLO census for one registered profile."""
    parser = argparse.ArgumentParser(
        description="Count StableHLO operations for a profiling profile."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--platform", choices=PLATFORMS, default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    with HostCampaignLock():
        result = build_census_campaign(
            args.campaign_dir,
            platform=args.platform,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
