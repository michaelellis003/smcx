# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Capture a fenced CPU JAX trace for one selected raw cell.

jax-mps 0.10.10's PJRT profiler is an explicit unsupported stub, so this tool
must never label a JAX profiler artifact as an MPS device trace:
https://github.com/tillahoffmann/jax-mps/blob/v0.10.10/src/pjrt_plugin/pjrt_profiler.cc#L57-L79
"""

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # Allow direct ``python .../trace.py`` use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.profiling.common import (
    DEFAULT_SEED,
    Cell,
    campaign_identity,
    canonical_json,
    expected_device_identity,
    profiling_runtime_flags,
    validate_result,
    worker_environment,
)
from benchmarks.profiling.locking import HostCampaignLock
from benchmarks.profiling.report import build_report
from benchmarks.profiling.run import raw_filename

_PROFILER_DOCUMENTATION = "https://docs.jax.dev/en/latest/profiling.html"


def cell_from_result(result: Mapping[str, Any]) -> Cell:
    """Recover the exact cell from one successful, correctness-passing row."""
    validate_result(result)
    if result["failure"] is not None or not result["correctness"]["passed"]:
        raise ValueError("tracing requires a successful, correct raw result")
    return Cell(
        workload=str(result["workload"]),
        platform=str(result["platform_requested"]),
        block=int(result["block"]),
        warmups=int(result["warmups"]),
        repeats=int(result["repeats"]),
        execution_mode=str(result["execution_mode"]),
        parameters=dict(result["parameters"]),
        correctness_replicates=int(result["correctness_replicates"]),
    )


def _read_result(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read one raw result and retain its exact bytes for provenance."""
    encoded = path.read_bytes()
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise ValueError("raw result must be a JSON object")
    return result, encoded


def _aggregate_matches_cell(
    aggregate: Mapping[str, Any],
    cell: Cell,
) -> bool:
    """Return whether one report aggregate contains the selected cell."""
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


def load_trace_target(
    campaign_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Load one exact, non-smoke, timing-eligible campaign member."""
    campaign_dir = Path(campaign_dir).resolve()
    report = build_report(campaign_dir)
    if report["profile"] == "smoke":
        raise ValueError("profiler traces require a non-smoke campaign")

    result_path = Path(result_path).resolve()
    if result_path.parent != (campaign_dir / "raw").resolve():
        raise ValueError("trace result must live in the campaign raw directory")
    result, result_encoded = _read_result(result_path)
    cell = cell_from_result(result)
    if cell.platform != "cpu":
        raise ValueError("JAX profiler trace capture is CPU-only")
    if result_path.name != raw_filename(cell):
        raise ValueError("trace result does not use its canonical raw filename")

    manifest_path = campaign_dir / "manifest.json"
    manifest_encoded = manifest_path.read_bytes()
    manifest = json.loads(manifest_encoded)
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must be a JSON object")
    manifest_cells = manifest.get("cells", [])
    exact_member = isinstance(manifest_cells, list) and any(
        isinstance(item, Mapping)
        and canonical_json(item) == canonical_json(cell._asdict())
        for item in manifest_cells
    )
    if not exact_member:
        raise ValueError("trace result is not an exact manifest member")

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
            "trace result belongs to a timing-ineligible aggregate"
        )
    return {
        "cell": cell,
        "dispatch_mode": result["dispatch_mode"],
        "manifest_sha256": hashlib.sha256(manifest_encoded).hexdigest(),
        "plan_sha256": manifest["plan_sha256"],
        "profile": report["profile"],
        "result": result,
        "result_sha256": hashlib.sha256(result_encoded).hexdigest(),
    }


def _require_current_identity(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Prevent a trace from being attributed to different code or hardware."""
    identity = campaign_identity()
    if result["source"] != identity["source"]:
        raise ValueError("raw result source does not match current source")
    if result["versions"] != identity["packages"]:
        raise ValueError("raw result packages do not match current packages")
    if any(
        result["environment"].get(name) != value
        for name, value in identity["host"].items()
    ):
        raise ValueError("raw result host does not match current host")
    return identity


def _activate_environment(platform: str) -> None:
    """Apply the same sanitized runtime environment as timing workers."""
    sanitized = worker_environment(platform)
    for name in tuple(os.environ):
        if name not in sanitized:
            os.environ.pop(name)
    os.environ.update(sanitized)


def capture_trace(
    cell: Cell,
    *,
    identity: Mapping[str, Any],
    output_dir: Path,
    provenance: Mapping[str, Any],
    steps: int,
) -> dict[str, Any]:
    """Warm one exact operation, then capture fenced device work."""
    if cell.platform != "cpu":
        raise ValueError("JAX profiler trace capture is CPU-only")
    if steps < 1:
        raise ValueError("steps must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("trace output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    _activate_environment(cell.platform)

    # Imports must follow environment activation so JAX initializes one backend.
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
        raise RuntimeError("trace device differs from the scheduled platform")
    prepared = prepare_workload(
        cell.workload,
        parameters=cell.parameters,
        seed=DEFAULT_SEED,
    )
    arguments = jax.device_put(tuple(prepared.arguments), device)
    operation = (
        jax.jit(prepared.operation).lower(*arguments).compile()
        if cell.execution_mode == "whole_program_jit"
        else prepared.operation
    )
    jax.block_until_ready(operation(*arguments))

    with jax.profiler.trace(
        output_dir,
        create_perfetto_trace=True,
    ):
        for step in range(steps):
            with jax.profiler.StepTraceAnnotation(
                "smcx_profile_step",
                step_num=step,
            ):
                jax.block_until_ready(operation(*arguments))

    metadata = {
        "campaign_identity": dict(identity),
        "cell": cell._asdict(),
        "device_id": int(getattr(device, "id", 0)),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "jax_profiler_documentation": _PROFILER_DOCUMENTATION,
        "provenance": dict(provenance),
        "runtime_flags": profiling_runtime_flags(),
        "steps": steps,
    }
    with (output_dir / "metadata.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Capture a trace from an explicitly selected successful raw result."""
    parser = argparse.ArgumentParser(
        description="Trace one selected, correctness-passing profiling cell."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", default=3, type=int)
    args = parser.parse_args(argv)

    with HostCampaignLock():
        target = load_trace_target(
            args.campaign_dir,
            result_path=args.result,
        )
        cell = target["cell"]
        identity = _require_current_identity(target["result"])
        capture_trace(
            cell,
            identity=identity,
            output_dir=args.output_dir,
            provenance={
                "dispatch_mode": target["dispatch_mode"],
                "manifest_sha256": target["manifest_sha256"],
                "plan_sha256": target["plan_sha256"],
                "profile": target["profile"],
                "result_sha256": target["result_sha256"],
            },
            steps=args.steps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
