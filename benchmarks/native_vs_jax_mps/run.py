# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Supervisor for isolated native MLX and jax-mps benchmark workers."""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

if __package__ in (None, ""):  # allow direct `python .../run.py` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.native_vs_jax_mps.common import (
    BOOTSTRAP_SEED,
    PINNED_VERSIONS,
    SCHEMA_VERSION,
    WORKLOAD_GRIDS,
    balanced_orders,
)

ARMS = ("mlx_gpu", "mlx_cpu", "jax_mps_sync", "jax_mps_async", "jax_cpu")

# Registered but held out of the pre-registered verdict matrix: report-only
# counter-experiments driven separately, never expanded into profile cells.
REPORT_ONLY_WORKLOADS = frozenset({"lgssm_pf_nohist"})


class Profile(NamedTuple):
    """Registered measurement schedule for a named benchmark profile."""

    blocks: int
    repeats: int
    warmups: int
    all_sizes: bool


PROFILES = {
    "smoke": Profile(blocks=1, repeats=1, warmups=1, all_sizes=False),
    "full": Profile(blocks=5, repeats=7, warmups=1, all_sizes=True),
}

# The end-to-end filter Kalman gate needs many independent keys; the protocol
# fixes twenty and runs them only in the first block so correctness is
# established once per cell without inflating every timed process.
LGSSM_CORRECTNESS_REPLICATES = 20

# A worker that hangs is retained as a failed cell rather than stalling the
# whole matrix; the operator supervises an interactive run and can raise this.
CELL_TIMEOUT_S = 1800


class Cell(NamedTuple):
    """One workload/size/arm/block combination the supervisor executes."""

    workload: str
    size: int
    arm: str
    block: int
    warmups: int
    repeats: int
    correctness_replicates: int


def build_worker_command(
    *,
    root: Path,
    arm: str,
    block: int,
    capture_ir: bool = False,
    correctness_replicates: int = 0,
    repeats: int,
    size: int,
    warmups: int,
    workload: str,
) -> list[str]:
    """Construct one fully pinned fresh-process worker command."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if capture_ir and arm.startswith("mlx_"):
        raise ValueError("IR capture is JAX-only; MLX exposes no StableHLO")

    if arm.startswith("mlx_"):
        command = [
            sys.executable,
            str(root / "benchmarks/native_vs_jax_mps/mlx_worker.py"),
        ]
    else:
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.13",
            "--with",
            f"jax=={PINNED_VERSIONS['jax']}",
            "--with",
            f"jaxlib=={PINNED_VERSIONS['jaxlib']}",
        ]
        if arm.startswith("jax_mps_"):
            command.extend(["--with", f"jax-mps=={PINNED_VERSIONS['jax-mps']}"])
        command.extend([
            "python",
            str(root / "benchmarks/native_vs_jax_mps/jax_worker.py"),
        ])

    command.extend([
        "--arm",
        arm,
        "--block",
        str(block),
        "--repeats",
        str(repeats),
        "--size",
        str(size),
        "--warmups",
        str(warmups),
        "--workload",
        workload,
    ])
    if capture_ir:
        command.append("--capture-ir")
    if correctness_replicates:
        command.extend([
            "--correctness-replicates",
            str(correctness_replicates),
        ])
    return command


def worker_environment(
    arm: str,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a sanitized environment for one explicit backend arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    environment = dict(os.environ if base is None else base)
    environment.pop("JAX_MPS_ASYNC_DISPATCH", None)
    environment.pop("JAX_PLATFORM_NAME", None)
    environment.pop("JAX_PLATFORMS", None)

    if arm.startswith("jax_mps_"):
        environment["JAX_PLATFORMS"] = "mps"
    elif arm == "jax_cpu":
        environment["JAX_PLATFORMS"] = "cpu"
    if arm == "jax_mps_async":
        environment["JAX_MPS_ASYNC_DISPATCH"] = "1"
    return environment


def plan_cells(profile: str, *, seed: int = BOOTSTRAP_SEED) -> list[Cell]:
    """Expand a registered profile into its deterministic ordered cells."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    settings = PROFILES[profile]

    cells: list[Cell] = []
    schedule = [
        w for w in sorted(WORKLOAD_GRIDS) if w not in REPORT_ONLY_WORKLOADS
    ]
    for index, workload in enumerate(schedule):
        grid = WORKLOAD_GRIDS[workload]
        sizes = grid if settings.all_sizes else (min(grid),)
        for size in sizes:
            # Rotate the seeded shuffle per cell so no arm systematically
            # gets the coolest machine across workloads.
            orders = balanced_orders(
                ARMS, blocks=settings.blocks, seed=seed + index
            )
            for block, order in enumerate(orders):
                for arm in order:
                    replicates = (
                        LGSSM_CORRECTNESS_REPLICATES
                        if workload == "lgssm_pf" and block == 0
                        else 0
                    )
                    cells.append(
                        Cell(
                            workload=workload,
                            size=size,
                            arm=arm,
                            block=block,
                            warmups=settings.warmups,
                            repeats=settings.repeats,
                            correctness_replicates=replicates,
                        )
                    )
    return cells


def raw_filename(cell: Cell) -> str:
    """Return the deterministic one-process-per-file raw result name."""
    return f"{cell.workload}_n{cell.size}_{cell.arm}_b{cell.block}.json"


def build_manifest(
    profile: str,
    cells: list[Cell],
    *,
    seed: int,
) -> dict:
    """Return the pre-execution manifest recording the frozen cell order."""
    return {
        "cells": [cell._asdict() for cell in cells],
        "profile": profile,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "versions": PINNED_VERSIONS,
    }


def _failure_record(cell: Cell, *, reason: str) -> dict:
    """Return a valid result envelope marking a cell as failed."""
    return {
        "arm": cell.arm,
        "backend": "unknown",
        "block": cell.block,
        "cold_s": None,
        "correctness": {"passed": False},
        "dispatch_mode": "none",
        "failure": {"reason": reason},
        "parameters": {"size": cell.size},
        "peak_memory_bytes": None,
        "schema_version": SCHEMA_VERSION,
        "summary": {},
        "times_s": [],
        "versions": {},
        "workload": cell.workload,
    }


def run_cell_subprocess(
    cell: Cell,
    *,
    root: Path,
    timeout_s: float = CELL_TIMEOUT_S,
) -> dict:
    """Run one cell in a fresh pinned process and parse its result."""
    command = build_worker_command(
        root=root,
        arm=cell.arm,
        block=cell.block,
        correctness_replicates=cell.correctness_replicates,
        repeats=cell.repeats,
        size=cell.size,
        warmups=cell.warmups,
        workload=cell.workload,
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=True,
            env=worker_environment(cell.arm),
            text=True,
            timeout=timeout_s,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except subprocess.CalledProcessError as error:
        return _failure_record(
            cell,
            reason=f"exit {error.returncode}: {error.stderr.strip()[-500:]}",
        )
    except subprocess.TimeoutExpired:
        return _failure_record(cell, reason=f"timeout after {timeout_s}s")
    except (json.JSONDecodeError, IndexError) as error:
        return _failure_record(cell, reason=f"unparseable output: {error}")


def supervise(
    profile: str,
    *,
    root: Path,
    output_dir: Path,
    seed: int = BOOTSTRAP_SEED,
    runner: Callable[[Cell], dict] | None = None,
) -> dict:
    """Run a profile, persisting the manifest and one raw file per process."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = plan_cells(profile, seed=seed)

    manifest_path = output_dir / "manifest.json"
    manifest = build_manifest(profile, cells, seed=seed)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    if runner is None:
        runner = lambda cell: run_cell_subprocess(cell, root=root)  # noqa: E731

    for cell in cells:
        raw_path = raw_dir / raw_filename(cell)
        if raw_path.exists():
            try:
                json.loads(raw_path.read_text())
            except json.JSONDecodeError:
                pass
            else:
                continue  # resume: never overwrite a completed raw block
        record = runner(cell)
        raw_path.write_text(json.dumps(record, sort_keys=True))

    completed = 0
    failed = 0
    for cell in cells:
        raw_path = raw_dir / raw_filename(cell)
        if not raw_path.exists():
            continue
        completed += 1
        try:
            record = json.loads(raw_path.read_text())
        except json.JSONDecodeError:
            continue
        if record.get("failure") is not None:
            failed += 1

    return {
        "cells": len(cells),
        "completed": completed,
        "failed": failed,
        "manifest_path": str(manifest_path),
        "profile": profile,
        "raw_dir": str(raw_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark supervisor from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the native MLX versus jax-mps benchmark matrix.",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the manifest and print the plan without running workers",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]

    if args.dry_run:
        cells = plan_cells(args.profile)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(args.profile, cells, seed=BOOTSTRAP_SEED)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        print(
            json.dumps(
                {"cells": len(cells), "dry_run": True, "profile": args.profile},
                sort_keys=True,
            )
        )
        return 0

    summary = supervise(args.profile, root=root, output_dir=args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
