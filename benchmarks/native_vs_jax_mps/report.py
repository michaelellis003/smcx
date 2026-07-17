# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first report generator for the native versus jax-mps matrix."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

if __package__ in (None, ""):  # allow direct `python .../report.py` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.native_vs_jax_mps.common import (
    BOOTSTRAP_SEED,
    bootstrap_ratio_ci,
)
from benchmarks.native_vs_jax_mps.run import PROFILES, plan_cells

# Pre-registered verdict thresholds (PROTOCOL.md, "Comparison and verdict").
RATIO_THRESHOLD = 1.5
STRONG_THRESHOLD = 3.0
MEMORY_BUDGET = 1.25
NOT_SUPPORTED_RATIO = 1.2
DRAWS = 10_000

SMC_MOTIFS = ("scan", "random", "gather_scatter", "systematic")
NEGATIVE_CONTROLS = ("eltwise_reduce", "matmul")

NATIVE_ARM = "mlx_gpu"
COMPATIBILITY_ARMS = ("jax_mps_sync", "jax_mps_async")

# Cap the failed/missing cell listing so a broken run does not print thousands
# of lines; the dropped count is always reported (no silent truncation).
MISSING_LIST_CAP = 25


class ArmSummary(NamedTuple):
    """Aggregate of every block a single workload/size/arm produced."""

    workload: str
    size: int
    arm: str
    process_medians: tuple[float, ...]
    primary_median: float | None
    peak_memory_bytes: float | None
    n_blocks: int
    n_passed: int
    n_failed: int


class CellComparison(NamedTuple):
    """Native-versus-best-compatibility comparison at one workload/size."""

    workload: str
    size: int
    native_available: bool
    compat_arm: str | None
    ratio: dict | None
    memory_ratio: float | None
    native_primary: float | None
    compat_primary: float | None
    reason: str


def summarize_arm(records: list[dict]) -> ArmSummary:
    """Collapse one arm's block records into a robust primary estimate."""
    if not records:
        raise ValueError("cannot summarize an empty record list")
    first = records[0]

    medians = [
        float(record["summary"]["median_s"])
        for record in records
        if record["failure"] is None and "median_s" in record["summary"]
    ]
    peaks = [
        float(record["peak_memory_bytes"])
        for record in records
        if record["failure"] is None and record["peak_memory_bytes"] is not None
    ]
    return ArmSummary(
        workload=first["workload"],
        size=first["parameters"]["size"],
        arm=first["arm"],
        process_medians=tuple(medians),
        primary_median=float(np.median(medians)) if medians else None,
        peak_memory_bytes=float(np.median(peaks)) if peaks else None,
        n_blocks=len(records),
        n_passed=sum(
            1 for record in records if record["correctness"].get("passed")
        ),
        n_failed=sum(1 for record in records if record["failure"] is not None),
    )


def _arm_correct(summary: ArmSummary | None, expected_blocks: int) -> bool:
    """Return whether an arm ran all blocks and passed every gate."""
    return (
        summary is not None
        and summary.n_blocks == expected_blocks
        and summary.n_failed == 0
        and summary.n_passed == expected_blocks
        and len(summary.process_medians) == expected_blocks
    )


def compare_cell(
    native: ArmSummary | None,
    safe: ArmSummary | None,
    asynchronous: ArmSummary | None,
    *,
    expected_blocks: int,
    workload: str | None = None,
    size: int | None = None,
    draws: int = DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> CellComparison:
    """Pick the faster correct jax-mps arm and compare it against native MLX."""
    # A fully missing cell has no summary to read the labels from, so accept
    # them explicitly and fall back to any present arm only for convenience.
    present = next(
        (arm for arm in (native, safe, asynchronous) if arm is not None), None
    )
    if workload is None:
        workload = present.workload if present is not None else ""
    if size is None:
        size = present.size if present is not None else 0

    native_available = _arm_correct(native, expected_blocks)
    candidates = [
        arm
        for arm in (safe, asynchronous)
        if arm is not None and _arm_correct(arm, expected_blocks)
    ]
    compat = min(candidates, key=lambda arm: arm.primary_median, default=None)

    ratio = None
    memory_ratio = None
    compat_primary = None
    reasons = []
    if not native_available:
        reasons.append("native cell missing or failed correctness")
    if compat is None:
        reasons.append("no jax-mps arm passed correctness")

    if native_available and compat is not None and native is not None:
        ratio = bootstrap_ratio_ci(
            native=native.process_medians,
            compatibility=compat.process_medians,
            draws=draws,
            seed=seed,
        )
        compat_primary = compat.primary_median
        best_peak = min(
            (
                arm.peak_memory_bytes
                for arm in candidates
                if arm.peak_memory_bytes is not None
            ),
            default=None,
        )
        if (
            native.peak_memory_bytes is not None
            and best_peak is not None
            and best_peak > 0
        ):
            memory_ratio = native.peak_memory_bytes / best_peak

    return CellComparison(
        workload=workload,
        size=size,
        native_available=native_available,
        compat_arm=compat.arm if compat is not None else None,
        ratio=ratio,
        memory_ratio=memory_ratio,
        native_primary=native.primary_median if native is not None else None,
        compat_primary=compat_primary,
        reason="; ".join(reasons),
    )


def workload_advantage(comparisons: list[CellComparison]) -> dict:
    """Decide the persistent native advantage at the two largest sizes."""
    workload = comparisons[0].workload
    by_size = {comparison.size: comparison for comparison in comparisons}
    large_sizes = sorted(by_size)[-2:]

    persistent = True
    reasons = []
    for size in large_sizes:
        comparison = by_size[size]
        if not comparison.native_available:
            persistent = False
            reasons.append(f"N={size}: native failed correctness")
            continue
        if comparison.compat_arm is None or comparison.ratio is None:
            persistent = False
            reasons.append(f"N={size}: no correct jax-mps arm")
            continue
        if comparison.ratio["low"] < RATIO_THRESHOLD:
            persistent = False
            reasons.append(
                f"N={size}: ratio lower bound "
                f"{comparison.ratio['low']:.2f} < {RATIO_THRESHOLD}"
            )
        if (
            comparison.memory_ratio is None
            or comparison.memory_ratio > MEMORY_BUDGET
        ):
            persistent = False
            reasons.append(f"N={size}: peak memory over the 1.25x budget")

    return {
        "workload": workload,
        "persistent": persistent,
        "large_sizes": large_sizes,
        "reason": "; ".join(reasons),
    }


def ecosystem_verdict(comparisons_by_workload: dict[str, list]) -> dict:
    """Apply the pre-registered native-ecosystem verdict over SMC workloads."""
    advantages = {
        workload: workload_advantage(comparisons)
        for workload, comparisons in comparisons_by_workload.items()
        if comparisons
    }

    lgssm = comparisons_by_workload.get("lgssm_pf", [])
    lgssm_by_size = {comparison.size: comparison for comparison in lgssm}
    large = sorted(lgssm_by_size)[-2:]

    lgssm_persistent = advantages.get("lgssm_pf", {}).get("persistent", False)
    motif_count = sum(
        1 for motif in SMC_MOTIFS if advantages.get(motif, {}).get("persistent")
    )

    within = [
        lgssm_by_size[size].ratio is not None
        and lgssm_by_size[size].ratio["estimate"] <= NOT_SUPPORTED_RATIO
        for size in large
    ]
    not_supported = bool(large) and all(within)

    lows = [
        lgssm_by_size[size].ratio["low"]
        if lgssm_by_size[size].ratio is not None
        else 0.0
        for size in large
    ]
    strong = (
        lgssm_persistent
        and len(large) >= 2
        and all(low >= STRONG_THRESHOLD for low in lows)
    )

    if not_supported:
        verdict = "not_supported"
    elif lgssm_persistent and motif_count >= 2 and strong:
        verdict = "strongly_supported"
    elif lgssm_persistent and motif_count >= 2:
        verdict = "supported"
    else:
        verdict = "mixed"

    return {
        "verdict": verdict,
        "advantages": advantages,
        "lgssm_persistent": lgssm_persistent,
        "motif_count": motif_count,
        "strong": strong,
    }


def _ratio_cell(comparison: CellComparison) -> str:
    """Render one ratio interval or the reason it is unavailable."""
    if comparison.ratio is None:
        return comparison.reason or "n/a"
    ratio = comparison.ratio
    return f"{ratio['low']:.2f} / {ratio['estimate']:.2f} / {ratio['high']:.2f}"


def _workload_table(
    workload: str,
    comparisons: list[CellComparison],
    advantage: dict | None,
) -> list[str]:
    """Render the per-size comparison table for one workload."""
    lines = [
        f"### {workload}",
        "",
        "| N | native median (s) | best jax arm | jax median (s) | "
        "ratio low/est/high | mem ratio |",
        "|---|---|---|---|---|---|",
    ]
    for comparison in sorted(comparisons, key=lambda item: item.size):
        native = (
            f"{comparison.native_primary:.6g}"
            if comparison.native_primary is not None
            else "—"
        )
        compat = (
            f"{comparison.compat_primary:.6g}"
            if comparison.compat_primary is not None
            else "—"
        )
        memory = (
            f"{comparison.memory_ratio:.2f}"
            if comparison.memory_ratio is not None
            else "—"
        )
        lines.append(
            f"| {comparison.size} | {native} | "
            f"{comparison.compat_arm or '—'} | {compat} | "
            f"{_ratio_cell(comparison)} | {memory} |"
        )
    if advantage is not None:
        note = (
            "persistent native advantage"
            if advantage["persistent"]
            else (advantage["reason"] or "no persistent advantage")
        )
        lines.extend(["", f"Persistent native advantage: {note}."])
    lines.append("")
    return lines


def render_report(
    manifest: dict,
    comparisons_by_workload: dict[str, list],
    verdict: dict,
    missing: list[tuple[str, int, str]],
) -> str:
    """Render the full correctness-first Markdown report."""
    lines = [
        "# Native MLX versus jax-mps — results",
        "",
        f"Profile: `{manifest.get('profile', 'unknown')}`. "
        f"Balanced-order seed: {manifest.get('seed', 'unknown')}.",
        "",
        "## Verdict",
        "",
        f"Native SMC ecosystem case: **{verdict['verdict']}** "
        f"(LGSSM-PF persistent: {verdict['lgssm_persistent']}; "
        f"supporting motifs: {verdict['motif_count']}/4; "
        f"strong: {verdict['strong']}).",
        "",
        "The negative controls below calibrate the harness and never count "
        "toward this verdict.",
        "",
        "## SMC workloads",
        "",
    ]
    advantages = verdict["advantages"]
    smc = [
        workload
        for workload in comparisons_by_workload
        if workload not in NEGATIVE_CONTROLS
    ]
    for workload in sorted(smc):
        lines.extend(
            _workload_table(
                workload,
                comparisons_by_workload[workload],
                advantages.get(workload),
            )
        )

    lines.extend(["## Negative controls", ""])
    controls = [
        workload
        for workload in comparisons_by_workload
        if workload in NEGATIVE_CONTROLS
    ]
    for workload in sorted(controls):
        lines.extend(
            _workload_table(workload, comparisons_by_workload[workload], None)
        )

    lines.extend(["## Missing or failed cells", ""])
    if not missing:
        lines.append("None: every registered cell produced a valid result.")
    else:
        lines.append(f"{len(missing)} registered cells are missing or failed:")
        lines.append("")
        for workload, size, arm in missing[:MISSING_LIST_CAP]:
            lines.append(f"- `{workload}` N={size} arm=`{arm}`")
        if len(missing) > MISSING_LIST_CAP:
            lines.append(
                f"- … and {len(missing) - MISSING_LIST_CAP} more "
                "(rerun `run.py` with the same output directory to resume)."
            )
    lines.append("")
    return "\n".join(lines)


def generate_report(results_dir: Path) -> str:
    """Load a results directory and produce the Markdown report."""
    results_dir = Path(results_dir)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    profile = manifest["profile"]
    expected_blocks = PROFILES[profile].blocks

    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for path in sorted((results_dir / "raw").glob("*.json")):
        record = json.loads(path.read_text())
        key = (record["workload"], record["parameters"]["size"], record["arm"])
        groups[key].append(record)
    summaries = {key: summarize_arm(recs) for key, recs in groups.items()}

    expected = {
        (cell.workload, cell.size, cell.arm) for cell in plan_cells(profile)
    }
    missing = sorted(expected - set(groups))

    sizes_by_workload: dict[str, set[int]] = defaultdict(set)
    for workload, size, _arm in expected:
        sizes_by_workload[workload].add(size)

    comparisons_by_workload: dict[str, list[CellComparison]] = defaultdict(list)
    for workload, sizes in sizes_by_workload.items():
        for size in sorted(sizes):
            comparisons_by_workload[workload].append(
                compare_cell(
                    summaries.get((workload, size, NATIVE_ARM)),
                    summaries.get((workload, size, COMPATIBILITY_ARMS[0])),
                    summaries.get((workload, size, COMPATIBILITY_ARMS[1])),
                    expected_blocks=expected_blocks,
                    workload=workload,
                    size=size,
                )
            )

    smc = {
        workload: comparisons
        for workload, comparisons in comparisons_by_workload.items()
        if workload not in NEGATIVE_CONTROLS
    }
    verdict = ecosystem_verdict(smc)
    return render_report(manifest, comparisons_by_workload, verdict, missing)


def main(argv: list[str] | None = None) -> int:
    """Render a report for a results directory given on the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Render the native MLX versus jax-mps benchmark report.",
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = generate_report(args.results_dir)
    if args.output is None:
        print(report)
    else:
        args.output.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
