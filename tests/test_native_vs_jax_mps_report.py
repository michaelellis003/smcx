# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native MLX versus jax-mps benchmark report generator."""

import json

import pytest

from benchmarks.native_vs_jax_mps.common import WORKLOAD_GRIDS
from benchmarks.native_vs_jax_mps.report import (
    CellComparison,
    compare_cell,
    ecosystem_verdict,
    generate_report,
    summarize_arm,
    workload_advantage,
)


def _arm_records(
    workload,
    size,
    arm,
    medians,
    *,
    passed=True,
    peak=1000.0,
    failures=0,
):
    """Build minimal per-block worker records for one arm."""
    records = []
    for block, median in enumerate(medians):
        records.append({
            "arm": arm,
            "workload": workload,
            "parameters": {"size": size},
            "block": block,
            "summary": {"median_s": median},
            "peak_memory_bytes": peak,
            "correctness": {"passed": passed},
            "failure": None,
        })
    for offset in range(failures):
        records.append({
            "arm": arm,
            "workload": workload,
            "parameters": {"size": size},
            "block": len(medians) + offset,
            "summary": {},
            "peak_memory_bytes": None,
            "correctness": {"passed": False},
            "failure": {"reason": "boom"},
        })
    return records


def _summary(workload, size, arm, medians, **kwargs):
    return summarize_arm(_arm_records(workload, size, arm, medians, **kwargs))


def _comp(
    workload,
    size,
    ratio_low,
    *,
    ratio_est=None,
    memory_ratio=1.0,
    native_available=True,
    compat_arm="jax_mps_sync",
):
    """Build one CellComparison directly for verdict-logic tests."""
    estimate = ratio_low if ratio_est is None else ratio_est
    ratio = None
    if compat_arm is not None:
        ratio = {"estimate": estimate, "high": estimate + 0.5, "low": ratio_low}
    return CellComparison(
        workload=workload,
        size=size,
        native_available=native_available,
        compat_arm=compat_arm,
        ratio=ratio,
        memory_ratio=memory_ratio,
        native_primary=0.01,
        compat_primary=0.02,
        reason="",
    )


def test_summarize_arm_computes_primary_median_and_correctness():
    summary = summarize_arm(
        _arm_records("scan", 10_000, "mlx_gpu", [0.03, 0.01, 0.02, 0.04, 0.05])
    )

    assert summary.arm == "mlx_gpu"
    assert summary.process_medians == (0.03, 0.01, 0.02, 0.04, 0.05)
    assert summary.primary_median == pytest.approx(0.03)
    assert summary.n_passed == 5
    assert summary.n_failed == 0


def test_summarize_arm_marks_failed_blocks_incorrect():
    summary = summarize_arm(
        _arm_records("scan", 10_000, "mlx_gpu", [0.03, 0.02], failures=1)
    )

    assert summary.n_blocks == 3
    assert summary.n_failed == 1
    assert summary.process_medians == (0.03, 0.02)


def test_compare_cell_picks_faster_correct_jax_arm():
    native = _summary("scan", 10_000, "mlx_gpu", [0.01] * 5)
    safe = _summary("scan", 10_000, "jax_mps_sync", [0.05] * 5)
    fast = _summary("scan", 10_000, "jax_mps_async", [0.03] * 5)

    comparison = compare_cell(native, safe, fast, expected_blocks=5)

    assert comparison.compat_arm == "jax_mps_async"
    assert comparison.ratio is not None
    assert comparison.ratio["estimate"] == pytest.approx(3.0)


def test_compare_cell_falls_back_when_faster_jax_arm_is_incorrect():
    native = _summary("scan", 10_000, "mlx_gpu", [0.01] * 5)
    safe = _summary("scan", 10_000, "jax_mps_sync", [0.05] * 5)
    fast = _summary("scan", 10_000, "jax_mps_async", [0.03] * 5, passed=False)

    comparison = compare_cell(native, safe, fast, expected_blocks=5)

    assert comparison.compat_arm == "jax_mps_sync"


def test_compare_cell_keeps_labels_for_a_fully_missing_cell():
    comparison = compare_cell(
        None,
        None,
        None,
        expected_blocks=5,
        workload="lgssm_pf",
        size=1_000_000,
    )

    assert comparison.workload == "lgssm_pf"
    assert comparison.size == 1_000_000
    assert comparison.native_available is False
    assert comparison.compat_arm is None


def test_compare_cell_reports_no_compat_when_both_jax_incorrect():
    native = _summary("scan", 10_000, "mlx_gpu", [0.01] * 5)
    safe = _summary("scan", 10_000, "jax_mps_sync", [0.05] * 5, passed=False)
    fast = _summary("scan", 10_000, "jax_mps_async", [0.03] * 5, passed=False)

    comparison = compare_cell(native, safe, fast, expected_blocks=5)

    assert comparison.compat_arm is None
    assert comparison.ratio is None


def test_workload_advantage_requires_lower_bound_at_both_large_sizes():
    comparisons = [
        _comp("scan", 10_000, 1.0),
        _comp("scan", 100_000, 1.6),
        _comp("scan", 1_000_000, 2.0),
    ]

    assert workload_advantage(comparisons)["persistent"] is True


def test_workload_advantage_fails_when_one_large_size_below_threshold():
    comparisons = [
        _comp("scan", 100_000, 1.6),
        _comp("scan", 1_000_000, 1.2),
    ]

    assert workload_advantage(comparisons)["persistent"] is False


def test_workload_advantage_rejects_when_memory_exceeds_budget():
    comparisons = [
        _comp("scan", 100_000, 2.0, memory_ratio=1.5),
        _comp("scan", 1_000_000, 2.0, memory_ratio=1.5),
    ]

    assert workload_advantage(comparisons)["persistent"] is False


def test_workload_advantage_rejects_incorrect_native_cell():
    comparisons = [
        _comp("scan", 100_000, 2.0),
        _comp("scan", 1_000_000, 2.0, native_available=False, compat_arm=None),
    ]

    assert workload_advantage(comparisons)["persistent"] is False


def _persistent(workload, low=2.0):
    return [
        _comp(workload, size, low) for size in sorted(WORKLOAD_GRIDS[workload])
    ]


def _no_advantage(workload):
    return [
        _comp(workload, size, 1.0) for size in sorted(WORKLOAD_GRIDS[workload])
    ]


def test_ecosystem_supported_needs_lgssm_and_two_motifs():
    comparisons = {
        "lgssm_pf": _persistent("lgssm_pf"),
        "scan": _persistent("scan"),
        "random": _persistent("random"),
        "gather_scatter": _no_advantage("gather_scatter"),
        "systematic": _no_advantage("systematic"),
    }

    assert ecosystem_verdict(comparisons)["verdict"] == "supported"


def test_ecosystem_strong_needs_lgssm_lower_bound_three():
    comparisons = {
        "lgssm_pf": _persistent("lgssm_pf", low=3.0),
        "scan": _persistent("scan"),
        "random": _persistent("random"),
    }

    assert ecosystem_verdict(comparisons)["verdict"] == "strongly_supported"


def test_ecosystem_not_supported_when_jax_within_1_2x():
    comparisons = {
        "lgssm_pf": [
            _comp("lgssm_pf", size, 0.9, ratio_est=1.05)
            for size in sorted(WORKLOAD_GRIDS["lgssm_pf"])
        ],
    }

    assert ecosystem_verdict(comparisons)["verdict"] == "not_supported"


def test_ecosystem_mixed_when_lgssm_persists_but_motifs_insufficient():
    comparisons = {
        "lgssm_pf": _persistent("lgssm_pf"),
        "scan": _no_advantage("scan"),
    }

    assert ecosystem_verdict(comparisons)["verdict"] == "mixed"


def test_generate_report_separates_controls_and_flags_missing(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"profile": "smoke", "seed": 1, "versions": {}})
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    records = (
        _arm_records("scan", 10_000, "mlx_gpu", [0.01])
        + _arm_records("scan", 10_000, "jax_mps_sync", [0.05])
        + _arm_records("scan", 10_000, "jax_mps_async", [0.04])
        + _arm_records("matmul", 256, "mlx_gpu", [0.02])
    )
    for index, record in enumerate(records):
        (raw_dir / f"cell_{index}.json").write_text(json.dumps(record))

    report = generate_report(tmp_path)

    assert "Verdict" in report
    assert "Negative controls" in report
    assert "missing" in report.lower()
