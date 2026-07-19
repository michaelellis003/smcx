# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first aggregation contracts for profiling reports."""

import hashlib
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from benchmarks.profiling.common import (
    PLATFORMS,
    SCHEMA_VERSION,
    WORKLOADS,
    Cell,
    build_manifest,
    campaign_identity,
    plan_cells,
    profiling_runtime_flags,
    summarize,
    worker_environment,
)
from benchmarks.profiling.preflight import estimate_plan
from benchmarks.profiling.report import build_report, render_markdown
from benchmarks.profiling.report import main as report_main
from benchmarks.profiling.run import raw_filename

_TEST_CAMPAIGN_IDENTITY = campaign_identity()
_TEST_CAMPAIGN_IDENTITY["source"]["git_dirty"] = False


def _write_manifest(
    output_dir: Path,
    profile: str,
    cells: list[Cell],
) -> None:
    output_dir.mkdir(exist_ok=True)
    requested = {cell.platform for cell in cells}
    platforms = tuple(
        platform for platform in PLATFORMS if platform in requested
    )
    registered_cells = plan_cells(
        profile,
        platforms=platforms,
        seed=20260719,
    )
    manifest = build_manifest(
        profile,
        registered_cells,
        platforms=platforms,
        seed=20260719,
    )
    manifest["preflight"] = estimate_plan(
        registered_cells,
        timeout_s=1_800.0,
    )
    manifest["campaign_identity"] = deepcopy(_TEST_CAMPAIGN_IDENTITY)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    (output_dir / "raw").mkdir()


def _record(
    cell: Cell,
    process_median: float,
    *,
    correctness: bool = True,
    failure: dict | None = None,
) -> dict:
    spec = WORKLOADS[cell.workload]
    runtime_flags = profiling_runtime_flags(
        worker_environment(cell.platform, base={})
    )
    runtime_environment = {
        **_TEST_CAMPAIGN_IDENTITY["host"],
        "device_id": 0,
        "device_kind": "cpu" if cell.platform == "cpu" else "gpu",
        "runtime_flags": runtime_flags,
    }
    if cell.repeats == 1:
        times = [process_median]
    else:
        times = [
            process_median + 0.2 * (index / (cell.repeats - 1) - 0.5)
            for index in range(cell.repeats)
        ]
    host_shell = cell.execution_mode == "host_shell"
    replicated = {
        "gate": "not_requested",
        "passed": True,
        "replicates": 0,
    }
    if cell.correctness_replicates:
        replicated = {
            "error": -0.01,
            "evidence_ratios": [0.99] * cell.correctness_replicates,
            "mean_ratio": 0.99,
            "passed": correctness and failure is None,
            "replicates": cell.correctness_replicates,
            "standard_deviation": 0.1,
            "upper_allowance": 0.05,
        }
        validation_provenance = {
            "backend": cell.platform,
            "dispatch_mode": (
                "asynchronous" if cell.platform == "cpu" else "safe"
            ),
            "environment": deepcopy(runtime_environment),
            "source": deepcopy(_TEST_CAMPAIGN_IDENTITY["source"]),
            "versions": deepcopy(_TEST_CAMPAIGN_IDENTITY["packages"]),
        }
    work_metrics = {
        "minimum_ess": 64.0 + cell.block,
        "state_leaf_count": 1,
        "weight_regime": "calibrated",
    }
    if spec.algorithm in {"bootstrap", "guided"}:
        work_metrics["resampling_event_count"] = 3
    return {
        "algorithm": spec.algorithm,
        "backend": cell.platform if failure is None else "unknown",
        "block": cell.block,
        "correctness": {
            "passed": correctness and failure is None,
            "maximum_error": 0.01,
            "replicated": replicated,
            **(
                {"validation_provenance": validation_provenance}
                if cell.correctness_replicates
                else {}
            ),
        },
        "correctness_level": (
            spec.replicated_correctness_level
            if cell.correctness_replicates
            else "structural"
        ),
        "correctness_replicates": cell.correctness_replicates,
        "dispatch_mode": ("asynchronous" if cell.platform == "cpu" else "safe"),
        "environment": {
            **runtime_environment,
            "post_cell": {
                "power_status": "Now drawing from 'AC Power'",
                "thermal_status": (
                    "No thermal warning level has been recorded\n"
                    "No performance warning level has been recorded"
                ),
            },
            "post_timing": {
                "power_status": "Now drawing from 'AC Power'",
                "thermal_status": (
                    "No thermal warning level has been recorded\n"
                    "No performance warning level has been recorded"
                ),
            },
            "pre_timing": {
                "power_status": "Now drawing from 'AC Power'",
                "thermal_status": (
                    "No thermal warning level has been recorded\n"
                    "No performance warning level has been recorded"
                ),
            },
        },
        "execution_mode": cell.execution_mode,
        "failure": failure,
        "first_execution_s": None if failure else process_median + 0.5,
        "lifecycle": (
            {}
            if failure
            else {
                "backend_compile_s": None if host_shell else 0.2,
                "lowering_s": None if host_shell else 0.1,
                "unavailable_reason": (
                    "host_controlled" if host_shell else None
                ),
            }
        ),
        "memory": (
            {}
            if failure
            else {
                "device_stats": {"peak_bytes_in_use": 2_048},
                "executable_analysis": {
                    "peak_memory_in_bytes": 1_024,
                },
                "process_max_rss_bytes": 10_000 + cell.block,
            }
        ),
        "model": spec.model,
        "parameters": dict(cell.parameters),
        "platform_requested": cell.platform,
        "repeats": cell.repeats,
        "schema_version": SCHEMA_VERSION,
        "source": deepcopy(_TEST_CAMPAIGN_IDENTITY["source"]),
        "steady_summary": {} if failure else summarize(times),
        "steady_times_s": [] if failure else times,
        "versions": (
            {} if failure else deepcopy(_TEST_CAMPAIGN_IDENTITY["packages"])
        ),
        "work_metrics": {} if failure else work_metrics,
        "workload": cell.workload,
        "warmups": cell.warmups,
    }


def _write_record(output_dir: Path, _name: str, record: dict) -> None:
    cell = Cell(
        workload=record["workload"],
        platform=record["platform_requested"],
        block=record["block"],
        warmups=record["warmups"],
        repeats=record["repeats"],
        execution_mode=record["execution_mode"],
        parameters=record["parameters"],
        correctness_replicates=record["correctness_replicates"],
    )
    (output_dir / "raw" / raw_filename(cell)).write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )


def _baseline_cells(workload: str, platforms: tuple[str, ...]) -> list[Cell]:
    return [
        cell
        for cell in plan_cells("baseline", platforms=platforms, seed=20260719)
        if cell.workload == workload
    ]


def test_aggregate_uses_fresh_process_medians_and_matches_backends(
    tmp_path: Path,
) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu", "mps"))
    _write_manifest(tmp_path, "baseline", cells)
    medians = {
        "cpu": [1.0, 2.0, 3.0, 4.0, 100.0],
        "mps": [2.0, 4.0, 6.0, 8.0, 200.0],
    }
    for cell in cells:
        value = medians[cell.platform][cell.block]
        record = _record(cell, value)
        if cell.platform == "mps":
            record["work_metrics"]["minimum_ess"] += 0.25
        _write_record(
            tmp_path,
            f"{cell.platform}-{cell.block}.json",
            record,
        )

    report = build_report(tmp_path)
    by_platform = {
        aggregate["platform"]: aggregate
        for aggregate in report["aggregates"]
        if aggregate["workload"] == "bootstrap_lgssm"
    }
    cpu = by_platform["cpu"]
    assert cpu["timing_eligible"] is True
    assert cpu["steady"]["median_s"] == pytest.approx(3.0)
    assert cpu["steady"]["q1_s"] == pytest.approx(2.0)
    assert cpu["steady"]["q3_s"] == pytest.approx(4.0)
    assert cpu["steady"]["mad_s"] == pytest.approx(1.0)
    assert cpu["lifecycle"]["lowering_s"]["median"] == pytest.approx(0.1)
    assert cpu["memory"]["process_max_rss_bytes"]["median"] == 10_002
    assert cpu["work"]["minimum_ess"]["median"] == pytest.approx(66.0)
    assert cpu["work"]["weight_regime"]["value"] == "calibrated"
    assert cpu["parameters"] == cells[0].parameters
    assert cpu["execution_mode"] == "whole_program_jit"
    assert cpu["dispatch_modes"] == ["asynchronous"]
    assert cpu["correctness"]["levels"] == [
        "oracle_accuracy",
        "structural",
    ]
    assert cpu["correctness"]["passed_blocks"] == 5
    assert cpu["correctness"]["accuracy"][0]["replicates"] == 20
    assert cpu["correctness"]["accuracy"][0]["metrics"][
        "mean_ratio"
    ] == pytest.approx(0.99)
    evidence_summary = cpu["correctness"]["accuracy"][0]["metrics"][
        "evidence_ratios"
    ]
    assert evidence_summary["count"] == 20
    assert evidence_summary["mean"] == pytest.approx(0.99)

    assert len(report["comparisons"]) == 1
    comparison = report["comparisons"][0]
    assert comparison["cpu_median_s"] == pytest.approx(3.0)
    assert comparison["mps_median_s"] == pytest.approx(6.0)
    assert comparison["mps_over_cpu"] == pytest.approx(2.0)
    assert comparison["adaptive_work"] == {
        block: {"resampling_event_count": 3} for block in range(5)
    }
    assert {item["device_kind"] for item in report["environments"]} == {
        "cpu",
        "gpu",
    }
    assert report["sources"] == [_TEST_CAMPAIGN_IDENTITY["source"]]
    assert report["preflight"]["timing_worker_processes"] == len(
        plan_cells("baseline", platforms=("cpu", "mps"), seed=20260719)
    )
    assert {item["platform"] for item in report["timing_states"]} == {
        "cpu",
        "mps",
    }

    markdown = render_markdown(report, report_date=date(2026, 7, 19))
    assert "Configuration, correctness, and accuracy" in markdown
    assert '"num_particles":10000' in markdown
    assert "whole_program_jit" in markdown
    assert "asynchronous" in markdown
    assert "oracle_accuracy" in markdown
    assert "AC Power" in markdown
    assert "No thermal warning" in markdown
    assert "worker processes" in markdown
    assert "Executable peak, MiB" in markdown
    assert "Device peak, MiB" in markdown
    assert "passed (R=20" in markdown
    assert "mean_ratio=0.99" in markdown
    assert '"evidence_ratios"' not in markdown


def test_inferential_timing_requires_ac_and_no_thermal_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cells = plan_cells(
        "baseline",
        platforms=("cpu",),
        order_seed=20260719,
    )
    _write_manifest(tmp_path, "baseline", cells)
    for index, cell in enumerate(cells):
        record = _record(cell, 1.0)
        if cell.workload == "bootstrap_lgssm" and cell.block == 2:
            record["environment"]["post_cell"]["thermal_status"] = (
                "CPU_Speed_Limit = 70"
            )
        _write_record(tmp_path, f"{index}.json", record)

    report = build_report(tmp_path)
    aggregate = next(
        item
        for item in report["aggregates"]
        if item["workload"] == "bootstrap_lgssm" and item["platform"] == "cpu"
    )
    assert not aggregate["timing_eligible"]
    assert aggregate["timing_state_failures"] == [
        {
            "block": 2,
            "reason": "post_cell reports a thermal/performance warning",
        }
    ]
    assert report["timing_state_failures"]
    assert report["complete"]
    assert report["correct"]
    assert not report["performance_eligible"]
    assert report_main(["--input-dir", str(tmp_path)]) == 1
    capsys.readouterr()
    markdown = render_markdown(report, report_date=date(2026, 7, 19))
    assert "Campaign timing ineligible" in markdown


def test_incomplete_and_incorrect_cells_remain_visible(tmp_path: Path) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu",))
    _write_manifest(tmp_path, "baseline", cells)
    _write_record(tmp_path, "good.json", _record(cells[0], 1.0))
    _write_record(
        tmp_path,
        "worker-failure.json",
        _record(
            cells[1],
            2.0,
            failure={"kind": "timeout", "message": "expired"},
        ),
    )
    _write_record(
        tmp_path,
        "wrong.json",
        _record(cells[2], 3.0, correctness=False),
    )
    (tmp_path / "raw" / "broken.json").write_text("{not-json")

    report = build_report(tmp_path)
    assert report["complete"] is False
    assert report["correct"] is False
    assert (
        len([
            item
            for item in report["missing_cells"]
            if item["workload"] == "bootstrap_lgssm"
        ])
        == 2
    )
    assert len(report["invalid_records"]) == 1
    assert len(report["worker_failures"]) == 1
    assert len(report["correctness_failures"]) == 1
    aggregate = report["aggregates"][0]
    assert aggregate["timing_eligible"] is False
    assert aggregate["steady"] is None

    markdown = render_markdown(report, report_date=date(2026, 7, 19))
    assert "Incomplete campaign" in markdown
    assert "Correctness failures" in markdown
    assert "timeout" in markdown


def test_oracle_accuracy_failure_is_labeled_as_validation_evidence(
    tmp_path: Path,
) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu",))
    _write_manifest(tmp_path, "baseline", cells)
    for index, cell in enumerate(cells):
        _write_record(
            tmp_path,
            f"{index}.json",
            _record(cell, 1.0, correctness=cell.block != 0),
        )

    report = build_report(tmp_path)
    assert report["correctness_failures"][0]["correctness_level"] == (
        "oracle_accuracy"
    )
    markdown = render_markdown(report, report_date=date(2026, 7, 19))
    assert "`oracle_accuracy` validation failure" in markdown
    assert "not proofs of implementation correctness" in markdown


def test_cross_backend_comparison_requires_identical_parameters(
    tmp_path: Path,
) -> None:
    cpu = _baseline_cells("bootstrap_lgssm", ("cpu",))
    mps = [
        cell._replace(parameters={**cell.parameters, "num_particles": 20_000})
        for cell in _baseline_cells("bootstrap_lgssm", ("mps",))
    ]
    cells = [*cpu, *mps]
    _write_manifest(tmp_path, "baseline", cells)
    for index, cell in enumerate(cells):
        _write_record(tmp_path, f"{index}.json", _record(cell, 1.0))

    report = build_report(tmp_path)
    assert report["comparisons"] == []


def test_cross_backend_comparison_requires_matching_adaptive_work(
    tmp_path: Path,
) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu", "mps"))
    _write_manifest(tmp_path, "baseline", cells)
    for index, cell in enumerate(cells):
        record = _record(cell, 1.0)
        if cell.platform == "mps" and cell.block == 2:
            record["work_metrics"]["resampling_event_count"] = 4
        _write_record(tmp_path, f"{index}.json", record)

    report = build_report(tmp_path)
    assert report["comparisons"] == []
    assert report["comparison_exclusions"] == [
        {
            "parameters": cells[0].parameters,
            "reason": "discrete adaptive work differs between CPU and MPS",
            "workload": "bootstrap_lgssm",
        }
    ]


@pytest.mark.parametrize("workload", ["auxiliary_lgssm", "liu_west_unknown_ar"])
def test_cross_backend_comparison_requires_instrumented_lookahead_work(
    tmp_path: Path,
    workload: str,
) -> None:
    cells = _baseline_cells(workload, ("cpu", "mps"))
    _write_manifest(tmp_path, "baseline", cells)
    for index, cell in enumerate(cells):
        _write_record(tmp_path, f"{index}.json", _record(cell, 1.0))

    report = build_report(tmp_path)
    assert report["comparisons"] == []
    assert len(report["comparison_exclusions"]) == 1
    assert (
        "resampling_event_count" in report["comparison_exclusions"][0]["reason"]
    )


def test_representation_profile_reports_matched_arm_overhead(
    tmp_path: Path,
) -> None:
    cells = plan_cells(
        "representation",
        platforms=("cpu",),
        seed=20260719,
    )
    _write_manifest(tmp_path, "representation", cells)
    for index, cell in enumerate(cells):
        is_tree = cell.workload == "bootstrap_tracking_pytree"
        history_cost = 0.5 if cell.parameters["store_history"] else 0.0
        process_median = 1.0 + history_cost + (0.25 if is_tree else 0.0)
        record = _record(cell, process_median)
        if cell.workload == "liu_west_unknown_ar":
            record["work_metrics"]["resampling_event_count"] = 99
        _write_record(
            tmp_path,
            f"{index}.json",
            record,
        )

    report = build_report(tmp_path)
    comparisons = report["arm_comparisons"]
    assert len(comparisons) == 4
    no_history = next(
        comparison
        for comparison in comparisons
        if not comparison["parameters"]["store_history"]
        and comparison["parameters"]["covariance_regime"] == "correlated"
    )
    assert no_history["kind"] == "representation"
    assert no_history["ratio"] == pytest.approx(1.25)
    assert no_history["numerator_workload"] == "bootstrap_tracking_pytree"
    assert no_history["denominator_workload"] == "bootstrap_tracking_dense"

    history_comparisons = report["history_comparisons"]
    assert len(history_comparisons) == 5
    liu_west = next(
        comparison
        for comparison in history_comparisons
        if comparison["workload"] == "liu_west_unknown_ar"
    )
    assert liu_west["history_on_over_off"] == pytest.approx(1.5)
    assert liu_west["parameters"] == {
        "num_particles": 10_000,
        "parameter_dimension": 1,
        "resampling_threshold": 1.1,
        "shrinkage": 0.95,
        "timesteps": 100,
    }


def test_arm_comparison_requires_matching_adaptive_work(
    tmp_path: Path,
) -> None:
    cells = plan_cells(
        "representation",
        platforms=("cpu",),
        seed=20260719,
    )
    _write_manifest(tmp_path, "representation", cells)
    for index, cell in enumerate(cells):
        record = _record(cell, 1.0)
        if (
            cell.workload == "bootstrap_tracking_pytree"
            and cell.parameters["covariance_regime"] == "correlated"
            and not cell.parameters["store_history"]
            and cell.block == 2
        ):
            record["work_metrics"]["resampling_event_count"] = 4
        _write_record(tmp_path, f"{index}.json", record)

    report = build_report(tmp_path)
    assert len(report["arm_comparisons"]) == 3
    assert report["arm_comparison_exclusions"] == [
        {
            "denominator_workload": "bootstrap_tracking_dense",
            "kind": "representation",
            "numerator_workload": "bootstrap_tracking_pytree",
            "parameters": next(
                cell.parameters
                for cell in cells
                if cell.workload.startswith("bootstrap_tracking_")
                and cell.parameters["covariance_regime"] == "correlated"
                and not cell.parameters["store_history"]
            ),
            "platform": "cpu",
            "reason": "discrete adaptive work differs between comparison arms",
        }
    ]


@pytest.mark.parametrize("mutation", ["omit", "reorder"])
def test_manifest_plan_digest_rejects_cell_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu",))
    _write_manifest(tmp_path, "baseline", cells)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "omit":
        manifest["cells"] = manifest["cells"][:-1]
    else:
        manifest["cells"][0], manifest["cells"][-1] = (
            manifest["cells"][-1],
            manifest["cells"][0],
        )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(ValueError, match="plan_sha256"):
        build_report(tmp_path)


@pytest.mark.parametrize("mutation", ["omit", "reorder", "empty"])
def test_manifest_rejects_a_self_consistent_unregistered_plan(
    tmp_path: Path,
    mutation: str,
) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu",))
    _write_manifest(tmp_path, "baseline", cells)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "omit":
        manifest["cells"] = manifest["cells"][:-1]
    elif mutation == "reorder":
        manifest["cells"][0], manifest["cells"][-1] = (
            manifest["cells"][-1],
            manifest["cells"][0],
        )
    else:
        manifest["cells"] = []
    manifest["plan_sha256"] = hashlib.sha256(
        json.dumps(
            manifest["cells"],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(ValueError, match="registered plan"):
        build_report(tmp_path)


def test_manifest_rejects_a_tampered_preflight_estimate(tmp_path: Path) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu",))
    _write_manifest(tmp_path, "baseline", cells)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preflight"]["total_worker_processes"] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    with pytest.raises(ValueError, match="preflight"):
        build_report(tmp_path)


def test_noncanonical_and_unexpected_records_are_not_silently_used(
    tmp_path: Path,
) -> None:
    cells = [
        cell
        for cell in plan_cells("smoke", platforms=("cpu",), seed=20260719)
        if cell.workload == "bootstrap_lgssm"
    ]
    _write_manifest(tmp_path, "smoke", cells)
    expected = _record(cells[0], 0.5)
    _write_record(tmp_path, "first.json", expected)
    canonical_path = tmp_path / "raw" / raw_filename(cells[0])
    (tmp_path / "raw" / "duplicate.json").write_bytes(
        canonical_path.read_bytes()
    )
    unexpected_cell = cells[0]._replace(
        parameters={**cells[0].parameters, "num_particles": 999}
    )
    _write_record(
        tmp_path,
        "unexpected.json",
        _record(unexpected_cell, 0.5),
    )

    report = build_report(tmp_path)
    assert report["complete"] is False
    assert report["duplicate_records"] == []
    assert len(report["invalid_records"]) == 1
    assert "canonical raw filename" in report["invalid_records"][0]["error"]
    assert len(report["unexpected_records"]) == 1
    assert (
        len([
            item
            for item in report["missing_cells"]
            if item["workload"] == "bootstrap_lgssm"
        ])
        == 0
    )
    bootstrap = next(
        aggregate
        for aggregate in report["aggregates"]
        if aggregate["workload"] == "bootstrap_lgssm"
    )
    assert bootstrap["timing_eligible"] is True


def test_swapped_canonical_raw_files_are_rejected(tmp_path: Path) -> None:
    cells = plan_cells(
        "smoke",
        platforms=("cpu",),
        order_seed=20260719,
    )
    _write_manifest(tmp_path, "smoke", cells)
    first, second = cells[:2]
    first_record = _record(first, 0.5)
    second_record = _record(second, 0.5)
    (tmp_path / "raw" / raw_filename(first)).write_text(
        json.dumps(second_record)
    )
    (tmp_path / "raw" / raw_filename(second)).write_text(
        json.dumps(first_record)
    )

    report = build_report(tmp_path)

    assert len(report["invalid_records"]) == 2
    assert all(
        "canonical raw filename" in item["error"]
        for item in report["invalid_records"]
    )
    assert report["matched_cells"] == 0


def test_source_identity_mismatch_is_never_timing_eligible(
    tmp_path: Path,
) -> None:
    cells = [
        cell
        for cell in plan_cells("smoke", platforms=("cpu",), seed=20260719)
        if cell.workload == "bootstrap_lgssm"
    ]
    _write_manifest(tmp_path, "smoke", cells)
    record = _record(cells[0], 0.5)
    record["source"]["source_sha256"] = "0" * 64
    _write_record(tmp_path, "mismatch.json", record)

    report = build_report(tmp_path)
    assert report["reproducible"] is False
    assert len(report["identity_mismatches"]) == 1
    assert report["aggregates"][0]["timing_eligible"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("runtime", "runtime flags differ from the sanitized contract"),
        ("timing_kind", "timing device differs from the scheduled platform"),
        ("timing_id", "timing device differs from the scheduled platform"),
        (
            "validation_device",
            "validation device differs from the timing device",
        ),
    ],
)
def test_runtime_and_validation_provenance_mismatch_is_ineligible(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    cells = _baseline_cells("bootstrap_lgssm", ("cpu",))
    _write_manifest(tmp_path, "baseline", cells)
    record = _record(cells[0], 0.5)
    if mutation == "runtime":
        record["environment"]["runtime_flags"]["JAX_ENABLE_X64"] = "true"
    elif mutation == "timing_kind":
        record["environment"]["device_kind"] = "test-cpu"
    elif mutation == "timing_id":
        record["environment"]["device_id"] = 1
    else:
        provenance = record["correctness"]["validation_provenance"]
        provenance["environment"]["device_id"] = 1
    _write_record(tmp_path, "mismatch.json", record)

    report = build_report(tmp_path)
    assert report["reproducible"] is False
    assert report["identity_mismatches"][0]["error"] == message


def test_smoke_markdown_is_explicitly_noninferential(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cells = plan_cells("smoke", platforms=("cpu",), seed=20260719)
    _write_manifest(tmp_path, "smoke", cells)
    for index, cell in enumerate(cells):
        _write_record(tmp_path, f"{index}.json", _record(cell, 0.5))

    report = build_report(tmp_path)
    markdown = render_markdown(report, report_date=date(2026, 7, 19))
    assert report["performance_eligible"]
    assert report_main(["--input-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert "NON-INFERENTIAL SMOKE RUN" in markdown
    assert "must not be used for performance rankings" in markdown
    assert "2026-07-19" in markdown
