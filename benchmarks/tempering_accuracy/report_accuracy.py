# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Strict accuracy and work aggregation for campaign reports."""

import math
from collections.abc import Mapping
from typing import Any, NamedTuple, cast

import jax.random as jr
import numpy as np

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.analysis import (
    AccuracyAnalysis,
    ReplicateEstimate,
    analyze_accuracy,
)
from benchmarks.tempering_accuracy.core import (
    GaussianTarget,
    accuracy_keys,
    build_target,
)
from benchmarks.tempering_accuracy.plan import (
    CampaignCell,
    current_cells,
    matched_cells,
    work_count,
)
from benchmarks.tempering_accuracy.report_data import (
    CampaignData,
    campaign_requests,
)
from benchmarks.tempering_accuracy.report_timing import (
    Summary,
    TimingReport,
    _summary,
    analyze_timing,
)

_RUN_FIELDS = {
    "key_index",
    "key_words",
    "posterior_mean",
    "posterior_covariance",
    "log_evidence",
    "temperatures",
    "reweighting_ess",
    "acceptance_rates",
    "work",
    "structural",
}
_CHECK_FIELDS = {
    "backend_ok",
    "shapes_ok",
    "dtypes_ok",
    "finite_ok",
    "trace_shapes_ok",
    "normalized_log_weights_ok",
    "equal_log_weights_ok",
    "temperature_trace_ok",
    "ess_bounds_ok",
    "acceptance_bounds_ok",
    "passed",
}
_STRUCTURAL_FIELDS = _CHECK_FIELDS | {
    "final_log_weight_lse_error",
    "uniform_log_weight_error",
}
_WORK_NAMES = (
    "stages",
    "mean_acceptance",
    "min_ess_fraction",
    "total_pairs",
    "resampling_events",
    "ancestor_draws",
)


class CellReport(NamedTuple):
    """Timing, accuracy, and work evidence for one standard cell."""

    cell: CampaignCell
    timing: TimingReport
    status: str
    analysis: AccuracyAnalysis | None
    work: dict[str, Summary] | None


class CampaignReport(NamedTuple):
    """All standard-cell analyses with their immutable source campaign."""

    campaign: CampaignData
    cells: tuple[CellReport, ...]


def _schema_error(message: str) -> ValueError:
    return ValueError(f"accuracy run {message}")


def _structural(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _STRUCTURAL_FIELDS:
        raise _schema_error("has an invalid structural schema")
    structural = cast(dict[str, Any], value)
    if any(type(structural[name]) is not bool for name in _CHECK_FIELDS):
        raise _schema_error("has non-boolean structural checks")
    for name in _STRUCTURAL_FIELDS - _CHECK_FIELDS:
        item = structural[name]
        if item is not None and type(item) not in {int, float}:
            raise _schema_error("has an invalid structural error")
    return structural["passed"]


def _parse_run(
    cell: CampaignCell,
    value: Mapping[str, Any],
    index: int,
    target: GaussianTarget,
) -> tuple[ReplicateEstimate, tuple[float, ...]]:
    if set(value) != _RUN_FIELDS or canonical_json(
        value.get("key_index")
    ) != canonical_json(index):
        raise _schema_error("does not match its committed index")
    words = [
        int(word) for word in np.asarray(jr.key_data(accuracy_keys()[index]))
    ]
    if canonical_json(value["key_words"]) != canonical_json(words):
        raise _schema_error("does not match its committed key")
    temperatures = np.asarray(value["temperatures"], dtype=np.float64)
    ess = np.asarray(value["reweighting_ess"], dtype=np.float64)
    acceptance = np.asarray(value["acceptance_rates"], dtype=np.float64)
    if not temperatures.size or not (
        temperatures.shape == ess.shape == acceptance.shape
    ):
        raise _schema_error("has invalid trace shapes")
    expected = work_count(cell, len(temperatures))._asdict()
    if canonical_json(value["work"]) != canonical_json(expected):
        raise _schema_error("does not match registered work")
    passed = _structural(value["structural"])
    mean = np.asarray(value["posterior_mean"], dtype=np.float64)
    covariance = np.asarray(value["posterior_covariance"], dtype=np.float64)
    if mean.shape != (cell.dimension,) or covariance.shape != (
        cell.dimension,
        cell.dimension,
    ):
        raise _schema_error("has invalid summary shapes")
    log_evidence = value["log_evidence"]
    if log_evidence is not None and type(log_evidence) not in {int, float}:
        raise _schema_error("has invalid evidence")
    log_evidence = math.nan if log_evidence is None else log_evidence
    with np.errstate(over="ignore", invalid="ignore"):
        evidence_ratio = float(np.exp(log_evidence - target.log_evidence))
    estimate = ReplicateEstimate(
        mean,
        covariance,
        evidence_ratio,
        passed,
        len(temperatures),
        expected["total_pairs"],
    )
    work = (
        float(len(temperatures)),
        float(np.mean(acceptance)),
        float(np.min(ess) / cell.reference_particles),
        float(expected["total_pairs"]),
        float(expected["resampling_events"]),
        float(expected["ancestor_draws"]),
    )
    return estimate, work


def analyze_accuracy_cell(
    cell: CampaignCell,
    result: Mapping[str, Any] | None,
    timing: TimingReport,
) -> CellReport:
    """Validate and analyze one cell without dropping failed replicates."""
    if result is None:
        return CellReport(cell, timing, "not_run_after_stop", None, None)
    failure, runs = result.get("failure"), result.get("runs")
    if not isinstance(runs, list):
        raise _schema_error("has no run list")
    if (failure and failure.get("kind") != "structural_failure") or len(
        runs
    ) != 32:
        return CellReport(cell, timing, "failed_execution", None, None)
    runs = cast(list[Mapping[str, Any]], runs)
    target = build_target(
        cell.geometry,
        cell.dimension,
        np.float64 if cell.lane == "cpu_f64" else np.float32,
    )
    parsed = tuple(
        _parse_run(cell, run, index, target) for index, run in enumerate(runs)
    )
    seconds = (
        timing.steady.values
        if timing.status == "eligible" and timing.steady is not None
        else None
    )
    analysis = analyze_accuracy(
        [item[0] for item in parsed],
        target,
        cell.lane,
        steady_block_median_seconds=seconds,
    )
    work = {
        name: _summary([item[1][position] for item in parsed])
        for position, name in enumerate(_WORK_NAMES)
    }
    status = "failed_structural" if failure else analysis.status
    return CellReport(cell, timing, status, analysis, work)


def aggregate_campaign(campaign: CampaignData) -> CampaignReport:
    """Build exactly 84 standard-cell reports from an immutable prefix."""
    requests = campaign_requests()
    if len(campaign.results) > len(requests):
        raise ValueError("campaign contains too many results")
    cells = (*current_cells(), *matched_cells())
    timing: dict[CampaignCell, list[tuple[int, Mapping[str, Any]]]] = {
        cell: [] for cell in cells
    }
    accuracy: dict[CampaignCell, Mapping[str, Any]] = {}
    for request, result in zip(requests, campaign.results, strict=False):
        if request.phase == "timing":
            assert request.block is not None
            timing[request.cell].append((request.block, result))
        elif request.phase == "accuracy":
            accuracy[request.cell] = result
    identity = campaign.manifest["campaign_identity"]
    reports = []
    for cell in cells:
        timing_results = tuple(result for _, result in sorted(timing[cell]))
        timing_report = analyze_timing(cell, timing_results, identity)
        reports.append(
            analyze_accuracy_cell(cell, accuracy.get(cell), timing_report)
        )
    return CampaignReport(campaign, tuple(reports))
