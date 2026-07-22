# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Accuracy aggregation contracts for the tempering-accuracy report."""

import jax.random as jr
import numpy as np
import pytest

from benchmarks.tempering_accuracy.core import accuracy_keys, build_target
from benchmarks.tempering_accuracy.plan import current_cells, work_count
from benchmarks.tempering_accuracy.report_accuracy import (
    aggregate_campaign,
    analyze_accuracy_cell,
)
from benchmarks.tempering_accuracy.report_data import CampaignData
from benchmarks.tempering_accuracy.report_timing import Summary, TimingReport

_CELL = current_cells()[0]
_STRUCTURAL = {
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


def _structural(passed=True):
    result: dict[str, object] = dict.fromkeys(_STRUCTURAL, True)
    result.update(
        passed=passed,
        final_log_weight_lse_error=0.0,
        uniform_log_weight_error=0.0,
    )
    return result


def _run(index, *, passed=True):
    target = build_target(_CELL.geometry, _CELL.dimension, np.float64)
    key = accuracy_keys()[index]
    stages = 2
    return {
        "key_index": index,
        "key_words": [int(word) for word in np.asarray(jr.key_data(key))],
        "posterior_mean": target.posterior_mean.tolist(),
        "posterior_covariance": target.posterior_covariance.tolist(),
        "log_evidence": target.log_evidence,
        "temperatures": [0.5, 1.0],
        "reweighting_ess": [750.0, 750.0],
        "acceptance_rates": [0.4, 0.4],
        "work": work_count(_CELL, stages)._asdict(),
        "structural": _structural(passed),
    }


def _timing(status="eligible"):
    if status != "eligible":
        return TimingReport(status, None, None, None, None)
    summary = Summary((0.1, 0.2, 0.3, 0.4, 0.5), 0.3, 0.2, 0.4, 0.2, 0.1, 0.5)
    return TimingReport(status, summary, summary, summary, None)


def _accuracy(*, failure=None):
    return {"failure": failure, "runs": [_run(index) for index in range(32)]}


def test_accuracy_uses_exact_keys_registered_work_and_eligible_timing():
    report = analyze_accuracy_cell(_CELL, _accuracy(), _timing())

    assert report.status == "eligible"
    assert report.analysis is not None and report.work is not None
    assert report.analysis.mean_loss.median_steady_seconds == pytest.approx(0.3)
    assert report.work["stages"].median == 2
    assert report.work["total_pairs"].median == work_count(_CELL, 2).total_pairs

    untimed = analyze_accuracy_cell(
        _CELL, _accuracy(), _timing("ineligible_runtime")
    )
    assert untimed.analysis is not None
    assert untimed.analysis.mean_loss.median_steady_seconds is None


@pytest.mark.parametrize("case", ("key", "structural", "work"))
def test_accuracy_validates_the_full_run_schema(case):
    result = _accuracy()
    if case == "key":
        result["runs"][0]["key_index"] = False
    elif case == "structural":
        del result["runs"][0]["structural"]["backend_ok"]
    else:
        result["runs"][0]["work"]["total_pairs"] += 1
    with pytest.raises(ValueError, match="accuracy run"):
        analyze_accuracy_cell(_CELL, result, _timing())


def test_accuracy_failure_precedence_retains_execution_and_structural_status():
    execution = _accuracy(failure={"kind": "execution_failure"})
    execution["runs"] = execution["runs"][:3]
    assert analyze_accuracy_cell(_CELL, execution, _timing()).status == (
        "failed_execution"
    )

    structural = _accuracy(failure={"kind": "structural_failure"})
    structural["runs"][0]["structural"] = _structural(False)
    structural["runs"][0]["posterior_mean"][0] = None
    assert analyze_accuracy_cell(_CELL, structural, _timing()).status == (
        "failed_structural"
    )


def test_nonfinite_accuracy_follows_structural_precedence():
    result = _accuracy()
    result["runs"][0]["posterior_mean"][0] = None
    assert analyze_accuracy_cell(_CELL, result, _timing()).status == (
        "failed_nonfinite"
    )


def test_campaign_has_exactly_84_cells_and_does_not_impute_missing_tail():
    campaign = CampaignData(
        {"campaign_identity": {}}, "", "", False, (0, 507), None, (), ()
    )
    report = aggregate_campaign(campaign)
    assert len(report.cells) == 84
    assert all(cell.status == "not_run_after_stop" for cell in report.cells)
    assert all(cell.analysis is None for cell in report.cells)
    assert report.campaign.results == ()
