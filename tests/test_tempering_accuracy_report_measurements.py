# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Sanitized raw-measurement contracts for the tempering report."""

import json

import pytest

from benchmarks.tempering_accuracy import artifacts, report_measurements
from benchmarks.tempering_accuracy.plan import current_cells
from benchmarks.tempering_accuracy.report_data import CampaignData
from tests.test_tempering_accuracy_report_accuracy import _run
from tests.test_tempering_accuracy_report_timing import _timing

_CELL = current_cells()[0]


def _timing_record():
    timing = _timing(_CELL, 1)
    memory = timing["memory"]
    memory["device_stats"] = {"peak_bytes_in_use": 123, "private": 999}
    memory["executable_analysis"] = None
    return timing


def test_public_measurements_preserve_values_without_private_payloads():
    timing = report_measurements._timing(_timing_record(), _CELL.lane)
    request = artifacts.CampaignRequest("accuracy", _CELL, None)
    runs = report_measurements._accuracy_runs(request, [_run(0)])
    prefix = {
        "eligible": False,
        "first_execution_s": 4.0,
        "steady_times_s": [1.0, 2.0],
    }
    boundary = {"power_status": "AC", "thermal_status": "nominal"}
    expected_failure = {
        "kind": "execution_failure",
        "key_index": 2,
        "key_words": [1, 2],
        "key_indices": [2, 3],
        "failed_call": {"role": "steady", "index": 2},
        "timing_prefix": prefix,
        "boundaries": dict.fromkeys(
            ("pre_timing", "post_timing", "post_cell"), boundary
        ),
        "changed_domains": ["source", "host"],
        "worker_failure": {"kind": "worker_exit"},
    }
    failure = report_measurements._failure(
        expected_failure
        | {
            "message": "secret /private/path",
            "stdout_tail": "secret stdout",
            "stderr_tail": "secret stderr",
            "worker_failure": {
                "kind": "worker_exit",
                "stderr_tail": "secret nested stderr",
            },
        }
    )

    assert timing["steady_times_s"] == [1.0] * 7
    assert timing["memory"]["device_peak_bytes_in_use"] == 123
    assert runs[0]["key_index"] == 0
    assert runs[0]["temperatures"] == [0.5, 1.0]
    assert failure == expected_failure
    encoded = json.dumps({"timing": timing, "runs": runs, "failure": failure})
    assert "posterior_mean" not in encoded
    assert "posterior_covariance" not in encoded
    assert "private" not in encoded and "secret" not in encoded


def test_measurements_fail_closed_on_nested_schema():
    with pytest.raises(ValueError):
        report_measurements._schema_version(True)
    request = artifacts.CampaignRequest("accuracy", _CELL, None)
    run = _run(0)
    run["unexpected"] = "hidden"
    with pytest.raises(ValueError):
        report_measurements._accuracy_runs(request, [run])


def test_incomplete_prefix_is_not_imputed():
    campaign = CampaignData({}, "", "", False, (0, 507), None, (), ())
    assert report_measurements.build_measurements(campaign)["requests"] == []
