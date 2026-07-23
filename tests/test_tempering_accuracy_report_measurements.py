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
_BOUNDARY = {"power_status": "AC", "thermal_status": "nominal"}


def _post_timing_failure():
    return {
        "kind": "execution_failure",
        "exception_type": "RuntimeError",
        "message": "secret /private/path",
        "failed_stage": "post_timing_extraction",
        "timing_prefix": {
            "eligible": False,
            "first_execution_s": 4.0,
            "steady_times_s": [1.0] * 7,
        },
        "environment": {
            "pre_timing": _BOUNDARY,
            "post_timing": _BOUNDARY,
            "failure_boundary": _BOUNDARY,
        },
    }


def test_current_worker_timing_failure_is_retained_directly_and_nested():
    failure = _post_timing_failure()
    public = {
        name: value
        for name, value in failure.items()
        if name not in {"exception_type", "message"}
    }
    assert report_measurements._failure(failure) == public

    drift = {
        "kind": "source_identity_changed_after_launch",
        "changed_domains": ["source"],
        "expected_source_sha256": "a" * 64,
        "observed_source_sha256": "b" * 64,
        "worker_failure": failure,
    }
    assert report_measurements._failure(drift) == {
        "kind": drift["kind"],
        "changed_domains": ["source"],
        "worker_failure": public,
    }


def test_public_measurements_preserve_values_without_private_payloads():
    raw_timing = _timing(_CELL, 1)
    raw_timing["memory"]["process_max_rss_before_measurement_bytes"] = 900
    raw_timing["memory"]["device_stats"] = {"peak_bytes_in_use": 123}
    raw_timing["memory"]["executable_analysis"] = None
    timing = report_measurements._timing(raw_timing, _CELL.lane)
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
            "worker_failure": {
                "kind": "worker_exit",
                "stderr_tail": "secret nested stderr",
            },
        }
    )

    assert timing["steady_times_s"] == [1.0] * 7
    assert timing["memory"]["process_max_rss_before_measurement_bytes"] == 900
    assert timing["memory"]["device_peak_bytes_in_use"] == 123
    assert (runs[0]["key_index"], runs[0]["temperatures"]) == (0, [0.5, 1.0])
    assert failure == expected_failure
    encoded = json.dumps({"timing": timing, "runs": runs, "failure": failure})
    assert "posterior_" not in encoded and "secret" not in encoded


def test_measurements_fail_closed_on_nested_schema():
    with pytest.raises(ValueError):
        report_measurements._schema_version(True)
    campaign = CampaignData({}, "", "", False, (0, 507), None, (), ())
    assert report_measurements.build_measurements(campaign)["requests"] == []
