# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Retry-attempt loading contracts for the tempering campaign report."""

import hashlib
import json

import pytest

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import bind_request, request_dict
from benchmarks.tempering_accuracy.report_attempts import (
    AttemptEvidence,
    load_attempts,
)
from benchmarks.tempering_accuracy.report_data import campaign_requests

_DIGEST = "d" * 64


def _failure(kind):
    if kind == "metal_prelaunch_ineligible":
        return {
            "kind": kind,
            "prelaunch": {
                "power_status": "Now drawing from 'AC Power'",
                "thermal_status": "No thermal warning level",
            },
        }
    if kind == "campaign_identity_changed_before_launch":
        return {
            "kind": kind,
            "expected_source_sha256": "a" * 64,
            "observed_source_sha256": "b" * 64,
        }
    return {
        "kind": kind,
        "exception_type": "OSError",
        "message": "secret at /private/tmp/campaign",
    }


def _record(index, retry, kind="launch_error"):
    request = bind_request(campaign_requests()[index], _DIGEST)
    return {
        "schema_version": 1,
        "request_index": index,
        "retry_index": retry,
        "request": request_dict(request),
        "failure": _failure(kind),
    }


def _write(attempts, index, retry, record=None, name=None):
    value = _record(index, retry) if record is None else record
    encoded = (canonical_json(value) + "\n").encode()
    path = attempts / (name or f"{index:03d}-{retry:03d}.json")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def test_missing_directory_is_an_explicit_empty_inventory(tmp_path):
    inventory = load_attempts(tmp_path, _DIGEST)

    assert inventory.entries == ()
    assert inventory.sha256 == hashlib.sha256(b"[]").hexdigest()


def test_attempts_are_bound_hashed_ordered_and_sanitized(tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    first = _write(attempts, 3, 0)
    second = _write(attempts, 3, 1, _record(3, 1, "metal_prelaunch_ineligible"))
    third = _write(
        attempts,
        10,
        0,
        _record(10, 0, "campaign_identity_changed_before_launch"),
    )

    inventory = load_attempts(tmp_path, _DIGEST)

    assert inventory.entries == (
        AttemptEvidence(3, 0, first, "launch_error"),
        AttemptEvidence(3, 1, second, "metal_prelaunch_ineligible"),
        AttemptEvidence(
            10, 0, third, "campaign_identity_changed_before_launch"
        ),
    )
    sanitized = [entry._asdict() for entry in inventory.entries]
    expected = hashlib.sha256(canonical_json(sanitized).encode()).hexdigest()
    assert inventory.sha256 == expected
    assert "/private/" not in canonical_json(sanitized)
    assert "power_status" not in canonical_json(sanitized)


@pytest.mark.parametrize(
    "case",
    (
        "noncanonical",
        "filename",
        "schema",
        "index_type",
        "request",
        "failure_kind",
        "failure_schema",
        "metal_schema",
        "extra_field",
    ),
)
def test_attempt_schema_and_identity_are_exact(tmp_path, case):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    record = _record(3, 0)
    name = None
    if case == "noncanonical":
        path = attempts / "003-000.json"
        path.write_text(json.dumps(record, indent=2))
    else:
        if case == "filename":
            name = "3-000.json"
        elif case == "schema":
            record["schema_version"] = True
        elif case == "index_type":
            record["retry_index"] = False
        elif case == "request":
            record["request"] = _record(4, 0)["request"]
        elif case == "failure_kind":
            record["failure"] = _failure("timeout")
        elif case == "failure_schema":
            record["failure"]["unexpected"] = None
        elif case == "metal_schema":
            record["failure"] = {
                "kind": "metal_prelaunch_ineligible",
                "exception_type": "OSError",
                "message": "wrong shape",
            }
        else:
            record["unexpected"] = None
        _write(attempts, 3, 0, record, name)

    with pytest.raises(ValueError, match="attempt"):
        load_attempts(tmp_path, _DIGEST)


def test_retry_indices_are_contiguous_and_entries_are_regular_files(tmp_path):
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    _write(attempts, 3, 1)
    with pytest.raises(ValueError, match="contiguous"):
        load_attempts(tmp_path, _DIGEST)

    (attempts / "003-001.json").unlink()
    (attempts / "notes").mkdir()
    with pytest.raises(ValueError, match="unexpected"):
        load_attempts(tmp_path, _DIGEST)


def test_broken_attempt_directory_symlink_is_not_treated_as_missing(tmp_path):
    (tmp_path / "attempts").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="unexpected"):
        load_attempts(tmp_path, _DIGEST)
