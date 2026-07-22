# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifact contracts for the tempering-accuracy campaign."""

import hashlib
from collections import Counter
from pathlib import Path

import pytest

import benchmarks.tempering_accuracy.artifacts as artifacts

_DIGEST = "a" * 64
_PLAN_SHA256 = (
    "ce573478ea79bd5b8cca7bf2d73c164e1a55ea784342996627c9fe01f55e1ca9"
)
_IDENTITY = {
    name: {} for name in ("source", "lock", "packages", "python", "host")
}


def _payload(request, digest=_DIGEST):
    bound = artifacts.bind_request(request, digest)
    return {
        "schema_version": 1,
        "request": artifacts.request_dict(bound),
        "failure": {"kind": "retained_test_failure"},
        "timing": None,
        "runs": [],
    }


def test_requests_freeze_exact_order_and_balanced_blocks():
    requests = artifacts.campaign_requests()
    assert len(requests) == len(set(requests)) == 508
    assert Counter(request.phase for request in requests) == {
        "smoke": 4,
        "timing": 420,
        "accuracy": 84,
    }
    assert Counter(request.cell.arm for request in requests) == {
        "current_systematic": 436,
        "matched_multinomial": 72,
    }
    assert requests[424].phase == "accuracy"
    encoded = artifacts.canonical_json([
        artifacts.request_dict(request) for request in requests
    ])
    assert hashlib.sha256(encoded.encode()).hexdigest() == _PLAN_SHA256
    timing = [request for request in requests if request.phase == "timing"]
    for arm, size in (("current_systematic", 72), ("matched_multinomial", 12)):
        observed = Counter(
            request.block for request in timing if request.cell.arm == arm
        )
        assert observed == {block: size for block in range(5)}


def test_manifest_hash_names_and_waste_free_exclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "campaign_identity", lambda root: _IDENTITY)
    manifest = artifacts.build_manifest(tmp_path)
    assert manifest["plan_sha256"] == _PLAN_SHA256
    assert len(manifest["requests"]) == 508
    exclusion = manifest["exclusions"][0]
    assert exclusion["status"] == "blocked_backend_correctness"
    assert exclusion["tracking_issue"] == 38
    assert tuple(exclusion["blocked_request_counts"].values()) == (2, 60, 12)
    assert (len(exclusion["smoke_cells"]), len(exclusion["cells"])) == (2, 12)
    expected = hashlib.sha256(
        (artifacts.canonical_json(manifest) + "\n").encode()
    ).hexdigest()
    assert artifacts.manifest_sha256(manifest) == expected
    names = list(map(artifacts.raw_filename, artifacts.campaign_requests()))
    assert len(names) == len(set(names)) == 508


def test_campaign_identity_covers_source_lock_python_packages_and_host():
    root = Path(__file__).resolve().parents[1]
    identity = artifacts.campaign_identity(root)
    assert "benchmarks/profiling/locking.py" in identity["source"]["files"]
    assert {"ml-dtypes", "scipy"} <= identity["packages"].keys()
    assert "tfp-nightly" not in identity["packages"]
    lock = (root / "uv.lock").read_bytes()
    assert identity["lock"]["sha256"] == hashlib.sha256(lock).hexdigest()


def test_manifest_and_raw_results_are_exclusive_and_resumable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(artifacts, "campaign_identity", lambda root: _IDENTITY)
    manifest = artifacts.build_manifest(tmp_path)
    digest = artifacts.ensure_manifest(tmp_path, manifest)
    assert artifacts.ensure_manifest(tmp_path, manifest) == digest
    with pytest.raises(ValueError, match="different manifest"):
        artifacts.ensure_manifest(tmp_path, manifest | {"campaign": "other"})
    request = artifacts.campaign_requests()[0]
    assert artifacts.load_raw_result(tmp_path, request, digest) is None
    payload = _payload(request, digest)
    path = artifacts.write_raw_result(tmp_path, request, digest, payload)
    assert artifacts.load_raw_result(tmp_path, request, digest) == payload
    with pytest.raises(FileExistsError):
        artifacts.write_raw_result(tmp_path, request, digest, payload)
    foreign = artifacts.canonical_json(_payload(request, "b" * 64))
    path.write_text(foreign + "\n")
    with pytest.raises(ValueError, match="invalid raw result"):
        artifacts.load_raw_result(tmp_path, request, digest)
