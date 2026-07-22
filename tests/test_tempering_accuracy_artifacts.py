# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifact contracts for the tempering-accuracy campaign."""

import hashlib
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
    encoded = artifacts.canonical_json([
        artifacts.request_dict(request) for request in requests
    ])
    assert hashlib.sha256(encoded.encode()).hexdigest() == _PLAN_SHA256


def test_manifest_hash_names_and_waste_free_exclusion(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "campaign_identity", lambda root: _IDENTITY)
    manifest = artifacts.build_manifest(tmp_path)
    exclusion = manifest["exclusions"][0]
    assert exclusion["status"] == "blocked_backend_correctness"
    assert exclusion["tracking_issue"] == 38
    assert tuple(exclusion["blocked_request_counts"].values()) == (2, 60, 12)
    assert (len(exclusion["smoke_cells"]), len(exclusion["cells"])) == (2, 12)
    assert manifest["algorithm_contract"] == {
        "proposal_covariance_source": "weighted_pre_resample_cloud",
        "proposal_scale": "2.38^2 / dimension",
        "target_ess": 0.5,
    }
    names = list(map(artifacts.raw_filename, artifacts.campaign_requests()))
    assert len(names) == len(set(names)) == 508


@pytest.mark.parametrize("missing", ("status", "commit"))
def test_campaign_identity_fails_closed_without_git_identity(
    monkeypatch, missing
):
    root = Path(__file__).resolve().parents[1]

    def command_value(command, **kwargs):
        kind = "status" if "status" in command else "commit"
        values = {"status": "", "commit": "c" * 40}
        return None if kind == missing else values[kind]

    monkeypatch.setattr(artifacts, "_command_value", command_value)
    with pytest.raises(RuntimeError, match="git identity"):
        artifacts.campaign_identity(root)


@pytest.mark.parametrize(("index", "field"), ((0, "schema"), (4, "block")))
def test_raw_result_identity_is_type_strict(tmp_path, index, field):
    request = artifacts.campaign_requests()[index]
    payload = _payload(request)
    if field == "schema":
        payload["schema_version"] = True
    else:
        payload["request"]["block"] = False
    with pytest.raises(ValueError, match="manifest request"):
        artifacts.write_raw_result(tmp_path, request, _DIGEST, payload)


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
