# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Campaign parsing contracts for the tempering-accuracy report."""

import hashlib

import pytest

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy import artifacts
from benchmarks.tempering_accuracy.report_data import load_campaign


def _identity():
    return {
        "source": {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "sha256": "a" * 64,
            "files": ["src/smcx/tempering.py"],
        },
        "lock": {"path": "uv.lock", "sha256": "b" * 64},
        "packages": {},
        "python": {},
        "host": {"os": "Darwin", "machine": "arm64"},
    }


def _payload(request, digest):
    return {
        "schema_version": 1,
        "request": artifacts.request_dict(
            artifacts.bind_request(request, digest)
        ),
        "failure": {"kind": "execution_failure", "message": "retained"},
        "timing": None,
        "runs": [],
    }


def test_load_campaign_accepts_only_a_canonical_contiguous_prefix(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifacts, "campaign_identity", lambda root: _identity()
    )
    manifest = artifacts.build_manifest(tmp_path)
    digest = artifacts.ensure_manifest(tmp_path, manifest)
    requests = artifacts.campaign_requests()
    for request in requests[:2]:
        artifacts.write_raw_result(
            tmp_path, request, digest, _payload(request, digest)
        )

    campaign = load_campaign(tmp_path)
    assert not campaign.complete
    assert campaign.not_run_after_stop == (2, 507)
    assert [entry.ordinal for entry in campaign.inventory] == [0, 1]
    encoded = canonical_json([
        entry._asdict() for entry in campaign.inventory
    ]).encode()
    assert campaign.raw_sha256 == hashlib.sha256(encoded).hexdigest()

    request = requests[3]
    artifacts.write_raw_result(
        tmp_path, request, digest, _payload(request, digest)
    )
    with pytest.raises(ValueError, match="contiguous prefix"):
        load_campaign(tmp_path)
