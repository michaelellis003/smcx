# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifact contracts for the tempering-accuracy campaign."""

import hashlib
import json
from collections import Counter

import benchmarks.tempering_accuracy.artifacts as artifacts
import pytest

from benchmarks.tempering_accuracy.plan import (
    cell_id,
    current_cells,
    matched_cells,
)

_DIGEST = "a" * 64


def _identity():
    return {
        "source": {},
        "lock": {},
        "packages": {},
        "python": {},
        "host": {},
    }


def _failure_payload(request, manifest_sha256=_DIGEST):
    bound = artifacts.bind_request(request, manifest_sha256)
    return {
        "schema_version": 1,
        "request": artifacts.request_dict(bound),
        "failure": {"kind": "retained_test_failure"},
        "timing": None,
        "runs": [],
    }


def test_campaign_requests_freeze_exact_508_order():
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
    assert [request.phase for request in requests].index("accuracy") == 424
    boundaries = (0, 4, 364, 424, 507)
    assert [cell_id(requests[index].cell) for index in boundaries] == [
        "current_systematic-g0-d4-n1000-cpu_f64-systematic-s5",
        "current_systematic-g0-d32-n1000-mps_f32-systematic-s20",
        "matched_multinomial-g0-d32-n1000-cpu_f64-multinomial-s20",
        "current_systematic-g0-d4-n1000-cpu_f64-systematic-s5",
        "matched_multinomial-g1-d128-n10000-mps_f32-multinomial-s20",
    ]
    assert artifacts.WorkerRequest._fields == (
        "manifest_sha256",
        "phase",
        "cell",
        "block",
    )
    encoded = artifacts.canonical_json([
        artifacts.request_dict(request) for request in requests
    ])
    assert hashlib.sha256(encoded.encode()).hexdigest() == (
        "ce573478ea79bd5b8cca7bf2d73c164e1a55ea784342996627c9fe01f55e1ca9"
    )


def test_timing_requests_use_balanced_five_block_rotations():
    timing = [
        request
        for request in artifacts.campaign_requests()
        if request.phase == "timing"
    ]
    for arm, cells in (
        ("current_systematic", current_cells()),
        ("matched_multinomial", matched_cells()),
    ):
        blocks = [
            [
                request.cell
                for request in timing
                if request.block == block and request.cell.arm == arm
            ]
            for block in range(5)
        ]
        assert all(set(block) == set(cells) for block in blocks)
        assert all(
            len({block[index] for block in blocks}) == 5
            for index in range(len(cells))
        )


def test_manifest_records_identity_and_blocked_waste_free_lane(
    monkeypatch,
    tmp_path,
):
    identity = _identity()
    monkeypatch.setattr(artifacts, "campaign_identity", lambda root: identity)

    manifest = artifacts.build_manifest(tmp_path)

    assert manifest["campaign_identity"] == identity
    assert manifest["order_seed"] == 20_260_719
    assert manifest["plan_sha256"] == (
        "ce573478ea79bd5b8cca7bf2d73c164e1a55ea784342996627c9fe01f55e1ca9"
    )
    assert len(manifest["requests"]) == 508
    exclusion = manifest["exclusions"][0]
    assert exclusion["status"] == "blocked_backend_correctness"
    assert exclusion["tracking_issue"] == 38
    assert exclusion["blocked_request_counts"] == {
        "smoke": 2,
        "timing": 60,
        "accuracy": 12,
    }
    assert len(exclusion["smoke_cells"]) == 2
    assert len(exclusion["cells"]) == 12


def test_canonical_json_manifest_hash_and_raw_names_are_frozen():
    assert artifacts.canonical_json({"z": 3, "a": [1, 2]}) == (
        '{"a":[1,2],"z":3}'
    )
    with pytest.raises(ValueError):
        artifacts.canonical_json({"bad": float("nan")})

    manifest = {"z": 3, "a": [1, 2]}
    expected = hashlib.sha256(b'{"a":[1,2],"z":3}\n').hexdigest()
    assert artifacts.manifest_sha256(manifest) == expected

    requests = artifacts.campaign_requests()
    names = [artifacts.raw_filename(request) for request in requests]
    assert len(names) == len(set(names)) == 508
    assert names[0] == (
        "smoke-current_systematic-g0-d4-n1000-cpu_f64-systematic-s5-"
        "2cfd8e189f75362c.json"
    )
    assert names[-1] == (
        "accuracy-matched_multinomial-g1-d128-n10000-mps_f32-"
        "multinomial-s20-823bf41c65d47b53.json"
    )


def test_campaign_identity_covers_source_lock_packages_python_and_host(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "src/smcx/module.py"
    benchmark = tmp_path / "benchmarks/tempering_accuracy/plan.py"
    lock_source = tmp_path / "benchmarks/profiling/locking.py"
    source.parent.mkdir(parents=True)
    benchmark.parent.mkdir(parents=True)
    lock_source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    benchmark.write_text("plan = 1\n")
    lock_source.write_text("lock = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    lock = tmp_path / "uv.lock"
    lock.write_text("locked\n")
    monkeypatch.setattr(artifacts, "_host_identity", lambda: {"host": True})
    monkeypatch.setattr(artifacts, "_package_versions", lambda: {"smcx": "1"})
    monkeypatch.setattr(
        artifacts,
        "_command_value",
        lambda command, **kwargs: "" if "status" in command else "c" * 40,
    )

    first = artifacts.campaign_identity(tmp_path)
    source.write_text("value = 2\n")
    second = artifacts.campaign_identity(tmp_path)

    assert set(first) == {"source", "lock", "packages", "python", "host"}
    assert first["source"]["git_commit"] == "c" * 40
    assert first["source"]["git_dirty"] is False
    assert "benchmarks/profiling/locking.py" in first["source"]["files"]
    assert first["lock"]["sha256"] == hashlib.sha256(b"locked\n").hexdigest()
    assert first["source"]["sha256"] != second["source"]["sha256"]


def test_manifest_and_raw_results_are_exclusive_and_resumable(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        artifacts, "campaign_identity", lambda root: _identity()
    )
    manifest = artifacts.build_manifest(tmp_path)
    digest = artifacts.ensure_manifest(tmp_path / "campaign", manifest)
    assert digest == artifacts.manifest_sha256(manifest)
    assert artifacts.ensure_manifest(tmp_path / "campaign", manifest) == digest
    with pytest.raises(ValueError, match="different manifest"):
        artifacts.ensure_manifest(
            tmp_path / "campaign", manifest | {"campaign": "other"}
        )

    request = artifacts.campaign_requests()[0]
    assert (
        artifacts.load_raw_result(tmp_path / "campaign", request, digest)
        is None
    )
    payload = _failure_payload(request, digest)
    path = artifacts.write_raw_result(
        tmp_path / "campaign", request, digest, payload
    )
    assert (
        artifacts.load_raw_result(tmp_path / "campaign", request, digest)
        == payload
    )
    with pytest.raises(FileExistsError):
        artifacts.write_raw_result(
            tmp_path / "campaign", request, digest, payload
        )
    expected = (artifacts.canonical_json(payload) + "\n").encode()
    assert path.read_bytes() == expected


def test_resume_rejects_noncanonical_or_foreign_raw_results(tmp_path):
    request = artifacts.campaign_requests()[0]
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / artifacts.raw_filename(request)
    payload = _failure_payload(request)
    path.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError, match="invalid raw result"):
        artifacts.load_raw_result(tmp_path, request, _DIGEST)

    foreign = artifacts.canonical_json(_failure_payload(request, "b" * 64))
    path.write_text(foreign + "\n")
    with pytest.raises(ValueError, match="invalid raw result"):
        artifacts.load_raw_result(tmp_path, request, _DIGEST)
