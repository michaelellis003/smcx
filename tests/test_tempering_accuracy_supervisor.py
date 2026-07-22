# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Host supervisor contracts for issue #30."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import benchmarks.tempering_accuracy.supervisor as supervisor
from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import (
    CampaignRequest,
    bind_request,
    campaign_requests,
    request_dict,
)
from benchmarks.tempering_accuracy.transport import WorkerAttempt

_DIGEST = "a" * 64
_ROOT = Path(__file__).resolve().parents[1]
_IDENTITY: dict[str, Any] = {
    "source": {
        "git_commit": "b" * 40,
        "git_dirty": False,
        "sha256": "c" * 64,
        "files": ["src/smcx/tempering.py"],
    },
    "lock": {"path": "uv.lock", "sha256": "d" * 64},
    "packages": {
        name: "1.0"
        for name in (
            "jax",
            "jax-mps",
            "jaxlib",
            "ml-dtypes",
            "numpy",
            "scipy",
            "smcx",
        )
    },
    "python": {
        "implementation": "CPython",
        "version": "3.13",
        "executable": "/python",
    },
    "host": {
        "os": "Darwin",
        "os_release": "25.2.0",
        "machine": "arm64",
        "macos": "26.2",
        "macos_build": "25C56",
        "cpu_model": "Apple M3 Pro",
        "hardware_model": "Mac15,7",
        "physical_memory_bytes": 38_654_705_664,
    },
}
_ELIGIBLE_BOUNDARY = {
    "power_status": "Now drawing from 'AC Power'",
    "thermal_status": (
        "No thermal warning level; No performance warning level"
    ),
}


def _payload(request, *, failure=None):
    return {
        "schema_version": 1,
        "request": request_dict(bind_request(request, _DIGEST)),
        "failure": failure,
        "timing": None,
        "runs": [],
    }


def _timing_payload(request):
    payload = _payload(request)
    payload["timing"] = {
        "environment": {
            name: deepcopy(_ELIGIBLE_BOUNDARY)
            for name in ("pre_timing", "post_timing", "post_cell")
        }
    }
    return payload


def _configure(monkeypatch, requests):
    monkeypatch.setattr(supervisor, "_ROOT", _ROOT)
    monkeypatch.setattr(supervisor, "_virtualization_status", lambda: "0")
    monkeypatch.setattr(
        supervisor, "campaign_identity", lambda root: deepcopy(_IDENTITY)
    )
    monkeypatch.setattr(
        supervisor,
        "build_manifest",
        lambda root: {"campaign_identity": deepcopy(_IDENTITY)},
    )
    monkeypatch.setattr(supervisor, "ensure_manifest", lambda *args: _DIGEST)
    monkeypatch.setattr(
        supervisor, "campaign_requests", lambda: tuple(requests)
    )
    monkeypatch.setattr(
        supervisor, "_prelaunch_snapshot", lambda: {}, raising=False
    )


@pytest.mark.parametrize("timeout", (0, -1, math.inf, math.nan))
def test_timeout_must_be_positive_and_finite(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        supervisor.run_campaign(tmp_path, timeout_s=timeout)


def test_output_directory_cannot_overlap_attested_source():
    with pytest.raises(supervisor.CampaignError, match="output_dir"):
        supervisor.run_campaign(_ROOT)


def test_cli_surface_and_nonmutating_dry_run(monkeypatch, tmp_path):
    _configure(monkeypatch, ())
    monkeypatch.setattr(
        supervisor,
        "ensure_manifest",
        lambda *args: pytest.fail("dry-run persisted a manifest"),
    )
    assert supervisor.run_campaign(tmp_path, dry_run=True) == 0
    calls = []
    monkeypatch.setattr(
        supervisor,
        "run_campaign",
        lambda output, *, timeout_s, dry_run: (
            calls.append((output, timeout_s, dry_run)) or 0
        ),
    )
    assert (
        supervisor.main([str(tmp_path), "--timeout-s", "2", "--dry-run"]) == 0
    )
    assert calls == [(tmp_path, 2.0, True)]
    with pytest.raises(SystemExit):
        supervisor.main([str(tmp_path), "--phase", "smoke"])


@pytest.mark.parametrize(
    "fault", ("identity", "dirty", "package", "host", "virtual")
)
def test_hard_preflight_precedes_manifest(monkeypatch, tmp_path, fault):
    identity = deepcopy(_IDENTITY)
    if fault == "identity":
        identity["source"]["sha256"] = None
    elif fault == "dirty":
        identity["source"]["git_dirty"] = True
    elif fault == "package":
        identity["packages"]["jax-mps"] = None
    elif fault == "host":
        identity["host"]["cpu_model"] = "VirtualApple"
    built = False
    monkeypatch.setattr(supervisor, "campaign_identity", lambda root: identity)
    monkeypatch.setattr(
        supervisor,
        "_virtualization_status",
        lambda: "1" if fault == "virtual" else "0",
    )

    def build(root):
        nonlocal built
        built = True
        return {}

    monkeypatch.setattr(supervisor, "build_manifest", build)
    with pytest.raises(supervisor.CampaignError):
        supervisor.run_campaign(tmp_path)
    assert not built


@pytest.mark.parametrize(
    "field", ("macos", "macos_build", "os_release", "physical_memory_bytes")
)
def test_hard_preflight_requires_stable_host_metadata(field):
    identity = deepcopy(_IDENTITY)
    identity["host"][field] = None

    with pytest.raises(supervisor.CampaignError, match="identity"):
        supervisor._require_preflight(identity)


def test_lock_spans_manifest_and_exact_phase_order(monkeypatch, tmp_path):
    requests = campaign_requests()
    _configure(monkeypatch, requests)
    events = []
    identity_calls = 0

    class Lock:
        def __enter__(self):
            events.append("locked")

        def __exit__(self, *args):
            events.append("released")

    def identity(root):
        nonlocal identity_calls
        identity_calls += 1
        assert events == ["locked"]
        return deepcopy(_IDENTITY)

    seen = []

    def run(root, request, *, timeout_s):
        assert events == ["locked"]
        seen.append(CampaignRequest(request.phase, request.cell, request.block))
        return WorkerAttempt(_payload(seen[-1]), False)

    monkeypatch.setattr(supervisor, "HostCampaignLock", Lock)
    monkeypatch.setattr(supervisor, "campaign_identity", identity)
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    monkeypatch.setattr(supervisor, "write_raw_result", lambda *args: None)
    monkeypatch.setattr(supervisor, "run_worker", run)
    assert supervisor.run_campaign(tmp_path) == 0
    assert tuple(seen) == requests
    assert [request.phase for request in seen] == sorted(
        (request.phase for request in seen),
        key=("smoke", "timing", "accuracy").index,
    )
    assert events == ["locked", "released"]
    assert identity_calls == 1 + 2 * len(requests)


@pytest.mark.parametrize("mode", ("prefix", "gap", "corrupt"))
def test_resume_accepts_only_a_valid_success_prefix(
    monkeypatch, tmp_path, mode
):
    requests = campaign_requests()[:3]
    _configure(monkeypatch, requests)
    values = [_payload(requests[0]), None, None]
    if mode == "gap":
        values[2] = _payload(requests[2])
    calls = []

    def load(output, request, digest):
        index = requests.index(request)
        if mode == "corrupt" and index == 1:
            raise ValueError("corrupt")
        return values[index]

    monkeypatch.setattr(supervisor, "load_raw_result", load)
    monkeypatch.setattr(supervisor, "write_raw_result", lambda *args: None)
    monkeypatch.setattr(
        supervisor,
        "run_worker",
        lambda root, request, **kwargs: (
            calls.append(request),
            WorkerAttempt(
                _payload(
                    CampaignRequest(request.phase, request.cell, request.block)
                ),
                False,
            ),
        )[1],
    )
    if mode in ("gap", "corrupt"):
        with pytest.raises((ValueError, supervisor.CampaignError)):
            supervisor.run_campaign(tmp_path)
    else:
        assert supervisor.run_campaign(tmp_path) == 0
        assert len(calls) == 2


def test_supervisor_import_does_not_import_jax():
    code = (
        "import sys; import benchmarks.tempering_accuracy.supervisor; "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=_ROOT, check=True)


def test_metal_prelaunch_requires_ac_and_both_no_warnings(monkeypatch):
    monkeypatch.setattr(
        supervisor, "_command_value", lambda command: "Battery Power"
    )
    with pytest.raises(supervisor._RetryableError):
        supervisor._prelaunch_snapshot()
    monkeypatch.setattr(
        supervisor,
        "_command_value",
        lambda command: (
            "Now drawing from 'AC Power'"
            if command[-1] == "batt"
            else "No thermal warning level"
        ),
    )
    with pytest.raises(supervisor._RetryableError):
        supervisor._prelaunch_snapshot()


def test_metal_timing_retains_supervisor_prelaunch(monkeypatch, tmp_path):
    request = next(
        item
        for item in campaign_requests()
        if item.phase == "timing" and item.cell.lane == "mps_f32"
    )
    _configure(monkeypatch, (request,))
    snapshot = {"power_status": "AC", "thermal_status": "cool"}
    monkeypatch.setattr(supervisor, "_prelaunch_snapshot", lambda: snapshot)
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    payload = _timing_payload(request)
    monkeypatch.setattr(
        supervisor,
        "run_worker",
        lambda *args, **kwargs: WorkerAttempt(payload, False),
    )
    raw = []
    monkeypatch.setattr(
        supervisor, "write_raw_result", lambda *args: raw.append(args[-1])
    )
    assert supervisor.run_campaign(tmp_path) == 0
    assert raw[0]["timing"]["environment"]["supervisor_prelaunch"] == snapshot


@pytest.mark.parametrize("boundary", ("pre_timing", "post_timing", "post_cell"))
def test_ineligible_metal_boundary_retains_timing_and_stops(
    monkeypatch, tmp_path, boundary
):
    request = next(
        item
        for item in campaign_requests()
        if item.phase == "timing" and item.cell.lane == "mps_f32"
    )
    requests = (request, campaign_requests()[-1])
    _configure(monkeypatch, requests)
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    payload = _timing_payload(request)
    payload["timing"]["environment"][boundary]["power_status"] = "Battery"
    calls = []
    monkeypatch.setattr(
        supervisor,
        "run_worker",
        lambda *args, **kwargs: (
            calls.append(1) or WorkerAttempt(payload, False)
        ),
    )
    raw = []
    monkeypatch.setattr(
        supervisor, "write_raw_result", lambda *args: raw.append(args[-1])
    )

    assert supervisor.run_campaign(tmp_path) == 1
    assert len(calls) == len(raw) == 1
    assert raw[0]["failure"]["kind"] == ("metal_timing_environment_ineligible")
    assert raw[0]["failure"]["boundaries"][boundary]["power_status"] == (
        "Battery"
    )
    assert raw[0]["timing"] == payload["timing"] | {
        "environment": payload["timing"]["environment"]
        | {"supervisor_prelaunch": {}}
    }


def test_retryable_launch_attempt_is_exclusive_and_rerunnable(
    monkeypatch, tmp_path
):
    request = campaign_requests()[0]
    _configure(monkeypatch, (request,))
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    monkeypatch.setattr(
        supervisor,
        "write_raw_result",
        lambda *args: pytest.fail("retryable attempt became a raw result"),
    )
    payload = _payload(
        request,
        failure={"kind": "launch_error", "message": "x" * 5_000},
    )
    launches = []
    monkeypatch.setattr(
        supervisor,
        "run_worker",
        lambda *args, **kwargs: (
            launches.append(1) or WorkerAttempt(payload, True)
        ),
    )
    assert supervisor.run_campaign(tmp_path) == 1
    assert supervisor.run_campaign(tmp_path) == 1
    paths = sorted((tmp_path / "attempts").iterdir())
    assert [path.name for path in paths] == ["000-000.json", "000-001.json"]
    records = [json.loads(path.read_text()) for path in paths]
    assert all(
        path.read_text() == canonical_json(record) + "\n"
        for path, record in zip(paths, records, strict=True)
    )
    assert records[0]["request"] == request_dict(bind_request(request, _DIGEST))
    assert len(records[0]["failure"]["message"]) == 4_096
    assert "timing" not in records[0] and len(launches) == 2


@pytest.mark.parametrize(
    ("mode", "kind"),
    (
        ("oserror", "launch_error"),
        ("prelaunch", "metal_prelaunch_ineligible"),
        ("source", "campaign_identity_changed_before_launch"),
    ),
)
def test_prelaunch_failures_leave_raw_missing(
    monkeypatch, tmp_path, mode, kind
):
    request = campaign_requests()[0]
    if mode == "prelaunch":
        request = next(
            item
            for item in campaign_requests()
            if item.phase == "timing" and item.cell.lane == "mps_f32"
        )
    _configure(monkeypatch, (request,))
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    monkeypatch.setattr(
        supervisor,
        "write_raw_result",
        lambda *args: pytest.fail("prelaunch failure became raw"),
    )
    if mode == "oserror":
        monkeypatch.setattr(
            supervisor,
            "run_worker",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("launch")),
        )
    elif mode == "prelaunch":
        monkeypatch.setattr(
            supervisor,
            "_prelaunch_snapshot",
            lambda: (_ for _ in ()).throw(
                supervisor._RetryableError({"kind": kind})
            ),
        )
    else:
        changed = deepcopy(_IDENTITY)
        changed["source"]["sha256"] = "e" * 64
        identities = iter((deepcopy(_IDENTITY), changed))
        monkeypatch.setattr(
            supervisor, "campaign_identity", lambda root: next(identities)
        )
    assert supervisor.run_campaign(tmp_path) == 1
    attempt = json.loads(next((tmp_path / "attempts").iterdir()).read_text())
    assert attempt["failure"]["kind"] == kind


@pytest.mark.parametrize(
    "kind",
    (
        "timeout",
        "worker_exit",
        "malformed_output",
        "execution_failure",
        "structural_failure",
    ),
)
def test_immutable_raw_failure_stops_without_retry(monkeypatch, tmp_path, kind):
    requests = campaign_requests()[:2]
    _configure(monkeypatch, requests)
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    calls = []

    def run(root, bound, **kwargs):
        request = CampaignRequest(bound.phase, bound.cell, bound.block)
        calls.append(request)
        return WorkerAttempt(_payload(request, failure={"kind": kind}), False)

    raw = []
    monkeypatch.setattr(supervisor, "run_worker", run)
    monkeypatch.setattr(
        supervisor, "write_raw_result", lambda *args: raw.append(args[-1])
    )
    assert supervisor.run_campaign(tmp_path) == 1
    assert calls == [requests[0]] and raw[0]["failure"]["kind"] == kind


def test_existing_scientific_failure_is_a_complete_prefix(
    monkeypatch, tmp_path
):
    request = campaign_requests()[0]
    _configure(monkeypatch, (request,))
    monkeypatch.setattr(
        supervisor,
        "load_raw_result",
        lambda *args: _payload(request, failure={"kind": "scientific"}),
    )
    monkeypatch.setattr(supervisor, "run_worker", pytest.fail)
    assert supervisor.run_campaign(tmp_path) == 1


def test_postlaunch_identity_drift_is_an_immutable_failure(
    monkeypatch, tmp_path
):
    request = campaign_requests()[0]
    _configure(monkeypatch, (request,))
    changed = deepcopy(_IDENTITY)
    changed["source"]["sha256"] = "e" * 64
    changed["packages"]["jax"] = "changed"
    identities = iter((deepcopy(_IDENTITY), deepcopy(_IDENTITY), changed))
    monkeypatch.setattr(
        supervisor, "campaign_identity", lambda root: next(identities)
    )
    monkeypatch.setattr(supervisor, "load_raw_result", lambda *args: None)
    monkeypatch.setattr(
        supervisor,
        "run_worker",
        lambda *args, **kwargs: WorkerAttempt(
            _payload(
                request,
                failure={"kind": "worker_exit", "stderr_tail": "x" * 5_000},
            ),
            False,
        ),
    )
    raw = []
    monkeypatch.setattr(
        supervisor, "write_raw_result", lambda *args: raw.append(args[-1])
    )
    assert supervisor.run_campaign(tmp_path) == 1
    failure = raw[0]["failure"]
    assert failure["kind"] == "source_identity_changed_after_launch"
    assert failure["changed_domains"] == ["source", "packages"]
    assert failure["worker_failure"]["kind"] == "worker_exit"
    assert len(failure["worker_failure"]["stderr_tail"]) == 4_096
