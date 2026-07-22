# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Host supervisor contracts for issue #30."""

import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import benchmarks.tempering_accuracy.supervisor as supervisor
import pytest

from benchmarks.tempering_accuracy.artifacts import (
    CampaignRequest,
    bind_request,
    campaign_requests,
    request_dict,
)
from benchmarks.tempering_accuracy.transport import WorkerAttempt

_DIGEST = "a" * 64
_ROOT = Path(__file__).resolve().parents[1]
_IDENTITY = {
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
        "machine": "arm64",
        "cpu_model": "Apple M3 Pro",
        "hardware_model": "Mac15,7",
    },
}


def _payload(request, *, failure=None):
    return {
        "schema_version": 1,
        "request": request_dict(bind_request(request, _DIGEST)),
        "failure": failure,
        "timing": None,
        "runs": [],
    }


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


@pytest.mark.parametrize("timeout", (0, -1, math.inf, math.nan))
def test_timeout_must_be_positive_and_finite(tmp_path, timeout):
    with pytest.raises(ValueError, match="timeout"):
        supervisor.run_campaign(tmp_path, timeout_s=timeout)


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


def test_lock_spans_manifest_and_exact_phase_order(monkeypatch, tmp_path):
    requests = campaign_requests()
    _configure(monkeypatch, requests)
    events = []

    class Lock:
        def __enter__(self):
            events.append("locked")

        def __exit__(self, *args):
            events.append("released")

    def identity(root):
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
