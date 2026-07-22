# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process transport contracts for issue #30."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.tempering_accuracy.transport import (
    RESULT_MARKER,
    parse_worker_stdout,
    run_worker,
    worker_environment,
)

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import (
    CampaignRequest,
    bind_request,
    request_dict,
)
from benchmarks.tempering_accuracy.plan import current_smoke_cells

_DIGEST = "a" * 64
_ROOT = Path(__file__).resolve().parents[1]
_CELL = next(cell for cell in current_smoke_cells() if cell.lane == "cpu_f64")
_REQUEST = bind_request(CampaignRequest("smoke", _CELL, None), _DIGEST)


def _payload(*, failure=None):
    return {
        "schema_version": 1,
        "request": request_dict(_REQUEST),
        "failure": failure,
        "timing": None,
        "runs": [],
    }


def _stdout(payload):
    return RESULT_MARKER + canonical_json(payload) + "\n"


@pytest.mark.parametrize(
    ("lane", "platform", "x64"),
    (("cpu_f64", "cpu", "true"), ("mps_f32", "mps", "false")),
)
def test_worker_environment_is_fresh_and_lane_specific(
    monkeypatch, lane, platform, x64
):
    inherited = (
        "JAX_MPS_ASYNC_DISPATCH",
        "XLA_FLAGS",
        "PJRT_DEVICE",
        "OMP_NUM_THREADS",
        "MPS_PROFILE",
        "MLX_METAL_DEBUG",
    )
    for name in inherited:
        monkeypatch.setenv(name, "inherited")

    environment = worker_environment(lane)

    assert environment == {
        "JAX_DISABLE_JIT": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_ENABLE_X64": x64,
        "JAX_PLATFORMS": platform,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": "/tmp",
    }
    assert not set(inherited) & environment.keys()


def test_transport_import_does_not_import_jax():
    code = (
        "import sys; import benchmarks.tempering_accuracy.transport; "
        "assert 'jax' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_run_worker_uses_registered_process_boundary(monkeypatch):
    payload = _payload()

    def fake_run(command, **kwargs):
        assert Path(command[0]).is_absolute()
        assert command[1:4] == [
            "-m",
            "benchmarks.tempering_accuracy.worker",
            "--request-json",
        ]
        assert command[4] == canonical_json(request_dict(_REQUEST))
        assert kwargs == {
            "cwd": _ROOT,
            "env": worker_environment("cpu_f64"),
            "capture_output": True,
            "text": True,
            "check": False,
            "shell": False,
            "timeout": 12.5,
        }
        return subprocess.CompletedProcess(command, 0, _stdout(payload), "")

    monkeypatch.setattr(
        "benchmarks.tempering_accuracy.transport.subprocess.run", fake_run
    )
    attempt = run_worker(_ROOT, _REQUEST, timeout_s=12.5)
    assert attempt.payload == payload
    assert attempt.retryable is False


@pytest.mark.parametrize("fault", ("schema", "digest", "request"))
def test_parser_rejects_result_identity_mismatch(fault):
    payload = copy.deepcopy(_payload())
    if fault == "schema":
        payload["schema_version"] = 2
    elif fault == "digest":
        payload["request"]["manifest_sha256"] = "b" * 64
    else:
        payload["request"]["phase"] = "accuracy"
    with pytest.raises(ValueError, match="manifest request"):
        parse_worker_stdout(_stdout(payload), _REQUEST)


@pytest.mark.parametrize("stdout", ("", "{marker}{json}{marker}{json}"))
def test_parser_requires_exactly_one_marker(stdout):
    encoded = stdout.format(
        marker=RESULT_MARKER, json=canonical_json(_payload()) + "\n"
    )
    with pytest.raises(ValueError, match="exactly one"):
        parse_worker_stdout(encoded, _REQUEST)


@pytest.mark.parametrize(
    ("mode", "kind", "retryable"),
    (
        ("timeout", "timeout", False),
        ("launch", "launch_error", True),
        ("exit", "worker_exit", False),
        ("malformed", "malformed_output", False),
    ),
)
def test_transport_classifies_process_failures(
    monkeypatch, mode, kind, retryable
):
    def fake_run(command, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(
                command, kwargs["timeout"], "x" * 5_000, b"y" * 5_000
            )
        if mode == "launch":
            raise OSError("cannot launch")
        if mode == "exit":
            return subprocess.CompletedProcess(
                command, 7, "x" * 5_000, "y" * 5_000
            )
        return subprocess.CompletedProcess(command, 0, "no marker", "trace")

    monkeypatch.setattr(
        "benchmarks.tempering_accuracy.transport.subprocess.run", fake_run
    )
    attempt = run_worker(_ROOT, _REQUEST, timeout_s=1.0)

    assert attempt.retryable is retryable
    assert attempt.payload["failure"]["kind"] == kind
    if mode in ("timeout", "exit"):
        assert len(attempt.payload["failure"]["stdout_tail"]) == 4_096
        assert len(attempt.payload["failure"]["stderr_tail"]) == 4_096


def test_valid_worker_failure_remains_result_data(monkeypatch):
    payload = _payload(failure={"kind": "execution_failure"})
    monkeypatch.setattr(
        "benchmarks.tempering_accuracy.transport.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, _stdout(payload), "diagnostic"
        ),
    )

    attempt = run_worker(_ROOT, _REQUEST, timeout_s=1.0)

    assert attempt == (payload, False)
