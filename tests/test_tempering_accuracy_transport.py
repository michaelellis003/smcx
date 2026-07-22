# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process transport contracts for issue #30."""

import subprocess
import sys
from pathlib import Path

import pytest

import benchmarks.tempering_accuracy.transport as transport
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
    return transport.RESULT_MARKER + canonical_json(payload) + "\n"


@pytest.mark.parametrize(
    ("lane", "platform", "x64"),
    (("cpu_f64", "cpu", "true"), ("mps_f32", "mps", "false")),
)
def test_worker_environment_is_fresh_and_lane_specific(
    monkeypatch, lane, platform, x64
):
    inherited = (
        "JAX_MPS_ASYNC_DISPATCH XLA_FLAGS PJRT_DEVICE OMP_NUM_THREADS "  # noqa: SIM905
        "MPS_PROFILE MLX_METAL_DEBUG".split()
    )
    for name in inherited:
        monkeypatch.setenv(name, "inherited")

    environment = transport.worker_environment(lane)

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


def test_transport_import_does_not_import_jax():
    code = (
        "import sys; import benchmarks.tempering_accuracy.transport; "
        "assert 'jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=_ROOT, check=True)


def test_run_worker_uses_registered_process_boundary(monkeypatch):
    payload = _payload(failure={"kind": "execution_failure"})

    def fake_run(command, **kwargs):
        assert command[0] == str(Path(sys.executable).absolute())
        assert command[1:4] == [
            "-m",
            "benchmarks.tempering_accuracy.worker",
            "--request-json",
        ]
        assert command[4] == canonical_json(request_dict(_REQUEST))
        assert kwargs == {
            "cwd": _ROOT,
            "env": transport.worker_environment("cpu_f64"),
            "capture_output": True,
            "text": True,
            "check": False,
            "shell": False,
            "timeout": 12.5,
        }
        return subprocess.CompletedProcess(command, 0, _stdout(payload), "")

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    attempt = transport.run_worker(_ROOT, _REQUEST, timeout_s=12.5)
    assert attempt == (payload, False)


@pytest.mark.parametrize(
    "fault", ("schema", "digest", "request", "missing", "duplicate")
)
def test_parser_rejects_invalid_marker_or_identity(fault):
    payload = _payload()
    if fault == "schema":
        payload["schema_version"] = 2
    elif fault == "digest":
        payload["request"]["manifest_sha256"] = "b" * 64
    elif fault == "request":
        payload["request"]["phase"] = "accuracy"
    stdout = "" if fault == "missing" else _stdout(payload)
    if fault == "duplicate":
        stdout *= 2
    with pytest.raises(ValueError):
        transport.parse_worker_stdout(stdout, _REQUEST)


@pytest.mark.parametrize(
    ("mode", "kind", "retryable", "captured_lengths"),
    (
        ("timeout", "timeout", False, (4_096, 4_096)),
        ("launch", "launch_error", True, None),
        ("exit", "worker_exit", False, (4_096, 4_096)),
        ("malformed", "malformed_output", False, (9, 5)),
    ),
)
def test_transport_classifies_process_failures(
    monkeypatch, mode, kind, retryable, captured_lengths
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

    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    attempt = transport.run_worker(_ROOT, _REQUEST, timeout_s=1.0)

    assert attempt.retryable is retryable
    assert attempt.payload["failure"]["kind"] == kind
    if captured_lengths is not None:
        failure = attempt.payload["failure"]
        captured = failure["stdout_tail"], failure["stderr_tail"]
        assert tuple(map(len, captured)) == captured_lengths
