# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""JAX-free fresh-process transport for the tempering campaign."""

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import (
    SCHEMA_VERSION,
    WorkerRequest,
    request_dict,
    validate_worker_result,
)
from benchmarks.tempering_accuracy.plan import CampaignCell

RESULT_MARKER = "SMCX_TEMPERING_RESULT="
_CAPTURE_LIMIT = 4_096
_RUNTIME_PREFIXES = tuple(
    "JAX_ XLA_ PJRT_ OMP_ KMP_ MKL_ OPENBLAS_ GOTO_ BLIS_ VECLIB_ "  # noqa: SIM905
    "NUMEXPR_ MPS_ MLX_ MTL_ METAL_".split()
)


class WorkerAttempt(NamedTuple):
    """One completed worker launch or a retryable launch failure."""

    payload: dict[str, Any]
    retryable: bool


def runtime_controls(environment: Mapping[str, str]) -> dict[str, str]:
    """Select backend and numerical-library controls for evidence."""
    return {
        name: environment[name]
        for name in sorted(environment)
        if name.startswith(_RUNTIME_PREFIXES)
    }


def worker_environment(lane: str) -> dict[str, str]:
    """Build a worker environment without inheriting process controls."""
    if lane not in ("cpu_f64", "mps_f32"):
        raise ValueError(f"unknown worker lane: {lane}") from None
    platform = lane.split("_")[0]
    x64 = str(lane == "cpu_f64").lower()
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "JAX_PLATFORMS": platform,
        "JAX_ENABLE_X64": x64,
        "JAX_DISABLE_JIT": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
    }


def decode_worker_request(encoded: str) -> WorkerRequest:
    """Decode one canonical manifest-bound request."""
    try:
        value = json.loads(encoded)
        request = WorkerRequest(
            value["manifest_sha256"],
            value["phase"],
            CampaignCell(**value["cell"]),
            value["block"],
        )
        if encoded != canonical_json(request_dict(request)):
            raise ValueError
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid canonical worker request") from error
    return request


def parse_worker_stdout(stdout: str, request: WorkerRequest) -> dict[str, Any]:
    """Extract and validate exactly one canonical worker result."""
    records = [
        line.removeprefix(RESULT_MARKER)
        for line in stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if len(records) != 1:
        raise ValueError("worker must emit exactly one result marker")
    try:
        value = json.loads(records[0])
        if not isinstance(value, dict) or records[0] != canonical_json(value):
            raise ValueError("worker result is not canonical JSON")
    except json.JSONDecodeError as error:
        raise ValueError("worker result is not JSON") from error
    validate_worker_result(value, request)
    return value


def _tail(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return (value or "")[-_CAPTURE_LIMIT:]


def _failure(
    request: WorkerRequest, kind: str, **evidence: Any
) -> dict[str, Any]:
    failure = {"kind": kind, **evidence}
    return {
        "schema_version": SCHEMA_VERSION,
        "request": request_dict(request),
        "failure": failure,
        "timing": None,
        "runs": [],
    }


def _failed_attempt(
    request: WorkerRequest, kind: str, retryable: bool = False, **evidence: Any
) -> WorkerAttempt:
    return WorkerAttempt(_failure(request, kind, **evidence), retryable)


def run_worker(
    root: Path, request: WorkerRequest, *, timeout_s: float
) -> WorkerAttempt:
    """Run one blocking fresh worker and classify its transport result."""
    root = Path(root).resolve()
    command = [
        str(Path(sys.executable).absolute()),
        "-m",
        "benchmarks.tempering_accuracy.worker",
        "--request-json",
        canonical_json(request_dict(request)),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env=worker_environment(request.cell.lane),
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        return _failed_attempt(
            request,
            "timeout",
            timeout_s=timeout_s,
            stdout_tail=_tail(error.stdout),
            stderr_tail=_tail(error.stderr),
        )
    except OSError as error:
        return _failed_attempt(
            request,
            "launch_error",
            True,
            exception_type=type(error).__name__,
            message=str(error),
        )
    if result.returncode != 0:
        return _failed_attempt(
            request,
            "worker_exit",
            returncode=result.returncode,
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
    try:
        return WorkerAttempt(parse_worker_stdout(result.stdout, request), False)
    except ValueError as error:
        return _failed_attempt(
            request,
            "malformed_output",
            exception_type=type(error).__name__,
            message=str(error),
            stdout_tail=_tail(result.stdout),
            stderr_tail=_tail(result.stderr),
        )
