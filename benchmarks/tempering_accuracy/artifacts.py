# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Immutable artifacts for the tempering-accuracy campaign."""

import hashlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.profiling.common import (
    _command_value,
    canonical_json,
    host_environment,
    package_versions,
)
from benchmarks.tempering_accuracy.plan import (
    ORDER_SEED,
    CampaignCell,
    cell_id,
    current_cells,
    current_smoke_cells,
    matched_cells,
    timing_blocks,
    waste_free_cells,
    waste_free_smoke_cells,
)

SCHEMA_VERSION = 1
_FIELDS = {"schema_version", "request", "failure", "timing", "runs"}


class CampaignRequest(NamedTuple):
    """One manifest entry before binding to a manifest digest."""

    phase: str
    cell: CampaignCell
    block: int | None


class WorkerRequest(NamedTuple):
    """One manifest-bound fresh-worker invocation."""

    manifest_sha256: str
    phase: str
    cell: CampaignCell
    block: int | None


def request_dict(
    request: CampaignRequest | WorkerRequest,
) -> dict[str, object]:
    """Convert one request to its JSON object."""
    result: dict[str, object] = {
        "phase": request.phase,
        "cell": request.cell._asdict(),
        "block": request.block,
    }
    if isinstance(request, WorkerRequest):
        result = {"manifest_sha256": request.manifest_sha256, **result}
    return result


def campaign_requests() -> tuple[CampaignRequest, ...]:
    """Return the exact 508 requests in execution order."""
    result = [
        CampaignRequest("smoke", cell, None) for cell in current_smoke_cells()
    ]
    for cells in (current_cells(), matched_cells()):
        result.extend(
            CampaignRequest("timing", cell, block)
            for block, ordered in enumerate(timing_blocks(cells))
            for cell in ordered
        )
    for cells in (current_cells(), matched_cells()):
        result.extend(CampaignRequest("accuracy", cell, None) for cell in cells)
    return tuple(result)


def bind_request(request: CampaignRequest, digest: str) -> WorkerRequest:
    """Bind a planned request to a persisted manifest."""
    valid = len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )
    if not valid:
        raise ValueError("manifest_sha256 must be 64 lowercase hex characters")
    return WorkerRequest(digest, *request)


def raw_filename(request: CampaignRequest) -> str:
    """Return a stable collision-resistant result filename."""
    identity = canonical_json(request_dict(request)).encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    block = "" if request.block is None else f"-b{request.block:02d}"
    return f"{request.phase}-{cell_id(request.cell)}{block}-{digest}.json"


def _source_identity(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    paths = [
        *sorted((root / "src/smcx").rglob("*.py")),
        *sorted((root / "benchmarks/tempering_accuracy").glob("*.py")),
        root / "benchmarks/profiling/common.py",
        root / "benchmarks/profiling/locking.py",
        root / "pyproject.toml",
    ]
    lock = root / "uv.lock"
    if not lock.is_file() or any(not path.is_file() for path in paths):
        raise FileNotFoundError("campaign source or lock file is missing")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
    status = _command_value(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src/smcx",
            "benchmarks/tempering_accuracy",
            "benchmarks/profiling/common.py",
            "benchmarks/profiling/locking.py",
            "pyproject.toml",
            "uv.lock",
        ),
        cwd=root,
        allow_empty=True,
    )
    commit = _command_value(("git", "rev-parse", "HEAD"), cwd=root)
    if status is None or commit is None:
        raise RuntimeError("campaign git identity is unavailable")
    source = {
        "git_commit": commit,
        "git_dirty": bool(status),
        "sha256": digest.hexdigest(),
        "files": [path.relative_to(root).as_posix() for path in paths],
    }
    frozen_lock = {
        "path": "uv.lock",
        "sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }
    return source, frozen_lock


def campaign_identity(root: Path) -> dict[str, Any]:
    """Capture source, lock, package, Python, and host identity."""
    source, lock = _source_identity(Path(root).resolve())

    packages = package_versions()
    for name in ("python", "tfp-nightly"):
        packages.pop(name, None)
    packages.update({name: version(name) for name in ("ml-dtypes", "scipy")})
    return {
        "source": source,
        "lock": lock,
        "packages": packages,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "host": host_environment(),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    """Build the immutable pre-execution manifest."""
    requests = [request_dict(request) for request in campaign_requests()]
    exclusion = {
        "arm": "waste_free_multinomial",
        "status": "blocked_backend_correctness",
        "tracking_issue": 38,
        "blocked_request_counts": {"smoke": 2, "timing": 60, "accuracy": 12},
        "smoke_cells": [cell._asdict() for cell in waste_free_smoke_cells()],
        "cells": [cell._asdict() for cell in waste_free_cells()],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": "tempering_accuracy",
        "order_seed": ORDER_SEED,
        "plan_sha256": hashlib.sha256(
            canonical_json(requests).encode()
        ).hexdigest(),
        "requests": requests,
        "exclusions": [exclusion],
        "campaign_identity": campaign_identity(root),
    }


def _document(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(dict(value)) + "\n").encode()


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the exact bytes persisted as ``manifest.json``."""
    return hashlib.sha256(_document(manifest)).hexdigest()


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_document(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> str:
    """Create the manifest once or require exact bytes on resume."""
    path = Path(output_dir) / "manifest.json"
    expected = _document(manifest)
    try:
        if path.exists():
            raise FileExistsError
        _write_exclusive(path, manifest)
    except FileExistsError:
        if path.read_bytes() != expected:
            raise ValueError(
                "output directory contains a different manifest"
            ) from None
    return hashlib.sha256(expected).hexdigest()


def _validate_result(
    value: Mapping[str, Any], request: CampaignRequest, digest: str
) -> None:
    failure = value.get("failure")
    timing = value.get("timing")
    valid = (
        set(value) == _FIELDS
        and type(value.get("schema_version")) is int
        and value["schema_version"] == SCHEMA_VERSION
        and canonical_json(value.get("request"))
        == canonical_json(request_dict(bind_request(request, digest)))
        and isinstance(value.get("runs"), list)
        and (failure is None or isinstance(failure, dict))
        and (timing is None or isinstance(timing, dict))
    )
    if not valid:
        raise ValueError("result does not match its manifest request")


def load_raw_result(
    output_dir: Path, request: CampaignRequest, digest: str
) -> dict[str, Any] | None:
    """Load and validate one immutable result for resume."""
    path = Path(output_dir) / "raw" / raw_filename(request)
    if not path.exists():
        return None
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
        if not isinstance(value, dict) or encoded != _document(value):
            raise ValueError("result is not canonical JSON")
        _validate_result(value, request, digest)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid raw result; refusing to overwrite {path}: {error}"
        ) from error
    return value


def write_raw_result(
    output_dir: Path,
    request: CampaignRequest,
    digest: str,
    payload: Mapping[str, Any],
) -> Path:
    """Exclusively persist one validated immutable result."""
    value = dict(payload)
    _validate_result(value, request, digest)
    path = Path(output_dir) / "raw" / raw_filename(request)
    _write_exclusive(path, value)
    return path
