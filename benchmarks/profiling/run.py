# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Supervise isolated workers for the all-algorithm profiling campaign."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # Allow direct ``python .../run.py`` use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.profiling.common import (
    DEFAULT_ORDER_SEED,
    PLATFORMS,
    PROFILES,
    SCHEMA_VERSION,
    WORKLOADS,
    Cell,
    build_manifest,
    campaign_identity,
    canonical_json,
    expected_device_identity,
    plan_cells,
    profiling_runtime_flags,
    record_matches_cell,
    validate_result,
    worker_environment,
)
from benchmarks.profiling.locking import HostCampaignLock
from benchmarks.profiling.preflight import estimate_plan

CELL_TIMEOUT_S = 1_800.0
RESULT_PREFIX = "SMCX_PROFILE_RESULT="

Runner = Callable[[Cell], Mapping[str, Any]]
ValidationRunner = Callable[[Cell], Mapping[str, Any]]


class CampaignIdentityError(ValueError):
    """The frozen campaign environment changed or was misreported."""


def _expected_runtime_flags(platform: str) -> dict[str, str]:
    """Return the exact sanitized flags required for one worker backend."""
    return profiling_runtime_flags(worker_environment(platform, base={}))


def _expected_dispatch_mode(platform: str) -> str:
    """Return the dispatch mode implied by the sanitized worker flags."""
    if platform == "cpu":
        return "asynchronous"
    if _expected_runtime_flags(platform).get("JAX_MPS_ASYNC_DISPATCH") == "1":
        return "asynchronous"
    return "safe"


def _expected_device_kind(platform: str) -> str:
    """Return the pinned backend's exact JAX device-kind label."""
    return expected_device_identity(platform)[0]


def _validate_runtime_metadata(
    record: Mapping[str, Any],
    cell: Cell,
    *,
    phase: str,
) -> None:
    """Require exact flags, dispatch, and a concrete device identity."""
    environment = record.get("environment")
    if not isinstance(environment, Mapping):
        raise CampaignIdentityError(f"{phase} environment is not a mapping")
    if environment.get("runtime_flags") != _expected_runtime_flags(
        cell.platform
    ):
        raise CampaignIdentityError(
            f"{phase} runtime flags do not match the sanitized contract"
        )
    device_kind = environment.get("device_kind")
    expected_device_kind = _expected_device_kind(cell.platform)
    if device_kind != expected_device_kind:
        raise CampaignIdentityError(
            f"{phase} device_kind does not match {expected_device_kind}"
        )
    device_id = environment.get("device_id")
    expected_device_id = expected_device_identity(cell.platform)[1]
    if device_id != expected_device_id or isinstance(device_id, bool):
        raise CampaignIdentityError(
            f"{phase} device_id does not match {expected_device_id}"
        )
    expected_dispatch = _expected_dispatch_mode(cell.platform)
    if record.get("dispatch_mode") != expected_dispatch:
        raise CampaignIdentityError(
            f"{phase} dispatch does not match {expected_dispatch}"
        )


def _cell_json(cell: Cell) -> str:
    """Serialize a cell canonically for hashing and worker transport."""
    return canonical_json(cell._asdict())


def build_worker_command(
    *,
    root: Path,
    cell: Cell,
    phase: str = "timing",
) -> list[str]:
    """Return the complete fresh-process command for one profiling cell."""
    if phase not in {"timing", "validation"}:
        raise ValueError(f"unknown worker phase: {phase}")
    python = Path(sys.executable).with_name("python")
    return [
        str(python),
        str(Path(root) / "benchmarks/profiling/worker.py"),
        "--cell-json",
        _cell_json(cell),
        "--phase",
        phase,
    ]


def raw_filename(cell: Cell) -> str:
    """Return a stable, collision-resistant filename for one exact cell."""
    digest = hashlib.sha256(_cell_json(cell).encode()).hexdigest()[:16]
    return f"{cell.workload}_{cell.platform}_b{cell.block:02d}_{digest}.json"


def _failure_record(
    cell: Cell,
    *,
    kind: str,
    message: str,
) -> dict[str, Any]:
    """Build a schema-valid envelope for a worker-level failure."""
    spec = WORKLOADS[cell.workload]
    return {
        "algorithm": spec.algorithm,
        "backend": "unknown",
        "block": cell.block,
        "correctness": {"passed": False},
        "correctness_replicates": cell.correctness_replicates,
        "correctness_level": (
            spec.replicated_correctness_level
            if cell.correctness_replicates
            else "structural"
        ),
        "dispatch_mode": "unavailable",
        "environment": {},
        "execution_mode": cell.execution_mode,
        "failure": {"kind": kind, "message": message},
        "first_execution_s": None,
        "lifecycle": {},
        "memory": {},
        "model": spec.model,
        "parameters": dict(cell.parameters),
        "platform_requested": cell.platform,
        "repeats": cell.repeats,
        "schema_version": SCHEMA_VERSION,
        "source": {},
        "steady_summary": {},
        "steady_times_s": [],
        "versions": {},
        "work_metrics": {},
        "workload": cell.workload,
        "warmups": cell.warmups,
    }


def _tail(value: str | bytes | None, *, limit: int = 1_000) -> str:
    """Return a bounded diagnostic suffix from subprocess output."""
    if value is None:
        return ""
    text = value.decode(errors="replace") if isinstance(value, bytes) else value
    return text.strip()[-limit:]


def _parse_worker_output(stdout: str) -> dict[str, Any]:
    """Parse the last stable prefixed JSON record emitted by a worker."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(RESULT_PREFIX):
            payload = line.removeprefix(RESULT_PREFIX)
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("worker result must be a JSON object")
            return value

    # Retain compatibility with a deliberately quiet worker or injected
    # executable that emits one bare JSON line.
    if lines:
        value = json.loads(lines[-1])
        if isinstance(value, dict):
            return value
    raise ValueError("worker emitted no result JSON")


def _record_matches_cell(record: Mapping[str, Any], cell: Cell) -> bool:
    """Return whether a result belongs to the exact scheduled cell."""
    return record_matches_cell(record, cell)


def _validate_cell_record(record: Mapping[str, Any], cell: Cell) -> None:
    """Validate both the stable envelope and its scheduled-cell identity."""
    validate_result(record)
    if not _record_matches_cell(record, cell):
        raise ValueError("worker result does not match its scheduled cell")


def _validate_campaign_identity(
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    cell: Cell,
    *,
    require_validation_provenance: bool = False,
) -> None:
    """Reject a successful record from a different reproducibility envelope."""
    if record["failure"] is not None:
        return
    if record["source"] != identity["source"]:
        raise CampaignIdentityError(
            "worker source identity does not match manifest"
        )
    if record["versions"] != identity["packages"]:
        raise CampaignIdentityError(
            "worker package identity does not match manifest"
        )
    environment = record["environment"]
    if any(
        environment.get(name) != value
        for name, value in identity["host"].items()
    ):
        raise CampaignIdentityError(
            "worker host identity does not match manifest"
        )
    _validate_runtime_metadata(record, cell, phase="timing")
    if not require_validation_provenance:
        return
    correctness = record["correctness"]
    provenance = correctness.get("validation_provenance")
    if not isinstance(provenance, Mapping):
        raise CampaignIdentityError(
            "final result is missing validation provenance"
        )
    if provenance.get("source") != identity["source"]:
        raise CampaignIdentityError(
            "validation provenance source does not match manifest"
        )
    if provenance.get("versions") != identity["packages"]:
        raise CampaignIdentityError(
            "validation provenance packages do not match manifest"
        )
    if provenance.get("backend") != cell.platform:
        raise CampaignIdentityError(
            "validation provenance backend does not match the cell"
        )
    _validate_runtime_metadata(
        {
            "dispatch_mode": provenance.get("dispatch_mode"),
            "environment": provenance.get("environment"),
        },
        cell,
        phase="validation provenance",
    )
    provenance_environment = provenance["environment"]
    if (
        environment["device_id"],
        environment["device_kind"],
    ) != (
        provenance_environment["device_id"],
        provenance_environment["device_kind"],
    ):
        raise CampaignIdentityError(
            "timing and validation provenance devices do not match"
        )


def _validate_validation_record(
    record: Mapping[str, Any],
    cell: Cell,
    identity: Mapping[str, Any],
) -> None:
    """Validate a validation-only worker payload and campaign identity."""
    required = {
        "backend",
        "block",
        "correctness_level",
        "correctness_replicates",
        "dispatch_mode",
        "environment",
        "execution_mode",
        "parameters",
        "platform_requested",
        "replicated",
        "schema_version",
        "source",
        "versions",
        "workload",
    }
    missing = required - record.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"validation result is missing fields: {names}")
    expected_level = WORKLOADS[cell.workload].replicated_correctness_level
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("validation result has unsupported schema_version")
    observed_identity = {
        "backend": record["backend"],
        "block": record["block"],
        "correctness_level": record["correctness_level"],
        "correctness_replicates": record["correctness_replicates"],
        "execution_mode": record["execution_mode"],
        "parameters": record["parameters"],
        "platform_requested": record["platform_requested"],
        "workload": record["workload"],
    }
    expected_identity = {
        "backend": cell.platform,
        "block": cell.block,
        "correctness_level": expected_level,
        "correctness_replicates": cell.correctness_replicates,
        "execution_mode": cell.execution_mode,
        "parameters": cell.parameters,
        "platform_requested": cell.platform,
        "workload": cell.workload,
    }
    if canonical_json(observed_identity) != canonical_json(expected_identity):
        raise ValueError("validation result does not match scheduled cell")
    replicated = record["replicated"]
    if (
        not isinstance(replicated, Mapping)
        or not isinstance(replicated.get("passed"), bool)
        or replicated.get("replicates") != cell.correctness_replicates
    ):
        raise ValueError("validation result has invalid replicated gate")
    if record["source"] != identity["source"]:
        raise CampaignIdentityError(
            "validation source identity does not match manifest"
        )
    if record["versions"] != identity["packages"]:
        raise CampaignIdentityError(
            "validation package identity does not match manifest"
        )
    environment = record["environment"]
    if not isinstance(environment, Mapping) or any(
        environment.get(name) != value
        for name, value in identity["host"].items()
    ):
        raise CampaignIdentityError(
            "validation host identity does not match manifest"
        )
    _validate_runtime_metadata(record, cell, phase="validation")


def run_cell_subprocess(
    cell: Cell,
    *,
    root: Path,
    timeout_s: float = CELL_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute one fresh worker and retain failures as result envelopes."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")

    command = build_worker_command(root=root, cell=cell)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            cwd=Path(root),
            env=worker_environment(cell.platform),
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        diagnostic = _tail(error.stderr) or _tail(error.stdout)
        suffix = f": {diagnostic}" if diagnostic else ""
        return _failure_record(
            cell,
            kind="timeout",
            message=f"worker timed out after {timeout_s:g}s{suffix}",
        )
    except OSError as error:
        return _failure_record(
            cell,
            kind="launch_error",
            message=str(error),
        )

    if completed.returncode != 0:
        diagnostic = _tail(completed.stderr) or _tail(completed.stdout)
        return _failure_record(
            cell,
            kind="worker_exit",
            message=f"exit {completed.returncode}: {diagnostic}",
        )

    try:
        record = _parse_worker_output(completed.stdout)
        _validate_cell_record(record, cell)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        diagnostic = _tail(completed.stderr) or _tail(completed.stdout)
        return _failure_record(
            cell,
            kind="invalid_output",
            message=f"{error}; output: {diagnostic}",
        )
    return record


def run_validation_subprocess(
    cell: Cell,
    *,
    root: Path,
    identity: Mapping[str, Any],
    timeout_s: float = CELL_TIMEOUT_S,
) -> dict[str, Any]:
    """Execute one validation-only worker and validate its payload."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    if cell.correctness_replicates < 1:
        raise ValueError("validation requires at least one replicate")

    command = build_worker_command(
        root=root,
        cell=cell,
        phase="validation",
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            cwd=Path(root),
            env=worker_environment(cell.platform),
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        diagnostic = _tail(error.stderr) or _tail(error.stdout)
        suffix = f": {diagnostic}" if diagnostic else ""
        raise TimeoutError(
            f"validation timed out after {timeout_s:g}s{suffix}"
        ) from error
    except OSError as error:
        raise RuntimeError(f"validation launch failed: {error}") from error

    if completed.returncode != 0:
        diagnostic = _tail(completed.stderr) or _tail(completed.stdout)
        raise RuntimeError(
            f"validation worker exited {completed.returncode}: {diagnostic}"
        )
    try:
        record = _parse_worker_output(completed.stdout)
        _validate_validation_record(record, cell, identity)
    except CampaignIdentityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        diagnostic = _tail(completed.stderr) or _tail(completed.stdout)
        raise ValueError(
            f"invalid validation output: {error}; output: {diagnostic}"
        ) from error
    return record


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically create one JSON file without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_if_changed(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically update a mutable summary only when its content changed."""
    if path.exists() and _read_json(path) == value:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write the manifest once or require an identical existing plan."""
    if path.exists():
        if _read_json(path) != manifest:
            raise ValueError(
                "output directory contains a different profiling manifest"
            )
        return
    try:
        _write_json_exclusive(path, manifest)
    except FileExistsError:
        if _read_json(path) != manifest:
            raise ValueError(
                "output directory contains a different profiling manifest"
            ) from None


def _load_completed_result(
    path: Path,
    cell: Cell,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load an immutable completed cell, or return None when absent."""
    if not path.exists():
        return None
    try:
        record = _read_json(path)
        _validate_cell_record(record, cell)
        _validate_campaign_identity(
            record,
            identity,
            cell,
            require_validation_provenance=bool(cell.correctness_replicates),
        )
    except CampaignIdentityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            f"existing raw result is invalid; refusing to overwrite {path}: "
            f"{error}"
        ) from error
    return record


def _load_completed_validation(
    path: Path,
    cell: Cell,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load an immutable validation sidecar, or return None when absent."""
    if not path.exists():
        return None
    try:
        record = _read_json(path)
        _validate_validation_record(record, cell, identity)
    except CampaignIdentityError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(
            "existing validation result is invalid; refusing to overwrite "
            f"{path}: {error}"
        ) from error
    return record


def _validation_payload_from_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract a validation-only payload from an injected full result."""
    return {
        "backend": result["backend"],
        "block": result["block"],
        "correctness_level": result["correctness_level"],
        "correctness_replicates": result["correctness_replicates"],
        "dispatch_mode": result["dispatch_mode"],
        "environment": result["environment"],
        "execution_mode": result["execution_mode"],
        "parameters": result["parameters"],
        "platform_requested": result["platform_requested"],
        "replicated": result["correctness"]["replicated"],
        "schema_version": result["schema_version"],
        "source": result["source"],
        "versions": result["versions"],
        "workload": result["workload"],
    }


def _combine_timing_and_validation(
    timing: Mapping[str, Any],
    validation: Mapping[str, Any],
    cell: Cell,
) -> dict[str, Any]:
    """Combine disjoint timing and oracle work into one final raw record."""
    timing_environment = timing["environment"]
    validation_environment = validation["environment"]
    timing_device = (
        timing["backend"],
        timing_environment["device_id"],
        timing_environment["device_kind"],
    )
    validation_device = (
        validation["backend"],
        validation_environment["device_id"],
        validation_environment["device_kind"],
    )
    if timing_device != validation_device:
        raise ValueError("timing and validation device identities do not match")
    if timing["dispatch_mode"] != validation["dispatch_mode"]:
        raise ValueError("timing and validation dispatch modes do not match")
    result = copy.deepcopy(dict(timing))
    replicated = copy.deepcopy(dict(validation["replicated"]))
    correctness = copy.deepcopy(dict(result["correctness"]))
    correctness["replicated"] = replicated
    correctness["validation_provenance"] = {
        "backend": validation["backend"],
        "dispatch_mode": validation["dispatch_mode"],
        "environment": copy.deepcopy(validation_environment),
        "source": copy.deepcopy(validation["source"]),
        "versions": copy.deepcopy(validation["versions"]),
    }
    correctness["passed"] = bool(
        correctness.get("passed") and replicated["passed"]
    )
    result["correctness"] = correctness
    result["correctness_replicates"] = cell.correctness_replicates
    result["correctness_level"] = validation["correctness_level"]
    return result


def _promote_timing_failure(
    timing: Mapping[str, Any],
    cell: Cell,
) -> dict[str, Any]:
    """Restore the original schedule on a failed zero-replicate timing run."""
    result = copy.deepcopy(dict(timing))
    result["correctness_replicates"] = cell.correctness_replicates
    result["correctness_level"] = (
        WORKLOADS[cell.workload].replicated_correctness_level
        if cell.correctness_replicates
        else "structural"
    )
    return result


def _validation_failure(
    timing: Mapping[str, Any],
    cell: Cell,
    error: Exception,
) -> dict[str, Any]:
    """Preserve successful timing evidence while recording gate failure."""
    result = copy.deepcopy(dict(timing))
    correctness = copy.deepcopy(dict(result["correctness"]))
    correctness["passed"] = False
    correctness["replicated"] = {
        "completed_replicates": 0,
        "gate": "validation_failed",
        "passed": False,
        "replicates": cell.correctness_replicates,
    }
    result["correctness"] = correctness
    result["correctness_replicates"] = cell.correctness_replicates
    result["correctness_level"] = WORKLOADS[
        cell.workload
    ].replicated_correctness_level
    result["failure"] = {
        "kind": "validation_error",
        "message": f"{type(error).__name__}: {error}",
    }
    return result


def _resolve_order_seed(
    order_seed: int,
    legacy_seed: int | None,
) -> int:
    """Resolve the compatibility spelling without obscuring its semantics."""
    if legacy_seed is None:
        return order_seed
    if order_seed != DEFAULT_ORDER_SEED and order_seed != legacy_seed:
        raise ValueError("seed and order_seed disagree")
    return legacy_seed


def _validate_output_dir(root: Path, output_dir: Path) -> None:
    """Reject output that would mutate the attested implementation scope."""
    resolved_root = root.resolve()
    resolved_output = output_dir.resolve(strict=False)
    protected = (
        (resolved_root / "src/smcx").resolve(strict=False),
        (resolved_root / "benchmarks/profiling").resolve(strict=False),
    )
    if any(
        resolved_output == path or resolved_output.is_relative_to(path)
        for path in protected
    ):
        raise ValueError(
            "output_dir cannot be inside attested source directories"
        )


def _require_frozen_campaign_identity(identity: Mapping[str, Any]) -> None:
    """Fail before launch when source, host, or packages have changed."""
    if campaign_identity() != identity:
        raise CampaignIdentityError(
            "campaign identity changed after the manifest was frozen"
        )


def supervise(
    profile: str,
    *,
    platforms: Sequence[str] = PLATFORMS,
    root: Path,
    output_dir: Path,
    order_seed: int = DEFAULT_ORDER_SEED,
    seed: int | None = None,
    runner: Runner | None = None,
    validation_runner: ValidationRunner | None = None,
    timeout_s: float = CELL_TIMEOUT_S,
) -> dict[str, Any]:
    """Run one host-isolated, resumable profiling campaign."""
    with HostCampaignLock():
        return _supervise(
            profile,
            platforms=platforms,
            root=root,
            output_dir=output_dir,
            order_seed=order_seed,
            seed=seed,
            runner=runner,
            validation_runner=validation_runner,
            timeout_s=timeout_s,
        )


def _supervise(
    profile: str,
    *,
    platforms: Sequence[str] = PLATFORMS,
    root: Path,
    output_dir: Path,
    order_seed: int = DEFAULT_ORDER_SEED,
    seed: int | None = None,
    runner: Runner | None = None,
    validation_runner: ValidationRunner | None = None,
    timeout_s: float = CELL_TIMEOUT_S,
) -> dict[str, Any]:
    """Run all timing cells, then all validation, with resumable staging."""
    root = Path(root)
    output_dir = Path(output_dir)
    _validate_output_dir(root, output_dir)
    order_seed = _resolve_order_seed(order_seed, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = plan_cells(
        profile,
        platforms=platforms,
        order_seed=order_seed,
    )
    preflight = estimate_plan(cells, timeout_s=timeout_s)

    manifest = build_manifest(
        profile,
        cells,
        order_seed=order_seed,
        platforms=platforms,
    )
    manifest["preflight"] = preflight
    identity = manifest["campaign_identity"]
    if profile != "smoke" and identity["source"].get("git_dirty") is not False:
        raise ValueError(
            "inferential profiling requires a clean relevant source tree"
        )
    manifest_path = output_dir / "manifest.json"
    _ensure_manifest(manifest_path, manifest)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    timing_dir = output_dir / "timing"
    timing_dir.mkdir(exist_ok=True)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    timings: dict[str, dict[str, Any]] = {}

    # Phase one is deliberately matrix-wide. No replicated oracle work starts
    # until every missing cell has a completed immutable timing record.
    for cell in cells:
        name = raw_filename(cell)
        raw_path = raw_dir / name
        record = _load_completed_result(raw_path, cell, identity)
        if record is not None:
            completed[name] = record
            continue

        timing_cell = cell._replace(correctness_replicates=0)
        timing_path = timing_dir / name
        record = _load_completed_result(timing_path, timing_cell, identity)
        if record is None:
            _require_frozen_campaign_identity(identity)
            try:
                candidate = (
                    run_cell_subprocess(
                        timing_cell,
                        root=root,
                        timeout_s=timeout_s,
                    )
                    if runner is None
                    else dict(runner(timing_cell))
                )
                _validate_cell_record(candidate, timing_cell)
                _validate_campaign_identity(candidate, identity, timing_cell)
                record = candidate
            except CampaignIdentityError:
                raise
            except Exception as error:  # A failed cell must not end the matrix.
                record = _failure_record(
                    timing_cell,
                    kind="supervisor_error",
                    message=f"{type(error).__name__}: {error}",
                )
                _validate_cell_record(record, timing_cell)
            try:
                _write_json_exclusive(timing_path, record)
            except FileExistsError as error:
                record = _load_completed_result(
                    timing_path,
                    timing_cell,
                    identity,
                )
                if record is None:  # Defensive: it existed one line ago.
                    raise RuntimeError(
                        "timing result disappeared during write"
                    ) from error
        timings[name] = record

    # Phase two can no longer perturb a timing observation: all measurements
    # are complete before any expensive replicated oracle validation begins.
    results: list[dict[str, Any]] = []
    for cell in cells:
        name = raw_filename(cell)
        raw_path = raw_dir / name
        record = completed.get(name)
        if record is not None:
            results.append(record)
            continue

        timing = timings[name]
        if timing["failure"] is not None:
            record = _promote_timing_failure(timing, cell)
        elif cell.correctness_replicates == 0:
            record = timing
        else:
            validation_path = validation_dir / name
            validation = _load_completed_validation(
                validation_path,
                cell,
                identity,
            )
            try:
                if validation is None:
                    _require_frozen_campaign_identity(identity)
                    if validation_runner is not None:
                        validation = dict(validation_runner(cell))
                    elif runner is None:
                        validation = run_validation_subprocess(
                            cell,
                            root=root,
                            identity=identity,
                            timeout_s=timeout_s,
                        )
                    else:
                        injected = dict(runner(cell))
                        _validate_cell_record(injected, cell)
                        _validate_campaign_identity(injected, identity, cell)
                        validation = _validation_payload_from_result(injected)
                    _validate_validation_record(validation, cell, identity)
                    try:
                        _write_json_exclusive(validation_path, validation)
                    except FileExistsError as error:
                        validation = _load_completed_validation(
                            validation_path,
                            cell,
                            identity,
                        )
                        if validation is None:
                            raise RuntimeError(
                                "validation result disappeared during write"
                            ) from error
                record = _combine_timing_and_validation(
                    timing,
                    validation,
                    cell,
                )
            except CampaignIdentityError:
                raise
            except Exception as error:
                record = _validation_failure(timing, cell, error)

        _validate_cell_record(record, cell)
        _validate_campaign_identity(
            record,
            identity,
            cell,
            require_validation_provenance=bool(cell.correctness_replicates),
        )
        try:
            _write_json_exclusive(raw_path, record)
        except FileExistsError as error:
            record = _load_completed_result(raw_path, cell, identity)
            if record is None:  # Defensive: the link existed one line ago.
                raise RuntimeError(
                    "raw result disappeared during write"
                ) from error
        results.append(record)

    failed = sum(
        result["failure"] is not None or not result["correctness"]["passed"]
        for result in results
    )
    summary = {
        "cells": len(cells),
        "completed": len(results),
        "failed": failed,
        "manifest_path": str(manifest_path),
        "platforms": list(platforms),
        "preflight": preflight,
        "profile": profile,
        "raw_dir": str(raw_dir),
        "order_seed": order_seed,
    }
    _write_json_if_changed(output_dir / "summary.json", summary)
    return summary


def _write_dry_run(
    profile: str,
    *,
    platforms: Sequence[str],
    root: Path,
    output_dir: Path,
    order_seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    """Persist a manifest without starting any workers."""
    _validate_output_dir(root, output_dir)
    cells = plan_cells(
        profile,
        platforms=platforms,
        order_seed=order_seed,
    )
    preflight = estimate_plan(cells, timeout_s=timeout_s)
    manifest = build_manifest(
        profile,
        cells,
        order_seed=order_seed,
        platforms=platforms,
    )
    manifest["preflight"] = preflight
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_manifest(output_dir / "manifest.json", manifest)
    return {
        "cells": len(cells),
        "dry_run": True,
        "preflight": preflight,
        "platforms": list(platforms),
        "profile": profile,
        "order_seed": order_seed,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the profiling supervisor from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the current-JAX all-algorithm profiling matrix."
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument(
        "--platforms",
        choices=PLATFORMS,
        default=list(PLATFORMS),
        nargs="+",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--order-seed",
        "--seed",
        dest="order_seed",
        default=DEFAULT_ORDER_SEED,
        type=int,
    )
    parser.add_argument("--timeout-s", default=CELL_TIMEOUT_S, type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    if args.dry_run:
        summary = _write_dry_run(
            args.profile,
            platforms=args.platforms,
            root=root,
            output_dir=args.output_dir,
            order_seed=args.order_seed,
            timeout_s=args.timeout_s,
        )
    else:
        summary = supervise(
            args.profile,
            platforms=args.platforms,
            root=root,
            output_dir=args.output_dir,
            order_seed=args.order_seed,
            timeout_s=args.timeout_s,
        )
    print(json.dumps(summary, sort_keys=True))
    return int(not args.dry_run and summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
