# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""JAX-free host supervisor for the tempering-accuracy campaign."""

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from benchmarks.profiling.common import _command_value
from benchmarks.profiling.locking import HostCampaignLock
from benchmarks.tempering_accuracy.artifacts import (
    SCHEMA_VERSION,
    _write_exclusive,
    bind_request,
    build_manifest,
    campaign_identity,
    campaign_requests,
    ensure_manifest,
    load_raw_result,
    request_dict,
    write_raw_result,
)
from benchmarks.tempering_accuracy.transport import run_worker

DEFAULT_TIMEOUT_S = 7_200.0
_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_PACKAGES = (
    "jax",
    "jax-mps",
    "jaxlib",
    "ml-dtypes",
    "numpy",
    "scipy",
    "smcx",
)
_CAPTURE_LIMIT = 4_096


class CampaignError(RuntimeError):
    """Raised when registered campaign execution must stop."""


class _RetryableError(RuntimeError):
    """Carry bounded evidence for a launch that may be retried later."""

    def __init__(self, failure: Mapping[str, Any]) -> None:
        self.failure = dict(failure)


def _bounded(value: Any) -> Any:
    """Bound strings in retained supervisor evidence."""
    if isinstance(value, str):
        return value[-_CAPTURE_LIMIT:]
    if isinstance(value, Mapping):
        return {str(name): _bounded(item) for name, item in value.items()}
    return value


def _prelaunch_snapshot() -> dict[str, str | None]:
    """Require and capture eligible Metal power and thermal state."""
    snapshot = {
        "power_status": _command_value(("pmset", "-g", "batt")),
        "thermal_status": _command_value(("pmset", "-g", "therm")),
    }
    power = snapshot["power_status"] or ""
    thermal = snapshot["thermal_status"] or ""
    eligible = (
        "Now drawing from 'AC Power'" in power
        and "No thermal warning level" in thermal
        and "No performance warning level" in thermal
    )
    if not eligible:
        raise _RetryableError({
            "kind": "metal_prelaunch_ineligible",
            "prelaunch": _bounded(snapshot),
        })
    return cast(dict[str, str | None], _bounded(snapshot))


def _identity_failure(
    expected: Mapping[str, Any], kind: str
) -> dict[str, Any] | None:
    """Return bounded evidence when exact campaign identity changed."""
    try:
        observed = campaign_identity(_ROOT)
    except Exception as error:
        return {
            "kind": kind,
            "exception_type": type(error).__name__,
            "message": _bounded(str(error)),
        }
    if observed == expected:
        return None
    return {
        "kind": kind,
        "expected_source_sha256": expected.get("source", {}).get("sha256"),
        "observed_source_sha256": observed.get("source", {}).get("sha256"),
    }


def _write_attempt(
    output_dir: Path,
    request_index: int,
    bound: Any,
    failure: Mapping[str, Any],
) -> None:
    """Exclusively retain one manifest-bound retryable attempt."""
    retry_index = 0
    while True:
        record = {
            "schema_version": SCHEMA_VERSION,
            "request_index": request_index,
            "retry_index": retry_index,
            "request": request_dict(bound),
            "failure": _bounded(failure),
        }
        path = (
            output_dir
            / "attempts"
            / f"{request_index:03d}-{retry_index:03d}.json"
        )
        try:
            _write_exclusive(path, record)
            return
        except FileExistsError:
            retry_index += 1


def _raw_failure(
    bound: Any,
    failure: Mapping[str, Any],
    prelaunch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build one immutable manifest-bound supervisor failure."""
    evidence = dict(_bounded(failure))
    if prelaunch is not None:
        evidence["prelaunch"] = dict(prelaunch)
    return {
        "schema_version": SCHEMA_VERSION,
        "request": request_dict(bound),
        "failure": evidence,
        "timing": None,
        "runs": [],
    }


def _retain_prelaunch(
    payload: Mapping[str, Any], snapshot: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Attach a supervisor boundary snapshot to a worker payload."""
    result = dict(payload)
    if snapshot is None:
        return result
    if isinstance(result.get("timing"), Mapping):
        timing = dict(result["timing"])
        environment = dict(timing.get("environment", {}))
        environment["supervisor_prelaunch"] = dict(snapshot)
        timing["environment"] = environment
        result["timing"] = timing
    elif isinstance(result.get("failure"), Mapping):
        failure = dict(result["failure"])
        failure["prelaunch"] = dict(snapshot)
        result["failure"] = failure
    return result


def _virtualization_status() -> str | None:
    """Return the macOS hypervisor-presence flag without importing JAX."""
    return _command_value(("sysctl", "-n", "kern.hv_vmm_present"))


def _filled(value: object) -> bool:
    """Return whether an identity field is a nonempty string."""
    return isinstance(value, str) and bool(value)


def _require_preflight(identity: Mapping[str, Any]) -> None:
    """Require a clean, complete identity on a physical Apple host."""
    names = ("source", "lock", "packages", "python", "host")
    values = tuple(identity.get(name) for name in names)
    if any(not isinstance(value, Mapping) for value in values):
        raise CampaignError("campaign identity is incomplete")
    maps = [cast(Mapping[str, Any], value) for value in values]
    source, lock, packages, python, host = maps
    source_fields = (
        source.get("git_commit"),
        source.get("sha256"),
        lock.get("sha256"),
        python.get("implementation"),
        python.get("version"),
        python.get("executable"),
    )
    files = source.get("files")
    if (
        not all(_filled(value) for value in source_fields)
        or not isinstance(files, list)
        or not files
        or not all(_filled(path) for path in files)
    ):
        raise CampaignError("campaign identity is incomplete")
    if source.get("git_dirty") is not False:
        raise CampaignError("campaign source is not clean")
    if any(not _filled(packages.get(name)) for name in _REQUIRED_PACKAGES):
        raise CampaignError("required campaign packages are unavailable")
    physical_host = (
        host.get("os") == "Darwin"
        and host.get("machine") == "arm64"
        and str(host.get("cpu_model", "")).startswith("Apple ")
        and str(host.get("hardware_model", "")).startswith("Mac")
    )
    if not physical_host or _virtualization_status() != "0":
        raise CampaignError("campaign requires a physical Apple-silicon Mac")


def _resume_prefix(
    output_dir: Path, requests: tuple[Any, ...], digest: str
) -> tuple[int, bool]:
    """Validate immutable results and return the complete prefix length."""
    results = [
        load_raw_result(output_dir, request, digest) for request in requests
    ]
    missing = next(
        (index for index, result in enumerate(results) if result is None),
        len(results),
    )
    if any(result is not None for result in results[missing + 1 :]):
        raise CampaignError("raw campaign results contain a gap")
    failed = [
        index
        for index, result in enumerate(results[:missing])
        if result is not None and result["failure"] is not None
    ]
    if failed and failed != [missing - 1]:
        raise CampaignError("raw results continue after a failure")
    return missing, bool(failed)


def run_campaign(
    output_dir: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    dry_run: bool = False,
) -> int:
    """Run or resume the single frozen campaign in manifest order."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive and finite")
    output_dir = Path(output_dir).resolve()
    if (
        output_dir == _ROOT
        or _ROOT in output_dir.parents
        or output_dir in _ROOT.parents
    ):
        raise CampaignError("output_dir overlaps the attested source tree")
    with HostCampaignLock():
        try:
            identity = campaign_identity(_ROOT)
        except Exception as error:
            raise CampaignError("campaign identity is unavailable") from error
        _require_preflight(identity)
        manifest = build_manifest(_ROOT)
        if manifest.get("campaign_identity") != identity:
            raise CampaignError("campaign identity changed during preflight")
        if dry_run:
            return 0
        digest = ensure_manifest(output_dir, manifest)
        requests = campaign_requests()
        completed, failed = _resume_prefix(output_dir, requests, digest)
        if failed:
            return 1
        for index, request in enumerate(requests[completed:], start=completed):
            bound = bind_request(request, digest)
            snapshot = None
            try:
                if request.phase == "timing" and request.cell.lane == "mps_f32":
                    snapshot = _prelaunch_snapshot()
                before = _identity_failure(
                    identity, "campaign_identity_changed_before_launch"
                )
                if before is not None:
                    raise _RetryableError(before)
                try:
                    attempt = run_worker(_ROOT, bound, timeout_s=timeout_s)
                except OSError as error:
                    raise _RetryableError({
                        "kind": "launch_error",
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }) from error
            except _RetryableError as error:
                _write_attempt(output_dir, index, bound, error.failure)
                return 1
            if attempt.retryable:
                failure = attempt.payload.get("failure")
                evidence = (
                    failure
                    if isinstance(failure, Mapping)
                    else {"kind": "launch_error"}
                )
                _write_attempt(output_dir, index, bound, evidence)
                return 1
            after = _identity_failure(
                identity, "source_identity_changed_after_launch"
            )
            payload = (
                _raw_failure(bound, after, snapshot)
                if after is not None
                else _retain_prelaunch(attempt.payload, snapshot)
            )
            write_raw_result(output_dir, request, digest, payload)
            if payload["failure"] is not None:
                return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the frozen campaign from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    return run_campaign(
        arguments.output_dir,
        timeout_s=arguments.timeout_s,
        dry_run=arguments.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
