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
    bind_request,
    build_manifest,
    campaign_identity,
    campaign_requests,
    ensure_manifest,
    load_raw_result,
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


class CampaignError(RuntimeError):
    """Raised when registered campaign execution must stop."""


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
    output_dir = Path(output_dir)
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
        for request in requests[completed:]:
            attempt = run_worker(
                _ROOT,
                bind_request(request, digest),
                timeout_s=timeout_s,
            )
            if attempt.retryable:
                return 1
            write_raw_result(output_dir, request, digest, attempt.payload)
            if attempt.payload["failure"] is not None:
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
