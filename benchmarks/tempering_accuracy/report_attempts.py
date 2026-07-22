# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Strict loading and sanitization of retryable campaign attempts."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import (
    SCHEMA_VERSION,
    bind_request,
    request_dict,
)
from benchmarks.tempering_accuracy.report_data import campaign_requests

_FIELDS = {
    "schema_version",
    "request_index",
    "retry_index",
    "request",
    "failure",
}
_RETRYABLE_KINDS = {
    "launch_error",
    "metal_prelaunch_ineligible",
    "campaign_identity_changed_before_launch",
}


class AttemptEvidence(NamedTuple):
    """Sanitized digest and identity of one retryable attempt."""

    request_index: int
    retry_index: int
    sha256: str
    kind: str


class AttemptInventory(NamedTuple):
    """Aggregate digest of an ordered sanitized attempt inventory."""

    sha256: str
    entries: tuple[AttemptEvidence, ...]


def _is_digest(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _failure_schema(value: object) -> str | None:
    if not isinstance(value, dict) or type(value.get("kind")) is not str:
        return None
    failure = cast(dict[str, Any], value)
    kind = cast(str, failure["kind"])
    if kind not in _RETRYABLE_KINDS:
        return None
    exception_fields = {"kind", "exception_type", "message"}
    exception_variant = kind in {
        "launch_error",
        "campaign_identity_changed_before_launch",
    }
    if exception_variant and set(failure) == exception_fields:
        valid = set(failure) == exception_fields and all(
            type(failure[name]) is str for name in ("exception_type", "message")
        )
        return kind if valid else None
    if kind == "launch_error":
        return None
    if kind == "metal_prelaunch_ineligible":
        snapshot = failure.get("prelaunch")
        valid = (
            set(failure) == {"kind", "prelaunch"}
            and isinstance(snapshot, dict)
            and set(snapshot) == {"power_status", "thermal_status"}
            and all(
                item is None or type(item) is str for item in snapshot.values()
            )
        )
        return kind if valid else None
    digest_fields = {
        "kind",
        "expected_source_sha256",
        "observed_source_sha256",
    }
    valid = set(failure) == digest_fields and all(
        _is_digest(failure[name]) for name in digest_fields - {"kind"}
    )
    return kind if valid else None


def _document(path: Path) -> tuple[dict[str, Any], bytes]:
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded)
        canonical = (
            isinstance(value, dict)
            and encoded == (canonical_json(value) + "\n").encode()
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid attempt document: {path.name}") from error
    if not canonical:
        raise ValueError(f"attempt document is not canonical: {path.name}")
    return value, encoded


def _load_entry(
    path: Path,
    manifest_sha256: str,
    requests: Sequence[Any],
) -> AttemptEvidence:
    value, encoded = _document(path)
    request_index = value.get("request_index")
    retry_index = value.get("retry_index")
    valid_indices = (
        type(request_index) is int
        and 0 <= request_index < len(requests)
        and type(retry_index) is int
        and retry_index >= 0
    )
    if not valid_indices:
        raise ValueError(f"attempt indices are invalid: {path.name}")
    expected_name = f"{request_index:03d}-{retry_index:03d}.json"
    kind = _failure_schema(value.get("failure"))
    expected_request = request_dict(
        bind_request(requests[request_index], manifest_sha256)
    )
    valid = (
        set(value) == _FIELDS
        and type(value.get("schema_version")) is int
        and value["schema_version"] == SCHEMA_VERSION
        and path.name == expected_name
        and canonical_json(value.get("request"))
        == canonical_json(expected_request)
        and kind is not None
    )
    if not valid:
        raise ValueError(f"attempt does not match its identity: {path.name}")
    return AttemptEvidence(
        request_index,
        retry_index,
        hashlib.sha256(encoded).hexdigest(),
        kind,
    )


def _inventory(entries: Sequence[AttemptEvidence]) -> AttemptInventory:
    ordered = tuple(sorted(entries, key=lambda item: item[:2]))
    retries: dict[int, list[int]] = defaultdict(list)
    for entry in ordered:
        retries[entry.request_index].append(entry.retry_index)
    if any(values != list(range(len(values))) for values in retries.values()):
        raise ValueError("attempt retry indices must be contiguous")
    sanitized = [entry._asdict() for entry in ordered]
    digest = hashlib.sha256(canonical_json(sanitized).encode()).hexdigest()
    return AttemptInventory(digest, ordered)


def load_attempts(output_dir: Path, manifest_sha256: str) -> AttemptInventory:
    """Load a strict manifest-bound inventory without sensitive details."""
    if not _is_digest(manifest_sha256):
        raise ValueError("attempt manifest digest is invalid")
    directory = Path(output_dir) / "attempts"
    if not directory.exists():
        return _inventory(())
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("unexpected attempts directory entry")
    paths = tuple(directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("unexpected attempts directory entry")
    requests = campaign_requests()
    if len(requests) != 508:
        raise ValueError("attempt request plan is not registered")
    return _inventory([
        _load_entry(path, manifest_sha256, requests) for path in paths
    ])
