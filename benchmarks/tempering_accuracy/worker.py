# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process standard-arm worker for issue #30."""

from typing import Any, NamedTuple

from benchmarks.tempering_accuracy.plan import (
    CampaignCell,
    current_cells,
    current_smoke_cells,
    matched_cells,
)

SCHEMA_VERSION = 1


class WorkerRequest(NamedTuple):
    """One manifest-bound worker invocation."""

    manifest_sha256: str
    phase: str
    cell: CampaignCell
    block: int | None


def _request_dict(request: WorkerRequest) -> dict[str, object]:
    return {
        "manifest_sha256": request.manifest_sha256,
        "phase": request.phase,
        "cell": request.cell._asdict(),
        "block": request.block,
    }


def _validate_request(request: WorkerRequest) -> None:
    digest = request.manifest_sha256
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("manifest_sha256 must be 64 lowercase hex characters")
    standards = (*current_cells(), *matched_cells())
    if request.phase == "smoke":
        valid = request.cell in current_smoke_cells() and request.block is None
    elif request.phase == "timing":
        valid = (
            request.cell in standards
            and isinstance(request.block, int)
            and not isinstance(request.block, bool)
            and 0 <= request.block < 5
        )
    elif request.phase == "accuracy":
        valid = request.cell in standards and request.block is None
    else:
        raise ValueError(f"unknown phase: {request.phase}")
    if not valid:
        raise ValueError("request is not a registered phase/cell/block")


def execute_request(request: WorkerRequest) -> dict[str, Any]:
    """Execute one request and retain validation or runtime failures."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request": _request_dict(request),
        "failure": None,
        "timing": None,
        "runs": [],
    }
    try:
        _validate_request(request)
    except ValueError as error:
        payload["failure"] = {
            "kind": "invalid_request",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        return payload
    raise NotImplementedError("registered worker execution is not implemented")
