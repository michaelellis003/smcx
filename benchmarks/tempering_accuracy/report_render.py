# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic public evidence for the tempering campaign."""

import gzip
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any, NamedTuple

import numpy as np

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.plan import (
    cell_id,
    centering_summary_count,
    current_cells,
    matched_cells,
)
from benchmarks.tempering_accuracy.report_accuracy import CampaignReport
from benchmarks.tempering_accuracy.report_data import campaign_requests

_HOST_FIELDS = (
    "cpu_count",
    "cpu_model",
    "hardware_model",
    "machine",
    "macos",
    "macos_build",
    "os",
    "os_release",
    "physical_memory_bytes",
    "processor",
)


class AttemptEvidence(NamedTuple):
    """Sanitized identity of one retained retryable launch failure."""

    request_index: int
    retry_index: int
    sha256: str
    kind: str


def _jsonable(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, Mapping):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _kind(value: object) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError("failure kind is not a safe identifier")
    return value


def _attempts(
    values: Sequence[AttemptEvidence] | None,
) -> tuple[AttemptEvidence, ...]:
    if values is None:
        raise ValueError("a verified attempt inventory is required")
    attempts = tuple(values)
    keys = [(item.request_index, item.retry_index) for item in attempts]
    valid = keys == sorted(set(keys)) and all(
        type(item.request_index) is int
        and 0 <= item.request_index < 508
        and type(item.retry_index) is int
        and item.retry_index >= 0
        and len(item.sha256) == 64
        and all(character in "0123456789abcdef" for character in item.sha256)
        and _kind(item.kind)
        for item in attempts
    )
    if not valid:
        raise ValueError("attempt inventory is invalid")
    return attempts


def _safe_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    source, python, host = (
        identity["source"],
        identity["python"],
        identity["host"],
    )
    return {
        "git_commit": source["git_commit"],
        "git_dirty": source["git_dirty"],
        "packages": _jsonable(identity["packages"]),
        "python": {
            "implementation": python["implementation"],
            "version": python["version"],
        },
        "host": {name: _jsonable(host.get(name)) for name in _HOST_FIELDS},
    }


def _failures(report: CampaignReport) -> list[dict[str, Any]]:
    failures = []
    for ordinal, (request, result) in enumerate(
        zip(campaign_requests(), report.campaign.results, strict=False)
    ):
        failure = result.get("failure")
        if not isinstance(failure, Mapping):
            continue
        failures.append({
            "ordinal": ordinal,
            "phase": request.phase,
            "cell_id": cell_id(request.cell),
            "block": request.block,
            "kind": _kind(failure.get("kind")),
        })
    return failures


def build_evidence(
    report: CampaignReport,
    *,
    attempts: Sequence[AttemptEvidence] | None = None,
) -> dict[str, Any]:
    """Build the canonical, path-free public evidence document."""
    retained_attempts = _attempts(attempts)
    expected = (*current_cells(), *matched_cells())
    registered_gates = sum(centering_summary_count(cell) for cell in expected)
    if tuple(item.cell for item in report.cells) != expected:
        raise ValueError("report cells do not match the registered campaign")
    campaign = report.campaign
    if campaign.complete and (
        len(campaign.inventory) != 508 or len(campaign.results) != 508
    ):
        raise ValueError("complete campaign must contain all 508 results")
    identity = campaign.manifest["campaign_identity"]
    analyses = [item.analysis for item in report.cells if item.analysis]
    gates = [
        gate
        for analysis in analyses
        for gate in (
            *analysis.mean_gates,
            *analysis.covariance_gates,
            analysis.evidence_gate,
        )
    ]
    current = report.cells[: len(current_cells())]
    verdict = (
        "incomplete"
        if not campaign.complete
        else (
            "current_rwm_eligible"
            if all(item.status == "eligible" for item in current)
            else "current_rwm_not_eligible"
        )
    )
    return {
        "schema_version": 1,
        "verdict": verdict,
        "integrity": {
            "manifest_sha256": campaign.manifest_sha256,
            "plan_sha256": campaign.manifest["plan_sha256"],
            "source_sha256": identity["source"]["sha256"],
            "lock_sha256": identity["lock"]["sha256"],
            "raw_sha256": campaign.raw_sha256,
            "raw_leaves": [
                {"ordinal": item.ordinal, "sha256": item.sha256}
                for item in campaign.inventory
            ],
        },
        "environment": _safe_identity(identity),
        "algorithm_contract": _jsonable(
            campaign.manifest["algorithm_contract"]
        ),
        "execution": {
            "complete": campaign.complete,
            "result_count": len(campaign.results),
            "not_run_after_stop": campaign.not_run_after_stop,
        },
        "gate_counts": {
            "centering": {
                "passed": sum(gate.passed for gate in gates),
                "evaluated": len(gates),
                "registered": registered_gates,
            },
            "evidence_resolution": {
                "passed": sum(
                    analysis.evidence_resolution_width <= 0.10
                    for analysis in analyses
                ),
                "evaluated": len(analyses),
                "registered": len(expected),
            },
        },
        "status_counts": dict(
            sorted(Counter(item.status for item in report.cells).items())
        ),
        "cells": [
            {
                "id": cell_id(item.cell),
                "cell": _jsonable(item.cell),
                "status": item.status,
                "timing": _jsonable(item.timing),
                "accuracy": _jsonable(item.analysis),
                "work": _jsonable(item.work),
            }
            for item in report.cells
        ],
        "failures": _failures(report),
        "attempts": _jsonable(retained_attempts),
        "exclusions": _jsonable(campaign.manifest["exclusions"]),
    }


def evidence_gzip(
    report: CampaignReport,
    *,
    attempts: Sequence[AttemptEvidence] | None = None,
) -> bytes:
    """Return reproducible gzip bytes containing canonical JSON evidence."""
    encoded = (
        canonical_json(build_evidence(report, attempts=attempts)) + "\n"
    ).encode()
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as file:
        file.write(encoded)
    return output.getvalue()
