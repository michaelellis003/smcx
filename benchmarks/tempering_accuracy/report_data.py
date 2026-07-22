# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Strict loading of immutable tempering-accuracy campaign artifacts."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.artifacts import (
    CampaignRequest,
    load_raw_result,
    raw_filename,
    request_dict,
)
from benchmarks.tempering_accuracy.plan import (
    ORDER_SEED,
    current_cells,
    current_smoke_cells,
    matched_cells,
    timing_blocks,
    waste_free_cells,
    waste_free_smoke_cells,
)

_PLAN_SHA256 = (
    "ce573478ea79bd5b8cca7bf2d73c164e1a55ea784342996627c9fe01f55e1ca9"
)
_MANIFEST_FIELDS = {
    "schema_version",
    "campaign",
    "order_seed",
    "algorithm_contract",
    "plan_sha256",
    "requests",
    "exclusions",
    "campaign_identity",
}
_IDENTITY_FIELDS = {"source", "lock", "packages", "python", "host"}
_ALGORITHM_CONTRACT = {
    "proposal_covariance_source": "weighted_pre_resample_cloud",
    "proposal_scale": "2.38^2 / dimension",
    "target_ess": 0.5,
}


class InventoryEntry(NamedTuple):
    """Digest of one raw result's exact bytes."""

    ordinal: int
    filename: str
    sha256: str


class CampaignData(NamedTuple):
    """Validated manifest, immutable results, and digest inventory."""

    manifest: dict[str, Any]
    manifest_sha256: str
    raw_sha256: str
    complete: bool
    not_run_after_stop: tuple[int, int] | None
    stopping_failure: Mapping[str, Any] | None
    inventory: tuple[InventoryEntry, ...]
    results: tuple[dict[str, Any], ...]


def campaign_requests() -> tuple[CampaignRequest, ...]:
    """Independently reconstruct the exact 508-request execution plan."""
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


def _exclusion() -> dict[str, Any]:
    return {
        "arm": "waste_free_multinomial",
        "status": "blocked_backend_correctness",
        "tracking_issue": 38,
        "blocked_request_counts": {"smoke": 2, "timing": 60, "accuracy": 12},
        "smoke_cells": [cell._asdict() for cell in waste_free_smoke_cells()],
        "cells": [cell._asdict() for cell in waste_free_cells()],
    }


def _load_manifest(output_dir: Path) -> tuple[dict[str, Any], str]:
    path = output_dir / "manifest.json"
    if path.is_symlink():
        raise ValueError("manifest must not be a symlink")
    encoded = path.read_bytes()
    try:
        manifest = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("manifest is not canonical JSON") from error
    requests = campaign_requests()
    expected = [request_dict(request) for request in requests]
    plan_sha256 = hashlib.sha256(canonical_json(expected).encode()).hexdigest()
    valid = (
        isinstance(manifest, dict)
        and encoded == (canonical_json(manifest) + "\n").encode()
        and set(manifest) == _MANIFEST_FIELDS
        and type(manifest["schema_version"]) is int
        and manifest["schema_version"] == 1
        and manifest["campaign"] == "tempering_accuracy"
        and manifest["order_seed"] == ORDER_SEED
        and canonical_json(manifest["algorithm_contract"])
        == canonical_json(_ALGORITHM_CONTRACT)
        and len(requests) == 508
        and plan_sha256 == _PLAN_SHA256 == manifest["plan_sha256"]
        and canonical_json(manifest["requests"]) == canonical_json(expected)
        and canonical_json(manifest["exclusions"])
        == canonical_json([_exclusion()])
        and isinstance(manifest["campaign_identity"], dict)
        and set(manifest["campaign_identity"]) == _IDENTITY_FIELDS
    )
    if not valid:
        raise ValueError("manifest does not match the registered campaign")
    return manifest, hashlib.sha256(encoded).hexdigest()


def _raw_names(raw_dir: Path) -> set[str]:
    if not raw_dir.exists():
        return set()
    if raw_dir.is_symlink():
        raise ValueError("raw directory must not be a symlink")
    paths = tuple(raw_dir.iterdir()) if raw_dir.is_dir() else ()
    if any(path.is_symlink() for path in paths):
        raise ValueError("raw directory contains a symlink")
    if not raw_dir.is_dir() or any(not path.is_file() for path in paths):
        raise ValueError("raw directory contains an unexpected artifact")
    return {path.name for path in paths}


def load_campaign(output_dir: Path) -> CampaignData:
    """Load a complete campaign or its exact contiguous execution prefix."""
    output = Path(output_dir)
    manifest, digest = _load_manifest(output)
    requests = campaign_requests()
    filenames = [raw_filename(request) for request in requests]
    actual = _raw_names(output / "raw")
    if not actual <= set(filenames):
        raise ValueError("raw directory contains an unexpected artifact")
    present = [filename in actual for filename in filenames]
    count = sum(present)
    if present != [True] * count + [False] * (len(requests) - count):
        raise ValueError("raw results must form a contiguous prefix")

    inventory = []
    results = []
    for ordinal, request in enumerate(requests[:count]):
        path = output / "raw" / filenames[ordinal]
        result = load_raw_result(output, request, digest)
        assert result is not None
        results.append(result)
        inventory.append(
            InventoryEntry(
                ordinal,
                path.name,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    if any(result["failure"] is not None for result in results[:-1]):
        raise ValueError("raw results continue after a terminal failure")
    frozen_inventory = tuple(inventory)
    raw_sha256 = hashlib.sha256(
        canonical_json([item._asdict() for item in frozen_inventory]).encode()
    ).hexdigest()
    return CampaignData(
        manifest,
        digest,
        raw_sha256,
        count == len(requests),
        None if count == len(requests) else (count, len(requests) - 1),
        results[-1].get("failure") if results else None,
        frozen_inventory,
        tuple(results),
    )
