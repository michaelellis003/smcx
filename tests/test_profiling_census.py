# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""StableHLO census contracts for profiling phase attribution."""

import json

import jax
import pytest

import benchmarks.profiling.census as census_module
from benchmarks.profiling.census import build_census
from benchmarks.profiling.common import plan_cells
from benchmarks.profiling.run import raw_filename


def _smoke_cell(workload: str, platform: str = "cpu"):
    return next(
        cell
        for cell in plan_cells("smoke", platforms=(platform,), seed=20260719)
        if cell.workload == workload
    )


def _eligible_record(cell) -> dict:
    """Return the exact cell identity fields consumed by target loading."""
    return {
        "block": cell.block,
        "correctness": {"passed": True},
        "correctness_replicates": cell.correctness_replicates,
        "execution_mode": cell.execution_mode,
        "failure": None,
        "parameters": cell.parameters,
        "platform_requested": cell.platform,
        "repeats": cell.repeats,
        "warmups": cell.warmups,
        "workload": cell.workload,
    }


def test_census_counts_stablehlo_operations_for_outer_jit() -> None:
    platform = jax.default_backend()
    x64_before = jax.config.read("jax_enable_x64")
    census = build_census(_smoke_cell("bootstrap_lgssm", platform))
    assert jax.config.read("jax_enable_x64") is x64_before
    assert census["workload"] == "bootstrap_lgssm"
    assert census["platform"] == platform
    assert census["total_operations"] > 0
    assert census["operation_counts"]["while"] >= 1
    assert len(census["stablehlo_sha256"]) == 64
    assert census["stablehlo_bytes"] > 0
    assert census["environment"]["device_kind"]
    assert "JAX_ENABLE_X64" in census["environment"]["runtime_flags"]


def test_census_rejects_host_controlled_public_api() -> None:
    with pytest.raises(ValueError, match="whole_program_jit"):
        build_census(_smoke_cell("temper_gaussian"))


def test_census_targets_require_non_smoke_eligible_campaign(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        census_module,
        "build_report",
        lambda campaign_dir: {"profile": "smoke"},
    )
    with pytest.raises(ValueError, match="non-smoke"):
        census_module.load_census_targets(tmp_path, platform="cpu")


def test_census_targets_are_exact_passing_raw_members(
    tmp_path, monkeypatch
) -> None:
    cell = _smoke_cell("bootstrap_lgssm")
    manifest = {
        "campaign_identity": {"host": {}, "packages": {}, "source": {}},
        "cells": [cell._asdict()],
        "exclusions": [{"workload": "optional"}],
        "plan_sha256": "a" * 64,
        "profile": "baseline",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / raw_filename(cell)
    raw_path.write_text(json.dumps(_eligible_record(cell)))
    monkeypatch.setattr(census_module, "validate_result", lambda record: None)
    monkeypatch.setattr(
        census_module,
        "build_report",
        lambda campaign_dir: {
            "aggregates": [
                {
                    "execution_mode": cell.execution_mode,
                    "parameters": cell.parameters,
                    "platform": cell.platform,
                    "repeats": cell.repeats,
                    "timing_eligible": True,
                    "warmups": cell.warmups,
                    "workload": cell.workload,
                }
            ],
            "profile": "baseline",
        },
    )
    target = census_module.load_census_targets(tmp_path, platform="cpu")
    assert target["cells"] == [cell]
    assert target["exclusions"] == [{"workload": "optional"}]
    assert len(target["manifest_sha256"]) == 64
    assert len(target["raw_sha256"][raw_filename(cell)]) == 64


def test_census_target_rejects_swapped_canonical_raw_content(
    tmp_path, monkeypatch
) -> None:
    cell = _smoke_cell("bootstrap_lgssm")
    other = _smoke_cell("auxiliary_lgssm")
    manifest = {
        "campaign_identity": {"host": {}, "packages": {}, "source": {}},
        "cells": [cell._asdict()],
        "exclusions": [],
        "plan_sha256": "a" * 64,
        "profile": "baseline",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / raw_filename(cell)).write_text(
        json.dumps(_eligible_record(other))
    )
    monkeypatch.setattr(census_module, "validate_result", lambda record: None)
    monkeypatch.setattr(
        census_module,
        "build_report",
        lambda campaign_dir: {
            "aggregates": [
                {
                    "execution_mode": cell.execution_mode,
                    "parameters": cell.parameters,
                    "platform": cell.platform,
                    "repeats": cell.repeats,
                    "timing_eligible": True,
                    "warmups": cell.warmups,
                    "workload": cell.workload,
                }
            ],
            "profile": "baseline",
        },
    )

    with pytest.raises(ValueError, match="does not match its manifest cell"):
        census_module.load_census_targets(tmp_path, platform="cpu")


def test_census_cli_holds_host_lock_for_build_and_write(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []

    class RecordingLock:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            del args
            events.append("exit")

    monkeypatch.setattr(census_module, "HostCampaignLock", RecordingLock)

    def build(*args, **kwargs):
        del args, kwargs
        assert events == ["enter"]
        return {"records": []}

    monkeypatch.setattr(census_module, "build_census_campaign", build)
    output = tmp_path / "census.json"

    assert (
        census_module.main([
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--platform",
            "cpu",
            "--output",
            str(output),
        ])
        == 0
    )
    assert events == ["enter", "exit"]
    assert json.loads(output.read_text()) == {"records": []}
