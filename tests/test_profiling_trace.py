# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Contracts for post-baseline, explicitly selected profiler traces."""

import json

import pytest

import benchmarks.profiling.trace as trace_module
from benchmarks.profiling.common import plan_cells


def _eligible_record() -> dict:
    cell = plan_cells("smoke", platforms=("cpu",), seed=7)[0]
    return {
        "block": cell.block,
        "correctness": {"passed": True},
        "correctness_replicates": cell.correctness_replicates,
        "dispatch_mode": "asynchronous",
        "execution_mode": cell.execution_mode,
        "failure": None,
        "parameters": cell.parameters,
        "platform_requested": cell.platform,
        "repeats": cell.repeats,
        "warmups": cell.warmups,
        "workload": cell.workload,
    }


def test_trace_cell_uses_exact_successful_raw_result(monkeypatch) -> None:
    monkeypatch.setattr(trace_module, "validate_result", lambda record: None)
    record = _eligible_record()
    cell = trace_module.cell_from_result(record)
    assert cell._asdict() == {
        "workload": record["workload"],
        "platform": record["platform_requested"],
        "block": record["block"],
        "warmups": record["warmups"],
        "repeats": record["repeats"],
        "execution_mode": record["execution_mode"],
        "parameters": record["parameters"],
        "correctness_replicates": record["correctness_replicates"],
    }


def test_jax_trace_capture_rejects_mps_profiler_stub(tmp_path) -> None:
    cell = plan_cells("smoke", platforms=("mps",), seed=7)[0]

    with pytest.raises(ValueError, match="CPU-only"):
        trace_module.capture_trace(
            cell,
            identity={},
            output_dir=tmp_path / "trace",
            provenance={},
            steps=1,
        )

    assert not (tmp_path / "trace").exists()


@pytest.mark.parametrize("field", ["failure", "correctness"])
def test_trace_cell_rejects_ineligible_raw_result(monkeypatch, field) -> None:
    monkeypatch.setattr(trace_module, "validate_result", lambda record: None)
    record = _eligible_record()
    if field == "failure":
        record[field] = {"kind": "timeout"}
    else:
        record[field] = {"passed": False}
    with pytest.raises(ValueError, match="successful, correct"):
        trace_module.cell_from_result(record)


def test_trace_target_rejects_smoke_campaign(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        trace_module,
        "build_report",
        lambda campaign_dir: {"profile": "smoke"},
    )
    with pytest.raises(ValueError, match="non-smoke"):
        trace_module.load_trace_target(
            tmp_path,
            result_path=tmp_path / "raw/result.json",
        )


def test_trace_target_is_exact_manifest_member_and_eligible(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(trace_module, "validate_result", lambda record: None)
    cell = plan_cells("smoke", platforms=("cpu",), seed=7)[0]
    record = _eligible_record()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    result_path = raw_dir / trace_module.raw_filename(cell)
    result_path.write_text(json.dumps(record))
    manifest = {
        "cells": [cell._asdict()],
        "plan_sha256": "a" * 64,
        "profile": "baseline",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        trace_module,
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
    target = trace_module.load_trace_target(
        tmp_path,
        result_path=result_path,
    )
    assert target["cell"] == cell
    assert target["profile"] == "baseline"
    assert target["plan_sha256"] == "a" * 64
    assert len(target["manifest_sha256"]) == 64
    assert len(target["result_sha256"]) == 64

    renamed = raw_dir / "renamed.json"
    renamed.write_bytes(result_path.read_bytes())
    with pytest.raises(ValueError, match="canonical raw filename"):
        trace_module.load_trace_target(tmp_path, result_path=renamed)


def test_trace_target_manifest_identity_is_type_strict(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(trace_module, "validate_result", lambda record: None)
    registered = plan_cells("smoke", platforms=("cpu",), seed=7)[0]
    parameters = dict(registered.parameters)
    integer_name = next(
        name
        for name, value in parameters.items()
        if isinstance(value, int) and not isinstance(value, bool)
    )
    parameters[integer_name] = float(parameters[integer_name])
    altered = registered._replace(parameters=parameters)
    record = {
        **_eligible_record(),
        "parameters": parameters,
    }
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    result_path = raw_dir / trace_module.raw_filename(altered)
    result_path.write_text(json.dumps(record))
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "cells": [registered._asdict()],
            "plan_sha256": "a" * 64,
            "profile": "baseline",
        })
    )
    monkeypatch.setattr(
        trace_module,
        "build_report",
        lambda campaign_dir: {
            "aggregates": [
                {
                    "execution_mode": registered.execution_mode,
                    "parameters": registered.parameters,
                    "platform": registered.platform,
                    "repeats": registered.repeats,
                    "timing_eligible": True,
                    "warmups": registered.warmups,
                    "workload": registered.workload,
                }
            ],
            "profile": "baseline",
        },
    )

    with pytest.raises(ValueError, match="exact manifest member"):
        trace_module.load_trace_target(tmp_path, result_path=result_path)


def test_trace_cli_holds_host_lock_for_target_and_capture(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    cell = plan_cells("smoke", platforms=("cpu",), seed=7)[0]

    class RecordingLock:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *args):
            del args
            events.append("exit")

    monkeypatch.setattr(trace_module, "HostCampaignLock", RecordingLock)
    monkeypatch.setattr(
        trace_module,
        "load_trace_target",
        lambda *args, **kwargs: {
            "cell": cell,
            "dispatch_mode": "asynchronous",
            "manifest_sha256": "a" * 64,
            "plan_sha256": "b" * 64,
            "profile": "baseline",
            "result": {},
            "result_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        trace_module,
        "_require_current_identity",
        lambda result: {},
    )

    def capture(*args, **kwargs):
        del args, kwargs
        assert events == ["enter"]

    monkeypatch.setattr(trace_module, "capture_trace", capture)

    assert (
        trace_module.main([
            "--campaign-dir",
            str(tmp_path / "campaign"),
            "--result",
            str(tmp_path / "result.json"),
            "--output-dir",
            str(tmp_path / "trace"),
        ])
        == 0
    )
    assert events == ["enter", "exit"]
