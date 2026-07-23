# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Final publication contract for the tempering-accuracy campaign."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from benchmarks.tempering_accuracy import report_publish


def _evidence_bytes() -> tuple[dict[str, object], bytes]:
    evidence: dict[str, object] = {
        "schema_version": 1,
        "verdict": "incomplete",
        "execution": {
            "complete": False,
            "result_count": 17,
            "not_run_after_stop": 491,
        },
    }
    encoded = (json.dumps(evidence, sort_keys=True) + "\n").encode()
    return evidence, gzip.compress(encoded, mtime=0)


def _install_pipeline(monkeypatch, *, plot_failure: bool = False):
    expected, encoded = _evidence_bytes()
    campaign = SimpleNamespace(manifest_sha256="a" * 64)
    attempts = object()

    def render_plots(evidence, gate_path, cost_path):
        gate_path.write_bytes(b"gate")
        if plot_failure:
            raise RuntimeError("plot failed")
        cost_path.write_bytes(b"cost")

    pipeline: dict[str, Any] = {
        "load_campaign": Mock(return_value=campaign),
        "load_attempts": Mock(return_value=attempts),
        "aggregate_campaign": Mock(return_value=object()),
        "evidence_gzip": Mock(return_value=encoded),
        "render_markdown": Mock(return_value="complete=False\n"),
        "render_plots": Mock(side_effect=render_plots),
    }
    for name, replacement in pipeline.items():
        monkeypatch.setattr(report_publish, name, replacement)
    return expected, encoded, pipeline


def test_publish_report_uses_one_evidence_document_for_stopped_prefix(
    tmp_path, monkeypatch
):
    expected, encoded, pipeline = _install_pipeline(monkeypatch)
    campaign_dir = tmp_path / "campaign"
    output_dir = tmp_path / "results"

    paths = report_publish.publish_report(
        campaign_dir, output_dir, report_date="2026-07-22"
    )

    names = (
        "2026-07-22-tempering-accuracy.md",
        "2026-07-22-tempering-accuracy.json.gz",
        "2026-07-22-tempering-accuracy-gates.png",
        "2026-07-22-tempering-accuracy-cost.png",
    )
    assert paths == tuple(output_dir / name for name in names)
    assert {path.name for path in output_dir.iterdir()} == set(names)
    assert paths[1].read_bytes() == encoded
    assert "complete=False" in paths[0].read_text()
    pipeline["load_campaign"].assert_called_once_with(campaign_dir)
    pipeline["load_attempts"].assert_called_once_with(campaign_dir, "a" * 64)
    pipeline["evidence_gzip"].assert_called_once()
    markdown_evidence = pipeline["render_markdown"].call_args.args[0]
    assert markdown_evidence is pipeline["render_plots"].call_args.args[0]
    assert markdown_evidence == expected
    assert pipeline["render_markdown"].call_args.kwargs == {
        "report_date": "2026-07-22",
        "gate_figure": names[2],
        "cost_figure": names[3],
    }


def test_publish_report_rejects_invalid_date_or_existing_destination(tmp_path):
    with pytest.raises(ValueError, match="ISO"):
        report_publish.publish_report(
            tmp_path / "campaign", tmp_path, report_date="22 July 2026"
        )

    existing = tmp_path / "2026-07-22-tempering-accuracy.md"
    existing.write_text("keep\n")
    with pytest.raises(FileExistsError, match=existing.name):
        report_publish.publish_report(
            tmp_path / "campaign", tmp_path, report_date="2026-07-22"
        )
    assert existing.read_text() == "keep\n"


def test_render_failure_leaves_no_final_names(tmp_path, monkeypatch):
    _install_pipeline(monkeypatch, plot_failure=True)
    output_dir = tmp_path / "results"

    with pytest.raises(RuntimeError, match="plot failed"):
        report_publish.publish_report(
            tmp_path / "campaign", output_dir, report_date="2026-07-22"
        )

    assert list(output_dir.iterdir()) == []


def test_main_delegates_to_publisher(tmp_path, monkeypatch):
    calls = []

    def publish_report(campaign_dir, output_dir, *, report_date):
        calls.append((campaign_dir, output_dir, report_date))
        return (Path("a"), Path("b"), Path("c"), Path("d"))

    monkeypatch.setattr(report_publish, "publish_report", publish_report)
    result = report_publish.main([
        str(tmp_path / "campaign"),
        str(tmp_path),
        "--date",
        "2026-07-22",
    ])

    assert result == 0
    assert calls == [(tmp_path / "campaign", tmp_path, "2026-07-22")]
