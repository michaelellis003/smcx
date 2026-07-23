# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Published report rendering contracts for the tempering campaign."""

import gzip
import json
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy.analysis import (
    ReplicateEstimate,
    analyze_accuracy,
)
from benchmarks.tempering_accuracy.core import build_target
from benchmarks.tempering_accuracy.plan import current_cells, matched_cells
from benchmarks.tempering_accuracy.report_accuracy import (
    CampaignReport,
    CellReport,
)
from benchmarks.tempering_accuracy.report_data import (
    CampaignData,
    InventoryEntry,
    campaign_requests,
)
from benchmarks.tempering_accuracy.report_markdown import render_markdown
from benchmarks.tempering_accuracy.report_plots import (
    PlotSummary,
    render_plots,
)
from benchmarks.tempering_accuracy.report_render import (
    AttemptEvidence,
    build_evidence,
    evidence_gzip,
)
from benchmarks.tempering_accuracy.report_timing import Summary, TimingReport


def _campaign_report():
    cells = (*current_cells(), *matched_cells())
    cell = cells[0]
    target = build_target("G0", 4, np.float64)
    estimates = [
        ReplicateEstimate(
            target.posterior_mean,
            target.posterior_covariance,
            1.0,
            True,
            2,
            11_000,
        )
        for _ in range(32)
    ]
    analysis = analyze_accuracy(
        estimates, target, "cpu_f64", steady_block_median_seconds=[0.1] * 5
    )
    summary = Summary((1.0,) * 5, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0)
    timing = TimingReport("eligible", summary, summary, summary, None)
    reports = [
        CellReport(
            cell,
            timing,
            "eligible",
            analysis,
            {"stages": summary, "total_pairs": summary},
        )
    ]
    reports.extend(
        CellReport(
            other,
            TimingReport("not_run_after_stop", None, None, None, None),
            "not_run_after_stop",
            None,
            None,
        )
        for other in cells[1:]
    )
    requests = campaign_requests()
    results = [{"failure": None} for _ in requests]
    results[1] = {
        "failure": {
            "kind": "execution_failure",
            "exception_type": "RuntimeError",
            "message": "secret at /private/tmp/campaign",
        }
    }
    identity = {
        "source": {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "sha256": "a" * 64,
            "files": ["benchmarks/tempering_accuracy/worker.py"],
        },
        "lock": {"path": "uv.lock", "sha256": "b" * 64},
        "packages": {"jax": "1.2.3", "smcx": "1.6.0"},
        "python": {
            "implementation": "CPython",
            "version": "3.13.9",
            "executable": "/private/tmp/.venv/bin/python",
        },
        "host": {
            "os": "Darwin",
            "machine": "arm64",
            "hardware_model": "Mac15,6",
            "cpu_model": "Apple M3 Pro",
            "macos": "15.5",
        },
    }
    manifest = {
        "plan_sha256": "d" * 64,
        "algorithm_contract": {"target_ess": 0.5},
        "campaign_identity": identity,
        "exclusions": [{"arm": "waste_free", "status": "blocked"}],
    }
    inventory = tuple(
        InventoryEntry(index, f"raw-{index}.json", f"{index:064x}")
        for index in range(508)
    )
    campaign = CampaignData(
        manifest,
        "e" * 64,
        "f" * 64,
        True,
        None,
        None,
        inventory,
        tuple(results),
    )
    return CampaignReport(campaign, tuple(reports))


def test_evidence_is_canonical_reproducible_and_sanitized():
    report = _campaign_report()
    attempt = AttemptEvidence(1, 0, "9" * 64, "launch_error")
    with pytest.raises(ValueError, match="attempt inventory"):
        evidence_gzip(report)
    with pytest.raises(ValueError, match="safe identifier"):
        evidence_gzip(report, attempts=(attempt._replace(kind="/private"),))
    first = evidence_gzip(report, attempts=(attempt,))

    assert first == evidence_gzip(report, attempts=(attempt,))
    decoded = gzip.decompress(first)
    evidence = json.loads(decoded)
    assert decoded == (canonical_json(evidence) + "\n").encode()
    assert len(evidence["integrity"]["raw_leaves"]) == 508
    assert evidence["integrity"]["manifest_sha256"] == "e" * 64
    assert len(evidence["cells"]) == 84
    assert evidence["cells"][0]["accuracy"]["correctness_eligible"]
    assert np.isclose(evidence["cells"][0]["timing"]["steady"]["median"], 1)
    assert evidence["gate_counts"]["centering"] == {
        "passed": 9,
        "evaluated": 9,
        "registered": 6_228,
    }
    assert evidence["gate_counts"]["evidence_resolution"] == {
        "passed": 1,
        "evaluated": 1,
        "registered": 84,
    }
    assert evidence["failures"][0]["kind"] == "execution_failure"
    assert evidence["attempts"] == [attempt._asdict()]
    assert evidence["exclusions"] == [
        {"arm": "waste_free", "status": "blocked"}
    ]
    assert "/private/" not in decoded.decode()
    assert "secret" not in decoded.decode()


def _table_rows(markdown, heading):
    section = markdown.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if line.startswith("| ")][2:]


def _evidence():
    attempt = AttemptEvidence(1, 0, "9" * 64, "launch_error")
    return build_evidence(_campaign_report(), attempts=(attempt,))


def test_markdown_report_is_deterministic_complete_and_sanitized():
    evidence = _evidence()
    first = render_markdown(evidence, report_date="2026-07-22")
    assert first == render_markdown(evidence, report_date="2026-07-22")
    assert first.startswith("# Tempering accuracy — 2026-07-22\n")
    assert "current_rwm_not_eligible`\n\nGates: — · Cost: —" in first
    assert "| Centering | 6,228 | 9 | 9 |" in first
    for heading, count in {
        "Minimum passing sweep": 24,
        "Matched challenge": 12,
        "All cells": 84,
    }.items():
        assert len(_table_rows(first, heading)) == count
    assert "execution_failure" in first and "launch_error" in first
    assert "waste_free" in first and "CPU/MPS" not in first


def test_markdown_hides_timing_for_an_accuracy_ineligible_cell():
    evidence = _evidence()
    cell = evidence["cells"][0]
    cell["status"] = "failed_accuracy"
    cell["accuracy"]["status"] = "failed_accuracy"
    cell["accuracy"]["correctness_eligible"] = False
    evidence["status_counts"] = {"failed_accuracy": 1, "not_run_after_stop": 83}
    markdown = render_markdown(evidence, report_date="2026-07-22")
    row = _table_rows(markdown, "All cells")[0]
    values = [value.strip() for value in row.strip("|").split("|")]
    assert values[-4:] == ["—", "—", "—", "—"]


def _plot_evidence():
    evidence = _evidence()
    accuracy = evidence["cells"][0]["accuracy"]
    for name in ("mean", "covariance", "evidence"):
        accuracy[f"{name}_loss"]["rmse"] = 0.1
    return evidence


def test_plots_are_deterministic_and_report_omissions(tmp_path):
    evidence = _plot_evidence()
    first_gate = tmp_path / "first-gates.png"
    first_cost = tmp_path / "first-cost.png"
    second_gate = tmp_path / "second-gates.png"
    second_cost = tmp_path / "second-cost.png"

    summary = render_plots(evidence, first_gate, first_cost)
    render_plots(evidence, second_gate, second_cost)

    assert summary == PlotSummary(
        evaluated_gate_cells=1,
        unavailable_gate_cells=71,
        eligible_cost_cells=1,
        unavailable_cost_cells=71,
    )
    assert first_gate.read_bytes() == second_gate.read_bytes()
    assert first_cost.read_bytes() == second_cost.read_bytes()


def test_plots_do_not_impute_accuracy_ineligible_cost(tmp_path):
    evidence = _plot_evidence()
    eligible_gate = tmp_path / "eligible-gates.png"
    render_plots(evidence, eligible_gate, tmp_path / "cost.png")
    cell = evidence["cells"][0]
    cell["status"] = "failed_accuracy"
    cell["accuracy"]["status"] = "failed_accuracy"
    cell["accuracy"]["correctness_eligible"] = False

    summary = render_plots(
        evidence, tmp_path / "gates.png", tmp_path / "cost.png"
    )

    assert summary.eligible_cost_cells == 0
    assert summary.unavailable_cost_cells == 72
    assert eligible_gate.read_bytes() != (tmp_path / "gates.png").read_bytes()


def test_cost_plot_distinguishes_sweep_counts(tmp_path, monkeypatch):
    evidence = _plot_evidence()
    source = evidence["cells"][0]
    for index in (2, 4):
        item = evidence["cells"][index]
        item["status"] = "eligible"
        item["accuracy"] = deepcopy(source["accuracy"])
        item["work"] = deepcopy(source["work"])
    observed = []

    def capture(figure, path):
        if path.name == "cost.png":
            observed.extend(
                collection.get_alpha()
                for collection in figure.axes[0].collections
            )

    monkeypatch.setattr(
        "benchmarks.tempering_accuracy.report_plots._save", capture
    )
    render_plots(evidence, tmp_path / "gates.png", tmp_path / "cost.png")

    assert observed == [0.45, 0.7, 1.0]


def test_cost_plot_uses_nonnegative_rmse_scale(tmp_path, monkeypatch):
    evidence = _plot_evidence()
    observed = []

    def capture(figure, path):
        if path.name == "cost.png":
            observed.extend(axis.get_yscale() for axis in figure.axes)

    monkeypatch.setattr(
        "benchmarks.tempering_accuracy.report_plots._save", capture
    )
    render_plots(evidence, tmp_path / "gates.png", tmp_path / "cost.png")

    assert observed == ["log"] * 6
