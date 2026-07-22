# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Published report rendering contracts for the tempering campaign."""

import gzip
import json

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
from benchmarks.tempering_accuracy.report_render import (
    AttemptEvidence,
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
