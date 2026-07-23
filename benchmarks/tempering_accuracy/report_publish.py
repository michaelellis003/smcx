# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Publish the four public artifacts from one tempering campaign."""

import argparse
import gzip
import json
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from benchmarks.tempering_accuracy.report_accuracy import aggregate_campaign
from benchmarks.tempering_accuracy.report_attempts import load_attempts
from benchmarks.tempering_accuracy.report_data import load_campaign
from benchmarks.tempering_accuracy.report_markdown import render_markdown
from benchmarks.tempering_accuracy.report_plots import render_plots
from benchmarks.tempering_accuracy.report_render import evidence_gzip


def _validate_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("report date must be ISO 8601") from error
    if parsed.isoformat() != value:
        raise ValueError("report date must be ISO 8601")
    return value


def _destinations(
    output_dir: Path, report_date: str
) -> tuple[Path, Path, Path, Path]:
    stem = f"{report_date}-tempering-accuracy"
    return (
        output_dir / f"{stem}.md",
        output_dir / f"{stem}.json.gz",
        output_dir / f"{stem}-gates.png",
        output_dir / f"{stem}-cost.png",
    )


def _parse_evidence(encoded: bytes) -> dict[str, Any]:
    value = json.loads(gzip.decompress(encoded))
    if not isinstance(value, dict):
        raise ValueError("evidence document must be a JSON object")
    return value


def publish_report(
    campaign_dir: Path,
    output_dir: Path,
    *,
    report_date: str,
) -> tuple[Path, Path, Path, Path]:
    """Publish Markdown, evidence, and figures without overwriting files."""
    report_date = _validate_date(report_date)
    campaign_dir, output_dir = Path(campaign_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = _destinations(output_dir, report_date)
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"report destination exists: {path.name}")

    campaign = load_campaign(campaign_dir)
    attempts = load_attempts(campaign_dir, campaign.manifest_sha256)
    report = aggregate_campaign(campaign)
    encoded = evidence_gzip(report, attempts=attempts)
    evidence = _parse_evidence(encoded)

    gate_path, cost_path = destinations[2:]
    markdown = render_markdown(
        evidence,
        report_date=report_date,
        gate_figure=gate_path.name,
        cost_figure=cost_path.name,
    )
    with TemporaryDirectory(prefix=".tempering-report-", dir=output_dir) as raw:
        staging = Path(raw)
        staged = tuple(staging / path.name for path in destinations)
        staged_markdown, staged_evidence, staged_gate, staged_cost = staged
        staged_markdown.write_text(markdown, encoding="utf-8", newline="\n")
        staged_evidence.write_bytes(encoded)
        render_plots(evidence, staged_gate, staged_cost)

        published: list[Path] = []
        try:
            for source, destination in zip(staged, destinations, strict=True):
                os.link(source, destination)
                published.append(destination)
        except OSError:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    return destinations


def main(argv: Sequence[str] | None = None) -> int:
    """Publish a dated report from a campaign directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--date", dest="report_date", required=True)
    arguments = parser.parse_args(argv)
    publish_report(
        arguments.campaign_dir,
        arguments.output_dir,
        report_date=arguments.report_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
