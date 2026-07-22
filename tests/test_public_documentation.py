# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Checks for the published documentation surface."""

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _tracked(*paths: str) -> list[str]:
    return subprocess.run(
        ("git", "-C", str(ROOT), "ls-files", "--", *paths),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def test_internal_roadmap_is_not_tracked() -> None:
    assert not _tracked("ROADMAP.md")

    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "/ROADMAP.md" in ignored


def test_internal_decision_index_is_not_tracked() -> None:
    paths = (
        "docs/adr/0030-native-conditionally-linear-gaussian-rbpf.md",
        "docs/adr/0031-mps-bootstrap-update-containment.md",
        "docs/adr/index.md",
    )
    assert not _tracked(*paths)


def test_reporting_and_checkpoint_records_are_not_tracked() -> None:
    paths = (
        "docs/adr/0027-arviz-bridge-contract.md",
        "docs/adr/0028-streaming-filter-checkpoints.md",
    )
    assert not _tracked(*paths)


def test_public_text_excludes_internal_decision_labels() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "pyproject.toml",
    ]
    paths.extend(
        path
        for path in (ROOT / "docs").rglob("*.md")
        if "adr" not in path.parts
    )
    paths.extend((ROOT / "src" / "smcx").rglob("*.py"))
    paths.extend((ROOT / "benchmarks").rglob("*.py"))
    paths.extend((ROOT / "benchmarks").rglob("PROTOCOL.md"))

    for path in paths:
        assert "ADR-" not in path.read_text(), path


def test_internal_licensing_inventory_is_not_published() -> None:
    assert not (ROOT / "docs/research/licensing.md").exists()


def test_tutorials_execute_during_documentation_build() -> None:
    config = yaml.safe_load((ROOT / "properdocs.yml").read_text())
    excluded = {line.strip() for line in config["exclude_docs"].splitlines()}
    assert "adr/" in excluded
    jupyter = next(
        plugin["mkdocs-jupyter"]
        for plugin in config["plugins"]
        if isinstance(plugin, dict) and "mkdocs-jupyter" in plugin
    )

    assert jupyter["include"] == ["tutorials/*.md"]
    assert jupyter["execute"] is True
    assert jupyter["allow_errors"] is False
    assert jupyter["cache"] is False

    tutorial = ROOT / "docs/tutorials/filtering.md"
    front_matter = yaml.safe_load(tutorial.read_text().split("---", 2)[1])
    assert front_matter["jupyter"]["kernelspec"]["language"] == "python"
