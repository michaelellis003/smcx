# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Checks for the published documentation surface."""

from pathlib import Path

import yaml


def test_public_text_excludes_internal_decision_labels() -> None:
    paths = [Path("docs/index.md"), *Path("docs/guides").glob("*.md")]
    paths.extend(Path("docs/tutorials").glob("*.md"))
    paths.extend(Path("src/smcx").glob("*.py"))

    for path in paths:
        assert "ADR-" not in path.read_text(), path


def test_tutorials_execute_during_documentation_build() -> None:
    config = yaml.safe_load(Path("properdocs.yml").read_text())
    jupyter = next(
        plugin["mkdocs-jupyter"]
        for plugin in config["plugins"]
        if isinstance(plugin, dict) and "mkdocs-jupyter" in plugin
    )

    assert jupyter["include"] == ["tutorials/*.md"]
    assert jupyter["execute"] is True
    assert jupyter["allow_errors"] is False
    assert jupyter["cache"] is False

    tutorial = Path("docs/tutorials/filtering.md")
    front_matter = yaml.safe_load(tutorial.read_text().split("---", 2)[1])
    assert front_matter["jupyter"]["kernelspec"]["language"] == "python"
