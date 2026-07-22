# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Checks for the published documentation surface."""

from pathlib import Path


def test_public_text_excludes_internal_decision_labels() -> None:
    paths = [Path("docs/index.md"), *Path("docs/guides").glob("*.md")]
    paths.extend(Path("src/smcx").glob("*.py"))

    for path in paths:
        assert "ADR-" not in path.read_text(), path
