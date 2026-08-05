# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Published example code must run against the current API."""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    "path, minimum_blocks",
    [
        ("README.md", 1),
        ("docs/guides/sequential-inference.md", 4),
        ("docs/guides/parameter-estimation.md", 3),
        ("docs/guides/rao-blackwellized-pf.md", 4),
    ],
)
def test_python_blocks_execute(path, minimum_blocks):
    """Execute every python fence of a published page, in order.

    The blocks of a page share one namespace, matching how a reader
    would run them top to bottom. Printed values are not asserted;
    the contract is that the published examples execute against the
    shipped API.
    """
    text = (_ROOT / path).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert len(blocks) >= minimum_blocks
    namespace: dict[str, object] = {}
    exec(compile("\n".join(blocks), path, "exec"), namespace)
    assert any(
        name in namespace
        for name in ("posterior", "score", "particle", "history")
    )
