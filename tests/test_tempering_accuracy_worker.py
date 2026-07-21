# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen standard-worker contracts for issue #30."""

import pytest
from benchmarks.tempering_accuracy.worker import WorkerRequest, execute_request

from benchmarks.tempering_accuracy.plan import (
    current_cells,
    current_smoke_cells,
    waste_free_cells,
)

_MANIFEST = "a" * 64


@pytest.mark.parametrize(
    "request",
    (
        WorkerRequest("bad", "smoke", current_smoke_cells()[0], None),
        WorkerRequest(
            _MANIFEST,
            "smoke",
            current_smoke_cells()[0]._replace(sweeps=20),
            None,
        ),
        WorkerRequest(_MANIFEST, "timing", current_cells()[0], None),
        WorkerRequest(_MANIFEST, "accuracy", current_cells()[0], 0),
        WorkerRequest(_MANIFEST, "accuracy", waste_free_cells()[0], None),
    ),
)
def test_invalid_requests_become_retained_failure_payloads(request):
    payload = execute_request(request)

    assert payload["schema_version"] == 1
    assert payload["request"]["manifest_sha256"] == request.manifest_sha256
    assert payload["failure"]["kind"] == "invalid_request"
    assert payload["failure"]["exception_type"] == "ValueError"
    assert payload["timing"] is None
    assert payload["runs"] == []
