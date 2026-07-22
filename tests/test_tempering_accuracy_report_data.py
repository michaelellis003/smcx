# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Campaign parsing contracts for the tempering-accuracy report."""

import hashlib

import jax.random as jr
import numpy as np
import pytest
from benchmarks.tempering_accuracy.report_data import (
    _cell_report,
    load_campaign,
)

from benchmarks.profiling.common import canonical_json
from benchmarks.tempering_accuracy import artifacts
from benchmarks.tempering_accuracy.core import accuracy_keys, build_target
from benchmarks.tempering_accuracy.plan import current_cells, work_count


def _identity():
    return {
        "source": {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "sha256": "s" * 64,
            "files": ["src/smcx/tempering.py"],
        },
        "lock": {"path": "uv.lock", "sha256": "l" * 64},
        "packages": {},
        "python": {},
        "host": {"os": "Darwin", "machine": "arm64"},
    }


def _payload(request, digest, *, failure=None, timing=None, runs=()):
    bound = (
        {}
        if request is None
        else artifacts.request_dict(artifacts.bind_request(request, digest))
    )
    return {
        "schema_version": 1,
        "request": bound,
        "failure": failure,
        "timing": timing,
        "runs": list(runs),
    }


def _run(cell, index, *, passed=True, stages=2):
    target = build_target(cell.geometry, cell.dimension, np.float64)
    key = accuracy_keys()[index] if index is not None else jr.key(20_260_719)
    words = [int(word) for word in np.asarray(jr.key_data(key))]
    work = work_count(cell, stages)
    return {
        "key_index": index,
        "key_words": words,
        "posterior_mean": target.posterior_mean.tolist(),
        "posterior_covariance": target.posterior_covariance.tolist(),
        "log_evidence": target.log_evidence,
        "temperatures": np.linspace(0.5, 1, stages).tolist(),
        "reweighting_ess": [0.75 * cell.reference_particles] * stages,
        "acceptance_rates": [0.4] * stages,
        "work": work._asdict(),
        "structural": {"passed": passed},
    }


def _timing(cell, seconds):
    flags = {
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "JAX_DISABLE_JIT": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        **{
            name: None
            for name in (
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_MPS_ASYNC_DISPATCH",
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    return {
        "execution_mode": "host_shell",
        "backend_startup_burns": 1,
        "warmups": 0,
        "repeats": 7,
        "first_execution_s": seconds * 2,
        "steady_times_s": [seconds] * 7,
        "backend": "cpu",
        "dispatch_mode": "asynchronous",
        "environment": {
            "device_id": 0,
            "device_kind": "cpu",
            "runtime_flags": flags,
            "runtime_state": {
                "backend": "cpu",
                "x64": True,
                "disable_jit": False,
                "cache_enabled": False,
                "cache_dir": None,
                "async": None,
            },
            "pre_timing": {},
            "post_timing": {},
            "post_cell": {},
        },
        "memory": {},
    }


def test_load_campaign_accepts_only_a_canonical_contiguous_prefix(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        artifacts, "campaign_identity", lambda root: _identity()
    )
    manifest = artifacts.build_manifest(tmp_path)
    digest = artifacts.ensure_manifest(tmp_path, manifest)
    requests = artifacts.campaign_requests()
    failure = {"kind": "execution_failure", "message": "retained"}
    for request in requests[:2]:
        artifacts.write_raw_result(
            tmp_path,
            request,
            digest,
            _payload(request, digest, failure=failure),
        )

    campaign = load_campaign(tmp_path)
    assert not campaign.complete
    assert campaign.not_run_after_stop == (2, 507)
    assert [entry.ordinal for entry in campaign.inventory] == [0, 1]
    encoded = canonical_json([
        entry._asdict() for entry in campaign.inventory
    ]).encode()
    assert campaign.raw_sha256 == hashlib.sha256(encoded).hexdigest()

    request = requests[3]
    artifacts.write_raw_result(
        tmp_path, request, digest, _payload(request, digest, failure=failure)
    )
    with pytest.raises(ValueError, match="contiguous prefix"):
        load_campaign(tmp_path)


def test_cell_report_builds_registered_timing_accuracy_and_work_summaries():
    cell = current_cells()[0]
    timing = tuple(
        _payload(
            None,
            "d" * 64,
            timing=_timing(cell, seconds),
            runs=[_run(cell, None)],
        )
        for seconds in (1, 2, 3, 4, 5)
    )
    accuracy = _payload(
        None,
        "d" * 64,
        runs=[_run(cell, index) for index in range(32)],
    )

    report = _cell_report(cell, timing, accuracy, _identity())

    assert report.timing_status == "eligible"
    assert report.timing.steady.median == pytest.approx(3)
    assert report.timing.steady.iqr == pytest.approx(2)
    assert report.accuracy_status == "eligible"
    assert report.accuracy.correctness_eligible
    assert report.work["stages"].median == pytest.approx(2)
    assert report.work["total_pairs"].median == work_count(cell, 2).total_pairs


@pytest.mark.parametrize(
    ("failure", "passed", "expected"),
    (
        ({"kind": "execution_failure"}, False, "failed_execution"),
        ({"kind": "structural_failure"}, False, "failed_structural"),
        ({"kind": "source_identity_drift"}, True, "ineligible_identity"),
    ),
)
def test_timing_failure_precedence(failure, passed, expected):
    cell = current_cells()[0]
    payloads = tuple(
        _payload(
            None,
            "d" * 64,
            failure=failure if block == 0 else None,
            timing=_timing(cell, 1) if failure is None or block else None,
            runs=[_run(cell, None, passed=passed if block == 0 else True)],
        )
        for block in range(5)
    )
    report = _cell_report(cell, payloads, None, _identity())
    assert report.timing_status == expected
    assert report.accuracy_status == "not_run_after_stop"
