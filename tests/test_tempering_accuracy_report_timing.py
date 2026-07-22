# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Timing eligibility contracts for the tempering-accuracy report."""

import copy

import pytest

from benchmarks.tempering_accuracy.plan import current_cells
from benchmarks.tempering_accuracy.report_timing import analyze_timing

_CPU, _MPS = current_cells()[:2]
_PACKAGES = {
    name: "1.0"
    for name in (
        "jax",
        "jax-mps",
        "jaxlib",
        "ml-dtypes",
        "numpy",
        "scipy",
        "smcx",
    )
}
_POWER = "Now drawing from 'AC Power'"
_THERMAL = "No thermal warning level\nNo performance warning level"


def _identity():
    return {
        "source": {
            "git_commit": "c" * 40,
            "git_dirty": False,
            "sha256": "a" * 64,
            "files": ["src/smcx/tempering.py"],
        },
        "lock": {"path": "uv.lock", "sha256": "b" * 64},
        "packages": dict(_PACKAGES),
        "python": {
            "implementation": "CPython",
            "version": "3.13.9",
            "executable": "/python",
        },
        "host": {"os": "Darwin", "machine": "arm64"},
    }


def _timing(cell, block):
    backend = "cpu" if cell.lane == "cpu_f64" else "mps"
    x64 = cell.lane == "cpu_f64"
    boundary = {"power_status": _POWER, "thermal_status": _THERMAL}
    return {
        "execution_mode": "host_shell",
        "backend_startup_burns": 1,
        "warmups": 0,
        "repeats": 7,
        "first_execution_s": float(2 * block),
        "steady_times_s": [float(block)] * 7,
        "backend": backend,
        "dispatch_mode": "asynchronous" if backend == "cpu" else "safe",
        "environment": {
            "device_id": 0,
            "device_kind": "cpu" if backend == "cpu" else "gpu",
            "runtime_flags": {
                "JAX_PLATFORMS": backend,
                "JAX_ENABLE_X64": str(x64).lower(),
                "JAX_DISABLE_JIT": "false",
                "JAX_ENABLE_COMPILATION_CACHE": "false",
            },
            "runtime_state": {
                "backend": backend,
                "x64": x64,
                "disable_jit": False,
                "cache_enabled": False,
                "cache_dir": None,
                "async": None,
            },
            "pre_timing": dict(boundary),
            "post_timing": dict(boundary),
            "post_cell": dict(boundary),
        },
        "memory": {
            "device_stats": (
                {"peak_bytes_in_use": 2_000 + block}
                if backend == "mps"
                else None
            ),
            "process_max_rss_before_measurement_bytes": 500 + block,
            "process_max_rss_bytes": 1_000 + block,
        },
    }


def _blocks(cell):
    return tuple(
        {
            "failure": None,
            "timing": _timing(cell, block),
            "runs": [{"structural": {"passed": True}}],
        }
        for block in range(1, 6)
    )


def test_eligible_timing_summarizes_five_blocks_and_memory_scopes():
    report = analyze_timing(_MPS, _blocks(_MPS), _identity())

    assert report.status == "eligible"
    assert report.first and report.steady and report.process_rss
    assert report.first.values == (2.0, 4.0, 6.0, 8.0, 10.0)
    assert (report.steady.median, report.steady.iqr) == (3.0, 2.0)
    assert report.process_rss.median == 1_003
    assert report.mps_peak is not None
    assert report.mps_peak.median == 2_003


@pytest.mark.parametrize(
    ("failures", "dirty", "expected"),
    (
        (("structural_failure", "execution_failure"), True, "failed_execution"),
        (("structural_failure",), True, "failed_structural"),
        (("source_identity_drift",), False, "ineligible_identity"),
        ((), True, "ineligible_identity"),
    ),
)
def test_timing_status_precedence(failures, dirty, expected):
    blocks = list(copy.deepcopy(_blocks(_CPU)))
    for block, kind in zip(blocks, failures, strict=False):
        block.update(failure={"kind": kind}, timing=None, runs=[])
    identity = _identity()
    identity["source"]["git_dirty"] = dirty
    assert analyze_timing(_CPU, tuple(blocks), identity).status == expected


@pytest.mark.parametrize(
    ("package", "value"), (("jaxlib", None), ("jax", None))
)
def test_timing_identity_requires_every_numerical_package(package, value):
    identity = _identity()
    identity["packages"][package] = value
    assert analyze_timing(_CPU, _blocks(_CPU), identity).status == (
        "ineligible_identity"
    )


@pytest.mark.parametrize("case", ("extra_flag", "state", "time", "rss"))
def test_timing_runtime_contract_is_exact(case):
    blocks = list(copy.deepcopy(_blocks(_CPU)))
    timing = blocks[0]["timing"]
    if case == "extra_flag":
        timing["environment"]["runtime_flags"]["OMP_NUM_THREADS"] = "1"
    elif case == "state":
        timing["environment"]["runtime_state"]["x64"] = False
    elif case == "time":
        timing["steady_times_s"][0] = 0.0
    else:
        timing["memory"]["process_max_rss_bytes"] = 0
    assert analyze_timing(_CPU, tuple(blocks), _identity()).status == (
        "ineligible_runtime"
    )


@pytest.mark.parametrize("boundary", ("pre_timing", "post_timing", "post_cell"))
def test_metal_requires_all_three_power_and_thermal_boundaries(boundary):
    blocks = list(copy.deepcopy(_blocks(_MPS)))
    blocks[0]["timing"]["environment"][boundary]["power_status"] = None
    assert analyze_timing(_MPS, tuple(blocks), _identity()).status == (
        "ineligible_environment"
    )


def test_incomplete_timing_is_not_imputed():
    report = analyze_timing(_CPU, _blocks(_CPU)[:4], _identity())
    assert report.status == "not_run_after_stop"
    assert report.steady is report.process_rss is None
