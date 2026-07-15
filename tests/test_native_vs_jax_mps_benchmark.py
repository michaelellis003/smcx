# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native MLX versus jax-mps benchmark harness."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.native_vs_jax_mps.common import (
    balanced_orders,
    bootstrap_ratio_ci,
    summarize,
    validate_result,
)


def test_summarize_retains_robust_statistics():
    summary = summarize([1.0, 2.0, 3.0, 4.0, 100.0])

    assert summary == {
        "iqr_s": 2.0,
        "mad_s": 1.0,
        "median_s": 3.0,
        "min_s": 1.0,
        "q1_s": 2.0,
        "q3_s": 4.0,
    }


def test_balanced_orders_rotate_every_arm_through_every_position():
    arms = ("mlx_gpu", "jax_mps_sync", "jax_mps_async")
    orders = balanced_orders(arms, blocks=4, seed=20260715)

    assert len(orders) == 4
    assert all(sorted(order) == sorted(arms) for order in orders)
    for position in range(len(arms)):
        assert {orders[block][position] for block in range(3)} == set(arms)


def test_bootstrap_ratio_ci_is_exact_for_constant_process_medians():
    estimate = bootstrap_ratio_ci(
        native=[1.0] * 5,
        compatibility=[2.0] * 5,
        draws=100,
        seed=20260715,
    )

    assert estimate == {"estimate": 2.0, "high": 2.0, "low": 2.0}


def test_validate_result_rejects_a_summary_without_raw_timings():
    result = {
        "arm": "mlx_gpu",
        "backend": "mlx",
        "block": 0,
        "cold_s": 0.1,
        "correctness": {"passed": True},
        "dispatch_mode": "native",
        "failure": None,
        "parameters": {"n": 10_000},
        "peak_memory_bytes": 1024,
        "schema_version": 1,
        "summary": {"median_s": 0.01},
        "versions": {"mlx": "0.32.0"},
        "workload": "eltwise_reduce",
    }

    with pytest.raises(ValueError, match="times_s"):
        validate_result(result)


def test_mlx_cpu_worker_emits_valid_tiny_result():
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/native_vs_jax_mps/mlx_worker.py"),
            "--arm",
            "mlx_cpu",
            "--block",
            "0",
            "--repeats",
            "2",
            "--size",
            "16",
            "--warmups",
            "1",
            "--workload",
            "eltwise_reduce",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    validate_result(result)
    assert result["arm"] == "mlx_cpu"
    assert result["correctness"]["passed"]
    assert len(result["times_s"]) == 2
