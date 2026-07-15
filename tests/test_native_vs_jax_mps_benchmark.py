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
from benchmarks.native_vs_jax_mps.run import (
    build_worker_command,
    worker_environment,
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


@pytest.mark.parametrize(
    ("workload", "size"),
    [
        ("gather_scatter", 16),
        ("matmul", 8),
        ("random", 1_000),
        ("scan", 16),
        ("systematic", 16),
    ],
)
def test_mlx_cpu_worker_smokes_every_kernel_motif(workload, size):
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
            "1",
            "--size",
            str(size),
            "--warmups",
            "1",
            "--workload",
            workload,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    validate_result(result)
    assert result["correctness"]["passed"]


def test_jax_mps_command_pins_the_isolated_compatibility_stack():
    root = Path(__file__).parents[1]
    command = build_worker_command(
        root=root,
        arm="jax_mps_sync",
        block=2,
        repeats=7,
        size=10_000,
        warmups=1,
        workload="eltwise_reduce",
    )

    assert command[:5] == ["uv", "run", "--no-project", "--python", "3.13"]
    assert "jax==0.10.2" in command
    assert "jaxlib==0.10.2" in command
    assert "jax-mps==0.10.9" in command
    assert command[-2:] == ["--workload", "eltwise_reduce"]


def test_worker_environment_exposes_safe_and_async_mps_separately():
    safe = worker_environment("jax_mps_sync", base={"PATH": "/bin"})
    asynchronous = worker_environment("jax_mps_async", base={"PATH": "/bin"})

    assert safe["JAX_PLATFORMS"] == "mps"
    assert "JAX_MPS_ASYNC_DISPATCH" not in safe
    assert asynchronous["JAX_PLATFORMS"] == "mps"
    assert asynchronous["JAX_MPS_ASYNC_DISPATCH"] == "1"


def test_jax_worker_cli_is_inspectable_without_jax_installed():
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/native_vs_jax_mps/jax_worker.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert "--arm" in completed.stdout
    assert "--workload" in completed.stdout


@pytest.mark.parametrize(
    ("workload", "size"),
    [
        ("gather_scatter", 16),
        ("matmul", 8),
        ("random", 1_000),
        ("scan", 16),
        ("systematic", 16),
    ],
)
def test_isolated_jax_cpu_worker_smokes_every_kernel_motif(workload, size):
    root = Path(__file__).parents[1]
    command = build_worker_command(
        root=root,
        arm="jax_cpu",
        block=0,
        repeats=1,
        size=size,
        warmups=1,
        workload=workload,
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=worker_environment("jax_cpu"),
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    validate_result(result)
    assert result["backend"] == "cpu"
    assert result["correctness"]["passed"]


@pytest.mark.parametrize("arm", ["mlx_cpu", "jax_cpu"])
def test_cpu_workers_smoke_the_matched_lgssm_filter(arm):
    root = Path(__file__).parents[1]
    command = build_worker_command(
        root=root,
        arm=arm,
        block=0,
        repeats=1,
        size=256,
        warmups=1,
        workload="lgssm_pf",
    )
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=worker_environment(arm),
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    validate_result(result)
    assert result["correctness"]["passed"]
