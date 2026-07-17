# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native MLX versus jax-mps benchmark harness."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.native_vs_jax_mps.common import (
    BOOTSTRAP_SEED,
    PINNED_VERSIONS,
    SCHEMA_VERSION,
    WORKLOAD_GRIDS,
    balanced_orders,
    bootstrap_ratio_ci,
    kalman_gate,
    summarize,
    validate_result,
)
from benchmarks.native_vs_jax_mps.run import (
    ARMS,
    Cell,
    build_manifest,
    build_worker_command,
    main,
    plan_cells,
    raw_filename,
    supervise,
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


def test_kalman_gate_uses_the_preregistered_one_sided_jensen_budget():
    gate = kalman_gate(
        log_evidence=[-10.2, -10.0, -9.8, -10.0],
        oracle=-10.0,
    )

    assert gate["passed"]
    assert gate["replicates"] == 4
    assert gate["lower_error_bound"] < 0.0
    assert gate["upper_error_bound"] > 0.0


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
        correctness_replicates=20,
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
    assert result["correctness"]["replicates"] == 20


def test_plan_cells_smoke_uses_smallest_registered_sizes():
    cells = plan_cells("smoke")

    assert {cell.workload for cell in cells} == set(WORKLOAD_GRIDS)
    assert {cell.arm for cell in cells} == set(ARMS)
    assert {cell.block for cell in cells} == {0}
    for cell in cells:
        assert cell.size == min(WORKLOAD_GRIDS[cell.workload])
        assert cell.repeats == 1
        assert cell.warmups == 1


def test_plan_cells_full_covers_every_size_block_and_arm():
    cells = plan_cells("full")

    for workload, grid in WORKLOAD_GRIDS.items():
        sizes = {cell.size for cell in cells if cell.workload == workload}
        assert sizes == set(grid)
    assert {cell.block for cell in cells} == set(range(5))
    matmul = [c for c in cells if c.workload == "matmul" and c.size == 256]
    assert {cell.repeats for cell in matmul} == {7}
    assert {cell.warmups for cell in matmul} == {1}
    keys = {(c.workload, c.size, c.block, c.arm) for c in cells}
    assert len(keys) == len(cells)


def test_plan_cells_assigns_r20_only_to_block_zero_lgssm():
    for cell in plan_cells("full"):
        if cell.workload == "lgssm_pf" and cell.block == 0:
            assert cell.correctness_replicates == 20
        else:
            assert cell.correctness_replicates == 0


def test_plan_cells_balances_arms_and_is_deterministic():
    assert plan_cells("full") == plan_cells("full")

    columns: dict[tuple[str, int], dict[int, list[str]]] = {}
    for cell in plan_cells("full"):
        block_map = columns.setdefault((cell.workload, cell.size), {})
        block_map.setdefault(cell.block, []).append(cell.arm)

    for block_map in columns.values():
        for position in range(len(ARMS)):
            column = {block_map[block][position] for block in range(len(ARMS))}
            assert column == set(ARMS)


def test_raw_filename_is_unique_and_stable():
    cells = plan_cells("full")
    names = [raw_filename(cell) for cell in cells]

    assert len(set(names)) == len(names)
    assert all(name.endswith(".json") for name in names)
    assert raw_filename(cells[0]) == raw_filename(cells[0])


def test_build_manifest_persists_ordered_cells_and_pins():
    cells = plan_cells("smoke")
    manifest = build_manifest("smoke", cells, seed=BOOTSTRAP_SEED)

    assert manifest["profile"] == "smoke"
    assert manifest["seed"] == BOOTSTRAP_SEED
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["versions"] == PINNED_VERSIONS
    assert len(manifest["cells"]) == len(cells)
    assert [entry["arm"] for entry in manifest["cells"]] == [
        cell.arm for cell in cells
    ]


def test_supervise_resumes_without_overwriting_completed_raw(tmp_path):
    root = Path(__file__).parents[1]
    cells = plan_cells("smoke")
    target = cells[0]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sentinel = {"failure": None, "sentinel": True}
    (raw_dir / raw_filename(target)).write_text(json.dumps(sentinel))

    called: list[Cell] = []

    def runner(cell: Cell) -> dict:
        called.append(cell)
        return {"failure": None, "arm": cell.arm, "fresh": True}

    supervise("smoke", root=root, output_dir=tmp_path, runner=runner)

    assert target not in called
    assert len(called) == len(cells) - 1
    preserved = json.loads((raw_dir / raw_filename(target)).read_text())
    assert preserved == sentinel


def test_supervise_retains_failure_records(tmp_path):
    root = Path(__file__).parents[1]
    cells = plan_cells("smoke")

    def runner(cell: Cell) -> dict:
        if cell == cells[0]:
            return {"failure": {"reason": "boom"}, "arm": cell.arm}
        return {"failure": None, "arm": cell.arm}

    summary = supervise("smoke", root=root, output_dir=tmp_path, runner=runner)

    first_raw = json.loads(
        (tmp_path / "raw" / raw_filename(cells[0])).read_text()
    )
    assert first_raw["failure"] == {"reason": "boom"}
    assert summary["failed"] == 1
    assert summary["completed"] == len(cells)


def test_main_dry_run_writes_manifest_without_workers(tmp_path):
    exit_code = main([
        "--profile",
        "smoke",
        "--output-dir",
        str(tmp_path),
        "--dry-run",
    ])

    assert exit_code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["profile"] == "smoke"
    assert not (tmp_path / "raw").exists()
