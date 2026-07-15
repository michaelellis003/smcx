# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Supervisor for isolated native MLX and jax-mps benchmark workers."""

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from benchmarks.native_vs_jax_mps.common import PINNED_VERSIONS

ARMS = ("mlx_gpu", "mlx_cpu", "jax_mps_sync", "jax_mps_async", "jax_cpu")


def build_worker_command(
    *,
    root: Path,
    arm: str,
    block: int,
    repeats: int,
    size: int,
    warmups: int,
    workload: str,
) -> list[str]:
    """Construct one fully pinned fresh-process worker command."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    if arm.startswith("mlx_"):
        command = [
            sys.executable,
            str(root / "benchmarks/native_vs_jax_mps/mlx_worker.py"),
        ]
    else:
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.13",
            "--with",
            f"jax=={PINNED_VERSIONS['jax']}",
            "--with",
            f"jaxlib=={PINNED_VERSIONS['jaxlib']}",
        ]
        if arm.startswith("jax_mps_"):
            command.extend(["--with", f"jax-mps=={PINNED_VERSIONS['jax-mps']}"])
        command.extend([
            "python",
            str(root / "benchmarks/native_vs_jax_mps/jax_worker.py"),
        ])

    command.extend([
        "--arm",
        arm,
        "--block",
        str(block),
        "--repeats",
        str(repeats),
        "--size",
        str(size),
        "--warmups",
        str(warmups),
        "--workload",
        workload,
    ])
    return command


def worker_environment(
    arm: str,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a sanitized environment for one explicit backend arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    environment = dict(os.environ if base is None else base)
    environment.pop("JAX_MPS_ASYNC_DISPATCH", None)
    environment.pop("JAX_PLATFORM_NAME", None)
    environment.pop("JAX_PLATFORMS", None)

    if arm.startswith("jax_mps_"):
        environment["JAX_PLATFORMS"] = "mps"
    elif arm == "jax_cpu":
        environment["JAX_PLATFORMS"] = "cpu"
    if arm == "jax_mps_async":
        environment["JAX_MPS_ASYNC_DISPATCH"] = "1"
    return environment
