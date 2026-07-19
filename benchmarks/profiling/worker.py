# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process worker for one current-JAX profiling cell."""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # Allow direct ``python .../worker.py`` use.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.profiling.common import (
    DEFAULT_SEED,
    PLATFORMS,
    SCHEMA_VERSION,
    WORKLOADS,
    Cell,
    host_environment,
    package_versions,
    profiling_runtime_flags,
    source_metadata,
    summarize,
    validate_result,
    worker_environment,
)

RESULT_PREFIX = "SMCX_PROFILE_RESULT="


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the serialized fresh-process cell."""
    parser = argparse.ArgumentParser(
        description="Execute one isolated smcx profiling cell.",
    )
    parser.add_argument("--cell-json", required=True)
    parser.add_argument(
        "--phase",
        choices=("timing", "validation"),
        default="timing",
    )
    return parser.parse_args(argv)


def _decode_cell(payload: str) -> Cell:
    """Decode and structurally validate a supervisor-provided cell."""
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("cell-json must be valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("cell-json must encode an object")
    try:
        cell = Cell(**raw)
    except TypeError as error:
        raise ValueError(
            "cell-json does not match the Cell contract"
        ) from error

    if cell.workload not in WORKLOADS:
        raise ValueError(f"unknown workload: {cell.workload}")
    if cell.platform not in PLATFORMS:
        raise ValueError(f"unknown platform: {cell.platform}")
    if cell.execution_mode != WORKLOADS[cell.workload].execution_mode:
        raise ValueError("cell execution_mode does not match workload registry")
    if cell.block < 0 or cell.warmups < 0 or cell.repeats < 1:
        raise ValueError(
            "block/warmups must be non-negative and repeats positive"
        )
    if cell.correctness_replicates < 0:
        raise ValueError("correctness_replicates must be non-negative")
    if not isinstance(cell.parameters, dict):
        raise ValueError("cell parameters must be an object")
    return cell


def _max_rss_bytes() -> int:
    """Return process high-water RSS normalized to bytes."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _memory_analysis(executable: Any) -> dict[str, int] | None:
    """Return executable-memory estimates when the backend exposes them."""
    try:
        stats = executable.memory_analysis()
    except Exception:
        return None
    if stats is None:
        return None
    names = (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "output_size_in_bytes",
        "peak_memory_in_bytes",
        "temp_size_in_bytes",
    )
    return {
        name: int(value)
        for name in names
        if (value := getattr(stats, name, None)) is not None
    }


def _device_memory(device: Any) -> dict[str, int] | None:
    """Return allocator counters when the selected device exposes them."""
    try:
        stats = device.memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {
        str(name): int(value)
        for name, value in stats.items()
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
    }


def _dispatch_mode(platform_name: str) -> str:
    """Describe the active dispatch contract without importing the plugin."""
    if platform_name == "cpu":
        return "asynchronous"
    if os.environ.get("JAX_MPS_ASYNC_DISPATCH") == "1":
        return "asynchronous"
    return "safe"


def _versions(jax: Any) -> dict[str, str | None]:
    """Capture the runtime versions needed to interpret a cell."""
    versions = package_versions()
    versions["jax"] = jax.__version__
    return versions


def _system_value(*command: str) -> str | None:
    """Return a short platform command value when the command exists."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _timing_environment() -> dict[str, str | None]:
    """Capture power and thermal state at one measurement boundary."""
    return {
        "power_status": _system_value("pmset", "-g", "batt"),
        "thermal_status": _system_value("pmset", "-g", "therm"),
    }


def _runtime_environment(
    device: Any,
    *,
    pre_timing: Mapping[str, Any] | None = None,
    post_timing: Mapping[str, Any] | None = None,
    post_cell: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture stable runtime metadata and optional timing boundaries."""
    environment = {
        **host_environment(),
        "device_id": int(getattr(device, "id", 0)),
        "device_kind": str(getattr(device, "device_kind", "unknown")),
        "runtime_flags": profiling_runtime_flags(),
    }
    if pre_timing is not None:
        environment["pre_timing"] = dict(pre_timing)
    if post_timing is not None:
        environment["post_timing"] = dict(post_timing)
    if post_cell is not None:
        environment["post_cell"] = dict(post_cell)
    return environment


def _require_sanitized_runtime(platform: str) -> None:
    """Refuse to emit evidence from an unattested interpreter environment."""
    expected = profiling_runtime_flags(worker_environment(platform, base={}))
    actual = profiling_runtime_flags()
    if actual != expected:
        raise RuntimeError(
            "worker runtime flags do not match the sanitized contract"
        )
    inherited = {
        name for name in ("PYTHONHOME", "PYTHONPATH") if name in os.environ
    }
    if inherited:
        names = ", ".join(sorted(inherited))
        raise RuntimeError(f"worker inherited forbidden Python paths: {names}")


def _jsonable(value: Any) -> Any:
    """Convert device/NumPy scalar diagnostics to stable JSON values."""
    if isinstance(value, Mapping):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return str(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _burn_backend(jax: Any, jnp: Any, device: Any) -> None:
    """Remove one-time backend/plugin startup from workload lifecycle timing."""
    burn_arg = jax.device_put(jnp.ones(8, dtype=jnp.float32), device)
    executable = jax.jit(lambda value: value + 1).lower(burn_arg).compile()
    jax.block_until_ready(executable(burn_arg))


def _measure_whole_program(
    operation: Any,
    arguments: tuple[Any, ...],
    *,
    jax: Any,
    repeats: int,
    warmups: int,
) -> tuple[Any, dict[str, float | str | None], float, list[float], Any]:
    """Measure an outer-jittable workload's complete compile lifecycle."""
    lowering_started = time.perf_counter()
    lowered = jax.jit(operation).lower(*arguments)
    lowering_s = time.perf_counter() - lowering_started

    compile_started = time.perf_counter()
    executable = lowered.compile()
    backend_compile_s = time.perf_counter() - compile_started

    first_started = time.perf_counter()
    output = executable(*arguments)
    jax.block_until_ready(output)
    first_execution_s = time.perf_counter() - first_started

    for _ in range(max(warmups - 1, 0)):
        jax.block_until_ready(executable(*arguments))

    steady_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = executable(*arguments)
        jax.block_until_ready(output)
        steady_times.append(time.perf_counter() - started)

    lifecycle: dict[str, float | str | None] = {
        "backend_compile_s": backend_compile_s,
        "lowering_s": lowering_s,
        "unavailable_reason": None,
    }
    return output, lifecycle, first_execution_s, steady_times, executable


def _measure_host_shell(
    operation: Any,
    arguments: tuple[Any, ...],
    *,
    jax: Any,
    repeats: int,
    warmups: int,
) -> tuple[Any, dict[str, float | str | None], float, list[float], None]:
    """Measure a host-controlled public API end to end without an outer jit."""
    first_started = time.perf_counter()
    output = operation(*arguments)
    jax.block_until_ready(output)
    first_execution_s = time.perf_counter() - first_started

    for _ in range(max(warmups - 1, 0)):
        jax.block_until_ready(operation(*arguments))

    steady_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = operation(*arguments)
        jax.block_until_ready(output)
        steady_times.append(time.perf_counter() - started)

    lifecycle: dict[str, float | str | None] = {
        "backend_compile_s": None,
        "lowering_s": None,
        "unavailable_reason": "host_controlled",
    }
    return output, lifecycle, first_execution_s, steady_times, None


_RESAMPLER_COMMITTED_KEY_COUNT = 8
_RESAMPLER_EXTENSION_TAG = 0x534D4358


def _correctness_keys(
    jax: Any,
    *,
    workload: str,
    count: int,
) -> Any:
    """Return fixed validation keys without rerolling resampler failures."""
    root = jax.random.key(DEFAULT_SEED + 1)
    if not workload.startswith("resample_"):
        return jax.random.split(root, count)

    committed = jax.random.split(root, _RESAMPLER_COMMITTED_KEY_COUNT)
    if count <= _RESAMPLER_COMMITTED_KEY_COUNT:
        return committed[:count]
    extension_root = jax.random.fold_in(root, _RESAMPLER_EXTENSION_TAG)
    extension = jax.numpy.stack([
        jax.random.fold_in(extension_root, index)
        for index in range(count - _RESAMPLER_COMMITTED_KEY_COUNT)
    ])
    return jax.numpy.concatenate((committed, extension))


def _run_correctness_replicates(
    prepared: Any,
    arguments: tuple[Any, ...],
    *,
    count: int,
    device: Any,
    executable: Any,
    jax: Any,
    workload: str,
) -> list[Any]:
    """Run independent, fenced outputs outside all measured regions."""
    keys = _correctness_keys(jax, workload=workload, count=count)
    outputs = []
    operation = executable if executable is not None else prepared.operation
    for key in keys:
        replicate_arguments = (
            jax.device_put(key, device),
            *arguments[1:],
        )
        output = operation(*replicate_arguments)
        jax.block_until_ready(output)
        outputs.append(output)
    return outputs


def run_cell(cell: Cell) -> dict[str, Any]:
    """Execute one prepared cell and return a validated result envelope."""
    _require_sanitized_runtime(cell.platform)
    # Imports occur only after the supervisor-selected environment is active.
    import jax
    import jax.numpy as jnp

    from benchmarks.profiling.workloads import prepare_workload

    actual_backend = jax.default_backend()
    if actual_backend != cell.platform:
        raise RuntimeError(
            f"requested {cell.platform} backend, got {actual_backend}"
        )
    device = jax.devices()[0]
    _burn_backend(jax, jnp, device)

    prepared = prepare_workload(
        cell.workload,
        parameters=cell.parameters,
        seed=DEFAULT_SEED,
    )
    if prepared.execution_mode != cell.execution_mode:
        raise ValueError("prepared execution_mode does not match cell")

    arguments = jax.device_put(tuple(prepared.arguments), device)
    jax.block_until_ready(arguments)
    rss_before = _max_rss_bytes()
    pre_timing = _timing_environment()

    if cell.execution_mode == "whole_program_jit":
        measured = _measure_whole_program(
            prepared.operation,
            arguments,
            jax=jax,
            repeats=cell.repeats,
            warmups=cell.warmups,
        )
    elif cell.execution_mode == "host_shell":
        measured = _measure_host_shell(
            prepared.operation,
            arguments,
            jax=jax,
            repeats=cell.repeats,
            warmups=cell.warmups,
        )
    else:
        raise ValueError(f"unknown execution_mode: {cell.execution_mode}")
    post_timing = _timing_environment()

    output, lifecycle, first_execution_s, steady_times, executable = measured
    # Capture memory before correctness extraction may copy large outputs to
    # host or construct diagnostic reductions outside the timed operation.
    memory = {
        "device_stats": _device_memory(device),
        "executable_analysis": (
            None if executable is None else _memory_analysis(executable)
        ),
        "process_max_rss_before_measurement_bytes": rss_before,
        "process_max_rss_bytes": _max_rss_bytes(),
    }
    correctness = _jsonable(prepared.check(output))
    if not isinstance(correctness, dict):
        raise TypeError("workload correctness check must return a mapping")
    if cell.correctness_replicates:
        outputs = _run_correctness_replicates(
            prepared,
            arguments,
            count=cell.correctness_replicates,
            device=device,
            executable=executable,
            jax=jax,
            workload=cell.workload,
        )
        replicated = _jsonable(prepared.check_replicates(outputs))
        if not isinstance(replicated, dict):
            raise TypeError(
                "replicated correctness check must return a mapping"
            )
        correctness["replicated"] = replicated
        correctness["passed"] = bool(
            correctness.get("passed") and replicated.get("passed")
        )
    else:
        correctness["replicated"] = {
            "gate": "not_requested",
            "passed": True,
            "replicates": 0,
        }
    work_metrics = _jsonable(prepared.measure_work(output))
    post_cell = _timing_environment()

    result = {
        "algorithm": prepared.algorithm,
        "backend": actual_backend,
        "block": cell.block,
        "correctness": correctness,
        "correctness_replicates": cell.correctness_replicates,
        "correctness_level": (
            WORKLOADS[cell.workload].replicated_correctness_level
            if cell.correctness_replicates
            else "structural"
        ),
        "dispatch_mode": _dispatch_mode(cell.platform),
        "environment": _runtime_environment(
            device,
            pre_timing=pre_timing,
            post_timing=post_timing,
            post_cell=post_cell,
        ),
        "execution_mode": cell.execution_mode,
        "failure": None,
        "first_execution_s": first_execution_s,
        "lifecycle": lifecycle,
        "memory": memory,
        "model": prepared.model,
        "parameters": dict(cell.parameters),
        "platform_requested": cell.platform,
        "repeats": cell.repeats,
        "schema_version": SCHEMA_VERSION,
        "source": source_metadata(),
        "steady_summary": summarize(steady_times),
        "steady_times_s": steady_times,
        "versions": _versions(jax),
        "work_metrics": work_metrics,
        "workload": cell.workload,
        "warmups": cell.warmups,
    }
    result = _jsonable(result)
    validate_result(result)
    return result


def run_validation(cell: Cell) -> dict[str, Any]:
    """Run only replicated oracle validation in a fresh process."""
    if cell.correctness_replicates < 1:
        raise ValueError("validation requires at least one replicate")
    _require_sanitized_runtime(cell.platform)

    # Imports occur only after the supervisor-selected environment is active.
    import jax
    import jax.numpy as jnp

    from benchmarks.profiling.workloads import prepare_workload

    actual_backend = jax.default_backend()
    if actual_backend != cell.platform:
        raise RuntimeError(
            f"requested {cell.platform} backend, got {actual_backend}"
        )
    device = jax.devices()[0]
    _burn_backend(jax, jnp, device)

    prepared = prepare_workload(
        cell.workload,
        parameters=cell.parameters,
        seed=DEFAULT_SEED,
    )
    if prepared.execution_mode != cell.execution_mode:
        raise ValueError("prepared execution_mode does not match cell")

    arguments = jax.device_put(tuple(prepared.arguments), device)
    jax.block_until_ready(arguments)
    executable = None
    if cell.execution_mode == "whole_program_jit":
        executable = jax.jit(prepared.operation).lower(*arguments).compile()
    elif cell.execution_mode != "host_shell":
        raise ValueError(f"unknown execution_mode: {cell.execution_mode}")

    outputs = _run_correctness_replicates(
        prepared,
        arguments,
        count=cell.correctness_replicates,
        device=device,
        executable=executable,
        jax=jax,
        workload=cell.workload,
    )
    replicated = _jsonable(prepared.check_replicates(outputs))
    if not isinstance(replicated, dict):
        raise TypeError("replicated correctness check must return a mapping")
    if replicated.get("replicates") != cell.correctness_replicates:
        raise ValueError(
            "replicated correctness result does not match scheduled count"
        )

    result = {
        "backend": actual_backend,
        "block": cell.block,
        "correctness_level": (
            WORKLOADS[cell.workload].replicated_correctness_level
        ),
        "correctness_replicates": cell.correctness_replicates,
        "dispatch_mode": _dispatch_mode(cell.platform),
        "environment": _runtime_environment(device),
        "execution_mode": cell.execution_mode,
        "parameters": dict(cell.parameters),
        "platform_requested": cell.platform,
        "replicated": replicated,
        "schema_version": SCHEMA_VERSION,
        "source": source_metadata(),
        "versions": _versions(jax),
        "workload": cell.workload,
    }
    return _jsonable(result)


def main(argv: list[str] | None = None) -> int:
    """Execute the serialized cell and print one prefixed JSON record."""
    args = _parse_args(argv)
    cell = _decode_cell(args.cell_json)
    result = run_cell(cell) if args.phase == "timing" else run_validation(cell)
    print(
        RESULT_PREFIX
        + json.dumps(
            result,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
