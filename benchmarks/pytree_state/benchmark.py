# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Benchmark dense versus structured latent-state PyTrees.

Each cell runs in a fresh process with the persistent compilation cache
disabled. The worker separately measures lowering, backend compilation,
first fenced execution, and seven fenced steady executions by default.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DIMENSION = 16
_LAYOUTS = ("dense", "2-leaf", "4-leaf", "16-leaf")
_RESULT_PREFIX = "SMCX_PYTREE_BENCH_RESULT="
_ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    """Parse orchestrator and private worker arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=10_000, type=int)
    parser.add_argument("--timesteps", default=100, type=int)
    parser.add_argument("--repeats", default=7, type=int)
    parser.add_argument("--warmups", default=1, type=int)
    parser.add_argument("--blocks", default=5, type=int)
    parser.add_argument("--seed", default=20260719, type=int)
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=("cpu", "mps"),
        default=("cpu", "mps"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--platform", choices=("cpu", "mps"))
    parser.add_argument("--layout", choices=_LAYOUTS)
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _max_rss_bytes() -> int:
    """Return the process high-water RSS normalized to bytes."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _summary(samples: list[float]) -> dict[str, float]:
    """Summarize fenced timing samples without hiding dispersion."""
    ordered = sorted(samples)
    midpoint = len(ordered) // 2
    lower = ordered[:midpoint]
    upper = ordered[midpoint + len(ordered) % 2 :]
    q1 = statistics.median(lower)
    q3 = statistics.median(upper)
    return {
        "iqr_s": q3 - q1,
        "median_s": statistics.median(ordered),
        "min_s": min(ordered),
        "q1_s": q1,
        "q3_s": q3,
    }


def _memory_analysis(executable: Any) -> dict[str, int] | None:
    """Return backend executable-memory estimates when implemented."""
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
    """Return available allocator counters from the active backend."""
    try:
        stats = device.memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {
        str(name): int(value)
        for name, value in stats.items()
        if isinstance(value, int)
    }


def _checksum(arrays: list[tuple[str, Any]], jax: Any) -> str:
    """Hash shape, dtype, and bytes of device output arrays."""
    digest = hashlib.sha256()
    for name, value in arrays:
        host = jax.device_get(value)
        digest.update(name.encode())
        digest.update(str(host.shape).encode())
        digest.update(str(host.dtype).encode())
        digest.update(host.tobytes())
    return digest.hexdigest()


def _callbacks(layout: str, jnp: Any, jr: Any) -> tuple[Any, Any, Any]:
    """Build mathematically equivalent callbacks for one tree layout."""
    num_leaves = 1 if layout == "dense" else int(layout.split("-")[0])
    width = _DIMENSION // num_leaves

    def initial_sampler(key: Any, num_particles: int) -> Any:
        dense = jr.normal(
            key,
            (num_particles, _DIMENSION),
            dtype=jnp.float32,
        )
        if num_leaves == 1:
            return dense
        return tuple(
            dense[:, start : start + width]
            for start in range(0, _DIMENSION, width)
        )

    def transition_sampler(key: Any, state: Any) -> Any:
        del key
        if num_leaves == 1:
            return state + jnp.float32(0.01)
        return tuple(leaf + jnp.float32(0.01) for leaf in state)

    def log_observation_fn(emission: Any, state: Any) -> Any:
        first = state[0] if num_leaves == 1 else state[0][0]
        residual = (emission[0] - first) / jnp.float32(0.5)
        return -jnp.float32(0.5) * residual**2

    return initial_sampler, transition_sampler, log_observation_fn


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    """Run one fresh-process platform/layout cell."""
    if args.platform is None or args.layout is None:
        raise ValueError("worker requires --platform and --layout")

    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from smcx import bootstrap_filter

    jax.config.update("jax_enable_x64", False)
    device = jax.devices()[0]

    # Remove one-time backend/plugin startup from the measured workload compile.
    burn_arg = jax.device_put(jnp.ones(8, dtype=jnp.float32), device)
    burn = jax.jit(lambda value: value + 1).lower(burn_arg).compile()
    jax.block_until_ready(burn(burn_arg))

    initial, transition, log_observation = _callbacks(args.layout, jnp, jr)
    emissions = jnp.sin(
        jnp.arange(args.timesteps, dtype=jnp.float32) * jnp.float32(0.1)
    )[:, None]
    key = jr.key(args.seed)
    key = jax.device_put(key, device)
    emissions = jax.device_put(emissions, device)

    def operation(run_key: Any, observations: Any) -> Any:
        return bootstrap_filter(
            run_key,
            initial,
            transition,
            log_observation,
            observations,
            num_particles=args.n,
            resampling_threshold=1.1,
            store_history=False,
        )

    rss_before = _max_rss_bytes()
    lowered_start = time.perf_counter()
    lowered = jax.jit(operation).lower(key, emissions)
    lowering_s = time.perf_counter() - lowered_start
    compile_start = time.perf_counter()
    executable = lowered.compile()
    backend_compile_s = time.perf_counter() - compile_start

    first_start = time.perf_counter()
    output = executable(key, emissions)
    jax.block_until_ready(output)
    first_execution_s = time.perf_counter() - first_start
    for _ in range(max(args.warmups - 1, 0)):
        jax.block_until_ready(executable(key, emissions))

    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        output = executable(key, emissions)
        jax.block_until_ready(output)
        samples.append(time.perf_counter() - start)

    # Capture execution memory before the untimed parity checksum allocates a
    # packed dense copy of structured output.
    device_memory = _device_memory(device)
    rss_end = _max_rss_bytes()
    particles = output.filtered_particles
    if args.layout != "dense":
        particles = jnp.concatenate(particles, axis=-1)
    digest = _checksum(
        [
            ("marginal_loglik", output.marginal_loglik),
            ("particles", particles),
            ("log_weights", output.filtered_log_weights),
            ("ancestors", output.ancestors),
            ("ess", output.ess),
            ("increments", output.log_evidence_increments),
        ],
        jax,
    )
    increment_sum = jnp.sum(output.log_evidence_increments)
    evidence_error = float(
        jax.device_get(jnp.abs(increment_sum - output.marginal_loglik))
    )
    ess_min = float(jax.device_get(jnp.min(output.ess)))
    ess_max = float(jax.device_get(jnp.max(output.ess)))

    return {
        "backend": jax.default_backend(),
        "backend_compile_s": backend_compile_s,
        "checksum": digest,
        "compilation_cache_enabled": bool(
            jax.config.values["jax_enable_compilation_cache"]
        ),
        "compile_latency_s": lowering_s + backend_compile_s,
        "device": str(device),
        "device_memory_stats": device_memory,
        "evidence_sum_abs_error": evidence_error,
        "ess_max": ess_max,
        "ess_min": ess_min,
        "executable_memory": _memory_analysis(executable),
        "first_execution_s": first_execution_s,
        "layout": args.layout,
        "lowering_s": lowering_s,
        "max_rss_before_compile_bytes": rss_before,
        "max_rss_end_bytes": rss_end,
        "parameters": {
            "dimension": _DIMENSION,
            "num_particles": args.n,
            "repeats": args.repeats,
            "store_history": False,
            "timesteps": args.timesteps,
            "warmups": args.warmups,
        },
        "platform_requested": args.platform,
        "steady": _summary(samples),
        "steady_samples_s": samples,
        "versions": {
            "jax": jax.__version__,
            "jax-mps": _package_version("jax-mps"),
            "jaxlib": _package_version("jaxlib"),
            "numpy": _package_version("numpy"),
            "python": platform.python_version(),
            "smcx": _package_version("smcx"),
        },
    }


def _command_output(*command: str) -> str | None:
    """Run a metadata command and return stripped stdout on success."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_digest() -> str:
    """Hash the production Python sources used by this run."""
    digest = hashlib.sha256()
    for path in sorted((_ROOT / "src" / "smcx").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    """Capture machine and source metadata once for the full run."""
    return {
        "chip": _command_output("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpu_count": os.cpu_count(),
        "git_commit": _command_output("git", "rev-parse", "HEAD"),
        "git_status_short": _command_output("git", "status", "--short"),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "memory_bytes": _command_output("sysctl", "-n", "hw.memsize"),
        "platform": platform.platform(),
        "power": _command_output("pmset", "-g", "batt"),
        "source_tree_sha256": _source_digest(),
        "thermal": _command_output("pmset", "-g", "therm"),
    }


def _run_cell(
    args: argparse.Namespace,
    block: int,
    platform_name: str,
    layout: str,
) -> dict:
    """Launch and parse one isolated worker process."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--platform",
        platform_name,
        "--layout",
        layout,
        "--n",
        str(args.n),
        "--timesteps",
        str(args.timesteps),
        "--repeats",
        str(args.repeats),
        "--warmups",
        str(args.warmups),
        "--seed",
        str(args.seed),
    ]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = platform_name
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment.pop("JAX_MPS_ASYNC_DISPATCH", None)
    result = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            record = json.loads(line.removeprefix(_RESULT_PREFIX))
            record["block"] = block
            return record
    return {
        "block": block,
        "failure": {
            "returncode": result.returncode,
            "stderr": result.stderr[-4000:],
            "stdout": result.stdout[-4000:],
        },
        "layout": layout,
        "platform_requested": platform_name,
    }


def _orchestrate(args: argparse.Namespace) -> dict[str, Any]:
    """Run all requested cells in recorded pseudorandom order."""
    cells = [
        (block, platform_name, layout)
        for block in range(args.blocks)
        for platform_name in args.platforms
        for layout in _LAYOUTS
    ]
    random.Random(args.seed).shuffle(cells)
    records = [_run_cell(args, *cell) for cell in cells]
    parity: dict[str, bool] = {}
    for platform_name in args.platforms:
        checksums = {
            record.get("checksum")
            for record in records
            if record.get("platform_requested") == platform_name
        }
        parity[platform_name] = len(checksums) == 1 and None not in checksums
    return {
        "cell_order": cells,
        "environment": _environment(),
        "exact_layout_parity": parity,
        "method": {
            "backend_startup_burn": True,
            "blocks": args.blocks,
            "dispatch": "safe (JAX_MPS_ASYNC_DISPATCH unset)",
            "fresh_process_per_cell": True,
            "primary_estimate": "median of fresh-process medians",
            "persistent_compilation_cache": False,
            "timing_fence": "jax.block_until_ready(full posterior PyTree)",
        },
        "records": records,
        "schema_version": 2,
    }


def main() -> None:
    """Run a worker cell or orchestrate the complete benchmark."""
    args = _parse_args()
    if not args.worker and args.repeats < 5:
        raise ValueError("benchmark protocol requires at least five repeats")
    if args.n < 1 or args.timesteps < 2 or args.warmups < 1 or args.blocks < 1:
        raise ValueError(
            "N >= 1, timesteps >= 2, warmups >= 1, and blocks >= 1 required"
        )
    if args.worker:
        print(_RESULT_PREFIX + json.dumps(_worker(args), sort_keys=True))
        return
    result = _orchestrate(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(args.output)


if __name__ == "__main__":
    main()
