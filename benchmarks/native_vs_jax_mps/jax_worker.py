# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process JAX CPU or jax-mps benchmark worker."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from common import SCHEMA_VERSION, summarize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("jax_cpu", "jax_mps_async", "jax_mps_sync"),
        required=True,
    )
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--warmups", required=True, type=int)
    parser.add_argument(
        "--workload", choices=("eltwise_reduce",), required=True
    )
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    """Return an installed package version without requiring jax-mps on CPU."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _peak_memory(device) -> int | None:
    """Read the best backend-reported peak-memory counter available."""
    stats = device.memory_stats()
    if not stats:
        return None
    for name in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
        if name in stats:
            return int(stats[name])
    return None


def main() -> None:
    """Run one benchmark block and emit one JSON result."""
    args = _parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)
    expected_backend = "cpu" if args.arm == "jax_cpu" else "mps"
    actual_backend = jax.default_backend()
    if actual_backend != expected_backend:
        raise RuntimeError(
            f"requested {expected_backend} backend, got {actual_backend}"
        )

    x_np = np.linspace(-3.0, 3.0, args.size, dtype=np.float32)
    x = jax.device_put(x_np)
    x.block_until_ready()

    def operation(value):
        return jnp.sum(jnp.tanh(value) * jax.nn.sigmoid(value) + 0.1 * value**2)

    compiled = jax.jit(operation)
    x64 = x_np.astype(np.float64)
    expected = float(
        np.sum(
            np.tanh(x64) / (1.0 + np.exp(-x64)) + 0.1 * x64**2,
            dtype=np.float64,
        )
    )

    started = time.perf_counter()
    output = compiled(x)
    output.block_until_ready()
    cold_s = time.perf_counter() - started

    actual = float(output)
    passed = bool(np.isclose(actual, expected, rtol=5e-5, atol=5e-6))

    for _ in range(args.warmups):
        compiled(x).block_until_ready()

    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        compiled(x).block_until_ready()
        times.append(time.perf_counter() - started)

    dispatch_mode = "cpu"
    if args.arm.startswith("jax_mps_"):
        dispatch_mode = (
            "async"
            if os.environ.get("JAX_MPS_ASYNC_DISPATCH") == "1"
            else "safe"
        )
    result = {
        "arm": args.arm,
        "backend": actual_backend,
        "block": args.block,
        "cold_s": cold_s,
        "correctness": {
            "actual": actual,
            "atol": 5e-6,
            "expected": expected,
            "passed": passed,
            "rtol": 5e-5,
        },
        "dispatch_mode": dispatch_mode,
        "failure": None,
        "parameters": {"size": args.size},
        "peak_memory_bytes": _peak_memory(jax.devices()[0]),
        "schema_version": SCHEMA_VERSION,
        "summary": summarize(times),
        "times_s": times,
        "versions": {
            "jax": jax.__version__,
            "jax-mps": _package_version("jax-mps"),
            "jaxlib": _package_version("jaxlib"),
            "python": platform.python_version(),
        },
        "workload": args.workload,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
