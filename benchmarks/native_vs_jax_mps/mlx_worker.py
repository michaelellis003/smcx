# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process native MLX benchmark worker."""

import argparse
import json
import platform
import time
from importlib.metadata import version

import mlx.core as mx
import numpy as np
from common import SCHEMA_VERSION, summarize


def _fence(value: mx.array) -> None:
    """Evaluate one result and wait for the device."""
    mx.eval(value)
    mx.synchronize()


def _eltwise_reduce(size: int):
    """Build the negative-control input, function, and f64 oracle."""
    x_np = np.linspace(-3.0, 3.0, size, dtype=np.float32)
    x = mx.array(x_np)

    def operation(value):
        return mx.sum(mx.tanh(value) * mx.sigmoid(value) + 0.1 * value**2)

    x64 = x_np.astype(np.float64)
    expected = float(
        np.sum(
            np.tanh(x64) / (1.0 + np.exp(-x64)) + 0.1 * x64**2,
            dtype=np.float64,
        )
    )
    return x, mx.compile(operation), expected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("mlx_cpu", "mlx_gpu"), required=True)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--warmups", required=True, type=int)
    parser.add_argument(
        "--workload", choices=("eltwise_reduce",), required=True
    )
    return parser.parse_args()


def main() -> None:
    """Run one benchmark block and emit one JSON result."""
    args = _parse_args()
    device = mx.cpu if args.arm == "mlx_cpu" else mx.gpu
    mx.set_default_device(mx.Device(device))
    value, operation, expected = _eltwise_reduce(args.size)

    mx.reset_peak_memory()
    started = time.perf_counter()
    output = operation(value)
    _fence(output)
    cold_s = time.perf_counter() - started

    actual = float(output.item())
    passed = bool(np.isclose(actual, expected, rtol=5e-5, atol=5e-6))

    for _ in range(args.warmups):
        _fence(operation(value))

    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        _fence(operation(value))
        times.append(time.perf_counter() - started)

    result = {
        "arm": args.arm,
        "backend": "mlx",
        "block": args.block,
        "cold_s": cold_s,
        "correctness": {
            "actual": actual,
            "atol": 5e-6,
            "expected": expected,
            "passed": passed,
            "rtol": 5e-5,
        },
        "dispatch_mode": "native",
        "failure": None,
        "parameters": {"size": args.size},
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "schema_version": SCHEMA_VERSION,
        "summary": summarize(times),
        "times_s": times,
        "versions": {
            "mlx": version("mlx"),
            "python": platform.python_version(),
        },
        "workload": args.workload,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
