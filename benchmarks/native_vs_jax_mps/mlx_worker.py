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

from smcx.resampling import _normalized_cdf, _searchsorted


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


def _gather_scatter(size: int):
    """Build deterministic indexed gather/scatter inputs and oracle."""
    rng = np.random.default_rng(20260715)
    base_np = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    indices_np = rng.integers(0, size, size=size, dtype=np.int32)
    updates_np = base_np[indices_np] * np.float32(0.01)

    def operation(base, indices, updates):
        return base.at[indices].add(updates)

    expected = base_np.astype(np.float64)
    np.add.at(expected, indices_np, updates_np.astype(np.float64))
    inputs = (mx.array(base_np), mx.array(indices_np), mx.array(updates_np))
    return inputs, mx.compile(operation), expected


def _matmul(size: int):
    """Build dense matrix inputs and an f64 checksum oracle."""
    rng = np.random.default_rng(20260715)
    left_np = rng.normal(0.0, 0.1, size=(size, size)).astype(np.float32)
    right_np = rng.normal(0.0, 0.1, size=(size, size)).astype(np.float32)

    def operation(left, right):
        return mx.sum(left @ right)

    expected = np.asarray(
        np.sum(
            left_np.astype(np.float64) @ right_np.astype(np.float64),
            dtype=np.float64,
        )
    )
    inputs = (mx.array(left_np), mx.array(right_np))
    return inputs, mx.compile(operation), expected


def _random(size: int):
    """Build a fixed-key normal draw with moment output."""
    key = mx.random.key(20260715)

    def operation(random_key):
        samples = mx.random.normal(shape=(size,), key=random_key)
        return mx.stack((mx.mean(samples), mx.var(samples)))

    return (key,), mx.compile(operation), None


def _scan(size: int):
    """Build MLX's production-style Python loop over a compiled step."""
    initial_np = np.linspace(-1.0, 1.0, size, dtype=np.float32)

    def step(state):
        return mx.tanh(0.99 * state + 0.01)

    compiled_step = mx.compile(step)

    def operation(initial):
        state = initial
        for _ in range(100):
            state = compiled_step(state)
            if mx.default_device() == mx.Device(mx.cpu):
                mx.eval(state)
            else:
                mx.async_eval(state)
        return mx.concatenate((state, mx.sum(state)[None]))

    expected_state = initial_np.astype(np.float64)
    for _ in range(100):
        expected_state = np.tanh(0.99 * expected_state + 0.01)
    expected = np.concatenate((expected_state, [np.sum(expected_state)]))
    return (mx.array(initial_np),), operation, expected


def _systematic(size: int):
    """Build supplied systematic queries and the native right-bisect."""
    weights_np = np.full(size, 1.0 / size, dtype=np.float32)
    queries_np = (np.arange(size, dtype=np.float64) + 0.37) / size
    particles_np = np.linspace(-2.0, 2.0, size, dtype=np.float32)

    def operation(weights, queries, particles):
        cdf = _normalized_cdf(weights)
        ancestors = _searchsorted(cdf, queries)
        return mx.take(particles, ancestors)

    cdf = np.cumsum(weights_np.astype(np.float64))
    cdf /= cdf[-1]
    ancestors = np.searchsorted(cdf, queries_np, side="right")
    ancestors = np.clip(ancestors, 0, size - 1)
    expected = particles_np.astype(np.float64)[ancestors]
    inputs = (
        mx.array(weights_np),
        mx.array(queries_np.astype(np.float32)),
        mx.array(particles_np),
    )
    return inputs, mx.compile(operation), expected


def _build_workload(name: str, size: int):
    """Build one explicitly registered workload."""
    if name == "eltwise_reduce":
        value, operation, expected = _eltwise_reduce(size)
        return (value,), operation, np.asarray(expected)
    if name == "gather_scatter":
        return _gather_scatter(size)
    if name == "matmul":
        return _matmul(size)
    if name == "random":
        return _random(size)
    if name == "scan":
        return _scan(size)
    if name == "systematic":
        return _systematic(size)
    raise ValueError(f"unsupported workload: {name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("mlx_cpu", "mlx_gpu"), required=True)
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--warmups", required=True, type=int)
    parser.add_argument(
        "--workload",
        choices=(
            "eltwise_reduce",
            "gather_scatter",
            "matmul",
            "random",
            "scan",
            "systematic",
        ),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    """Run one benchmark block and emit one JSON result."""
    args = _parse_args()
    device = mx.cpu if args.arm == "mlx_cpu" else mx.gpu
    mx.set_default_device(mx.Device(device))
    inputs, operation, expected = _build_workload(args.workload, args.size)

    mx.reset_peak_memory()
    started = time.perf_counter()
    output = operation(*inputs)
    _fence(output)
    cold_s = time.perf_counter() - started

    actual_array = np.asarray(output)
    if args.workload == "random":
        mean, variance = (float(value) for value in actual_array)
        mean_limit = 5.0 / np.sqrt(args.size)
        variance_limit = 5.0 * np.sqrt(2.0 / (args.size - 1))
        passed = bool(
            abs(mean) <= mean_limit and abs(variance - 1.0) <= variance_limit
        )
        actual = {"mean": mean, "variance": variance}
        expected_json = {"mean": 0.0, "variance": 1.0}
        atol = {"mean": mean_limit, "variance": variance_limit}
        rtol = 0.0
    else:
        passed = bool(np.allclose(actual_array, expected, rtol=5e-5, atol=5e-6))
        actual = float(np.sum(actual_array, dtype=np.float64))
        expected_json = float(np.sum(expected, dtype=np.float64))
        atol = 5e-6
        rtol = 5e-5

    for _ in range(args.warmups):
        _fence(operation(*inputs))

    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        _fence(operation(*inputs))
        times.append(time.perf_counter() - started)

    result = {
        "arm": args.arm,
        "backend": "mlx",
        "block": args.block,
        "cold_s": cold_s,
        "correctness": {
            "actual": actual,
            "atol": atol,
            "expected": expected_json,
            "passed": passed,
            "rtol": rtol,
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
