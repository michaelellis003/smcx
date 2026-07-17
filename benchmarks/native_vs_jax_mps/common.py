# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for the native MLX versus jax-mps benchmark."""

import random
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

# StableHLO textual form assigns each operation to an SSA value, either bare
# (`%0 = stablehlo.tanh ...`) or in the generic quoted form
# (`%3 = "stablehlo.reduce"(...)`). Declarations such as `func.func` have no
# `=` and are intentionally excluded from the operation census.
_STABLEHLO_OP = re.compile(r'=\s*"?([a-z_][\w]*\.[\w.]+)"?')

SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 20260715

PINNED_VERSIONS = {
    "jax": "0.10.2",
    "jax-mps": "0.10.9",
    "jaxlib": "0.10.2",
    "mlx": "0.32.0",
}

WORKLOAD_GRIDS = {
    "eltwise_reduce": (10_000, 1_000_000, 10_000_000),
    "gather_scatter": (10_000, 100_000, 1_000_000),
    "lgssm_pf": (10_000, 100_000, 1_000_000),
    # Report-only tuned counter-experiment (see run.REPORT_ONLY_WORKLOADS): the
    # same bootstrap filter with unconditional resampling and no stored history,
    # returning only the marginal log-likelihood. It gives jax-mps its strongest
    # fair implementation and never enters the pre-registered verdict matrix.
    "lgssm_pf_nohist": (10_000, 100_000, 1_000_000),
    "matmul": (256, 1_024, 2_048),
    "random": (10_000, 1_000_000, 10_000_000),
    "scan": (10_000, 100_000, 1_000_000),
    "systematic": (10_000, 100_000, 1_000_000),
}

LGSSM = {
    "a": 0.9,
    "m0": 0.0,
    "p0": 1.0,
    "q": 0.5,
    "r": 0.3,
    "timesteps": 100,
}

REQUIRED_RESULT_FIELDS = {
    "arm",
    "backend",
    "block",
    "cold_s",
    "correctness",
    "dispatch_mode",
    "failure",
    "parameters",
    "peak_memory_bytes",
    "schema_version",
    "summary",
    "times_s",
    "versions",
    "workload",
}


def lgssm_data() -> tuple[np.ndarray, float]:
    """Generate the committed-seed LGSSM data and exact f64 log evidence."""
    rng = np.random.default_rng(20260714)
    state = np.empty(int(LGSSM["timesteps"]), dtype=np.float64)
    state[0] = rng.normal(LGSSM["m0"], np.sqrt(LGSSM["p0"]))
    for index in range(1, state.size):
        state[index] = LGSSM["a"] * state[index - 1] + rng.normal(
            0.0, np.sqrt(LGSSM["q"])
        )
    observations = state + rng.normal(0.0, np.sqrt(LGSSM["r"]), state.size)

    mean = LGSSM["m0"]
    variance = LGSSM["p0"]
    log_evidence = 0.0
    for index, observation in enumerate(observations):
        if index:
            mean = LGSSM["a"] * mean
            variance = LGSSM["a"] ** 2 * variance + LGSSM["q"]
        innovation_variance = variance + LGSSM["r"]
        innovation = observation - mean
        log_evidence -= 0.5 * (
            np.log(2.0 * np.pi * innovation_variance)
            + innovation**2 / innovation_variance
        )
        gain = variance / innovation_variance
        mean += gain * innovation
        variance *= 1.0 - gain
    return observations.astype(np.float32), float(log_evidence)


def count_stablehlo_ops(text: str) -> dict[str, int]:
    """Count StableHLO operations by dialect-qualified name in lowered IR."""
    counts: dict[str, int] = {}
    for match in _STABLEHLO_OP.finditer(text):
        name = match.group(1)
        counts[name] = counts.get(name, 0) + 1
    return counts


def summarize(times: Sequence[float]) -> dict[str, float]:
    """Return robust statistics without discarding the raw observations."""
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("times must be finite and non-negative")

    median = float(np.median(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return {
        "iqr_s": float(q3 - q1),
        "mad_s": float(np.median(np.abs(values - median))),
        "median_s": median,
        "min_s": float(np.min(values)),
        "q1_s": float(q1),
        "q3_s": float(q3),
    }


def balanced_orders(
    arms: Sequence[str],
    *,
    blocks: int,
    seed: int,
) -> list[list[str]]:
    """Shuffle once, then rotate arms through process-order positions."""
    if not arms:
        raise ValueError("at least one arm is required")
    if len(set(arms)) != len(arms):
        raise ValueError("arms must be unique")
    if blocks < 1:
        raise ValueError("blocks must be positive")

    first = list(arms)
    random.Random(seed).shuffle(first)
    return [
        first[offset:] + first[:offset]
        for offset in (block % len(first) for block in range(blocks))
    ]


def bootstrap_ratio_ci(
    *,
    native: Sequence[float],
    compatibility: Sequence[float],
    draws: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Estimate a paired percentile interval for compatibility/native time."""
    native_values = np.asarray(native, dtype=np.float64)
    compatibility_values = np.asarray(compatibility, dtype=np.float64)
    if native_values.shape != compatibility_values.shape:
        raise ValueError("process-median sequences must have matching shapes")
    if native_values.ndim != 1 or native_values.size == 0:
        raise ValueError("process-median sequences must be non-empty and 1D")
    if draws < 1:
        raise ValueError("draws must be positive")
    if (
        not np.all(np.isfinite(native_values))
        or not np.all(np.isfinite(compatibility_values))
        or np.any(native_values <= 0)
        or np.any(compatibility_values <= 0)
    ):
        raise ValueError("process medians must be finite and positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        native_values.size,
        size=(draws, native_values.size),
    )
    ratios = np.median(compatibility_values[indices], axis=1) / np.median(
        native_values[indices], axis=1
    )
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {
        "estimate": float(
            np.median(compatibility_values) / np.median(native_values)
        ),
        "high": float(high),
        "low": float(low),
    }


def kalman_gate(
    *,
    log_evidence: Sequence[float],
    oracle: float,
) -> dict[str, Any]:
    """Apply the pre-registered one-sided Monte Carlo correctness gate."""
    values = np.asarray(log_evidence, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("log_evidence must contain at least two replicates")
    if not np.all(np.isfinite(values)) or not np.isfinite(oracle):
        raise ValueError("log evidence and oracle must be finite")

    standard_deviation = float(np.std(values, ddof=1))
    error = float(np.mean(values) - oracle)
    upper = 3.0 * standard_deviation / np.sqrt(values.size)
    lower = -(upper + 0.5 * standard_deviation**2)
    return {
        "error": error,
        "log_evidence": values.tolist(),
        "lower_error_bound": float(lower),
        "mean": float(np.mean(values)),
        "oracle": float(oracle),
        "passed": bool(lower <= error <= upper),
        "replicates": int(values.size),
        "standard_deviation": standard_deviation,
        "upper_error_bound": float(upper),
    }


def validate_result(result: dict[str, Any]) -> None:
    """Validate the stable worker-result envelope."""
    missing = REQUIRED_RESULT_FIELDS - result.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"result is missing required fields: {names}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if result["workload"] not in WORKLOAD_GRIDS:
        raise ValueError("unregistered workload")
    if result["failure"] is None:
        times = result["times_s"]
        if not isinstance(times, list) or not times:
            raise ValueError("times_s must retain non-empty raw timings")
        expected = summarize(times)
        if result["summary"] != expected:
            raise ValueError("summary does not match times_s")
        correctness = result["correctness"]
        if not isinstance(correctness, dict) or "passed" not in correctness:
            raise ValueError("correctness must contain passed")
