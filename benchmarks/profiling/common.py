# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Shared contracts for the current-JAX profiling harness.

This module deliberately does not import JAX.  The supervisor imports it
before choosing a worker backend, so importing the benchmark registry must not
initialize a backend in the parent process.
"""

import hashlib
import json
import math
import os
import platform as platform_module
import random
import subprocess
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260719
DEFAULT_ORDER_SEED = DEFAULT_SEED
SEED_CONTRACT = {
    "data_seed_base": DEFAULT_SEED,
    "data_seed_offsets": {
        "lgssm": 0,
        "stochastic_volatility": 1,
        "tracking": 3,
        "unknown_ar": 2,
    },
    "inference_seed": DEFAULT_SEED,
    "validation_seed": DEFAULT_SEED + 1,
}


def _package_version(name: str) -> str | None:
    """Return an installed distribution version without importing it."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    """Return the complete package identity relevant to this campaign."""
    return {
        "jax": _package_version("jax"),
        "jax-mps": _package_version("jax-mps"),
        "jaxlib": _package_version("jaxlib"),
        "numpy": np.__version__,
        "python": platform_module.python_version(),
        "smcx": _package_version("smcx"),
        "tfp-nightly": _package_version("tfp-nightly"),
    }


def _command_value(
    command: Sequence[str],
    *,
    allow_empty: bool = False,
    cwd: Path | None = None,
) -> str | None:
    """Return a short command value when its executable is available."""
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            cwd=cwd,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output if output or allow_empty else None


def host_environment() -> dict[str, Any]:
    """Return stable host identity fields without initializing JAX."""
    memory_text = _command_value(("sysctl", "-n", "hw.memsize"))
    return {
        "cpu_count": os.cpu_count(),
        "cpu_model": _command_value((
            "sysctl",
            "-n",
            "machdep.cpu.brand_string",
        )),
        "hardware_model": _command_value(("sysctl", "-n", "hw.model")),
        "machine": platform_module.machine(),
        "macos": platform_module.mac_ver()[0] or None,
        "macos_build": _command_value(("sw_vers", "-buildVersion")),
        "os": platform_module.system(),
        "os_release": platform_module.release(),
        "physical_memory_bytes": (
            int(memory_text) if memory_text is not None else None
        ),
        "processor": platform_module.processor() or None,
    }


def source_metadata(root: Path | None = None) -> dict[str, Any]:
    """Hash the actual implementation, protocol, and dependency lock."""
    repository = (
        Path(__file__).resolve().parents[2] if root is None else Path(root)
    )
    source_paths = [
        *sorted((repository / "src/smcx").rglob("*.py")),
        *sorted((repository / "benchmarks/profiling").glob("*.py")),
        repository / "benchmarks/profiling/PROTOCOL.md",
        repository / "pyproject.toml",
        repository / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        if not path.is_file():
            continue
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    lock_path = repository / "uv.lock"
    lock_digest = (
        hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if lock_path.is_file()
        else None
    )
    status = _command_value(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src/smcx",
            "benchmarks/profiling",
            "pyproject.toml",
            "uv.lock",
        ),
        allow_empty=True,
        cwd=repository,
    )
    return {
        "git_commit": _command_value(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
        ),
        "git_dirty": None if status is None else bool(status),
        "lock_sha256": lock_digest,
        "source_sha256": digest.hexdigest(),
    }


def campaign_identity() -> dict[str, Any]:
    """Return the immutable host, package, lock, and source identity."""
    return {
        "host": host_environment(),
        "packages": package_versions(),
        "source": source_metadata(),
    }


INFERENCE_ALGORITHMS = (
    "bootstrap",
    "auxiliary",
    "guided",
    "liu_west",
    "temper",
    "smc2",
)
RESAMPLERS = (
    "systematic",
    "stratified",
    "multinomial",
    "residual",
)
# Resamplers are independently timed public algorithms in this campaign, so
# the coverage contract includes them alongside the six inference entry
# points.  The narrower tuple remains available to callers that need the
# distinction.
ALGORITHMS = INFERENCE_ALGORITHMS + RESAMPLERS

PLATFORMS = ("cpu", "mps")
EXECUTION_MODES = ("whole_program_jit", "host_shell")


class Profile(NamedTuple):
    """Fresh-process measurement schedule for a registered profile."""

    blocks: int
    repeats: int
    warmups: int


PROFILES = {
    "smoke": Profile(blocks=1, repeats=1, warmups=1),
    "baseline": Profile(blocks=5, repeats=7, warmups=1),
    "filter-regimes": Profile(blocks=5, repeats=7, warmups=1),
    "representation": Profile(blocks=5, repeats=7, warmups=1),
    "scaling": Profile(blocks=5, repeats=7, warmups=1),
}


class WorkloadSpec(NamedTuple):
    """Static metadata and profile parameters for one workload."""

    algorithm: str
    model: str
    execution_mode: str
    smoke_parameters: Mapping[str, Any]
    baseline_parameters: Mapping[str, Any]
    baseline_correctness_replicates: int = 0
    replicated_correctness_level: str = "structural"
    profiles: tuple[str, ...] = ("smoke", "baseline")


_L1_SMOKE = {
    "num_particles": 128,
    "timesteps": 20,
    "observation_regime": "calibrated",
    "resampling_threshold": 0.5,
    "store_history": False,
}
_L1_BASELINE = {
    "num_particles": 10_000,
    "timesteps": 100,
    "observation_regime": "calibrated",
    "resampling_threshold": 0.5,
    "store_history": False,
}
_RESAMPLER_SMOKE = {
    "num_particles": 1_024,
    "weight_regime": "moderately_uneven",
}
_RESAMPLER_BASELINE = {
    "num_particles": 100_000,
    "weight_regime": "moderately_uneven",
}


WORKLOADS: dict[str, WorkloadSpec] = {
    "bootstrap_lgssm": WorkloadSpec(
        algorithm="bootstrap",
        model="lgssm",
        execution_mode="whole_program_jit",
        smoke_parameters=_L1_SMOKE,
        baseline_parameters=_L1_BASELINE,
        baseline_correctness_replicates=20,
        replicated_correctness_level="oracle_accuracy",
        profiles=(
            "smoke",
            "baseline",
            "filter-regimes",
            "scaling",
        ),
    ),
    "auxiliary_lgssm": WorkloadSpec(
        algorithm="auxiliary",
        model="lgssm",
        execution_mode="whole_program_jit",
        smoke_parameters=_L1_SMOKE,
        baseline_parameters=_L1_BASELINE,
        baseline_correctness_replicates=20,
        replicated_correctness_level="oracle_accuracy",
        profiles=("smoke", "baseline", "filter-regimes", "scaling"),
    ),
    "guided_lgssm": WorkloadSpec(
        algorithm="guided",
        model="lgssm",
        execution_mode="whole_program_jit",
        smoke_parameters=_L1_SMOKE,
        baseline_parameters=_L1_BASELINE,
        baseline_correctness_replicates=20,
        replicated_correctness_level="oracle_accuracy",
        profiles=("smoke", "baseline", "filter-regimes", "scaling"),
    ),
    "bootstrap_sv": WorkloadSpec(
        algorithm="bootstrap",
        model="stochastic_volatility",
        execution_mode="whole_program_jit",
        smoke_parameters={
            "num_particles": 128,
            "timesteps": 30,
            "resampling_threshold": 0.5,
            "store_history": False,
        },
        baseline_parameters={
            "num_particles": 10_000,
            "timesteps": 500,
            "resampling_threshold": 0.5,
            "store_history": False,
        },
        profiles=("smoke", "baseline", "scaling"),
    ),
    "liu_west_unknown_ar": WorkloadSpec(
        algorithm="liu_west",
        model="unknown_ar_lgssm",
        execution_mode="whole_program_jit",
        smoke_parameters={
            "num_particles": 128,
            "parameter_dimension": 1,
            "timesteps": 20,
            "resampling_threshold": 0.5,
            "shrinkage": 0.95,
            "store_history": False,
        },
        baseline_parameters={
            "num_particles": 10_000,
            "parameter_dimension": 1,
            "timesteps": 100,
            "resampling_threshold": 0.5,
            "shrinkage": 0.95,
            "store_history": False,
        },
        baseline_correctness_replicates=12,
        replicated_correctness_level="oracle_accuracy",
        profiles=("smoke", "baseline", "scaling"),
    ),
    "temper_gaussian": WorkloadSpec(
        algorithm="temper",
        model="gaussian_target",
        execution_mode="host_shell",
        smoke_parameters={
            "dimension": 4,
            "num_mcmc_steps": 1,
            "num_particles": 128,
            "target_ess": 0.5,
        },
        baseline_parameters={
            "dimension": 32,
            "num_mcmc_steps": 5,
            "num_particles": 10_000,
            "target_ess": 0.5,
        },
        baseline_correctness_replicates=12,
        replicated_correctness_level="oracle_accuracy",
        profiles=("smoke", "baseline", "scaling"),
    ),
    "smc2_forward": WorkloadSpec(
        algorithm="smc2",
        model="unknown_ar_lgssm",
        execution_mode="host_shell",
        smoke_parameters={
            "ess_threshold": 0.0,
            "num_pmmh_steps": 1,
            "num_theta": 8,
            "num_x": 16,
            "store_history": False,
            "timesteps": 8,
        },
        baseline_parameters={
            "ess_threshold": 0.0,
            "num_pmmh_steps": 1,
            "num_theta": 128,
            "num_x": 256,
            "store_history": False,
            "timesteps": 40,
        },
        baseline_correctness_replicates=8,
        replicated_correctness_level="oracle_accuracy",
        profiles=("smoke", "baseline", "scaling"),
    ),
    "smc2_forced": WorkloadSpec(
        algorithm="smc2",
        model="unknown_ar_lgssm",
        execution_mode="host_shell",
        smoke_parameters={
            "ess_threshold": 1.1,
            "num_pmmh_steps": 1,
            "num_theta": 8,
            "num_x": 16,
            "store_history": False,
            "timesteps": 8,
        },
        baseline_parameters={
            "ess_threshold": 1.1,
            "num_pmmh_steps": 1,
            "num_theta": 32,
            "num_x": 64,
            "store_history": False,
            "timesteps": 20,
        },
        baseline_correctness_replicates=8,
        replicated_correctness_level="oracle_accuracy",
    ),
    **{
        f"resample_{name}": WorkloadSpec(
            algorithm=name,
            model="normalized_weights",
            execution_mode="whole_program_jit",
            smoke_parameters=_RESAMPLER_SMOKE,
            baseline_parameters=_RESAMPLER_BASELINE,
            baseline_correctness_replicates=8,
            replicated_correctness_level="statistical",
            profiles=("smoke", "baseline", "scaling"),
        )
        for name in RESAMPLERS
    },
    "bootstrap_tracking_dense": WorkloadSpec(
        algorithm="bootstrap",
        model="tracking_lgssm_dense",
        execution_mode="whole_program_jit",
        smoke_parameters={
            "covariance_regime": "correlated",
            "num_particles": 128,
            "resampling_threshold": 0.5,
            "store_history": False,
            "timesteps": 12,
        },
        baseline_parameters={
            "covariance_regime": "correlated",
            "num_particles": 10_000,
            "resampling_threshold": 0.5,
            "store_history": False,
            "timesteps": 200,
        },
        baseline_correctness_replicates=20,
        replicated_correctness_level="oracle_accuracy",
        profiles=("representation",),
    ),
    "bootstrap_tracking_pytree": WorkloadSpec(
        algorithm="bootstrap",
        model="tracking_lgssm_pytree",
        execution_mode="whole_program_jit",
        smoke_parameters={
            "covariance_regime": "correlated",
            "num_particles": 128,
            "resampling_threshold": 0.5,
            "store_history": False,
            "timesteps": 12,
        },
        baseline_parameters={
            "covariance_regime": "correlated",
            "num_particles": 10_000,
            "resampling_threshold": 0.5,
            "store_history": False,
            "timesteps": 200,
        },
        baseline_correctness_replicates=20,
        replicated_correctness_level="oracle_accuracy",
        profiles=("representation",),
    ),
}


class Cell(NamedTuple):
    """One workload/backend/block executed in a fresh worker process."""

    workload: str
    platform: str
    block: int
    warmups: int
    repeats: int
    execution_mode: str
    parameters: dict[str, Any]
    correctness_replicates: int


def canonical_json(value: Any) -> str:
    """Return a type-preserving canonical JSON identity."""
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def record_matches_cell(record: Mapping[str, Any], cell: Cell) -> bool:
    """Return whether a result encodes the exact scheduled cell identity."""
    try:
        identity = {
            "block": record["block"],
            "correctness_replicates": record["correctness_replicates"],
            "execution_mode": record["execution_mode"],
            "parameters": record["parameters"],
            "platform": record["platform_requested"],
            "repeats": record["repeats"],
            "warmups": record["warmups"],
            "workload": record["workload"],
        }
        return canonical_json(identity) == canonical_json(cell._asdict())
    except (KeyError, TypeError, ValueError):
        return False


def expected_device_identity(platform: str) -> tuple[str, int]:
    """Return the only device kind and ID admitted for a worker backend."""
    if platform == "cpu":
        return "cpu", 0
    if platform == "mps":
        return "gpu", 0
    raise ValueError(f"unknown platform: {platform}")


def summarize(times: Sequence[float]) -> dict[str, float]:
    """Return robust timing statistics while retaining raw samples elsewhere."""
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
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


def replicated_evidence_ratio_gate(
    estimates: Sequence[float],
    *,
    oracle: float,
) -> dict[str, Any]:
    """Test unbiased evidence ratios against one at five estimator SEs."""
    values = np.asarray(estimates, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("estimates must contain at least two replicates")
    if not np.all(np.isfinite(values)) or not math.isfinite(oracle):
        raise ValueError("estimates and oracle must be finite")

    evidence_ratios = np.exp(values - oracle)
    standard_deviation = float(np.std(evidence_ratios, ddof=1))
    estimator_se = standard_deviation / math.sqrt(values.size)
    tolerance = max(5.0 * estimator_se, 5e-5)
    mean_ratio = float(np.mean(evidence_ratios))
    error = mean_ratio - 1.0
    return {
        "error": error,
        "estimator_se": estimator_se,
        "evidence_ratios": evidence_ratios.tolist(),
        "log_evidence_estimates": values.tolist(),
        "mean_ratio": mean_ratio,
        "oracle": float(oracle),
        "passed": bool(abs(error) <= tolerance),
        "replicates": int(values.size),
        "standard_deviation": standard_deviation,
        "tolerance": tolerance,
    }


def _balanced_orders(
    platforms: Sequence[str], *, blocks: int, seed: int
) -> list[list[str]]:
    """Return a seeded order, rotated so process positions stay balanced."""
    first = list(platforms)
    random.Random(seed).shuffle(first)
    return [
        first[offset:] + first[:offset]
        for offset in (block % len(first) for block in range(blocks))
    ]


def _variant_orders(
    count: int,
    *,
    blocks: int,
    seed: int,
) -> list[list[int]]:
    """Return seeded Latin rotations that spread cells over run position."""
    if count < 1:
        return []
    first = list(range(count))
    random.Random(seed).shuffle(first)
    if count == 1:
        return [first for _ in range(blocks)]
    step = max(1, math.ceil(count / blocks))
    while math.gcd(step, count) != 1:
        step += 1
        if step >= count:
            step = 1
            break
    return [
        first[offset:] + first[:offset]
        for offset in ((block * step) % count for block in range(blocks))
    ]


def _updated_parameters(
    parameters: Mapping[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    """Return a copied parameter mapping with explicit axis updates."""
    return {**parameters, **updates}


def _filter_regime_variants() -> list[tuple[str, dict[str, Any]]]:
    """Expand the preregistered standard-filter scenario matrix."""
    variants = []
    for workload in (
        "auxiliary_lgssm",
        "bootstrap_lgssm",
        "guided_lgssm",
    ):
        baseline = WORKLOADS[workload].baseline_parameters
        for observation_regime in ("calibrated", "diffuse", "sharp"):
            for threshold in (0.0, 0.5, 1.1):
                for store_history in (False, True):
                    variants.append((
                        workload,
                        _updated_parameters(
                            baseline,
                            observation_regime=observation_regime,
                            resampling_threshold=threshold,
                            store_history=store_history,
                        ),
                    ))
    return variants


def _scaling_variants() -> list[tuple[str, dict[str, Any]]]:
    """Expand all preregistered particle, dimension, and regime axes."""
    variants = []
    particle_counts = (1_000, 10_000, 100_000)
    for workload in (
        "auxiliary_lgssm",
        "bootstrap_lgssm",
        "guided_lgssm",
        "bootstrap_sv",
        "liu_west_unknown_ar",
    ):
        baseline = WORKLOADS[workload].baseline_parameters
        for num_particles in particle_counts:
            overrides: dict[str, Any] = {"num_particles": num_particles}
            if workload in {"auxiliary_lgssm", "liu_west_unknown_ar"}:
                overrides["resampling_threshold"] = 1.1
            variants.append((
                workload,
                _updated_parameters(
                    baseline,
                    **overrides,
                ),
            ))

    liu_west_baseline = WORKLOADS["liu_west_unknown_ar"].baseline_parameters
    # The N=1_000 particle-scaling cell already supplies d=1 exactly.
    for parameter_dimension in (4, 16, 64):
        variants.append((
            "liu_west_unknown_ar",
            _updated_parameters(
                liu_west_baseline,
                num_particles=1_000,
                parameter_dimension=parameter_dimension,
                resampling_threshold=1.1,
            ),
        ))

    temper_baseline = WORKLOADS["temper_gaussian"].baseline_parameters
    for num_particles in (1_000, 10_000):
        for dimension in (4, 32, 128):
            variants.append((
                "temper_gaussian",
                _updated_parameters(
                    temper_baseline,
                    dimension=dimension,
                    num_particles=num_particles,
                ),
            ))

    smc2_baseline = WORKLOADS["smc2_forward"].baseline_parameters
    for num_theta, num_x, timesteps in (
        (32, 64, 20),
        (128, 256, 40),
        (512, 512, 100),
    ):
        variants.append((
            "smc2_forward",
            _updated_parameters(
                smc2_baseline,
                num_theta=num_theta,
                num_x=num_x,
                timesteps=timesteps,
            ),
        ))

    for workload in sorted(
        name for name in WORKLOADS if name.startswith("resample_")
    ):
        baseline = WORKLOADS[workload].baseline_parameters
        for num_particles in (10_000, 100_000, 1_000_000):
            for weight_regime in (
                "moderately_uneven",
                "one_dominant",
                "uniform",
                "zero_tail",
            ):
                variants.append((
                    workload,
                    _updated_parameters(
                        baseline,
                        num_particles=num_particles,
                        weight_regime=weight_regime,
                    ),
                ))
    return variants


def _profile_variants(
    profile: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Return the exact mathematical cells registered for one profile."""
    if profile in {"smoke", "baseline"}:
        return [
            (
                workload,
                dict(
                    spec.smoke_parameters
                    if profile == "smoke"
                    else spec.baseline_parameters
                ),
            )
            for workload, spec in sorted(WORKLOADS.items())
            if profile in spec.profiles
        ]
    if profile == "filter-regimes":
        return _filter_regime_variants()
    if profile == "scaling":
        return _scaling_variants()
    if profile == "representation":
        variants = []
        for workload in (
            "bootstrap_tracking_dense",
            "bootstrap_tracking_pytree",
        ):
            baseline = WORKLOADS[workload].baseline_parameters
            for covariance_regime in ("correlated", "diagonal"):
                for store_history in (False, True):
                    variants.append((
                        workload,
                        _updated_parameters(
                            baseline,
                            covariance_regime=covariance_regime,
                            store_history=store_history,
                        ),
                    ))
        liu_west_baseline = WORKLOADS["liu_west_unknown_ar"].baseline_parameters
        for store_history in (False, True):
            variants.append((
                "liu_west_unknown_ar",
                _updated_parameters(
                    liu_west_baseline,
                    resampling_threshold=1.1,
                    store_history=store_history,
                ),
            ))
        return variants
    raise ValueError(f"unknown profile: {profile}")


def plan_cells(
    profile: str,
    *,
    platforms: Sequence[str] = PLATFORMS,
    order_seed: int = DEFAULT_ORDER_SEED,
    seed: int | None = None,
) -> list[Cell]:
    """Expand a profile into its deterministic fresh-process cell order."""
    if seed is not None:
        if order_seed != DEFAULT_ORDER_SEED and order_seed != seed:
            raise ValueError("seed and order_seed disagree")
        order_seed = seed
    if not isinstance(order_seed, int) or isinstance(order_seed, bool):
        raise ValueError("order_seed must be an integer")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if not platforms:
        raise ValueError("at least one platform is required")
    if len(set(platforms)) != len(platforms):
        raise ValueError("platforms must be unique")
    unknown = set(platforms) - set(PLATFORMS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown platform: {names}")

    settings = PROFILES[profile]
    variants = _profile_variants(profile)
    variant_orders = _variant_orders(
        len(variants),
        blocks=settings.blocks,
        seed=order_seed,
    )
    platform_orders = [
        _balanced_orders(
            platforms,
            blocks=settings.blocks,
            seed=order_seed + variant_index,
        )
        for variant_index in range(len(variants))
    ]
    cells: list[Cell] = []
    for block, variant_order in enumerate(variant_orders):
        for variant_index in variant_order:
            workload, parameters = variants[variant_index]
            spec = WORKLOADS[workload]
            for platform in platform_orders[variant_index][block]:
                correctness_replicates = (
                    spec.baseline_correctness_replicates
                    if (
                        profile != "smoke"
                        and block == 0
                        and not (
                            profile == "representation"
                            and bool(parameters.get("store_history"))
                        )
                        and not (
                            profile == "filter-regimes"
                            and workload
                            in {"bootstrap_lgssm", "auxiliary_lgssm"}
                            and not float(parameters["resampling_threshold"])
                        )
                    )
                    else 0
                )
                cells.append(
                    Cell(
                        workload=workload,
                        platform=platform,
                        block=block,
                        warmups=settings.warmups,
                        repeats=settings.repeats,
                        execution_mode=spec.execution_mode,
                        parameters=dict(parameters),
                        correctness_replicates=correctness_replicates,
                    )
                )
    return cells


def build_manifest(
    profile: str,
    cells: Sequence[Cell],
    *,
    order_seed: int = DEFAULT_ORDER_SEED,
    platforms: Sequence[str] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the pre-execution manifest that freezes exact cell order."""
    if seed is not None:
        if order_seed != DEFAULT_ORDER_SEED and order_seed != seed:
            raise ValueError("seed and order_seed disagree")
        order_seed = seed
    if not isinstance(order_seed, int) or isinstance(order_seed, bool):
        raise ValueError("order_seed must be an integer")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if platforms is None:
        platform_order = list(dict.fromkeys(cell.platform for cell in cells))
    else:
        platform_order = list(platforms)
    if not platform_order or len(set(platform_order)) != len(platform_order):
        raise ValueError("manifest platforms must be non-empty and unique")
    if set(platform_order) != {cell.platform for cell in cells}:
        raise ValueError("manifest platforms do not match scheduled cells")
    serialized_cells = [cell._asdict() for cell in cells]
    plan_sha256 = hashlib.sha256(
        json.dumps(
            serialized_cells,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "campaign_identity": campaign_identity(),
        "cells": serialized_cells,
        "exclusions": [],
        "order_seed": order_seed,
        "plan_sha256": plan_sha256,
        "platforms": platform_order,
        "profile": profile,
        "schema_version": SCHEMA_VERSION,
        "seed_contract": SEED_CONTRACT,
    }


REQUIRED_RESULT_FIELDS = {
    "algorithm",
    "backend",
    "block",
    "correctness",
    "correctness_replicates",
    "correctness_level",
    "dispatch_mode",
    "environment",
    "execution_mode",
    "failure",
    "first_execution_s",
    "lifecycle",
    "memory",
    "model",
    "parameters",
    "platform_requested",
    "repeats",
    "schema_version",
    "source",
    "steady_summary",
    "steady_times_s",
    "versions",
    "work_metrics",
    "workload",
    "warmups",
}


def _require_nonnegative(value: Any, *, name: str) -> None:
    """Require a finite, non-negative numeric lifecycle value."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def validate_result(result: Mapping[str, Any]) -> None:
    """Validate one stable worker-result envelope."""
    missing = REQUIRED_RESULT_FIELDS - result.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"result is missing required fields: {names}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")

    workload = result["workload"]
    if workload not in WORKLOADS:
        raise ValueError(f"unregistered workload: {workload}")
    spec = WORKLOADS[workload]
    if result["algorithm"] != spec.algorithm:
        raise ValueError("algorithm does not match workload registry")
    if result["model"] != spec.model:
        raise ValueError("model does not match workload registry")
    if result["execution_mode"] != spec.execution_mode:
        raise ValueError("execution_mode does not match workload registry")

    requested = result["platform_requested"]
    if requested not in PLATFORMS:
        raise ValueError(f"unknown requested platform: {requested}")
    if result["failure"] is None and result["backend"] != requested:
        raise ValueError("actual backend does not match requested platform")

    block = result["block"]
    if not isinstance(block, int) or isinstance(block, bool) or block < 0:
        raise ValueError("block must be a non-negative integer")

    schedule = (
        ("correctness_replicates", 0),
        ("repeats", 1),
        ("warmups", 0),
    )
    for name, minimum in schedule:
        value = result[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")

    correctness = result["correctness"]
    if not isinstance(correctness, Mapping) or not isinstance(
        correctness.get("passed"), bool
    ):
        raise ValueError("correctness must contain a boolean passed field")
    correctness_level = result["correctness_level"]
    if correctness_level not in {
        "oracle_accuracy",
        "statistical",
        "structural",
    }:
        raise ValueError("unknown correctness_level")
    expected_level = (
        spec.replicated_correctness_level
        if result["correctness_replicates"]
        else "structural"
    )
    if correctness_level != expected_level:
        raise ValueError(
            "correctness_level does not match the scheduled validation"
        )
    if result["failure"] is not None:
        if not isinstance(result["failure"], Mapping):
            raise ValueError("failure must be null or a mapping")
        if correctness["passed"]:
            raise ValueError("a failed result cannot pass correctness")
        # A worker may fail before backend startup, lowering, or allocation.
        # The stable envelope remains useful even when those nested records are
        # empty, so lifecycle validation applies only to completed workers.
        return
    replicated = correctness.get("replicated")
    if (
        not isinstance(replicated, Mapping)
        or not isinstance(replicated.get("passed"), bool)
        or replicated.get("replicates") != result["correctness_replicates"]
    ):
        raise ValueError(
            "correctness replicate result does not match scheduled count"
        )
    if correctness["passed"] and not replicated["passed"]:
        raise ValueError("a failed replicated gate cannot pass correctness")

    lifecycle = result["lifecycle"]
    if not isinstance(lifecycle, Mapping):
        raise ValueError("lifecycle must be a mapping")
    lifecycle_fields = {
        "backend_compile_s",
        "lowering_s",
        "unavailable_reason",
    }
    if lifecycle_fields - lifecycle.keys():
        raise ValueError("lifecycle is missing required fields")

    execution_mode = result["execution_mode"]
    if execution_mode == "host_shell":
        if (
            lifecycle["backend_compile_s"] is not None
            or lifecycle["lowering_s"] is not None
            or lifecycle["unavailable_reason"] != "host_controlled"
        ):
            raise ValueError(
                "host_shell lifecycle must use null compile timings and "
                "reason host_controlled"
            )
    elif execution_mode == "whole_program_jit":
        _require_nonnegative(
            lifecycle["backend_compile_s"], name="backend_compile_s"
        )
        _require_nonnegative(lifecycle["lowering_s"], name="lowering_s")
        if lifecycle["unavailable_reason"] is not None:
            raise ValueError(
                "whole_program_jit lifecycle cannot be unavailable"
            )
    else:  # defensive even though the workload registry already constrains it
        raise ValueError(f"unknown execution_mode: {execution_mode}")

    _require_nonnegative(result["first_execution_s"], name="first_execution_s")
    times = result["steady_times_s"]
    if not isinstance(times, Sequence) or len(times) != result["repeats"]:
        raise ValueError("steady_times_s must match the scheduled repeats")
    expected_summary = summarize(times)
    if result["steady_summary"] != expected_summary:
        raise ValueError("steady summary does not match steady_times_s")
    if not isinstance(result["parameters"], Mapping):
        raise ValueError("parameters must be a mapping")
    if not isinstance(result["memory"], Mapping):
        raise ValueError("memory must be a mapping")
    if not isinstance(result["environment"], Mapping):
        raise ValueError("environment must be a mapping")
    if not isinstance(result["source"], Mapping):
        raise ValueError("source must be a mapping")
    if not isinstance(result["versions"], Mapping):
        raise ValueError("versions must be a mapping")
    if not isinstance(result["work_metrics"], Mapping):
        raise ValueError("work_metrics must be a mapping")


def worker_environment(
    platform: str,
    *,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a sanitized environment selecting one explicit backend."""
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform: {platform}")
    environment = dict(os.environ if base is None else base)
    thread_variables = {
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
    for name in tuple(environment):
        if name.startswith(("JAX_", "XLA_")) or name in thread_variables:
            environment.pop(name)
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["JAX_PLATFORMS"] = platform
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment["JAX_ENABLE_X64"] = "false"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def profiling_runtime_flags(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return only execution flags that can alter profiling behavior."""
    values = os.environ if environment is None else environment
    thread_variables = {
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
    return {
        name: value
        for name, value in sorted(values.items())
        if name.startswith(("JAX_", "XLA_"))
        or name in thread_variables
        or name == "PYTHONNOUSERSITE"
    }
