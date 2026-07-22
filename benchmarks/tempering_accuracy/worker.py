# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process standard-arm worker for issue #30."""

import math
import os
import resource
import subprocess
import sys
import time
from typing import Any, NamedTuple

import jax
import jax.random as jr
import numpy as np

import smcx
from benchmarks.tempering_accuracy.core import (
    Callbacks,
    build_target,
    make_callbacks,
)
from benchmarks.tempering_accuracy.plan import (
    CampaignCell,
    WorkCount,
    current_cells,
    current_smoke_cells,
    matched_cells,
    work_count,
)
from smcx.types import ResamplingFn

SCHEMA_VERSION = 1
_TIMING_KEY = 20_260_719
_clock = time.perf_counter


class WorkerRequest(NamedTuple):
    """One manifest-bound worker invocation."""

    manifest_sha256: str
    phase: str
    cell: CampaignCell
    block: int | None


class StructuralChecks(NamedTuple):
    """Registered structural verdict for one public call."""

    backend_ok: bool
    shapes_ok: bool
    dtypes_ok: bool
    finite_ok: bool
    trace_shapes_ok: bool
    normalized_log_weights_ok: bool
    equal_log_weights_ok: bool
    temperature_trace_ok: bool
    ess_bounds_ok: bool
    acceptance_bounds_ok: bool
    final_log_weight_lse_error: float | None
    uniform_log_weight_error: float | None
    passed: bool


class RunRecord(NamedTuple):
    """Summary retained from one committed inference key."""

    key_index: int | None
    key_words: tuple[int, int]
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    log_evidence: float
    temperatures: np.ndarray
    reweighting_ess: np.ndarray
    acceptance_rates: np.ndarray
    work: WorkCount
    structural: StructuralChecks


class PreparedCall(NamedTuple):
    """Device-ready inputs reused across one worker invocation."""

    key: jax.Array
    callbacks: Callbacks
    resampler: ResamplingFn


class TimingRecord(NamedTuple):
    """Raw timing and resource evidence for one fresh-process block."""

    execution_mode: str
    backend_startup_burns: int
    warmups: int
    repeats: int
    first_execution_s: float
    steady_times_s: tuple[float, ...]
    backend: str
    dispatch_mode: str
    environment: dict[str, Any]
    memory: dict[str, Any]


def _request_dict(request: WorkerRequest) -> dict[str, object]:
    return {
        "manifest_sha256": request.manifest_sha256,
        "phase": request.phase,
        "cell": request.cell._asdict(),
        "block": request.block,
    }


def _validate_request(request: WorkerRequest) -> None:
    digest = request.manifest_sha256
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("manifest_sha256 must be 64 lowercase hex characters")
    standards = (*current_cells(), *matched_cells())
    if request.phase == "smoke":
        valid = request.cell in current_smoke_cells() and request.block is None
    elif request.phase == "timing":
        valid = (
            request.cell in standards
            and isinstance(request.block, int)
            and not isinstance(request.block, bool)
            and 0 <= request.block < 5
        )
    elif request.phase == "accuracy":
        valid = request.cell in standards and request.block is None
    else:
        raise ValueError(f"unknown phase: {request.phase}")
    if not valid:
        raise ValueError("request is not a registered phase/cell/block")


def _runtime_state() -> dict[str, object]:
    return {
        "backend": jax.default_backend(),
        "x64": bool(jax.config.values["jax_enable_x64"]),
        "cache_dir": jax.config.values["jax_compilation_cache_dir"],
        "async": os.environ.get("JAX_MPS_ASYNC_DISPATCH"),
    }


def _validate_timing_runtime(cell: CampaignCell) -> dict[str, object]:
    """Reject a timing process whose selected runtime is not registered."""
    state = _runtime_state()
    backend = "cpu" if cell.lane == "cpu_f64" else "mps"
    if state["backend"] != backend:
        raise RuntimeError("timing runtime selected the wrong backend")
    if bool(state["x64"]) != (cell.lane == "cpu_f64"):
        raise RuntimeError("timing runtime selected the wrong x64 mode")
    if state["cache_dir"] not in (None, ""):
        raise RuntimeError("timing runtime enabled a persistent cache")
    if backend == "mps" and state["async"] not in (None, "", "0"):
        raise RuntimeError("timing runtime enabled asynchronous MPS dispatch")
    return state


def _max_rss_bytes() -> int:
    """Return process maximum resident memory in bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1_024


def _system_value(*command: str) -> str | None:
    """Return one host-status command's output when available."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _timing_environment() -> dict[str, str | None]:
    """Capture power and thermal state at a measurement boundary."""
    return {
        "power_status": _system_value("pmset", "-g", "batt"),
        "thermal_status": _system_value("pmset", "-g", "therm"),
    }


def _runtime_flags() -> dict[str, str | None]:
    names = (
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_MPS_ASYNC_DISPATCH",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names}


def _device_memory(device: Any) -> dict[str, Any] | None:
    """Return the backend's allocator counters when it implements them."""
    try:
        stats = device.memory_stats()
    except (AttributeError, RuntimeError):
        return None
    return None if stats is None else dict(stats)


def _burn_backend() -> None:
    """Remove one-time backend startup without warming the workload."""
    device = jax.devices()[0]
    argument = jax.device_put(np.ones(8, dtype=np.float32), device)
    executable = jax.jit(lambda value: value + 1).lower(argument).compile()
    jax.block_until_ready(executable(argument))


def _structural_checks(
    posterior: smcx.TemperedPosterior,
    request: WorkerRequest,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> StructuralChecks:
    cell = request.cell
    arrays = tuple(np.asarray(jax.device_get(value)) for value in posterior)
    particles, log_weights, evidence, temperatures, ess, acceptance = arrays
    dtype_name = "float64" if cell.lane == "cpu_f64" else "float32"
    expected_dtype = np.dtype(dtype_name)
    trace_shapes_ok = bool(
        temperatures.ndim == ess.ndim == acceptance.ndim == 1
        and 0 < len(temperatures) == len(ess) == len(acceptance)
    )
    shapes_ok = bool(
        particles.shape == (cell.reference_particles, cell.dimension)
        and log_weights.shape == (cell.reference_particles,)
        and evidence.shape == ()
    )
    dtypes_ok = all(value.dtype == expected_dtype for value in arrays)
    finite_ok = bool(
        all(np.all(np.isfinite(value)) for value in arrays)
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(covariance))
    )
    if np.all(np.isfinite(log_weights)):
        maximum = float(np.max(log_weights))
        log_sum = maximum + math.log(
            float(np.sum(np.exp(log_weights - maximum)))
        )
        lse_error = abs(log_sum)
        uniform_error = float(
            np.max(np.abs(log_weights + math.log(cell.reference_particles)))
        )
    else:
        lse_error = None
        uniform_error = None
    normalized_ok = lse_error is not None and lse_error <= 2e-5
    equal_ok = uniform_error is not None and uniform_error <= 2e-5
    temperature_ok = bool(
        trace_shapes_ok
        and np.all(temperatures > 0)
        and np.all(np.diff(temperatures) > 0)
        and np.all(temperatures <= 1 + 1e-6)
        and abs(float(temperatures[-1]) - 1) <= 1e-6
    )
    ess_ok = bool(
        trace_shapes_ok
        and np.all(ess > 0)
        and np.all(ess <= cell.reference_particles * (1 + 5e-6))
    )
    acceptance_ok = bool(
        trace_shapes_ok and np.all(acceptance >= 0) and np.all(acceptance <= 1)
    )
    expected_backend = "cpu" if cell.lane == "cpu_f64" else "mps"
    checks = (
        jax.default_backend() == expected_backend,
        shapes_ok,
        dtypes_ok,
        finite_ok,
        trace_shapes_ok,
        normalized_ok,
        equal_ok,
        temperature_ok,
        ess_ok,
        acceptance_ok,
    )
    return StructuralChecks(
        *checks,
        lse_error,
        uniform_error,
        all(checks),
    )


def _prepare_call(request: WorkerRequest, key: jax.Array) -> PreparedCall:
    """Build callbacks and fence their captured arrays before measurement."""
    dtype = np.float64 if request.cell.lane == "cpu_f64" else np.float32
    target = build_target(request.cell.geometry, request.cell.dimension, dtype)
    callbacks = make_callbacks(target)
    resampler = (
        smcx.systematic
        if request.cell.resampler == "systematic"
        else smcx.multinomial
    )
    placed_key = jax.device_put(key, jax.devices()[0])
    jax.block_until_ready((placed_key, callbacks.device_inputs))
    return PreparedCall(placed_key, callbacks, resampler)


def _invoke_prepared(
    request: WorkerRequest,
    prepared: PreparedCall,
) -> smcx.TemperedPosterior:
    """Call the public host shell and fence its complete output PyTree."""
    posterior = smcx.temper(
        prepared.key,
        prepared.callbacks.initial_sampler,
        prepared.callbacks.log_prior,
        prepared.callbacks.log_likelihood,
        request.cell.reference_particles,
        num_mcmc_steps=request.cell.sweeps,
        target_ess=0.5,
        resampling_fn=prepared.resampler,
        max_stages=1_000,
    )
    return jax.block_until_ready(posterior)


def _record_posterior(
    request: WorkerRequest,
    key: jax.Array,
    key_index: int | None,
    posterior: smcx.TemperedPosterior,
) -> RunRecord:
    """Extract host summaries after any measured public call has finished."""
    particles = np.asarray(
        jax.device_get(posterior.particles), dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(particles, axis=0)
        covariance = np.cov(particles, rowvar=False, ddof=1)
    temperatures = np.asarray(jax.device_get(posterior.temperatures))
    stages = len(temperatures)
    key_words = tuple(int(word) for word in np.asarray(jr.key_data(key)))
    assert len(key_words) == 2
    return RunRecord(
        key_index,
        key_words,
        mean,
        covariance,
        float(posterior.marginal_loglik),
        temperatures,
        np.asarray(jax.device_get(posterior.ess)),
        np.asarray(jax.device_get(posterior.acceptance_rates)),
        work_count(request.cell, stages),
        _structural_checks(posterior, request, mean, covariance),
    )


def _run_once(
    request: WorkerRequest,
    key: jax.Array,
    key_index: int | None,
) -> RunRecord:
    prepared = _prepare_call(request, key)
    posterior = _invoke_prepared(request, prepared)
    return _record_posterior(request, prepared.key, key_index, posterior)


def _run_timing(request: WorkerRequest) -> tuple[TimingRecord, RunRecord]:
    """Measure one registered first call and seven steady public calls."""
    runtime = _validate_timing_runtime(request.cell)
    _burn_backend()
    prepared = _prepare_call(request, jr.key(_TIMING_KEY))
    device = jax.devices()[0]
    rss_before = _max_rss_bytes()
    pre_timing = _timing_environment()

    started = _clock()
    posterior = _invoke_prepared(request, prepared)
    first_execution = _clock() - started
    steady = []
    for _ in range(7):
        started = _clock()
        posterior = _invoke_prepared(request, prepared)
        steady.append(_clock() - started)

    post_timing = _timing_environment()
    device_stats = _device_memory(device)
    record = _record_posterior(request, prepared.key, None, posterior)
    post_cell = _timing_environment()
    durations = (first_execution, *steady)
    if any(not math.isfinite(value) or value < 0 for value in durations):
        raise RuntimeError("timing clock produced an invalid duration")
    environment = {
        "device_id": int(device.id),
        "device_kind": str(device.device_kind),
        "runtime_flags": _runtime_flags(),
        "pre_timing": pre_timing,
        "post_timing": post_timing,
        "post_cell": post_cell,
    }
    memory = {
        "device_stats": device_stats,
        "executable_analysis": None,
        "process_max_rss_before_measurement_bytes": rss_before,
        "process_max_rss_bytes": _max_rss_bytes(),
    }
    timing = TimingRecord(
        "host_shell",
        1,
        0,
        7,
        first_execution,
        tuple(steady),
        str(runtime["backend"]),
        "asynchronous" if request.cell.lane == "cpu_f64" else "safe",
        environment,
        memory,
    )
    return timing, record


def _jsonable(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {name: _jsonable(item) for name, item in value._asdict().items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def execute_request(request: WorkerRequest) -> dict[str, Any]:
    """Execute one request and retain validation or runtime failures."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request": _request_dict(request),
        "failure": None,
        "timing": None,
        "runs": [],
    }
    try:
        _validate_request(request)
    except ValueError as error:
        payload["failure"] = {
            "kind": "invalid_request",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        return payload
    try:
        if request.phase == "smoke":
            record = _run_once(request, jr.key(_TIMING_KEY), None)
        elif request.phase == "timing":
            timing, record = _run_timing(request)
            payload["timing"] = _jsonable(timing)
        else:
            raise NotImplementedError(f"{request.phase} execution is pending")
        payload["runs"] = [_jsonable(record)]
        if not record.structural.passed:
            payload["failure"] = {
                "kind": "structural_failure",
                "exception_type": None,
                "message": "registered structural checks failed",
            }
    except Exception as error:
        payload["failure"] = {
            "kind": "execution_failure",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    return payload
