# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process standard-arm worker for issue #30."""

import math
from typing import Any, NamedTuple

import jax
import jax.random as jr
import numpy as np

import smcx
from benchmarks.tempering_accuracy.core import build_target, make_callbacks
from benchmarks.tempering_accuracy.plan import (
    CampaignCell,
    WorkCount,
    current_cells,
    current_smoke_cells,
    matched_cells,
    work_count,
)

SCHEMA_VERSION = 1
_TIMING_KEY = 20_260_719


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


def _request_dict(request: WorkerRequest) -> dict[str, object]:
    return {
        "manifest_sha256": request.manifest_sha256,
        "phase": request.phase,
        "cell": request.cell._asdict(),
        "block": request.block,
    }


def _validate_request(request: WorkerRequest) -> None:
    digest = request.manifest_sha256
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
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


def _run_once(
    request: WorkerRequest,
    key: jax.Array,
    key_index: int | None,
) -> RunRecord:
    dtype = np.float64 if request.cell.lane == "cpu_f64" else np.float32
    target = build_target(request.cell.geometry, request.cell.dimension, dtype)
    callbacks = make_callbacks(target)
    resampler = (
        smcx.systematic
        if request.cell.resampler == "systematic"
        else smcx.multinomial
    )
    posterior = smcx.temper(
        key,
        callbacks.initial_sampler,
        callbacks.log_prior,
        callbacks.log_likelihood,
        request.cell.reference_particles,
        num_mcmc_steps=request.cell.sweeps,
        target_ess=0.5,
        resampling_fn=resampler,
        max_stages=1_000,
    )
    jax.block_until_ready(posterior)
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
        if request.phase != "smoke":
            raise NotImplementedError(f"{request.phase} execution is pending")
        record = _run_once(request, jr.key(_TIMING_KEY), None)
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
