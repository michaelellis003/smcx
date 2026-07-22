# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen standard-worker contracts for issue #30."""

import json
import math
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import benchmarks.tempering_accuracy.worker as worker
import smcx
from benchmarks.tempering_accuracy.plan import (
    current_cells,
    current_smoke_cells,
    matched_cells,
    waste_free_cells,
)
from benchmarks.tempering_accuracy.worker import WorkerRequest, execute_request

_MANIFEST = "a" * 64


def _active_cell(cells):
    lane = "mps_f32" if worker.jax.default_backend() == "mps" else "cpu_f64"
    return next(cell for cell in cells if cell.lane == lane)


def _fake_posterior(cell):
    dtype = jnp.dtype("float64" if cell.lane == "cpu_f64" else "float32")
    count = cell.reference_particles
    return smcx.TemperedPosterior(
        particles=jnp.zeros((count, cell.dimension), dtype=dtype),
        log_weights=jnp.full((count,), -math.log(count), dtype=dtype),
        marginal_loglik=jnp.asarray(-3.0, dtype=dtype),
        temperatures=jnp.asarray([0.4, 1.0], dtype=dtype),
        ess=jnp.asarray([500.0, 700.0], dtype=dtype),
        acceptance_rates=jnp.asarray([0.2, 0.3], dtype=dtype),
    )


def _valid_runtime(cell):
    return {
        "backend": cell.lane[:3],
        "x64": cell.lane == "cpu_f64",
        "cache_dir": None,
        "async": None,
        "disable_jit": False,
        "cache_enabled": False,
    }


@pytest.mark.parametrize(
    "worker_request",
    (
        WorkerRequest("bad", "smoke", current_smoke_cells()[0], None),
        WorkerRequest(cast(str, 7), "smoke", current_smoke_cells()[0], None),
        WorkerRequest(
            _MANIFEST,
            "smoke",
            current_smoke_cells()[0]._replace(sweeps=20),
            None,
        ),
        WorkerRequest(_MANIFEST, "timing", current_cells()[0], None),
        WorkerRequest(_MANIFEST, "accuracy", current_cells()[0], 0),
        WorkerRequest(_MANIFEST, "accuracy", waste_free_cells()[0], None),
    ),
)
def test_invalid_requests_become_retained_failure_payloads(worker_request):
    payload = execute_request(worker_request)

    assert payload["schema_version"] == 1
    assert payload["request"]["manifest_sha256"] == (
        worker_request.manifest_sha256
    )
    assert payload["failure"]["kind"] == "invalid_request"
    assert payload["failure"]["exception_type"] == "ValueError"
    assert payload["timing"] is None
    assert payload["runs"] == []


def test_smoke_dispatches_public_temper_and_retains_complete_summary(
    monkeypatch,
):
    cell = _active_cell(current_smoke_cells())
    calls = []

    def fake_temper(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_posterior(cell)

    monkeypatch.setattr(worker.smcx, "temper", fake_temper)
    payload = execute_request(WorkerRequest(_MANIFEST, "smoke", cell, None))

    assert payload["failure"] is None
    assert payload["timing"] is None
    assert len(calls) == len(payload["runs"]) == 1
    args, kwargs = calls[0]
    assert args[4] == 1_000
    assert kwargs == {
        "num_mcmc_steps": 5,
        "target_ess": 0.5,
        "resampling_fn": smcx.systematic,
        "max_stages": 1_000,
    }
    np.testing.assert_array_equal(
        jr.key_data(args[0]), jr.key_data(jr.key(20_260_719))
    )
    record = payload["runs"][0]
    assert record["key_index"] is None
    assert record["posterior_mean"] == [0.0] * 4
    assert record["posterior_covariance"] == np.zeros((4, 4)).tolist()
    assert record["log_evidence"] == pytest.approx(-3.0)
    assert record["temperatures"] == pytest.approx([0.4, 1.0])
    assert record["reweighting_ess"] == pytest.approx([500.0, 700.0])
    assert record["acceptance_rates"] == pytest.approx([0.2, 0.3])
    assert record["work"]["total_pairs"] == 11_000
    assert record["structural"]["passed"]
    json.dumps(payload, allow_nan=False)


def test_timing_fences_registered_matched_call_schedule(monkeypatch):
    cell = _active_cell(matched_cells())
    calls = []
    events = []
    ticks = iter(float(index) for index in range(16))

    def clock():
        events.append("clock")
        return next(ticks)

    def fence(value):
        events.append(type(value).__name__)
        return value

    def fake_temper(*args, **kwargs):
        events.append("temper")
        calls.append((args, kwargs))
        return _fake_posterior(cell)

    record_posterior = worker._record_posterior

    def record(*args):
        events.append("record")
        return record_posterior(*args)

    monkeypatch.setattr(worker.smcx, "temper", fake_temper)
    monkeypatch.setattr(worker, "_burn_backend", lambda: events.append("burn"))
    monkeypatch.setattr(worker, "_clock", clock, raising=False)
    monkeypatch.setattr(worker.jax, "block_until_ready", fence)
    monkeypatch.setattr(worker, "_timing_environment", lambda: {})
    monkeypatch.setattr(worker, "_max_rss_bytes", lambda: 123, raising=False)
    monkeypatch.setattr(worker, "_device_memory", lambda _: {}, raising=False)
    monkeypatch.setattr(worker, "_runtime_state", lambda: _valid_runtime(cell))
    monkeypatch.setattr(worker, "_record_posterior", record)

    payload = execute_request(WorkerRequest(_MANIFEST, "timing", cell, 0))

    assert payload["failure"] is None
    assert len(calls) == 8
    assert all(call[1]["resampling_fn"] is smcx.multinomial for call in calls)
    first_callbacks = calls[0][0][1:4]
    assert all(call[0][1:4] == first_callbacks for call in calls)
    assert all(
        np.array_equal(jr.key_data(call[0][0]), jr.key_data(jr.key(20_260_719)))
        for call in calls
    )
    assert events == [
        "burn",
        "tuple",
        *(["clock", "temper", "TemperedPosterior", "clock"] * 8),
        "record",
    ]
    timing = payload["timing"]
    assert timing["first_execution_s"] == pytest.approx(1.0)
    assert timing["steady_times_s"] == pytest.approx([1.0] * 7)
    assert timing["environment"]["runtime_state"] == _valid_runtime(cell)
    assert len(payload["runs"]) == 1
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("lane", "update"),
    (
        ("cpu_f64", {"backend": "mps"}),
        ("cpu_f64", {"x64": False}),
        ("cpu_f64", {"cache_dir": "/tmp/x"}),
        ("cpu_f64", {"disable_jit": True}),
        ("cpu_f64", {"cache_enabled": True}),
        ("mps_f32", {"async": "1"}),
    ),
)
def test_timing_refuses_unattested_runtime_before_temper(
    monkeypatch,
    lane,
    update,
):
    cell = next(cell for cell in current_cells() if cell.lane == lane)
    runtime = _valid_runtime(cell) | update
    calls = []
    monkeypatch.setattr(
        worker, "_runtime_state", lambda: runtime, raising=False
    )
    monkeypatch.setattr(worker.smcx, "temper", lambda *args: calls.append(args))

    payload = execute_request(WorkerRequest(_MANIFEST, "timing", cell, 0))

    assert payload["failure"]["kind"] == "execution_failure"
    assert payload["failure"]["exception_type"] == "RuntimeError"
    assert payload["timing"] is None
    assert payload["runs"] == []
    assert calls == []


@pytest.mark.parametrize(
    ("failure_call", "failed_call", "timing_prefix"),
    (
        (
            1,
            {"role": "first", "index": 0},
            {
                "eligible": False,
                "first_execution_s": None,
                "steady_times_s": [],
            },
        ),
        (
            3,
            {"role": "steady", "index": 1},
            {
                "eligible": False,
                "first_execution_s": 1.0,
                "steady_times_s": [1.0],
            },
        ),
    ),
)
def test_timing_retains_failed_call_without_retry(
    monkeypatch,
    failure_call,
    failed_call,
    timing_prefix,
):
    cell = _active_cell(current_cells())
    calls = []
    ticks = iter(float(index) for index in range(20))

    def fail_registered_call(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == failure_call:
            raise RuntimeError("registered timing failure")
        return _fake_posterior(cell)

    monkeypatch.setattr(worker.smcx, "temper", fail_registered_call)
    monkeypatch.setattr(worker, "_burn_backend", lambda: None, raising=False)
    monkeypatch.setattr(worker, "_clock", lambda: next(ticks), raising=False)
    monkeypatch.setattr(
        worker, "_timing_environment", lambda: {}, raising=False
    )
    monkeypatch.setattr(worker, "_max_rss_bytes", lambda: 123, raising=False)
    monkeypatch.setattr(worker, "_device_memory", lambda _: {}, raising=False)
    monkeypatch.setattr(worker, "_runtime_state", lambda: _valid_runtime(cell))

    payload = execute_request(WorkerRequest(_MANIFEST, "timing", cell, 0))

    assert len(calls) == failure_call
    assert payload["failure"] == {
        "kind": "execution_failure",
        "exception_type": "RuntimeError",
        "message": "registered timing failure",
        "failed_call": failed_call,
        "timing_prefix": timing_prefix,
    }
    assert payload["timing"] is None
    assert payload["runs"] == []


def test_system_value_returns_none_after_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == pytest.approx(5.0)
        raise worker.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(worker.subprocess, "run", timeout)

    assert worker._system_value("host-status") is None


def test_system_value_returns_none_after_nonzero_exit(monkeypatch):
    def nonzero(*args, **kwargs):
        assert kwargs["check"] is False
        return worker.subprocess.CompletedProcess(args[0], 1, stdout="ignored")

    monkeypatch.setattr(worker.subprocess, "run", nonzero)

    assert worker._system_value("host-status") is None


def test_runtime_flags_retain_registered_environment_values(monkeypatch):
    names = (
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_DISABLE_JIT",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_MPS_ASYNC_DISPATCH",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    expected = {name: f"registered-{index}" for index, name in enumerate(names)}
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    assert worker._runtime_flags() == expected
