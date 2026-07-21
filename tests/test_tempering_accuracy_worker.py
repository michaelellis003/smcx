# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen standard-worker contracts for issue #30."""

import json
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import benchmarks.tempering_accuracy.worker as worker
import smcx
from benchmarks.tempering_accuracy.plan import (
    current_cells,
    current_smoke_cells,
    waste_free_cells,
)
from benchmarks.tempering_accuracy.worker import WorkerRequest, execute_request

_MANIFEST = "a" * 64


@pytest.mark.parametrize(
    "worker_request",
    (
        WorkerRequest("bad", "smoke", current_smoke_cells()[0], None),
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
    cell = current_smoke_cells()[0]
    calls = []

    def fake_temper(*args, **kwargs):
        calls.append((args, kwargs))
        num_particles = args[4]
        dtype = jnp.float64
        return smcx.TemperedPosterior(
            particles=jnp.zeros((num_particles, cell.dimension), dtype=dtype),
            log_weights=jnp.full(
                (num_particles,), -math.log(num_particles), dtype=dtype
            ),
            marginal_loglik=jnp.asarray(-3.0, dtype=dtype),
            temperatures=jnp.asarray([0.4, 1.0], dtype=dtype),
            ess=jnp.asarray([500.0, 700.0], dtype=dtype),
            acceptance_rates=jnp.asarray([0.2, 0.3], dtype=dtype),
        )

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
    assert record["temperatures"] == [0.4, 1.0]
    assert record["reweighting_ess"] == [500.0, 700.0]
    assert record["acceptance_rates"] == [0.2, 0.3]
    assert record["work"]["total_pairs"] == 11_000
    assert record["structural"]["passed"]
    json.dumps(payload, allow_nan=False)
