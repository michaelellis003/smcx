# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Evidence-accumulation regressions for one-shot particle filters."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.containers import ParticleFilterResult

_CPU_PLACED_JIT_SCRIPT = """
import jax
import jax.numpy as jnp
import jax.random as jr

import smcx

assert jax.default_backend() == "mps"
cpu = jax.devices("cpu")[0]


def run(key, emissions):
    def initial_sampler(_key, count):
        return jnp.zeros((count, 1), dtype=emissions.dtype)

    def transition_sampler(_key, state):
        return state + 1

    def log_observation(_emission, state):
        return jnp.zeros((), dtype=state.dtype)

    return smcx.bootstrap_filter(
        key,
        initial_sampler,
        transition_sampler,
        log_observation,
        emissions,
        2,
        resampling_threshold=0.0,
    )


key = jax.device_put(jr.key(38), cpu)
emissions = jax.device_put(jnp.zeros((3, 1)), cpu)
posterior = jax.jit(run)(key, emissions)
jax.block_until_ready(posterior)
platforms = {
    device.platform for device in posterior.filtered_particles.devices()
}
assert platforms == {"cpu"}
"""


def _run_filter(
    filter_name: str,
    emissions: jax.Array,
) -> ParticleFilterResult:
    """Run one deterministic single-particle filter."""
    dtype = emissions.dtype

    def initial_sampler(_key, count):
        return jnp.zeros((count, 1), dtype=dtype)

    def transition_sampler(_key, state):
        return state

    def log_observation(emission, _state):
        return emission[0]

    if filter_name == "bootstrap":
        return smcx.bootstrap_filter(
            key=jr.key(111),
            initial_sampler=initial_sampler,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation,
            emissions=emissions,
            num_particles=1,
            resampling_threshold=0.0,
            store_history=False,
        )
    if filter_name == "auxiliary":
        return smcx.auxiliary_filter(
            key=jr.key(111),
            initial_sampler=initial_sampler,
            transition_sampler=transition_sampler,
            log_observation_fn=log_observation,
            log_auxiliary_fn=lambda _emission, _state: jnp.zeros((), dtype),
            emissions=emissions,
            num_particles=1,
            resampling_threshold=0.0,
            store_history=False,
        )
    if filter_name == "guided":
        return smcx.guided_filter(
            key=jr.key(111),
            initial_sampler=initial_sampler,
            proposal_sampler=lambda _key, state, _emission: state,
            log_proposal_fn=lambda _emission, _new, _old: jnp.zeros((), dtype),
            log_transition_fn=lambda _new, _old: jnp.zeros((), dtype),
            log_observation_fn=log_observation,
            emissions=emissions,
            num_particles=1,
            resampling_threshold=0.0,
            store_history=False,
        )
    if filter_name == "liu-west":
        return smcx.liu_west_filter(
            key=jr.key(111),
            initial_sampler=initial_sampler,
            transition_sampler=lambda _key, state, _params: state,
            log_observation_fn=lambda emission, _state, _params: emission[0],
            log_auxiliary_fn=lambda _emission, _state, _params: jnp.zeros(
                (), dtype
            ),
            param_initial_sampler=lambda _key, count: jnp.zeros(
                (count, 1), dtype
            ),
            emissions=emissions,
            num_particles=1,
            resampling_threshold=0.0,
            store_history=False,
        )
    raise AssertionError(f"unknown filter: {filter_name}")


@pytest.mark.parametrize(
    "filter_name",
    ["bootstrap", "auxiliary", "guided", "liu-west"],
)
def test_one_shot_filter_compensates_float32_cancellation(filter_name):
    emissions = jnp.asarray(
        [2**24, 1, -(2**24)],
        dtype=jnp.float32,
    )[:, None]

    with jax.enable_x64(False):
        posterior = _run_filter(filter_name, emissions)

    expected_increments = np.asarray(emissions[:, 0])
    np.testing.assert_array_equal(
        posterior.log_evidence_increments, expected_increments
    )
    np.testing.assert_array_equal(
        posterior.marginal_loglik,
        np.asarray(1.0, dtype=np.float32),
    )


@pytest.mark.skipif(
    jax.default_backend() != "mps",
    reason="requires both the physical MPS and CPU backends",
)
def test_jitted_cpu_filter_uses_argument_platform():
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "mps,cpu"
    env["JAX_ENABLE_X64"] = "false"
    result = subprocess.run(
        [sys.executable, "-c", _CPU_PLACED_JIT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
