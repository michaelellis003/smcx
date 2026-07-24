# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Long-horizon evidence regressions for one-shot particle filters."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.containers import ParticleFilterResult


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
def test_one_shot_filter_compensates_long_horizon_evidence(filter_name):
    emissions = jnp.full((100_000, 1), -100.003, dtype=jnp.float32)

    with jax.enable_x64(False):
        posterior = _run_filter(filter_name, emissions)

    # math.fsum is an independent higher-precision oracle. Cast once to the
    # public f32 result dtype. One ULP at the final total admits the rounding
    # of the retained correction back into one public f32 scalar.
    reference = np.asarray(
        math.fsum(map(float, np.asarray(posterior.log_evidence_increments))),
        dtype=np.float32,
    )
    final_ulp = float(abs(np.spacing(reference)))
    np.testing.assert_allclose(
        posterior.marginal_loglik,
        reference,
        rtol=0.0,
        atol=final_ulp,
    )
