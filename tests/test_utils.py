# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared private particle-filter helpers."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx._utils as utils
import smcx.bootstrap as bootstrap_module
import smcx.guided as guided_module
from smcx import bootstrap_filter, guided_filter
from smcx.resampling import systematic


def _run_standard_filter(
    algorithm: str,
    threshold: float,
    store_history: bool,
):
    def initial_sampler(key, num_particles):
        return jr.normal(key, (num_particles, 1))

    def transition_sampler(key, state):
        return 0.8 * state + 0.1 * jr.normal(key, state.shape)

    def log_observation_fn(emission, state):
        return -10.0 * (emission[0] - state[0]) ** 2

    emissions = jnp.array([[0.1], [-0.3], [0.2], [0.5], [-0.2]])
    if algorithm == "bootstrap":
        return bootstrap_filter(
            jr.key(23),
            initial_sampler,
            transition_sampler,
            log_observation_fn,
            emissions,
            16,
            resampling_threshold=threshold,
            store_history=store_history,
        )

    def proposal_sampler(key, state, emission):
        del emission
        return transition_sampler(key, state)

    def log_transition_fn(new_state, old_state):
        error = (new_state[0] - 0.8 * old_state[0]) / 0.1
        return -0.5 * error**2

    def log_proposal_fn(emission, new_state, old_state):
        del emission
        return log_transition_fn(new_state, old_state)

    return guided_filter(
        jr.key(23),
        initial_sampler,
        proposal_sampler,
        log_proposal_fn,
        log_transition_fn,
        log_observation_fn,
        emissions,
        16,
        resampling_threshold=threshold,
        store_history=store_history,
    )


@pytest.mark.parametrize(
    ("threshold", "expected_resample", "expected_ancestors"),
    [
        (0.0, False, [0, 1, 2, 3]),
        (5.0, True, [0, 1, 1, 2]),
    ],
    ids=["threshold-zero", "forced"],
)
def test_conditional_resample_preserves_fixed_key_output(
    threshold: float,
    expected_resample: bool,
    expected_ancestors: list[int],
) -> None:
    """Changing normalization control flow must preserve seeded outputs."""
    num_particles = 4
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    log_weights = jnp.log(
        jnp.array([0.25, 0.5, 0.1875, 0.0625], dtype=jnp.float32)
    )

    do_resample, ancestors = jax.jit(
        lambda key, weights: utils._conditional_resample(
            key,
            weights,
            jnp.asarray(utils.compute_ess(weights)),
            systematic,
            threshold,
            num_particles,
            identity,
        )
    )(jr.key(20260719), log_weights)

    assert bool(do_resample) is expected_resample
    np.testing.assert_array_equal(ancestors, expected_ancestors)


def test_conditional_resample_skips_normalize_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-resample runtime branch must not normalize a weight vector."""
    normalization_calls: list[None] = []
    original_normalize = utils.normalize

    def record_call(_value: object) -> None:
        normalization_calls.append(None)

    def observed_normalize(log_weights: jax.Array) -> jax.Array:
        jax.debug.callback(record_call, log_weights)
        return original_normalize(log_weights)

    monkeypatch.setattr(utils, "normalize", observed_normalize)
    num_particles = 4
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    log_weights = jnp.log(
        jnp.array([0.25, 0.5, 0.1875, 0.0625], dtype=jnp.float32)
    )

    result = jax.jit(
        lambda key, weights: utils._conditional_resample(
            key,
            weights,
            jnp.asarray(utils.compute_ess(weights)),
            systematic,
            0.0,
            num_particles,
            identity,
        )
    )(jr.key(20260719), log_weights)
    jax.block_until_ready(result)
    jax.effects_barrier()

    do_resample, ancestors = result
    assert not bool(do_resample)
    np.testing.assert_array_equal(ancestors, identity)
    assert normalization_calls == []


@pytest.mark.parametrize("algorithm", ["bootstrap", "guided"])
def test_standard_filter_computes_ess_once_per_timestep(
    algorithm: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each reported ESS value should require one vector reduction."""
    ess_calls: list[None] = []
    original_compute_ess = utils.compute_ess

    def record_call(_value: object) -> None:
        ess_calls.append(None)

    def observed_compute_ess(log_weights: jax.Array) -> jax.Array:
        jax.debug.callback(record_call, log_weights)
        return jnp.asarray(original_compute_ess(log_weights))

    filter_module = (
        bootstrap_module if algorithm == "bootstrap" else guided_module
    )
    monkeypatch.setattr(utils, "compute_ess", observed_compute_ess)
    monkeypatch.setattr(
        filter_module,
        "compute_ess",
        observed_compute_ess,
    )
    posterior = _run_standard_filter(algorithm, 0.5, False)
    jax.block_until_ready(posterior)
    jax.effects_barrier()

    assert len(ess_calls) == posterior.ess.shape[0]


@pytest.mark.parametrize("algorithm", ["bootstrap", "guided"])
@pytest.mark.parametrize(
    "threshold",
    [0.0, 0.5, 1.1],
    ids=["never", "adaptive", "forced"],
)
@pytest.mark.parametrize(
    "store_history",
    [True, False],
    ids=["full-history", "final-only"],
)
def test_standard_filter_ess_carry_preserves_recomputed_reference(
    algorithm: str,
    threshold: float,
    store_history: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Carried ESS must match the prior fixed-key recomputation path."""
    filter_module = (
        bootstrap_module if algorithm == "bootstrap" else guided_module
    )
    carried_conditional_resample = filter_module._conditional_resample

    def recomputing_conditional_resample(
        key,
        log_weights,
        _carried_ess,
        resampling_fn,
        absolute_threshold,
        num_particles,
        identity,
    ):
        recomputed_ess = jnp.asarray(utils.compute_ess(log_weights))
        return carried_conditional_resample(
            key,
            log_weights,
            recomputed_ess,
            resampling_fn,
            absolute_threshold,
            num_particles,
            identity,
        )

    monkeypatch.setattr(
        filter_module,
        "_conditional_resample",
        recomputing_conditional_resample,
    )
    reference = _run_standard_filter(algorithm, threshold, store_history)
    monkeypatch.setattr(
        filter_module,
        "_conditional_resample",
        carried_conditional_resample,
    )
    carried = _run_standard_filter(algorithm, threshold, store_history)

    reference_leaves = jax.tree.leaves(reference)
    carried_leaves = jax.tree.leaves(carried)
    assert len(reference_leaves) == len(carried_leaves)
    for reference_leaf, carried_leaf in zip(
        reference_leaves,
        carried_leaves,
        strict=True,
    ):
        np.testing.assert_array_equal(carried_leaf, reference_leaf)

    if math.isclose(threshold, 0.5) and store_history:
        ancestors = np.asarray(reference.ancestors[1:])
        resampled = np.any(ancestors != np.arange(16), axis=1)
        assert np.any(resampled)
        assert np.any(~resampled)
