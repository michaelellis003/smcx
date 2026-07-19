# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared private particle-filter helpers."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx._utils as utils
from smcx.resampling import systematic


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
