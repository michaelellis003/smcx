# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable bootstrap filtering (ADR-0028)."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx._utils import _validate_particle_cloud
from smcx.bootstrap import _bootstrap_step
from smcx.resampling import systematic

EMISSIONS = jnp.array([[0.2], [-0.4], [0.7], [0.1], [-0.2]])
NUM_PARTICLES = 16


def _initial(key, num_particles):
    return jr.normal(key, (num_particles, 1))


def _transition(key, state):
    return 0.8 * state + 0.3 * jr.normal(key, state.shape)


def _log_observation(emission, state):
    return -0.5 * (emission[0] - state[0]) ** 2


def _checkpoint():
    return smcx.bootstrap_init(
        jr.key(7), _initial, _log_observation, EMISSIONS[0], NUM_PARTICLES
    )[0]


def test_uncompiled_step_matches_compiled_step():
    """The pure core and supported compiled path agree within f32 error."""
    step_key, checkpoint = jr.key(19), _checkpoint()
    signature = _validate_particle_cloud(
        checkpoint.state.particles,
        NUM_PARTICLES,
        name="checkpoint particles",
    )
    eager = _bootstrap_step(
        step_key,
        checkpoint,
        _transition,
        _log_observation,
        EMISSIONS[1],
        systematic,
        0.5,
        None,
        signature,
    )
    compiled = smcx.bootstrap_step(
        step_key, checkpoint, _transition, _log_observation, EMISSIONS[1]
    )
    # The keys are fixed, so there is no MC error. Five f32 eps covers only
    # rounding from the separate eager and fused compiler paths.
    tolerance = float(5 * np.finfo(np.float32).eps)
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True
    ):
        np.testing.assert_allclose(
            eager_leaf, compiled_leaf, rtol=tolerance, atol=tolerance
        )


def test_init_and_step_report_resampling_semantics():
    """Initialization never resamples; a forced step reports that it did."""
    checkpoint, init_info = smcx.bootstrap_init(
        jr.key(7), _initial, _log_observation, EMISSIONS[0], NUM_PARTICLES
    )
    _, step_info = smcx.bootstrap_step(
        jr.key(8),
        checkpoint,
        _transition,
        _log_observation,
        EMISSIONS[1],
        resampling_threshold=1.0,
    )
    assert not init_info.resampled
    assert step_info.resampled


def test_init_and_step_raise_on_degenerate_weights():
    """Each public shell rejects an all-negative-infinity update."""
    impossible = lambda emission, state: -jnp.inf  # noqa: E731
    with pytest.raises(smcx.DegenerateWeightsError):
        smcx.bootstrap_init(
            jr.key(1), _initial, impossible, EMISSIONS[0], NUM_PARTICLES
        )
    with pytest.raises(smcx.DegenerateWeightsError):
        smcx.bootstrap_step(
            jr.key(2), _checkpoint(), _transition, impossible, EMISSIONS[1]
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("log_weights", "log_weights must be rank 1"),
        ("particles", "num_particles=16"),
        ("log_marginal_likelihood", "must be scalar"),
        ("ess", "must be scalar"),
        ("log_evidence_compensation", "must be scalar"),
    ],
)
def test_step_validates_checkpoint_structure(field, message):
    """Malformed checkpoint fields fail at the public entry point."""
    checkpoint = _checkpoint()
    if field == "particles":
        state = checkpoint.state._replace(particles=jnp.ones((15, 1)))
        checkpoint = checkpoint._replace(state=state)
    elif field in checkpoint.state._fields:
        value = jnp.ones((NUM_PARTICLES, 1)) if field == "log_weights" else []
        checkpoint = checkpoint._replace(
            state=checkpoint.state._replace(**{field: jnp.asarray(value)})
        )
    else:
        checkpoint = checkpoint._replace(**{field: jnp.ones(1)})
    with pytest.raises(ValueError, match=message):
        smcx.bootstrap_step(
            jr.key(3), checkpoint, _transition, _log_observation, EMISSIONS[1]
        )
