# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable bootstrap filtering (ADR-0028)."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.bootstrap import _bootstrap_step, _validate_checkpoint
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


def _advance(key, checkpoint, emission=EMISSIONS[1], **kwargs):
    observation = kwargs.pop("observation", _log_observation)
    return smcx.bootstrap_step(
        key, checkpoint, _transition, observation, emission, **kwargs
    )


def _record(checkpoint, info):
    return (*checkpoint.state[:2], *info[:2], info[-1])


def test_one_shot_equals_init_then_repeated_step():
    """The legacy key schedule exactly matches repeated public steps."""
    key = jr.key(2026)
    expected = smcx.bootstrap_filter(
        key, _initial, _transition, _log_observation, EMISSIONS, NUM_PARTICLES
    )
    step_root, init_key = jr.split(key)
    checkpoint, info = smcx.bootstrap_init(
        init_key, _initial, _log_observation, EMISSIONS[0], NUM_PARTICLES
    )
    records = [_record(checkpoint, info)]
    assert not info.resampled
    for step_key, emission in zip(
        jr.split(step_root, EMISSIONS.shape[0] - 1),
        EMISSIONS[1:],
        strict=True,
    ):
        checkpoint, info = _advance(step_key, checkpoint, emission)
        records.append(_record(checkpoint, info))
    actual = jax.tree.map(lambda *xs: jnp.stack(xs), *records)
    jax.tree.map(np.testing.assert_array_equal, actual, expected[1:])
    np.testing.assert_array_equal(
        checkpoint.state.log_marginal_likelihood, expected.marginal_loglik
    )
    _, forced = _advance(jr.key(8), _checkpoint(), resampling_threshold=1.0)
    assert forced.resampled


def test_uncompiled_step_matches_compiled_step():
    """The pure core and supported compiled path agree within f32 error."""
    step_key, checkpoint = jr.key(19), _checkpoint()
    eager = _bootstrap_step(
        step_key,
        checkpoint,
        _transition,
        _log_observation,
        EMISSIONS[1],
        systematic,
        0.5,
        None,
        _validate_checkpoint(checkpoint),
    )
    compiled = _advance(step_key, checkpoint)
    # Fixed keys remove MC error; five f32 eps covers compiler rounding.
    tolerance = float(5 * np.finfo(np.float32).eps)
    assert_close = partial(
        np.testing.assert_allclose, rtol=tolerance, atol=tolerance
    )
    jax.tree.map(assert_close, eager, compiled)


def test_init_and_step_raise_on_degenerate_weights():
    """Each public shell rejects an all-negative-infinity update."""
    impossible = lambda emission, state: -jnp.inf  # noqa: E731
    with pytest.raises(smcx.DegenerateWeightsError):
        smcx.bootstrap_init(
            jr.key(1), _initial, impossible, EMISSIONS[0], NUM_PARTICLES
        )
    with pytest.raises(smcx.DegenerateWeightsError):
        _advance(jr.key(2), _checkpoint(), observation=impossible)
    checkpoint = _checkpoint()
    state = checkpoint.state._replace(log_marginal_likelihood=jnp.nan)
    with pytest.raises(smcx.DegenerateWeightsError):
        _advance(jr.key(2), checkpoint._replace(state=state))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("log_weights", jnp.array(0.0), "log_weights must be rank 1"),
        ("log_weights", jnp.ones((16, 1)), "log_weights must be rank 1"),
        ("log_weights", jnp.ones(0), "at least one particle"),
        ("log_weights", jnp.ones(16, dtype=jnp.int32), "floating"),
        ("particles", jnp.ones((15, 1)), "num_particles=16"),
        ("log_marginal_likelihood", jnp.ones(1), "must be scalar"),
        ("log_marginal_likelihood", jnp.array(0), "floating"),
        ("ess", jnp.ones(1), "must be scalar"),
        ("ess", jnp.array(jnp.nan), "finite and nonnegative"),
        ("ess", jnp.array(-1.0), "finite and nonnegative"),
        ("log_evidence_compensation", jnp.ones(1), "must be scalar"),
        ("log_evidence_compensation", jnp.array(0), "floating"),
    ],
)
def test_step_validates_checkpoint_structure(field, value, message):
    """Malformed checkpoint fields fail at the public entry point."""
    checkpoint = _checkpoint()
    if field in checkpoint.state._fields:
        checkpoint = checkpoint._replace(
            state=checkpoint.state._replace(**{field: value})
        )
    else:
        checkpoint = checkpoint._replace(**{field: value})
    with pytest.raises(ValueError, match=message):
        _advance(jr.key(3), checkpoint)
