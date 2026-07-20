# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable bootstrap filtering (ADR-0028)."""

import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.bootstrap import _bootstrap_step, _validate_checkpoint
from smcx.resampling import systematic

EMISSIONS = jnp.array([[0.2], [-0.4], [0.7], [0.1], [-0.2], [0.3], [-0.6]])
NUM_PARTICLES = 16
BAD_KEY_DATA = jnp.ones((1, 3), dtype=jnp.uint32)


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


def _update(keys, checkpoint, emissions, **kwargs):
    return smcx.bootstrap_update(
        keys, checkpoint, _transition, _log_observation, emissions, **kwargs
    )


def _assert_tree_equal(actual, expected):
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


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


@pytest.mark.parametrize(
    ("threshold", "expect_resampling"), [(0.0, False), (1.0, True)]
)
def test_update_is_invariant_to_three_unequal_chunks(
    threshold, expect_resampling
):
    """Ordered keys preserve histories and conditional evidence by chunk."""
    keys = jr.split(jr.key(17), EMISSIONS.shape[0] - 1)
    initial = _checkpoint()
    whole_checkpoint, whole = _update(
        keys, initial, EMISSIONS[1:], resampling_threshold=threshold
    )
    whole = jax.device_get(whole)
    stepped, records = initial, []
    for step_key, emission in zip(keys, EMISSIONS[1:], strict=True):
        stepped, info = _advance(
            step_key, stepped, emission, resampling_threshold=threshold
        )
        records.append(_record(stepped, info))
    actual = jax.tree.map(lambda *xs: np.stack(xs), *records)
    _assert_tree_equal(actual, whole[1:])
    _assert_tree_equal(stepped, whole_checkpoint)
    checkpoint, chunks = initial, []
    for start, stop in ((0, 1), (1, 3), (3, 6)):
        checkpoint, chunk = _update(
            keys[start:stop],
            checkpoint,
            EMISSIONS[start + 1 : stop + 1],
            resampling_threshold=threshold,
        )
        chunks.append(chunk)

    _assert_tree_equal(checkpoint, whole_checkpoint)
    for field in (
        "filtered_particles",
        "filtered_log_weights",
        "ancestors",
        "ess",
        "log_evidence_increments",
    ):
        joined = jax.tree.map(
            lambda *xs: jnp.concatenate(xs),
            *(getattr(chunk, field) for chunk in chunks),
        )
        _assert_tree_equal(joined, getattr(whole, field))
    tolerance = float(5 * np.finfo(np.float32).eps)
    np.testing.assert_allclose(
        math.fsum(float(chunk.marginal_loglik) for chunk in chunks),
        whole.marginal_loglik,
        rtol=tolerance,
        atol=tolerance,
    )
    resampled = jnp.any(whole.ancestors != jnp.arange(NUM_PARTICLES))
    assert bool(resampled) is expect_resampling


def test_update_aligns_inputs_for_structured_state():
    """Each chunk input reaches the matching PyTree transition and weight."""
    inputs = jnp.array([[1.0], [2.0], [3.0]])
    emissions = jnp.zeros((3, 1))

    def initial(key, num_particles, input_t):
        del key
        return {
            "position": jnp.full((num_particles, 1), input_t[0]),
            "regime": jnp.arange(num_particles, dtype=jnp.int32),
        }

    def transition(key, state, input_t):
        del key
        position = state["position"] + input_t
        return {"position": position, "regime": state["regime"]}

    def log_observation(emission, state, input_t):
        del emission, state
        return -0.5 * input_t[0] ** 2

    checkpoint, _ = smcx.bootstrap_init(
        jr.key(3), initial, log_observation, emissions[0], 4, input_t=inputs[0]
    )
    checkpoint, posterior = smcx.bootstrap_update(
        jr.split(jr.key(4), 2),
        checkpoint,
        transition,
        log_observation,
        emissions[1:],
        inputs=inputs[1:],
    )
    expected = jnp.broadcast_to(jnp.array([3.0, 6.0])[:, None], (2, 4))
    assert jnp.array_equal(
        posterior.filtered_particles["position"][:, :, 0], expected
    )
    assert checkpoint.state.particles["regime"].dtype == jnp.int32


def test_checkpoint_preserves_compensated_evidence_across_chunks():
    """Large and small increments retain their correction when chunked."""
    large = -1e16 if jax.config.read("jax_enable_x64") else -1e8
    emissions = jnp.array([[large], [-1.0], [-2.0], [-3.0]])

    def transition(key, state):
        del key
        return state

    def log_observation(emission, state):
        del state
        return emission[0]

    keys = jr.split(jr.key(31), 3)
    initial, initial_info = smcx.bootstrap_init(
        jr.key(32), _initial, log_observation, emissions[0], 4
    )
    whole_checkpoint, whole = smcx.bootstrap_update(
        keys, initial, transition, log_observation, emissions[1:]
    )
    checkpoint = initial
    for start, stop in ((0, 1), (1, 3)):
        checkpoint, _ = smcx.bootstrap_update(
            keys[start:stop],
            checkpoint,
            transition,
            log_observation,
            emissions[start + 1 : stop + 1],
        )
    _assert_tree_equal(checkpoint, whole_checkpoint)
    assert whole_checkpoint.log_evidence_compensation != 0
    increments = [
        float(initial_info.log_evidence_increment),
        *map(float, whole.log_evidence_increments),
    ]
    actual = (
        whole_checkpoint.state.log_marginal_likelihood
        + whole_checkpoint.log_evidence_compensation
    )
    assert actual == jnp.asarray(math.fsum(increments), dtype=actual.dtype)


def test_update_final_only_matches_full_history():
    """Final-only chunks keep scalar traces and the same live state."""
    keys = jr.split(jr.key(41), EMISSIONS.shape[0] - 1)
    initial = _checkpoint()
    full_checkpoint, full = _update(keys, initial, EMISSIONS[1:])
    final_checkpoint, final = _update(
        keys, initial, EMISSIONS[1:], store_history=False
    )
    _assert_tree_equal(final_checkpoint, full_checkpoint)
    for field in ("filtered_particles", "filtered_log_weights", "ancestors"):
        expected = jax.tree.map(lambda value: value[-1:], getattr(full, field))
        _assert_tree_equal(getattr(final, field), expected)
    _assert_tree_equal(final.ess, full.ess)
    _assert_tree_equal(
        final.log_evidence_increments, full.log_evidence_increments
    )


def test_update_accepts_legacy_key_data_batch():
    """Typed and legacy batches with identical key data agree exactly."""
    checkpoint = _checkpoint()
    typed = _update(jr.split(jr.key(5), 2), checkpoint, EMISSIONS[1:3])
    legacy = _update(jr.split(jr.PRNGKey(5), 2), checkpoint, EMISSIONS[1:3])
    _assert_tree_equal(typed, legacy)


@pytest.mark.parametrize(
    ("keys", "emissions", "inputs", "message"),
    [
        (jr.key(1), EMISSIONS[1:2], None, "batched PRNG key"),
        (jr.PRNGKey(1), EMISSIONS[1:2], None, "batched PRNG key"),
        (jnp.ones(1), EMISSIONS[1:2], None, "batched PRNG key"),
        (BAD_KEY_DATA, EMISSIONS[1:2], None, "batched PRNG key"),
        (jr.split(jr.key(1), 0), EMISSIONS[1:1], None, "at least one"),
        (jr.split(jr.key(1), 1), EMISSIONS[1:3], None, "same leading"),
        (jr.split(jr.key(1), 2), EMISSIONS[1:3], jnp.ones((1, 1)), "leading"),
    ],
)
def test_update_validates_chunk_alignment(keys, emissions, inputs, message):
    """Documented chunk alignment errors fail at the public shell."""
    with pytest.raises(ValueError, match=message):
        _update(keys, _checkpoint(), emissions, inputs=inputs)


def test_update_validates_checkpoint_structure():
    """Chunk updates share the single-step checkpoint validation."""
    checkpoint = _checkpoint()._replace(ess=jnp.ones(1))
    with pytest.raises(ValueError, match="ess must be scalar"):
        _update(jr.split(jr.key(1), 1), checkpoint, EMISSIONS[1:2])
