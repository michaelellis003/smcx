# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public caller-owned particle-filter runner contracts."""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import smcx

_EMISSIONS = jnp.array([[10.0], [20.0], [30.0]])
_INPUTS = jnp.array([1.0, 2.0, 3.0])


def _transport_callbacks(num_particles, evidence_scale=1.0):
    identity = jnp.arange(num_particles, dtype=jnp.int32)
    reverse = identity[::-1]

    def make_record(particles, ancestors, increment):
        logits = -0.05 * jnp.square(particles[:, 0] - increment)
        log_weights, _ = smcx.log_normalize(logits)
        return log_weights, smcx.ParticleFilterRecord(
            particles, log_weights, ancestors, increment
        )

    def initialize(time_index, emission, input_t, key):
        del key
        particles = identity[:, None] + input_t + 0.01 * emission
        increment = evidence_scale * (
            input_t[0] + time_index + 0.01 * emission[0]
        )
        log_weights, record = make_record(particles, identity, increment)
        carry = {
            "algorithm": (particles, log_weights),
            "input_checksum": input_t[0],
        }
        return carry, record

    def step(carry, time_index, emission, input_t, key):
        del key
        particles = (
            carry["algorithm"][0][reverse]
            + input_t
            + time_index
            + 0.01 * emission
        )
        increment = evidence_scale * (
            input_t[0] + time_index + 0.01 * emission[0]
        )
        log_weights, record = make_record(particles, reverse, increment)
        next_carry = {
            "algorithm": (particles, log_weights),
            "input_checksum": carry["input_checksum"] + input_t[0],
        }
        return next_carry, record

    return initialize, step


def _valid_record(**changes):
    fields = {
        "particles": jnp.zeros((3, 1)),
        "log_weights": jnp.zeros(3),
        "ancestors": jnp.arange(3, dtype=jnp.int32),
        "log_evidence_increment": jnp.asarray(0.0),
    }
    return smcx.ParticleFilterRecord(**(fields | changes))


def _run_records(initial_record, step_record=None, *, num_timesteps=1):
    current_step_record = initial_record if step_record is None else step_record

    def initialize(time_index, emission, key):
        del time_index, emission, key
        return jnp.asarray(0.0), initial_record

    def step(carry, time_index, emission, key):
        del time_index, emission, key
        return carry, current_step_record

    return smcx.run_particle_filter(
        jr.key(0),
        initialize,
        step,
        jnp.zeros((num_timesteps, 1)),
    )


def _unexpected_callback(*args):
    raise AssertionError(args)


def test_runner_matches_bootstrap_filter_at_a_fixed_key():
    """The runner preserves the established filter key schedule."""
    num_particles = 8
    emissions = jnp.zeros((4, 1))

    def initial_sampler(key, count):
        return jr.normal(key, (count, 1))

    def transition_sampler(key, state):
        return 0.8 * state + 0.1 * jr.normal(key, state.shape)

    def log_observation(emission, state):
        del emission
        return -1.0 + 0.0 * state[0]

    def make_record(particles, ancestors, emission):
        log_obs = jax.vmap(log_observation, in_axes=(None, 0))(
            emission, particles
        )
        log_weights, log_sum = smcx.log_normalize(log_obs)
        increment = log_sum - jnp.log(jnp.asarray(num_particles))
        return log_weights, smcx.ParticleFilterRecord(
            particles, log_weights, ancestors, increment
        )

    def initialize(time_index, emission, key):
        del time_index
        particles = initial_sampler(key, num_particles)
        identity = jnp.arange(num_particles, dtype=jnp.int32)
        log_weights, record = make_record(particles, identity, emission)
        return (particles, log_weights), record

    def step(carry, time_index, emission, key):
        del time_index
        resample_key, transition_key = jr.split(key)
        ancestors = smcx.multinomial(
            resample_key, smcx.normalize(carry[1]), num_particles
        )
        selected = carry[0][ancestors]
        particle_keys = jr.split(transition_key, num_particles)
        particles = jax.vmap(transition_sampler)(particle_keys, selected)
        log_weights, record = make_record(particles, ancestors, emission)
        return (particles, log_weights), record

    key = jr.key(17)
    expected = smcx.bootstrap_filter(
        key,
        initial_sampler,
        transition_sampler,
        log_observation,
        emissions,
        num_particles,
        resampling_fn=smcx.multinomial,
        resampling_threshold=1.1,
    )
    actual = smcx.run_particle_filter(key, initialize, step, emissions)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected),
        jax.tree.leaves(actual),
        strict=True,
    ):
        assert jnp.array_equal(expected_leaf, actual_leaf)


def test_runner_executes_a_custom_kernel_with_time_aligned_inputs():
    num_particles = 16
    initialize, step = _transport_callbacks(num_particles)
    posterior = smcx.run_particle_filter(
        jr.key(5),
        initialize,
        step,
        _EMISSIONS,
        inputs=_INPUTS,
    )

    reverse = jnp.arange(num_particles - 1, -1, -1, dtype=jnp.int32)
    expected_ancestors = jnp.stack([
        jnp.arange(num_particles, dtype=jnp.int32),
        reverse,
        reverse,
    ])
    assert isinstance(posterior, smcx.ParticleFilterResult)
    assert posterior.filtered_particles.shape == (3, num_particles, 1)
    assert jnp.array_equal(posterior.ancestors, expected_ancestors)
    assert jnp.allclose(
        posterior.log_evidence_increments,
        jnp.array([1.1, 3.2, 5.3]),
    )
    assert jnp.allclose(
        posterior.ess,
        jax.vmap(smcx.ess)(posterior.filtered_log_weights),
    )
    assert jnp.array_equal(
        smcx.log_ml_increments(posterior),
        posterior.log_evidence_increments,
    )
    assert jnp.array_equal(smcx.particle_diversity(posterior), jnp.ones(3))
    assert "min_ess" in smcx.diagnose(posterior)


@pytest.mark.parametrize("num_timesteps", [1, 3])
def test_runner_final_only_matches_full_custom_history(num_timesteps):
    initialize, step = _transport_callbacks(16)
    args = (
        jr.key(8),
        initialize,
        step,
        _EMISSIONS[:num_timesteps],
    )
    full = smcx.run_particle_filter(*args, inputs=_INPUTS[:num_timesteps])
    final = smcx.run_particle_filter(
        *args,
        inputs=_INPUTS[:num_timesteps],
        store_history=False,
    )

    for field in (
        "filtered_particles",
        "filtered_log_weights",
        "ancestors",
    ):
        assert jnp.array_equal(
            getattr(full, field)[-1:],
            getattr(final, field),
        )
    for field in ("ess", "log_evidence_increments", "marginal_loglik"):
        assert jnp.array_equal(getattr(full, field), getattr(final, field))


def test_custom_runner_supports_jit_vmap_and_evidence_gradients():
    def run(key, evidence_scale=1.0):
        callbacks = _transport_callbacks(8, evidence_scale)
        return smcx.run_particle_filter(
            key, *callbacks, _EMISSIONS, inputs=_INPUTS
        )

    eager = run(jr.key(12))
    compiled = jax.jit(run)(jr.key(12))
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        assert jnp.array_equal(eager_leaf, compiled_leaf)

    batched = jax.vmap(run)(jr.split(jr.key(13), 2))
    assert batched.filtered_particles.shape == (2, 3, 8, 1)
    derivative = jax.grad(lambda scale: run(jr.key(14), scale).marginal_loglik)(
        jnp.asarray(1.0)
    )
    expected = jnp.sum(_INPUTS + jnp.arange(3) + 0.01 * _EMISSIONS[:, 0])
    assert jnp.allclose(derivative, expected)


def test_runner_compensates_cancellation_in_caller_evidence():
    dtype = jnp.asarray(0.0).dtype
    large = 1e16 if dtype == jnp.float64 else 1e8
    emissions = jnp.asarray([large, 1.0, -large])[:, None]
    particles = jnp.zeros((1, 1), dtype=dtype)

    def record(increment):
        return smcx.ParticleFilterRecord(
            particles,
            jnp.zeros(1, dtype=dtype),
            jnp.zeros(1, dtype=jnp.int32),
            increment,
        )

    def initialize(time_index, emission, key):
        del time_index, key
        return particles, record(emission[0])

    def step(carry, time_index, emission, key):
        del time_index, key
        return carry, record(emission[0])

    posterior = smcx.run_particle_filter(
        jr.key(20), initialize, step, emissions
    )
    # Cast the independent compensated oracle to the active f32/f64 dtype.
    expected = jnp.asarray(math.fsum(map(float, emissions[:, 0])), dtype=dtype)
    assert jnp.array_equal(posterior.marginal_loglik, expected)
    assert jnp.array_equal(posterior.log_evidence_increments, emissions[:, 0])


@pytest.mark.parametrize(
    ("emissions", "message"),
    [
        (jnp.empty((0, 1)), "emissions must contain at least one row"),
        (jnp.zeros(3), "emissions must have shape"),
    ],
)
def test_runner_rejects_malformed_emissions(emissions, message):
    with pytest.raises(ValueError, match=message):
        smcx.run_particle_filter(
            jr.key(0),
            _unexpected_callback,
            _unexpected_callback,
            emissions,
        )


@pytest.mark.parametrize(
    ("record", "error", "message"),
    [
        (
            _valid_record(log_weights=jnp.zeros((3, 1))),
            ValueError,
            "log_weights must be rank 1",
        ),
        (
            _valid_record(particles={}),
            ValueError,
            "particles must be a nonempty PyTree",
        ),
        (
            _valid_record(particles=jnp.zeros((2, 1))),
            ValueError,
            "leading dimension num_particles=3",
        ),
        (
            _valid_record(log_weights=jnp.zeros(3, dtype=jnp.int32)),
            ValueError,
            "log_weights must be floating",
        ),
        (
            _valid_record(ancestors=jnp.arange(2, dtype=jnp.int32)),
            ValueError,
            "ancestors must have length num_particles=3",
        ),
        (
            _valid_record(ancestors=jnp.arange(3.0)),
            ValueError,
            "ancestors must be integer",
        ),
        (
            _valid_record(log_evidence_increment=jnp.zeros(2)),
            ValueError,
            "log_evidence_increment must be scalar",
        ),
        (
            _valid_record(
                log_evidence_increment=jnp.asarray(0, dtype=jnp.int32)
            ),
            ValueError,
            "log_evidence_increment must be floating",
        ),
        (
            tuple(_valid_record()),
            TypeError,
            "must be a ParticleFilterRecord",
        ),
    ],
)
def test_runner_rejects_malformed_initial_records(record, error, message):
    with pytest.raises(error, match=message):
        _run_records(record)


def test_runner_rejects_step_particle_shape_drift():
    with pytest.raises(ValueError, match="must preserve shape"):
        _run_records(
            _valid_record(),
            _valid_record(particles=jnp.zeros((3, 2))),
            num_timesteps=2,
        )


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (jnp.zeros((3, 1, 1)), "inputs must have shape"),
        (jnp.zeros((2, 1)), "inputs must have leading dimension T=3"),
    ],
)
def test_runner_rejects_misaligned_inputs(inputs, message):
    with pytest.raises(ValueError, match=message):
        smcx.run_particle_filter(
            jr.key(0),
            _unexpected_callback,
            _unexpected_callback,
            jnp.zeros((3, 1)),
            inputs=inputs,
        )
