# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Boundary and transformation contracts for particle FFBS."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.containers import ParticleFilterPosterior


class _MixedParticles(NamedTuple):
    continuous: jax.Array
    category: jax.Array
    flag: jax.Array


def _posterior(particles: Any, log_weights: Any) -> ParticleFilterPosterior:
    weights = jnp.asarray(log_weights)
    ntime, num_particles = weights.shape
    return ParticleFilterPosterior(
        marginal_loglik=jnp.asarray(0.0),
        filtered_particles=particles,
        filtered_log_weights=weights,
        ancestors=jnp.tile(
            jnp.arange(num_particles, dtype=jnp.int32), (ntime, 1)
        ),
        ess=jnp.full((ntime,), num_particles, dtype=weights.dtype),
        log_evidence_increments=jnp.zeros(ntime, dtype=weights.dtype),
    )


def _uniform_log_weights(ntime: int, num_particles: int) -> jax.Array:
    return jnp.full((ntime, num_particles), -jnp.log(num_particles))


def _dense_posterior() -> ParticleFilterPosterior:
    return _posterior(
        jnp.arange(4.0).reshape(2, 2, 1), _uniform_log_weights(2, 2)
    )


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _sample(
    posterior: Any, callback: Any, num_draws: Any = 2, inputs: Any = None
) -> smcx.ParticleSmootherPosterior:
    return smcx.backward_simulation(
        jr.key(0), posterior, callback, None, num_draws=num_draws, inputs=inputs
    )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("filtered_log_weights", jnp.empty((0, 2)), "nonempty floating"),
        (
            "filtered_log_weights",
            jnp.zeros((2, 2, 1)),
            "shape \\(T, N\\)",
        ),
        (
            "filtered_log_weights",
            jnp.zeros((2, 2), dtype=jnp.int32),
            "nonempty floating",
        ),
        (
            "filtered_log_weights",
            jnp.zeros((2, 2), dtype=jnp.bfloat16),
            "at least float32 precision",
        ),
        ("filtered_particles", {}, "nonempty PyTree"),
        ("filtered_particles", {"bad": "value"}, "must be a JAX array"),
        ("filtered_particles", jnp.zeros(2), "leading time and particle axes"),
        (
            "filtered_particles",
            jnp.zeros((2, 3, 1)),
            "time and particle axes to match",
        ),
        (
            "log_evidence_increments",
            jnp.zeros(3),
            "full particle history",
        ),
    ],
)
def test_rejects_malformed_posterior_histories(
    field: str, replacement: Any, message: str
) -> None:
    posterior = _dense_posterior()._replace(**{field: replacement})

    with pytest.raises(ValueError, match=message):
        _sample(posterior, lambda *_: jnp.asarray(0.0))


def test_rejects_structural_posterior_lookalike() -> None:
    candidate: Any = _dense_posterior()._asdict()

    with pytest.raises(ValueError, match="posterior must be"):
        _sample(candidate, lambda *_: jnp.asarray(0.0))


@pytest.mark.parametrize(
    ("num_draws", "message"),
    [
        (True, "not a boolean"),
        (0, "must be positive"),
        (-1, "must be positive"),
    ],
)
def test_draw_count_validation_precedes_callback(
    num_draws: Any, message: str
) -> None:
    callback: Any = None

    with pytest.raises(ValueError, match=message):
        _sample(_dense_posterior(), callback, num_draws)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ([1.0, 2.0], "must be a JAX array"),
        (jnp.asarray(1.0), "shape \\(T,\\)"),
        (jnp.zeros((3, 1)), "leading dimension T=2"),
        (jnp.zeros((2, 0)), "input_dim >= 1"),
    ],
)
def test_input_validation_precedes_callback(inputs: Any, message: str) -> None:
    def must_not_run(*_args: Any) -> jax.Array:
        pytest.fail("callback ran before inputs were validated")

    with pytest.raises(ValueError, match=message):
        _sample(_dense_posterior(), must_not_run, inputs=inputs)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (jnp.zeros(1), "shape \\(2, 2\\)"),
        (jnp.asarray(0, dtype=jnp.int32), "must be floating"),
        (
            jnp.asarray(0.0, dtype=jnp.bfloat16),
            "at least float32 precision",
        ),
    ],
)
def test_rejects_malformed_callback_batches(value: Any, message: str) -> None:
    def log_transition(*_args: Any) -> Any:
        return value

    with pytest.raises(ValueError, match=message):
        _sample(_dense_posterior(), log_transition)


@pytest.mark.parametrize(
    ("callback", "message"),
    [
        (None, "log_transition must be callable"),
        (lambda *_: "not a JAX value", "log_transition output"),
    ],
)
def test_positive_time_normalizes_callback_type_errors(
    callback: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _sample(_dense_posterior(), callback)


def test_callback_authored_type_error_is_not_rewritten() -> None:
    def raises_inside(*_args: Any) -> jax.Array:
        raise TypeError("authored inside callback")

    with pytest.raises(TypeError, match="authored inside callback"):
        _sample(_dense_posterior(), raises_inside)


def test_two_times_match_the_labelled_key_schedule() -> None:
    particles = jnp.arange(8.0).reshape(2, 4, 1)
    weights = jnp.asarray([[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.2, 0.3]])
    log_weights = jnp.log(weights)
    posterior = _posterior(particles, log_weights)
    params = {"shift": jnp.asarray(4.0)}
    key = jr.key(902)
    num_draws = np.int64(5)

    def log_transition(state, prev_state, params, input_t):
        del input_t
        difference = state[0] - prev_state[0] - params["shift"]
        return -0.25 * difference**2

    actual = smcx.backward_simulation(
        key, posterior, log_transition, params, num_draws=num_draws
    )
    draw_keys = jax.vmap(lambda value: jr.split(value, int(num_draws)))(
        jr.split(key, 2)
    )
    terminal = jax.vmap(
        lambda draw_key: smcx.multinomial(draw_key, weights[-1], 1)[0]
    )(draw_keys[-1])

    def draw_previous(draw_key, terminal_index):
        next_value = particles[-1, terminal_index, 0]
        difference = next_value - particles[0, :, 0] - params["shift"]
        logits = log_weights[0] - 0.25 * difference**2
        return smcx.multinomial(draw_key, jnp.exp(logits - jnp.max(logits)), 1)[
            0
        ]

    previous = jax.vmap(draw_previous)(draw_keys[0], terminal)
    expected_indices = jnp.stack((previous, terminal), axis=1)
    expected_trajectories = particles[jnp.arange(2)[None, :], expected_indices]

    np.testing.assert_array_equal(actual.backward_indices, expected_indices)
    np.testing.assert_array_equal(
        actual.smoothed_trajectories, expected_trajectories
    )


def test_one_time_compiles_without_validating_transition() -> None:
    particles = jnp.asarray([[[1.0], [2.0], [3.0]]])
    posterior = _posterior(particles, jnp.asarray([[-jnp.inf, 0.0, -jnp.inf]]))
    callback: Any = None

    result = jax.jit(
        lambda key: smcx.backward_simulation(
            key, posterior, callback, None, num_draws=2
        )
    )(jr.key(7))

    np.testing.assert_array_equal(result.backward_indices, 1)
    np.testing.assert_array_equal(
        result.smoothed_trajectories, jnp.full((2, 1, 1), 2.0)
    )


def test_single_particle_returns_the_sole_path() -> None:
    particles = jnp.asarray([[[1.0]], [[2.0]], [[4.0]]])
    posterior = _posterior(particles, jnp.zeros((3, 1)))

    result = smcx.backward_simulation(
        jr.key(8),
        posterior,
        lambda state, prev, *_: -jnp.square(state[0] - 2.0 * prev[0]),
        None,
        num_draws=2,
    )

    np.testing.assert_array_equal(result.backward_indices, 0)
    np.testing.assert_array_equal(
        result.smoothed_trajectories,
        jnp.broadcast_to(particles[:, 0], (2, 3, 1)),
    )


@pytest.mark.parametrize("source", ["terminal", "backward"])
@pytest.mark.parametrize("kind", ["no_finite", "nan", "positive_inf"])
def test_eager_invalid_categorical_rows_raise(source: str, kind: str) -> None:
    bad = {
        "no_finite": -jnp.inf,
        "nan": jnp.nan,
        "positive_inf": jnp.inf,
    }[kind]
    posterior = _dense_posterior()

    def callback(*_args: Any) -> jax.Array:
        return jnp.asarray(0.0 if source == "terminal" else bad)

    if source == "terminal":
        row = jnp.full(2, bad) if kind == "no_finite" else jnp.asarray([0, bad])
        posterior = posterior._replace(
            filtered_log_weights=posterior.filtered_log_weights.at[-1].set(row)
        )

    with pytest.raises(
        smcx.DegenerateWeightsError, match="cannot be normalized"
    ):
        smcx.backward_simulation(
            jr.key(9), posterior, callback, None, num_draws=3
        )


def test_structured_state_composes_and_traced_failure_is_whole_record() -> None:
    ntime, num_particles, num_draws = 3, 4, 8
    category = jnp.tile(jnp.arange(num_particles, dtype=jnp.int32), (ntime, 1))
    particles = _MixedParticles(
        continuous=jnp.arange(ntime * num_particles * 2.0).reshape(
            ntime, num_particles, 2
        ),
        category=category,
        flag=category % 2 == 0,
    )
    terminal_weights = jnp.asarray([0.4, 0.1, 0.2, 0.3])
    log_weights = jnp.log(jnp.tile(terminal_weights, (ntime, 1)))
    posterior = _posterior(particles, log_weights)
    inputs = jnp.arange(ntime, dtype=jnp.int32)
    key = jr.key(902)
    terminal_indices = jax.vmap(
        lambda k: smcx.multinomial(k, terminal_weights, 1)[0]
    )(jr.split(jr.split(key, ntime)[-1], num_draws))
    invalid_category = terminal_indices[0]
    assert bool(jnp.any(terminal_indices != invalid_category))

    def log_transition(state, prev_state, params, input_t):
        assert input_t.dtype == jnp.int32
        difference = state.continuous - prev_state.continuous
        ordinary = -params["rate"] * jnp.sum(difference**2)
        invalid = (
            params["fail"]
            & (input_t[0] == ntime - 1)
            & (state.category == params["invalid_category"])
        )
        return jnp.where(invalid, -jnp.inf, ordinary)

    def run(draw_key, fail):
        params = {
            "rate": jnp.asarray(0.01),
            "fail": fail,
            "invalid_category": invalid_category,
        }
        return smcx.backward_simulation(
            draw_key,
            posterior,
            log_transition,
            params,
            num_draws=num_draws,
            inputs=inputs,
        )

    valid = run(key, jnp.asarray(False))
    compiled = jax.jit(run)(key, jnp.asarray(False))
    mapped = jax.jit(jax.vmap(run))(
        jnp.stack((key, key)), jnp.asarray([False, True])
    )

    _assert_tree_equal(compiled, valid)
    _assert_tree_equal(jax.tree.map(lambda leaf: leaf[0], mapped), valid)
    assert isinstance(valid.smoothed_trajectories, _MixedParticles)
    times = jnp.arange(ntime)[None, :]
    for history, drawn in zip(
        jax.tree.leaves(particles),
        jax.tree.leaves(valid.smoothed_trajectories),
        strict=True,
    ):
        np.testing.assert_array_equal(
            drawn, history[times, valid.backward_indices]
        )
    invalid = jax.tree.map(lambda leaf: leaf[1], mapped)
    np.testing.assert_array_equal(invalid.backward_indices, -1)
    assert bool(jnp.all(jnp.isnan(invalid.smoothed_trajectories.continuous)))
    np.testing.assert_array_equal(invalid.smoothed_trajectories.category, 0)
    np.testing.assert_array_equal(invalid.smoothed_trajectories.flag, False)


def test_gradient_follows_the_realized_inexact_gathers() -> None:
    base = jnp.arange(1.0, 13.0).reshape(3, 4, 1)
    log_weights = _uniform_log_weights(3, 4)

    def objective(scale):
        posterior = _posterior(scale * base, log_weights)
        result = smcx.backward_simulation(
            jr.key(904),
            posterior,
            lambda *_: jnp.asarray(0.0),
            None,
            num_draws=3,
        )
        return jnp.sum(result.smoothed_trajectories)

    one = jnp.asarray(1.0)
    expected = objective(one)
    np.testing.assert_array_equal(jax.grad(objective)(one), expected)
    np.testing.assert_array_equal(jax.jit(jax.grad(objective))(one), expected)
