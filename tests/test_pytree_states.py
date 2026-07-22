# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Structured latent-state PyTree contracts."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import Array

import smcx


class KalmanState(NamedTuple):
    """Per-particle sufficient statistics used as nested state leaves."""

    mean: Array
    covariance: Array


def _structured_initial(key, num_particles):
    del key
    position = jnp.arange(num_particles, dtype=float)[:, None]
    return {
        "position": position,
        "kalman": KalmanState(
            mean=jnp.concatenate([position + 100.0, position + 200.0], axis=1),
            covariance=(position[:, :, None] + 1.0) * jnp.eye(2),
        ),
    }


def _structured_transition(key, state):
    del key
    return {
        "position": state["position"] + 1.0,
        "kalman": KalmanState(
            mean=state["kalman"].mean + 1.0,
            covariance=state["kalman"].covariance,
        ),
    }


def _structured_log_observation(emission, state):
    return -4.0 * (emission[0] - state["position"][0]) ** 2


def _run_structured_filter(kind, key, *, store_history=True):
    emissions = jnp.array([[0.0], [2.0], [4.0]])
    if kind == "bootstrap":
        return smcx.bootstrap_filter(
            key,
            _structured_initial,
            _structured_transition,
            _structured_log_observation,
            emissions,
            8,
            resampling_threshold=1.1,
            store_history=store_history,
        )
    if kind == "auxiliary":
        return smcx.auxiliary_filter(
            key,
            _structured_initial,
            _structured_transition,
            _structured_log_observation,
            _structured_log_observation,
            emissions,
            8,
            resampling_threshold=1.1,
            store_history=store_history,
        )

    def proposal(key_i, state, emission):
        del emission
        return _structured_transition(key_i, state)

    def log_density(emission, new_state, old_state):
        del emission, new_state, old_state
        return jnp.asarray(0.0)

    return smcx.guided_filter(
        key,
        _structured_initial,
        proposal,
        log_density,
        lambda new, old: log_density(None, new, old),
        _structured_log_observation,
        emissions,
        8,
        resampling_threshold=1.1,
        store_history=store_history,
    )


def test_bootstrap_gathers_every_structured_state_leaf():
    emissions = jnp.array([[0.0], [2.0], [4.0]])
    posterior = _run_structured_filter("bootstrap", jr.key(4))

    particles = posterior.filtered_particles
    assert particles["position"].shape == (3, 8, 1)
    assert particles["kalman"].mean.shape == (3, 8, 2)
    assert particles["kalman"].covariance.shape == (3, 8, 2, 2)
    for time in range(1, emissions.shape[0]):
        ancestors = posterior.ancestors[time]
        assert jnp.array_equal(
            particles["position"][time],
            particles["position"][time - 1][ancestors] + 1.0,
        )
        assert jnp.array_equal(
            particles["kalman"].mean[time],
            particles["kalman"].mean[time - 1][ancestors] + 1.0,
        )
        assert jnp.array_equal(
            particles["kalman"].covariance[time],
            particles["kalman"].covariance[time - 1][ancestors],
        )


@pytest.mark.parametrize("kind", ["bootstrap", "auxiliary", "guided"])
def test_final_only_structured_history_matches_full_history(kind):
    full = _run_structured_filter(kind, jr.key(6), store_history=True)
    final = _run_structured_filter(kind, jr.key(6), store_history=False)
    for full_leaf, final_leaf in zip(
        jax.tree.leaves(full.filtered_particles),
        jax.tree.leaves(final.filtered_particles),
        strict=True,
    ):
        assert final_leaf.shape[0] == 1
        assert jnp.array_equal(full_leaf[-1], final_leaf[0])
    assert jnp.array_equal(full.ess, final.ess)
    assert jnp.array_equal(
        full.log_evidence_increments, final.log_evidence_increments
    )


def test_dense_state_is_exactly_equal_to_a_single_leaf_tree():
    emissions = jnp.linspace(-0.5, 0.5, 6)[:, None]

    def initial(key, num_particles):
        return jr.normal(key, (num_particles, 2))

    def transition(key, state):
        return 0.8 * state + 0.2 * jr.normal(key, state.shape)

    def log_observation(emission, state):
        return -0.5 * (emission[0] - state[0]) ** 2

    dense = smcx.bootstrap_filter(
        jr.key(91), initial, transition, log_observation, emissions, 32
    )
    structured = smcx.bootstrap_filter(
        jr.key(91),
        lambda key, n: {"state": initial(key, n)},
        lambda key, state: {"state": transition(key, state["state"])},
        lambda emission, state: log_observation(emission, state["state"]),
        emissions,
        32,
    )

    assert jnp.array_equal(
        dense.filtered_particles, structured.filtered_particles["state"]
    )
    for field in (
        "marginal_loglik",
        "filtered_log_weights",
        "ancestors",
        "ess",
        "log_evidence_increments",
    ):
        assert jnp.array_equal(
            getattr(dense, field), getattr(structured, field)
        )


@pytest.mark.parametrize("kind", ["bootstrap", "auxiliary", "guided"])
def test_structured_filter_is_eager_jit_and_vmap_compatible(kind):
    def run(key):
        return _run_structured_filter(kind, key)

    eager = run(jr.key(12))
    compiled = jax.jit(run)(jr.key(12))
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True
    ):
        assert jnp.array_equal(eager_leaf, compiled_leaf)

    batched = jax.vmap(run)(jr.split(jr.key(13), 2))
    assert batched.filtered_particles["position"].shape == (2, 3, 8, 1)
    assert batched.filtered_particles["kalman"].covariance.shape == (
        2,
        3,
        8,
        2,
        2,
    )


def test_structured_filter_supports_exogenous_inputs():
    inputs = jnp.array([1.0, 2.0, 3.0])

    def initial(key, num_particles, input_t):
        del key
        value = jnp.full((num_particles, 1), input_t[0])
        return {"value": value, "square": value**2}

    def transition(key, state, input_t):
        del key
        value = state["value"] + input_t
        return {"value": value, "square": value**2}

    def log_observation(emission, state, input_t):
        del emission
        return 0.0 * (state["value"][0] + input_t[0])

    posterior = smcx.bootstrap_filter(
        jr.key(0),
        initial,
        transition,
        log_observation,
        jnp.zeros((3, 1)),
        4,
        inputs=inputs,
    )
    expected = jnp.array([1.0, 3.0, 6.0])[:, None, None]
    assert jnp.array_equal(
        posterior.filtered_particles["value"],
        jnp.broadcast_to(expected, (3, 4, 1)),
    )
    assert jnp.array_equal(
        posterior.filtered_particles["square"],
        posterior.filtered_particles["value"] ** 2,
    )


def _run_invalid_initial(initial_sampler):
    return smcx.bootstrap_filter(
        jr.key(0),
        initial_sampler,
        lambda key, state: state,
        lambda emission, state: jnp.asarray(0.0),
        jnp.zeros((2, 1)),
        4,
    )


@pytest.mark.parametrize(
    ("initial_sampler", "message"),
    [
        (lambda key, n: {}, "nonempty PyTree"),
        (
            lambda key, n: {"good": jnp.zeros((n, 1)), "bad": "value"},
            "leaf ['bad'] must be a JAX array",
        ),
        (
            lambda key, n: {"bad": jnp.asarray(1.0)},
            "leading particle axis",
        ),
        (
            lambda key, n: {"bad": jnp.zeros((n + 1, 2))},
            "leading dimension num_particles=4",
        ),
    ],
)
def test_initial_state_rejects_invalid_trees(initial_sampler, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[")):
        _run_invalid_initial(initial_sampler)


@pytest.mark.parametrize("failure", ["structure", "shape", "dtype"])
def test_transition_rejects_state_contract_drift(failure):
    def initial(key, num_particles):
        del key
        return {"value": jnp.zeros((num_particles, 2))}

    def transition(key, state):
        del key
        if failure == "structure":
            return {"value": state["value"], "extra": state["value"]}
        if failure == "shape":
            return {"value": state["value"][:1]}
        return {"value": state["value"].astype(jnp.int32)}

    expected = {
        "structure": "PyTree structure",
        "shape": "preserve shape",
        "dtype": "preserve dtype",
    }[failure]
    with pytest.raises(ValueError, match=expected):
        smcx.bootstrap_filter(
            jr.key(0),
            initial,
            transition,
            lambda emission, state: jnp.asarray(0.0),
            jnp.zeros((2, 1)),
            4,
        )


def test_invalid_initial_tree_has_stable_error_under_jit():
    run = jax.jit(lambda: _run_invalid_initial(lambda key, n: None))
    with pytest.raises(ValueError, match="nonempty PyTree"):
        run()


def test_guided_proposal_rejects_state_contract_drift():
    def proposal(key, state, emission):
        del key, emission
        return {"different": state["position"]}

    with pytest.raises(ValueError, match="proposal_sampler output"):
        smcx.guided_filter(
            jr.key(0),
            _structured_initial,
            proposal,
            lambda emission, new, old: jnp.asarray(0.0),
            lambda new, old: jnp.asarray(0.0),
            _structured_log_observation,
            jnp.zeros((2, 1)),
            4,
        )


def test_simulate_returns_structured_state_history_under_jit():
    def initial(key):
        del key
        return {
            "position": jnp.array([0.0]),
            "kalman": KalmanState(jnp.array([10.0, 20.0]), jnp.eye(2)),
        }

    def transition(key, state):
        del key
        return {
            "position": state["position"] + 1.0,
            "kalman": KalmanState(
                state["kalman"].mean + 1.0,
                state["kalman"].covariance,
            ),
        }

    def emission(key, state):
        del key
        return state["position"]

    def run(key):
        return smcx.simulate(key, initial, transition, emission, 4)

    states, emissions = jax.jit(run)(jr.key(0))
    assert states["position"].shape == (4, 1)
    assert states["kalman"].mean.shape == (4, 2)
    assert states["kalman"].covariance.shape == (4, 2, 2)
    assert jnp.array_equal(states["position"][:, 0], jnp.arange(4.0))
    assert jnp.array_equal(states["position"], emissions)


def test_simulate_supports_structured_state_with_exogenous_inputs():
    inputs = jnp.array([1.0, 2.0, 3.0])

    def initial(key, input_t):
        del key
        return {"value": input_t, "square": input_t**2}

    def transition(key, state, input_t):
        del key
        value = state["value"] + input_t
        return {"value": value, "square": value**2}

    def emission(key, state, input_t):
        del key, input_t
        return state["value"]

    states, emissions = jax.jit(
        lambda key: smcx.simulate(
            key,
            initial,
            transition,
            emission,
            3,
            inputs=inputs,
        )
    )(jr.key(0))
    expected = jnp.array([1.0, 3.0, 6.0])[:, None]
    assert jnp.array_equal(states["value"], expected)
    assert jnp.array_equal(states["square"], expected**2)
    assert jnp.array_equal(emissions, expected)


def test_simulate_rejects_state_contract_drift():
    with pytest.raises(ValueError, match="transition_sampler output"):
        smcx.simulate(
            jr.key(0),
            lambda key: {"value": jnp.zeros((1,))},
            lambda key, state: {"value": jnp.zeros((2,))},
            lambda key, state: state["value"],
            2,
        )


def test_trajectory_and_predictive_operations_are_tree_aware():
    posterior = _run_structured_filter("bootstrap", jr.key(20))
    trajectories = smcx.reconstruct_trajectories(posterior)
    assert jax.tree.structure(trajectories) == jax.tree.structure(
        posterior.filtered_particles
    )

    ancestors = np.asarray(posterior.ancestors)
    selectors = np.empty_like(ancestors)
    selectors[-1] = np.arange(ancestors.shape[1])
    for time in range(ancestors.shape[0] - 1, 0, -1):
        selectors[time - 1] = ancestors[time, selectors[time]]
    times = np.arange(ancestors.shape[0])[:, None]
    for history_leaf, trajectory_leaf in zip(
        jax.tree.leaves(posterior.filtered_particles),
        jax.tree.leaves(trajectories),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(trajectory_leaf),
            np.asarray(history_leaf)[times, selectors],
        )

    predictive = smcx.posterior_predictive_sample(
        jr.key(21),
        posterior,
        _structured_transition,
        lambda key, state: state["position"],
        num_samples=5,
    )
    assert predictive.shape == (3, 5, 1)


def test_predictive_transition_rejects_state_contract_drift():
    posterior = _run_structured_filter("bootstrap", jr.key(23))

    def changed_transition(key, state):
        del key
        return (state["position"].astype(jnp.int32), jnp.zeros((2,)))

    with pytest.raises(ValueError, match="transition_sampler output"):
        smcx.posterior_predictive_sample(
            jr.key(24),
            posterior,
            changed_transition,
            lambda key, state: state[1],
            num_samples=5,
        )


@pytest.mark.parametrize(
    "diagnostic",
    [
        smcx.weighted_mean,
        smcx.weighted_variance,
        lambda posterior: smcx.weighted_quantile(posterior, jnp.array([0.5])),
        smcx.tail_ess,
        smcx.diagnose,
    ],
)
def test_euclidean_diagnostics_reject_structured_state(diagnostic):
    posterior = _run_structured_filter("bootstrap", jr.key(22))
    with pytest.raises(TypeError, match="dense array"):
        diagnostic(posterior)


@jax.tree_util.register_pytree_node_class
class RegisteredState:
    """Minimal custom PyTree with static metadata, like caller model state."""

    def __init__(self, value, label):
        """Store one dynamic leaf and one static label."""
        self.value = value
        self.label = label

    def tree_flatten(self):
        return (self.value,), self.label

    @classmethod
    def tree_unflatten(cls, label, children):
        return cls(children[0], label)


def test_registered_custom_pytree_preserves_static_metadata():
    def initial(key, num_particles):
        del key
        return RegisteredState(jnp.zeros((num_particles, 1)), "latent")

    def transition(key, state):
        del key
        return RegisteredState(state.value + 1.0, state.label)

    posterior = smcx.bootstrap_filter(
        jr.key(0),
        initial,
        transition,
        lambda emission, state: jnp.asarray(0.0),
        jnp.zeros((3, 1)),
        4,
    )
    assert isinstance(posterior.filtered_particles, RegisteredState)
    assert posterior.filtered_particles.label == "latent"
    assert posterior.filtered_particles.value.shape == (3, 4, 1)
    assert isinstance(posterior, smcx.ParticleFilterResult)
