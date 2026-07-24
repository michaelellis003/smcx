# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public entry points reject malformed structural inputs clearly."""

import jax.numpy as jnp
import jax.random as jr
import pytest

import smcx


def _initial_sampler(key, num_particles):
    del key
    return jnp.zeros((num_particles, 1))


def _transition_sampler(key, state):
    del key
    return state


def _log_observation(emission, state):
    del emission, state
    return jnp.asarray(0.0)


def _run_bootstrap(emissions, num_particles):
    return smcx.bootstrap_filter(
        jr.key(0),
        _initial_sampler,
        _transition_sampler,
        _log_observation,
        emissions,
        num_particles,
    )


def _run_auxiliary(emissions, num_particles):
    return smcx.auxiliary_filter(
        jr.key(0),
        _initial_sampler,
        _transition_sampler,
        _log_observation,
        _log_observation,
        emissions,
        num_particles,
    )


def _proposal_sampler(key, state, emission):
    del key, emission
    return state


def _log_proposal(emission, state, previous_state):
    del emission, state, previous_state
    return jnp.asarray(0.0)


def _log_transition(state, previous_state):
    del state, previous_state
    return jnp.asarray(0.0)


def _run_guided(emissions, num_particles):
    return smcx.guided_filter(
        jr.key(0),
        _initial_sampler,
        _proposal_sampler,
        _log_proposal,
        _log_transition,
        _log_observation,
        emissions,
        num_particles,
    )


def _param_initial_sampler(key, num_particles):
    del key
    return jnp.zeros((num_particles, 1))


def _param_transition(key, state, params):
    del key, params
    return state


def _param_log_observation(emission, state, params):
    del emission, state, params
    return jnp.asarray(0.0)


def _run_liu_west(emissions, num_particles, *, shrinkage=0.95):
    return smcx.liu_west_filter(
        jr.key(0),
        _initial_sampler,
        _param_transition,
        _param_log_observation,
        _param_log_observation,
        _param_initial_sampler,
        emissions,
        num_particles,
        shrinkage=shrinkage,
    )


_PARTICLE_FILTERS = (
    _run_bootstrap,
    _run_auxiliary,
    _run_guided,
    _run_liu_west,
)


@pytest.mark.parametrize("run_filter", _PARTICLE_FILTERS)
@pytest.mark.parametrize(
    ("emissions", "message"),
    [
        (jnp.zeros((0, 1)), "must contain at least one row"),
    ],
)
def test_particle_filters_validate_emissions(run_filter, emissions, message):
    with pytest.raises(ValueError, match=message):
        run_filter(emissions, 4)


@pytest.mark.parametrize("run_filter", _PARTICLE_FILTERS)
def test_particle_filters_require_positive_particle_count(run_filter):
    with pytest.raises(ValueError, match="num_particles must be >= 1"):
        run_filter(jnp.zeros((2, 1)), 0)


def test_bootstrap_requires_scalar_log_observation():
    def vector_log_observation(emission, state):
        del emission, state
        return jnp.zeros(2)

    with pytest.raises(
        ValueError, match="log_observation_fn output must have shape"
    ):
        smcx.bootstrap_filter(
            jr.key(0),
            _initial_sampler,
            _transition_sampler,
            vector_log_observation,
            jnp.zeros((2, 1)),
            4,
        )


def test_liu_west_validates_shrinkage_before_callbacks():
    with pytest.raises(ValueError, match="shrinkage must be in"):
        _run_liu_west(jnp.zeros((2, 1)), 1, shrinkage=1.1)


def test_simulate_requires_at_least_one_timestep():
    def initial(key):
        del key
        return jnp.zeros(1)

    def transition(key, state):
        del key
        return state

    def emission(key, state):
        del key
        return state

    with pytest.raises(ValueError, match="num_timesteps must be >= 1"):
        smcx.simulate(jr.key(0), initial, transition, emission, 0)


def test_temper_validates_initial_particle_cloud():
    def rank_one_initial(key, num_particles):
        del key
        return jnp.zeros(num_particles)

    with pytest.raises(ValueError, match=r"must have shape \(N, d\)"):
        smcx.temper(
            jr.key(0),
            rank_one_initial,
            lambda particle: -jnp.sum(particle**2),
            lambda particle: -jnp.sum(particle**2),
            4,
        )


@pytest.mark.parametrize(
    ("emissions", "num_theta", "num_x", "message"),
    [
        (jnp.zeros((0, 1)), 2, 2, "must contain at least one row"),
        (jnp.zeros((2, 1)), 0, 2, "num_theta must be >= 1"),
        (jnp.zeros((2, 1)), 2, 0, "num_x must be >= 1"),
    ],
)
def test_smc2_validates_public_structure(emissions, num_theta, num_x, message):
    def log_prior(params):
        return -jnp.sum(params**2)

    def initial(key, count, params):
        del key, params
        return jnp.zeros((count, 1))

    def transition(key, state, params):
        del key, params
        return state

    def log_observation(emission, state, params):
        del emission, state, params
        return jnp.asarray(0.0)

    with pytest.raises(ValueError, match=message):
        smcx.smc2(
            jr.key(0),
            _param_initial_sampler,
            log_prior,
            initial,
            transition,
            log_observation,
            emissions,
            num_theta,
            num_x,
        )
