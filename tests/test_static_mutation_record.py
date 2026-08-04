# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the StaticMutation parameter record."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx


class _State(NamedTuple):
    position: jax.Array


class _Info(NamedTuple):
    acceptance_rate: jax.Array


def _init(position, _logdensity_fn):
    return _State(position)


def _step(key, state, logdensity_fn):
    proposal = state.position + 0.1 * jr.normal(
        key, state.position.shape, dtype=state.position.dtype
    )
    accept = logdensity_fn(proposal) > logdensity_fn(state.position)
    position = jnp.where(accept, proposal, state.position)
    rate = jnp.asarray(accept, dtype=state.position.dtype)
    return _State(position), _Info(rate)


def _sample_prior(key, count):
    return jr.normal(key, (count, 1))


def _log_prior(position):
    return -0.5 * position[0] ** 2


def _log_likelihood(position):
    return -0.5 * (position[0] - 1.0) ** 2


def _log_increment(emission, position, _input_t):
    return -0.5 * (emission[0] - position[0]) ** 2


def _assert_posteriors_equal(actual, expected):
    for actual_field, expected_field in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_field), np.asarray(expected_field)
        )


def test_record_path_matches_pair_path_for_temper():
    """One record equals the legacy pair bitwise at a fixed key."""
    paired = smcx.temper(
        jr.key(7),
        _sample_prior,
        _log_prior,
        _log_likelihood,
        num_particles=16,
        num_mcmc_steps=2,
        target_ess=0.5,
        mutation_init_fn=_init,
        mutation_step_fn=_step,
    )
    record = smcx.temper(
        jr.key(7),
        _sample_prior,
        _log_prior,
        _log_likelihood,
        num_particles=16,
        num_mcmc_steps=2,
        target_ess=0.5,
        mutation=smcx.StaticMutation(_init, _step),
    )
    _assert_posteriors_equal(record, paired)


def test_record_path_matches_pair_path_for_ibis():
    """The record drives IBIS mutation identically to the pair."""
    emissions = jnp.asarray([0.4, -0.2, 0.9])
    paired = smcx.ibis(
        jr.key(11),
        _sample_prior,
        _log_prior,
        _log_increment,
        emissions,
        num_particles=32,
        num_mcmc_steps=2,
        resampling_threshold=1.5,
        mutation_init_fn=_init,
        mutation_step_fn=_step,
    )
    record = smcx.ibis(
        jr.key(11),
        _sample_prior,
        _log_prior,
        _log_increment,
        emissions,
        num_particles=32,
        num_mcmc_steps=2,
        resampling_threshold=1.5,
        mutation=smcx.StaticMutation(_init, _step),
    )
    assert bool(jnp.all(record.resampled))
    _assert_posteriors_equal(record, paired)


@pytest.mark.parametrize("driver", ["temper", "ibis"])
def test_record_and_pair_together_raise(driver):
    """Supplying the record beside either legacy kwarg is rejected."""
    record = smcx.StaticMutation(_init, _step)
    with pytest.raises(ValueError, match="mutation"):
        if driver == "temper":
            smcx.temper(
                jr.key(0),
                _sample_prior,
                _log_prior,
                _log_likelihood,
                num_particles=8,
                mutation=record,
                mutation_init_fn=_init,
                mutation_step_fn=_step,
            )
        else:
            smcx.ibis(
                jr.key(0),
                _sample_prior,
                _log_prior,
                _log_increment,
                jnp.asarray([0.1]),
                num_particles=8,
                mutation=record,
                mutation_init_fn=_init,
                mutation_step_fn=_step,
            )


@pytest.mark.parametrize("driver", ["temper", "ibis"])
@pytest.mark.parametrize("missing", ["both", "step"])
def test_record_with_missing_member_raises(driver, missing):
    """A record with a missing callback cannot silently disable moves."""
    record = smcx.StaticMutation(
        None if missing == "both" else _init,  # ty: ignore[invalid-argument-type]
        None,  # ty: ignore[invalid-argument-type]
    )
    with pytest.raises(ValueError, match="StaticMutation"):
        if driver == "temper":
            smcx.temper(
                jr.key(0),
                _sample_prior,
                _log_prior,
                _log_likelihood,
                num_particles=8,
                mutation=record,
            )
        else:
            smcx.ibis(
                jr.key(0),
                _sample_prior,
                _log_prior,
                _log_increment,
                jnp.asarray([0.1]),
                num_particles=8,
                mutation=record,
            )
