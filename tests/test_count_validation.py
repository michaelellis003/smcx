# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Uniform count validation across every counted entry point."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx


def _prior(key, count):
    return jr.normal(key, (count, 1))


def _log_prior(position):
    return -0.5 * position[0] ** 2


def _log_likelihood(position):
    return -0.5 * (position[0] - 1.0) ** 2


def _call(entry, count):
    if entry == "temper_particles":
        return smcx.temper(
            jr.key(0), _prior, _log_prior, _log_likelihood, count
        )
    if entry == "temper_steps":
        return smcx.temper(
            jr.key(0),
            _prior,
            _log_prior,
            _log_likelihood,
            8,
            num_mcmc_steps=count,
        )
    if entry == "liu_west":
        return smcx.liu_west_filter(
            jr.key(0),
            lambda key, n: jr.normal(key, (n, 1)),
            lambda key, state, params: state + params,
            lambda emission, state, params: (
                -0.5 * (emission[0] - state[0]) ** 2
            ),
            lambda emission, state, params: (
                -0.5 * (emission[0] - state[0]) ** 2
            ),
            lambda key, n: jr.normal(key, (n, 1)),
            jnp.asarray([[0.1], [0.2]]),
            count,
        )
    if entry == "smc2_theta":
        return smcx.smc2(
            jr.key(0),
            _prior,
            _log_prior,
            lambda key, n, theta: jr.normal(key, (n, 1)),
            lambda key, state, theta: state,
            lambda emission, state, theta: -0.5 * (emission[0] - state[0]) ** 2,
            jnp.asarray([[0.1]]),
            count,
            8,
        )
    if entry == "smc2_x":
        return smcx.smc2(
            jr.key(0),
            _prior,
            _log_prior,
            lambda key, n, theta: jr.normal(key, (n, 1)),
            lambda key, state, theta: state,
            lambda emission, state, theta: -0.5 * (emission[0] - state[0]) ** 2,
            jnp.asarray([[0.1]]),
            8,
            count,
        )
    if entry == "simulate":
        return smcx.simulate(
            jr.key(0),
            lambda key: jr.normal(key, (1,)),
            lambda key, state: state,
            lambda key, state: state,
            count,
        )
    if entry == "bootstrap_init":
        return smcx.bootstrap_init(
            jr.key(0),
            lambda key, n: jr.normal(key, (n, 1)),
            lambda emission, state: -0.5 * (emission[0] - state[0]) ** 2,
            jnp.asarray([0.1]),
            count,
        )
    raise AssertionError(entry)


_ENTRIES = (
    "temper_particles",
    "temper_steps",
    "liu_west",
    "smc2_theta",
    "smc2_x",
    "simulate",
    "bootstrap_init",
)


@pytest.mark.parametrize("entry", _ENTRIES)
def test_boolean_counts_are_rejected(entry):
    """True is not a usable count anywhere; it silently meant one."""
    with pytest.raises(ValueError, match="positive integer"):
        _call(entry, True)


@pytest.mark.parametrize("entry", _ENTRIES)
def test_float_counts_are_rejected_at_the_boundary(entry):
    """A float count gets the documented error, not a deep JAX one."""
    with pytest.raises(ValueError, match="positive integer"):
        _call(entry, 8.0)


@pytest.mark.parametrize("entry", _ENTRIES)
def test_numpy_integer_counts_stay_accepted(entry):
    """NumPy integers keep working through operator.index."""
    result = _call(entry, np.int64(2))
    assert result is not None


def test_smc2_pmmh_steps_reject_booleans_but_keep_zero():
    """The rejuvenation count keeps its zero floor while rejecting bools."""

    def run(count):
        return smcx.smc2(
            jr.key(0),
            _prior,
            _log_prior,
            lambda key, n, theta: jr.normal(key, (n, 1)),
            lambda key, state, theta: state,
            lambda emission, state, theta: -0.5 * (emission[0] - state[0]) ** 2,
            jnp.asarray([[0.1]]),
            4,
            4,
            num_pmmh_steps=count,
        )

    assert run(0) is not None
    with pytest.raises(ValueError, match="integer"):
        run(True)
    with pytest.raises(ValueError, match="integer"):
        run(1.0)
