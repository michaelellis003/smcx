# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Uniform threshold-type errors across every thresholded entry point."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import config as jaxtyping_config

import smcx


@pytest.fixture(autouse=True)
def _reach_the_product_boundary(monkeypatch):
    """Disable the suite's import hook so the product errors are testable.

    Default-config users have no hook; these tests assert the boundary
    they actually reach (the test_ibis nonnumeric-threshold precedent).
    """
    monkeypatch.setattr(jaxtyping_config, "jaxtyping_disable", True)


def _call(entry, threshold):
    if entry == "bootstrap":
        return smcx.bootstrap_filter(
            jr.key(0),
            lambda key, n: jr.normal(key, (n, 1)),
            lambda key, state: state,
            lambda emission, state: -0.5 * (emission[0] - state[0]) ** 2,
            jnp.asarray([[0.1]]),
            8,
            resampling_threshold=threshold,
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
            jnp.asarray([[0.1]]),
            8,
            resampling_threshold=threshold,
        )
    if entry == "temper":
        return smcx.temper(
            jr.key(0),
            lambda key, n: jr.normal(key, (n, 1)),
            lambda x: -0.5 * x[0] ** 2,
            lambda x: -0.5 * (x[0] - 1.0) ** 2,
            8,
            target_ess=threshold,
        )
    if entry == "smc2":
        return smcx.smc2(
            jr.key(0),
            lambda key, n: jr.normal(key, (n, 1)),
            lambda x: -0.5 * x[0] ** 2,
            lambda key, n, theta: jr.normal(key, (n, 1)),
            lambda key, state, theta: state,
            lambda emission, state, theta: -0.5 * (emission[0] - state[0]) ** 2,
            jnp.asarray([[0.1]]),
            4,
            4,
            ess_threshold=threshold,
        )
    if entry == "ibis":
        return smcx.ibis(
            jr.key(0),
            lambda key, n: jr.normal(key, (n, 1)),
            lambda x: -0.5 * x[0] ** 2,
            lambda emission, params, input_t: (
                -0.5 * (emission[0] - params[0]) ** 2
            ),
            jnp.asarray([0.1]),
            8,
            resampling_threshold=threshold,
        )
    raise AssertionError(entry)


_ENTRIES = ("bootstrap", "liu_west", "temper", "smc2", "ibis")


@pytest.mark.parametrize("entry", _ENTRIES)
def test_string_thresholds_get_the_documented_error(entry):
    """A string threshold raises ValueError, never a raw TypeError."""
    with pytest.raises(ValueError, match="must be"):
        _call(entry, "half")


@pytest.mark.parametrize("entry", _ENTRIES)
def test_none_thresholds_get_the_documented_error(entry):
    """None raises the documented error where None has no meaning."""
    if entry == "liu_west":
        pytest.skip("liu_west documents None as its transitional default")
    with pytest.raises(ValueError, match="must be"):
        _call(entry, None)


def test_smc2_callable_threshold_gets_the_documented_error():
    """smc2 does not accept criteria; a callable gets a clean error."""
    with pytest.raises(ValueError, match="must be"):
        _call("smc2", lambda log_weights, ess, t: True)
