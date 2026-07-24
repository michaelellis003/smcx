# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for caller-owned state-space resampling criteria."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

N = 4
IDENTITY = np.arange(N, dtype=np.int32)
REVERSE = IDENTITY[::-1]
EMISSIONS = jnp.zeros((4, 1), dtype=jnp.float32)


def _initial(key, num_particles):
    del key
    return jnp.arange(num_particles, dtype=jnp.float32)[:, None]


def _param_initial(key, num_particles):
    del key
    return jnp.zeros((num_particles, 1))


def _identity_move(key, state, *unused):
    del key, unused
    return state


def _flat(*args):
    return jnp.zeros((), dtype=args[-1].dtype)


def _lookahead(emission, state, *unused):
    del emission, unused
    return jnp.where(state[0] == 0, 12.0, 0.0)


def _reverse(key, weights, num_samples):
    del key, weights
    return jnp.arange(num_samples - 1, -1, -1, dtype=jnp.int32)


def _always(log_weights, current_ess, time_index):
    del log_weights, current_ess, time_index
    return True


def _never(log_weights, current_ess, time_index):
    del log_weights, current_ess, time_index
    return False


def _even_time(log_weights, current_ess, time_index):
    del log_weights, current_ess
    return time_index % 2 == 0


def _run_filter(kind, criterion=None, *, concentrated=False):
    emissions = EMISSIONS[:2] if concentrated else EMISSIONS
    kwargs = {} if criterion is None else {"resampling_threshold": criterion}
    if concentrated:
        kwargs["resampling_fn"] = _reverse
    auxiliary = _lookahead if concentrated else _flat
    args = (jr.key(7), _initial, _identity_move)
    if kind == "bootstrap":
        return smcx.bootstrap_filter(*args, _flat, emissions, N, **kwargs)
    if kind == "auxiliary":
        return smcx.auxiliary_filter(
            *args, _flat, auxiliary, emissions, N, **kwargs
        )
    if kind == "guided":
        return smcx.guided_filter(
            jr.key(7),
            _initial,
            _identity_move,
            _flat,
            _flat,
            _flat,
            emissions,
            N,
            **kwargs,
        )
    return smcx.liu_west_filter(
        *args,
        _flat,
        auxiliary,
        _param_initial,
        emissions,
        N,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("criterion", "rows"),
    [
        (_always, [REVERSE, REVERSE, REVERSE]),
        (_never, [IDENTITY, IDENTITY, IDENTITY]),
        (_even_time, [IDENTITY, REVERSE, IDENTITY]),
    ],
)
def test_bootstrap_criterion_controls_eager_and_jit_ancestors(criterion, rows):
    def run():
        return smcx.bootstrap_filter(
            jr.key(0),
            _initial,
            _identity_move,
            _flat,
            EMISSIONS,
            N,
            resampling_fn=_reverse,
            resampling_threshold=criterion,
        )

    expected = np.vstack([IDENTITY, *rows])
    np.testing.assert_array_equal(run().ancestors, expected)
    np.testing.assert_array_equal(jax.jit(run)().ancestors, expected)


@pytest.mark.parametrize(
    "kind", ["bootstrap", "auxiliary", "guided", "liu_west"]
)
def test_callable_default_preserves_fixed_key_output(kind):
    def default_rule(log_weights, current_ess, time_index):
        del time_index
        return current_ess < 0.5 * log_weights.shape[0]

    expected = _run_filter(kind)
    actual = _run_filter(kind, default_rule)
    jax.tree.map(
        lambda left, right: np.testing.assert_array_equal(left, right),
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "criterion",
    [
        lambda log_weights, current_ess, time_index: jnp.array([True]),
        lambda log_weights, current_ess, time_index: jnp.asarray(1.0),
    ],
)
def test_rejects_malformed_criterion_result(criterion):
    with pytest.raises(
        ValueError, match="resampling criterion must return a scalar Boolean"
    ):
        _run_filter("bootstrap", criterion)


def _first_stage_rule(log_weights, current_ess, time_index):
    normalized = jnp.isclose(jnp.sum(jnp.exp(log_weights)), 1.0)
    matching_ess = jnp.isclose(current_ess, smcx.ess(log_weights))
    concentrated = jnp.max(jnp.exp(log_weights)) > 0.9
    return normalized & matching_ess & concentrated & (time_index == 1)


@pytest.mark.parametrize("kind", ["auxiliary", "liu_west"])
def test_auxiliary_criteria_receive_first_stage_weights_and_ess(kind):
    posterior = _run_filter(kind, _first_stage_rule, concentrated=True)
    np.testing.assert_array_equal(posterior.ancestors[1], REVERSE)
