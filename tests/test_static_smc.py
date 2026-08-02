# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the private static-target population stage."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_allclose

from smcx._static_smc import (
    _static_smc_correction,
    _static_smc_stage,
    _StaticMoveResult,
    _StaticSMCState,
    _validated_acceptance_rate,
)
from smcx.exceptions import DegenerateWeightsError
from smcx.weights import ess, log_normalize


class _Population(NamedTuple):
    positions: jax.Array
    cache: jax.Array


def _state() -> _StaticSMCState:
    return _StaticSMCState(
        _Population(jnp.arange(7.0)[:, None], jnp.arange(7.0) + 10.0),
        jnp.log(jnp.arange(1, 8, dtype=jnp.float32) / 28),
        jnp.asarray(1.25),
        jnp.asarray(0.125),
    )


_INCREMENT = jnp.log(jnp.linspace(0.5, 2.0, 7, dtype=jnp.float32))


def _assert_tree_equal(actual, expected) -> None:
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


def _identity_resampling(_key, _weights, count):
    return jnp.arange(count, dtype=jnp.int32)


def _identity_move(_key, _source, _log_weights, selected):
    return _StaticMoveResult(selected, jnp.asarray(0.5), jnp.asarray(True))


def _run_stage(
    resampling_fn, move_fn, *, rule=None, index=None, state=None, inc=None
):
    return _static_smc_stage(
        _state() if state is None else state,
        _INCREMENT if inc is None else inc,
        jr.key(17),
        jr.key(23),
        resampling_fn=resampling_fn,
        resampling_threshold=rule,
        time_index=index,
        move_fn=move_fn,
    )


def test_correction_matches_oracle_and_compiled_execution():
    state, increment = _state(), _INCREMENT
    args = (state.log_weights, increment, *state[2:])
    eager = _static_smc_correction(*args)
    compiled = jax.jit(_static_smc_correction)(*args)
    # The oracle crosses several f32 log/exp operations; 1e-6 is fewer than
    # nine f32 eps at unit scale and covers their accumulated rounding.
    for actual, expected in zip(compiled, eager, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    raw = np.log(np.arange(1, 8) / 28 * np.linspace(0.5, 2.0, 7))
    increment_oracle = np.log(np.exp(raw).sum())
    np.testing.assert_allclose(
        eager.log_weights, raw - increment_oracle, rtol=1e-6
    )
    np.testing.assert_allclose(
        eager.log_evidence_increment, increment_oracle, rtol=1e-6
    )
    np.testing.assert_allclose(
        eager.log_evidence + eager.log_evidence_correction,
        math.fsum([1.25, 0.125, increment_oracle]),
        rtol=1e-6,
    )


def test_forced_stage_gathers_population_and_routes_exact_keys():
    state = _state()
    expected_weights, expected_increment = log_normalize(
        state.log_weights + _INCREMENT
    )
    ancestors = jnp.asarray([6, 0, 6, 3, 1, 5, 2], dtype=jnp.int32)
    selected = jax.tree.map(lambda leaf: leaf[ancestors], state.population)

    def resample(key, weights, count):
        np.testing.assert_array_equal(jr.key_data(key), jr.key_data(jr.key(17)))
        assert_allclose(weights, jnp.exp(expected_weights), rtol=1e-6)
        assert count == 7
        return ancestors

    def move(key, source, log_weights, seeds):
        np.testing.assert_array_equal(jr.key_data(key), jr.key_data(jr.key(23)))
        _assert_tree_equal(source, state.population)
        _assert_tree_equal(seeds, selected)
        assert_allclose(log_weights, expected_weights, rtol=1e-6)
        return _StaticMoveResult(seeds, jnp.asarray(-0.0), jnp.asarray(True))

    result, info = _run_stage(resample, move)
    _assert_tree_equal(result.population, selected)
    assert result.log_weights.dtype == expected_weights.dtype
    np.testing.assert_allclose(result.log_weights, -math.log(7))
    assert_allclose(info.selection_ess, ess(expected_weights), rtol=1e-6)
    assert_allclose(info.log_evidence_increment, expected_increment, rtol=1e-6)
    np.testing.assert_array_equal([info.ess, info.acceptance_rate], [7.0, -0.0])
    assert np.signbit(np.asarray(info.acceptance_rate))
    assert bool(info.resampled)


def test_callable_skip_receives_corrected_diagnostics_and_skips_callbacks():
    state = _state()
    expected_weights, _ = log_normalize(state.log_weights + _INCREMENT)
    index = jnp.asarray(4, dtype=jnp.int32)

    def criterion(log_weights, current_ess, time_index):
        assert_allclose(log_weights, expected_weights, rtol=1e-6)
        assert_allclose(current_ess, ess(expected_weights), rtol=1e-6)
        np.testing.assert_array_equal(time_index, index)
        return jnp.asarray(False)

    def fail(*_args):
        raise AssertionError("skipped callback was invoked")

    result, info = _run_stage(fail, fail, rule=criterion, index=index)
    _assert_tree_equal(result.population, state.population)
    assert_allclose(result.log_weights, expected_weights, rtol=1e-6)
    np.testing.assert_allclose(info.selection_ess, info.ess)
    assert not bool(info.resampled)
    np.testing.assert_array_equal(info.acceptance_rate, 0.0)


def test_stage_gates_degeneracy_and_invalid_ancestors():
    with pytest.raises(DegenerateWeightsError):
        _run_stage(
            _identity_resampling,
            _identity_move,
            inc=jnp.full((7,), -jnp.inf),
        )

    huge = jnp.asarray(jnp.finfo(jnp.float32).max)
    overflow = _state()._replace(
        log_evidence=huge, log_evidence_correction=jnp.zeros_like(huge)
    )
    with pytest.raises(DegenerateWeightsError):
        _run_stage(
            _identity_resampling,
            _identity_move,
            state=overflow,
            inc=jnp.full((7,), huge),
        )

    def invalid(_key, _weights, count):
        return jnp.full((count,), -1, dtype=jnp.int32)

    with pytest.raises(ValueError, match=r"entries must be in \[0, 7\)"):
        _run_stage(invalid, _identity_move)


@pytest.mark.parametrize(
    ("rate", "valid", "message"),
    [
        (jnp.ones(2), True, "scalar float"),
        (0.5, jnp.ones(2, dtype=jnp.bool_), "scalar Boolean"),
        (jnp.nan, True, "finite and in"),
        (1.1, True, "finite and in"),
        (jnp.nextafter(jnp.float32(0), jnp.float32(-1)), True, "finite and in"),
        (0.5, False, "finite and in"),
    ],
)
def test_move_diagnostic_validation(rate, valid, message):
    result = _StaticMoveResult(_state().population, jnp.asarray(rate), valid)
    with pytest.raises(ValueError, match=message):
        _validated_acceptance_rate(result)
