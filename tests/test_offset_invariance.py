# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Additive-offset invariance of weight combination (2026-08-06 review).

A log potential, auxiliary twist, transition density, or tempered
increment is defined only up to an additive constant, yet each was
added to normalized log weights before any centering, so a constant
whose magnitude exceeds the weights' floating-point resolution
absorbed the carried relative weight information: valid ``[0.9, 0.1]``
weights became uniform. The invariance the library can promise is
about the values as they *arrive*: a batch whose spread survives
floating point must combine with carried weights exactly as the same
batch shifted to sit near zero. (An offset a user callback adds to an
O(1) term absorbs inside the callback and is unrecoverable here.)

Every test therefore builds callbacks whose outputs are exactly
representable with and without a ``1e17`` offset — an exact constant,
or two values ``{c, c - 64}`` whose gap of four units in the last
place at that scale survives — and requires the weight-side outputs
of the shifted run to match the unshifted run bitwise, while scalar
evidence moves by the predicted constant within floating-point
resolution at the constant's own scale.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx

# The float32 leg needs a smaller level: 1e17 leaves float32 no room
# for the 64-unit spread (its unit in the last place there is ~8.6e9),
# while 64 is eight units in the last place next to 1e8. Both levels
# absorb an O(1) log weight in their dtype, which is the failure the
# fix removes.
X64 = bool(jax.config.read("jax_enable_x64"))
OFFSET = -1e17 if X64 else -1e8
SPREAD = 64.0  # exactly representable next to either level
EVIDENCE_RTOL = 1e-12 if X64 else 1e-6
SENTINEL = 999.0
NUM_PARTICLES = 64
EMISSIONS = jnp.asarray([[0.2], [SENTINEL], [0.7], [SENTINEL]])


def _initial(key, num_particles):
    return jr.normal(key, (num_particles, 1))


def _transition(key, state):
    return 0.9 * state + 0.5 * jr.normal(key, state.shape)


def _log_obs(offset):
    """Informative on ordinary rows, exactly constant on sentinels."""

    def log_observation_fn(emission, state):
        informative = -0.5 * jnp.sum((emission - state) ** 2)
        return jnp.where(emission[0] == SENTINEL, offset, informative)

    return log_observation_fn


def _weight_fields_identical(base, shifted):
    np.testing.assert_array_equal(
        np.asarray(base.filtered_log_weights),
        np.asarray(shifted.filtered_log_weights),
    )
    np.testing.assert_array_equal(
        np.asarray(base.ancestors), np.asarray(shifted.ancestors)
    )
    np.testing.assert_array_equal(np.asarray(base.ess), np.asarray(shifted.ess))


def _evidence_shifted(base, shifted, total_shift):
    # The scalar moves by the constant; the resolution is the
    # constant's own (a few units in the last place at its scale in
    # the active dtype), which is the sharpest claim a shifted sum
    # allows.
    assert float(shifted.marginal_loglik) == pytest.approx(
        float(base.marginal_loglik) + total_shift,
        abs=abs(total_shift) * EVIDENCE_RTOL,
    )


class TestBootstrapPotentialOffset:
    """Exactly-constant potentials must not disturb carried weights."""

    def _run(self, offset, threshold):
        return smcx.bootstrap_filter(
            jr.key(3),
            _initial,
            _transition,
            _log_obs(offset),
            EMISSIONS,
            NUM_PARTICLES,
            resampling_threshold=threshold,
        )

    @pytest.mark.parametrize("threshold", [0.0, 1.0])
    def test_weights_and_ancestors_are_offset_invariant(self, threshold):
        base = self._run(0.0, threshold)
        shifted = self._run(OFFSET, threshold)
        _weight_fields_identical(base, shifted)

    @pytest.mark.parametrize("threshold", [0.0, 1.0])
    def test_evidence_moves_by_the_constant_only(self, threshold):
        base = self._run(0.0, threshold)
        shifted = self._run(OFFSET, threshold)
        _evidence_shifted(base, shifted, 2.0 * OFFSET)


class TestAuxiliaryTwistOffset:
    """A constant twist cancels from selection, weights, and evidence."""

    def _run(self, aux_constant):
        def log_auxiliary_fn(emission, state):
            del emission, state
            return jnp.asarray(aux_constant)

        return smcx.auxiliary_filter(
            jr.key(5),
            _initial,
            _transition,
            _log_obs(0.0),
            log_auxiliary_fn,
            EMISSIONS,
            NUM_PARTICLES,
            resampling_threshold=1.0,
        )

    def test_twist_constant_leaves_everything_invariant(self):
        base = self._run(0.0)
        shifted = self._run(-OFFSET)
        _weight_fields_identical(base, shifted)
        np.testing.assert_array_equal(
            np.asarray(base.log_evidence_increments),
            np.asarray(shifted.log_evidence_increments),
        )


class TestGuidedCancellation:
    """Identical proposal and transition cancel at any magnitude."""

    def test_identical_transition_and_proposal_cancel_exactly(self):
        num_timesteps = EMISSIONS.shape[0]

        def proposal_sampler(key, state, emission):
            return _transition(key, state)

        def log_density(*args):
            return jnp.asarray(OFFSET)

        def log_observation_fn(emission, state):
            return jnp.asarray(-3.0)

        posterior = smcx.guided_filter(
            jr.key(11),
            _initial,
            proposal_sampler,
            log_density,
            log_density,
            log_observation_fn,
            EMISSIONS,
            NUM_PARTICLES,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments),
            np.full(num_timesteps, -3.0),
        )


class TestLiuWestAuxiliaryOffset:
    """The Liu-West first stage shares the twist-centering contract."""

    def _run(self, aux_constant, obs_offset=0.0):
        def param_initial_sampler(key, num_particles):
            return jr.uniform(key, (num_particles, 1), minval=0.5, maxval=1.5)

        def transition_sampler(key, state, params):
            return 0.9 * state + params[0] * jr.normal(key, state.shape)

        def log_observation_fn(emission, state, params):
            informative = -0.5 * jnp.sum((emission - state) ** 2)
            return jnp.where(emission[0] == SENTINEL, obs_offset, informative)

        def log_auxiliary_fn(emission, state, params):
            del emission, state, params
            return jnp.asarray(aux_constant)

        return smcx.liu_west_filter(
            jr.key(17),
            _initial,
            transition_sampler,
            log_observation_fn,
            log_auxiliary_fn,
            param_initial_sampler,
            EMISSIONS,
            NUM_PARTICLES,
            resampling_threshold=1.0,
            parameter_moves="on_selection",
        )

    def test_auxiliary_constant_leaves_everything_invariant(self):
        base = self._run(0.0)
        shifted = self._run(-OFFSET)
        np.testing.assert_array_equal(
            np.asarray(base.filtered_log_weights),
            np.asarray(shifted.filtered_log_weights),
        )
        np.testing.assert_array_equal(
            np.asarray(base.ancestors), np.asarray(shifted.ancestors)
        )
        np.testing.assert_array_equal(
            np.asarray(base.log_evidence_increments),
            np.asarray(shifted.log_evidence_increments),
        )

    def test_observation_constant_preserves_weights(self):
        base = self._run(0.0)
        shifted = self._run(0.0, obs_offset=OFFSET)
        np.testing.assert_array_equal(
            np.asarray(base.filtered_log_weights),
            np.asarray(shifted.filtered_log_weights),
        )
        _evidence_shifted(base, shifted, 2.0 * OFFSET)


class TestTemperedIncrementOffset:
    """A represented likelihood spread survives a huge common level."""

    def _run(self, level):
        def log_likelihood_fn(state):
            # Two exactly representable values: the spread drives a
            # genuine multi-stage schedule, the level must cancel.
            return jnp.where(
                state[0] > 1.0,
                jnp.asarray(level),
                jnp.asarray(level - SPREAD),
            )

        return smcx.temper(
            jr.key(23),
            _initial,
            lambda state: -0.5 * jnp.sum(state**2),
            log_likelihood_fn,
            NUM_PARTICLES,
            num_mcmc_steps=1,
        )

    def test_constant_level_shifts_evidence_only(self):
        base = self._run(0.0)
        shifted = self._run(OFFSET)
        assert len(np.asarray(base.temperatures)) > 1  # non-vacuity
        np.testing.assert_array_equal(
            np.asarray(base.temperatures), np.asarray(shifted.temperatures)
        )
        np.testing.assert_array_equal(
            np.asarray(base.log_weights), np.asarray(shifted.log_weights)
        )
        _evidence_shifted(base, shifted, OFFSET)


class TestBackwardTransitionOffset:
    """Backward reweighting sees the spread, never the level."""

    def _posterior(self):
        return smcx.bootstrap_filter(
            jr.key(29),
            _initial,
            _transition,
            _log_obs(0.0),
            EMISSIONS,
            NUM_PARTICLES,
        )

    @staticmethod
    def _log_transition(level):
        def log_transition(next_state, state, params, input_t):
            # Two exactly representable values at any level, so the
            # base and shifted backward laws are comparable bitwise.
            forward = jnp.sum((next_state - 0.9 * state) ** 2)
            return jnp.where(
                forward < 0.25,
                jnp.asarray(level),
                jnp.asarray(level - SPREAD),
            )

        return log_transition

    def test_smoothing_weights_are_offset_invariant(self):
        posterior = self._posterior()
        base = smcx.smoothing_weights(
            posterior, self._log_transition(0.0), None
        )
        shifted = smcx.smoothing_weights(
            posterior, self._log_transition(OFFSET), None
        )
        assert np.asarray(base).std() > 0.0  # non-vacuity
        np.testing.assert_array_equal(np.asarray(base), np.asarray(shifted))

    def test_backward_simulation_is_offset_invariant(self):
        posterior = self._posterior()
        base = smcx.backward_simulation(
            jr.key(31), posterior, self._log_transition(0.0), None, num_draws=8
        )
        shifted = smcx.backward_simulation(
            jr.key(31),
            posterior,
            self._log_transition(OFFSET),
            None,
            num_draws=8,
        )
        np.testing.assert_array_equal(
            np.asarray(base.smoothed_trajectories),
            np.asarray(shifted.smoothed_trajectories),
        )
