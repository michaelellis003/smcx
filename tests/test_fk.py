# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Feynman-Kac core: fixed-key characterization and derivation tests.

The characterization values were captured from ``bootstrap_filter``
at v1.16.0 (commit 8aa3ac8, before the loop was extracted into
``smcx.fk``) and pin the rewiring bitwise on CPU: any change to the
key schedule, weight rule, or evidence accumulation fails here.
Re-frozen 2026-08-06 for the deliberate potential-centering fix
(offset-invariant weight combination); the new values differ from
the v1.16.0 capture at unit-in-the-last-place level only, and the
key schedule and every resampling decision are unchanged.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import smcx
from smcx.fk import FeynmanKac, run_smc


def _scalar_model():
    def initial(key, n):
        return 0.8 * jr.normal(key, (n, 1))

    def transition(key, state):
        return 0.9 * state + 0.3 * jr.normal(key, state.shape)

    def log_obs(emission, state):
        residual = (emission[0] - state[0]) / 0.7
        return -0.5 * residual**2

    return initial, transition, log_obs


def _input_model():
    def initial(key, n, input_0):
        return (0.8 + 0.1 * input_0[0]) * jr.normal(key, (n, 1))

    def transition(key, state, input_t):
        return (
            0.9 * state + 0.1 * input_t[0] + 0.3 * jr.normal(key, state.shape)
        )

    def log_obs(emission, state, input_t):
        residual = (emission[0] - state[0] - 0.05 * input_t[0]) / 0.7
        return -0.5 * residual**2

    return initial, transition, log_obs


EMISSIONS = jnp.asarray([[0.2], [-0.1], [0.4], [0.05], [-0.3]])
INPUTS = jnp.asarray([0.5, -0.2, 0.1, 0.4, -0.5])


def _every_second(log_weights, current_ess, time_index):
    del log_weights, current_ess
    return time_index % 2 == 0


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="frozen CPU/x64 arithmetic contract",
)
class TestFixedKeyCharacterization:
    """The FK rewiring reproduces v1.16.0 outputs bitwise on CPU/x64."""

    def test_plain_filter_matches_pre_rewiring_values(self):
        initial, transition, log_obs = _scalar_model()
        posterior = smcx.bootstrap_filter(
            jr.key(7), initial, transition, log_obs, EMISSIONS, 8
        )

        # Characterization is deliberately bitwise: exact equality.
        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik, dtype=np.float64),
            -1.0650910019788342,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess, dtype=np.float64),
            [
                7.358808663574204,
                7.186536808200305,
                6.253748789587071,
                5.702218383013829,
                5.6649161548844,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            [
                -0.26557714548728906,
                -0.1908570369879842,
                -0.2578773563344198,
                -0.15491428765015977,
                -0.1958651755189814,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -2.371662394403776,
                -2.1139233459107087,
                -2.4337907438084647,
                -1.3662128134601033,
                -1.4336862444081753,
                -4.000778825077267,
                -1.9131331691718496,
                -3.2496055258429917,
            ],
        )

    def test_input_criterion_final_only_matches_pre_rewiring_values(self):
        initial, transition, log_obs = _input_model()
        posterior = smcx.bootstrap_filter(
            jr.key(11),
            initial,
            transition,
            log_obs,
            EMISSIONS,
            8,
            resampling_threshold=_every_second,
            inputs=INPUTS,
            store_history=False,
        )

        # Characterization is deliberately bitwise: exact equality.
        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik, dtype=np.float64),
            -1.381429307661791,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess, dtype=np.float64),
            [
                5.467540219699177,
                5.118317361827715,
                7.277171788942068,
                7.186992894457259,
                7.472179594273397,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ancestors)[-1],
            [0, 2, 2, 3, 4, 5, 6, 7],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -1.7216733460523552,
                -2.6110907544897852,
                -1.8261012131510044,
                -1.969392219947026,
                -2.2234362934075396,
                -2.1713339587629426,
                -1.999183208898672,
                -2.413637353221063,
            ],
        )


class TestFeynmanKacDerivation:
    """run_smc is the whole bootstrap driver, and log_g really routes."""

    def test_hand_built_bootstrap_fk_matches_public_filter(self):
        initial, transition, log_obs = _scalar_model()

        def m0(key, num_particles, context_t):
            del context_t
            return initial(key, num_particles)

        def m(key, parent, context_t):
            del context_t
            return transition(key, parent)

        def log_g(parent, state, context_t):
            del parent
            return log_obs(context_t[0], state)

        fk = FeynmanKac(m0=m0, m=m, log_g=log_g, contexts=(EMISSIONS,))
        via_fk = run_smc(jr.key(7), fk, 8)
        via_filter = smcx.bootstrap_filter(
            jr.key(7), initial, transition, log_obs, EMISSIONS, 8
        )

        for fk_field, filter_field in zip(via_fk, via_filter, strict=True):
            np.testing.assert_array_equal(
                np.asarray(fk_field), np.asarray(filter_field)
            )

    def test_custom_potential_changes_the_posterior(self):
        initial, transition, log_obs = _scalar_model()

        def m0(key, num_particles, context_t):
            del context_t
            return initial(key, num_particles)

        def m(key, parent, context_t):
            del context_t
            return transition(key, parent)

        def tempered_log_g(parent, state, context_t):
            del parent
            return 0.5 * log_obs(context_t[0], state)

        fk = FeynmanKac(m0=m0, m=m, log_g=tempered_log_g, contexts=(EMISSIONS,))
        tempered = run_smc(jr.key(7), fk, 8)
        plain = smcx.bootstrap_filter(
            jr.key(7), initial, transition, log_obs, EMISSIONS, 8
        )

        assert not np.array_equal(
            np.asarray(tempered.marginal_loglik),
            np.asarray(plain.marginal_loglik),
        )
        assert np.all(np.isfinite(np.asarray(tempered.marginal_loglik)))


def _aux_model():
    initial, transition, log_obs = _scalar_model()

    def log_aux(emission, state):
        residual = (emission[0] - 0.9 * state[0]) / 0.85
        return -0.5 * residual**2

    return initial, transition, log_obs, log_aux


def _guided_model():
    initial, _transition, log_obs = _scalar_model()

    def proposal(key, state, emission):
        center = 0.5 * (0.9 * state + emission)
        return center + 0.25 * jr.normal(key, state.shape)

    def log_prop(emission, new_state, old_state):
        center = 0.5 * (0.9 * old_state[0] + emission[0])
        return -0.5 * ((new_state[0] - center) / 0.25) ** 2

    def log_trans(new_state, old_state):
        return -0.5 * ((new_state[0] - 0.9 * old_state[0]) / 0.3) ** 2

    return initial, proposal, log_prop, log_trans, log_obs


@pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.read("jax_enable_x64"),
    reason="frozen CPU/x64 arithmetic contract",
)
class TestTwistAndCompositeCharacterization:
    """Auxiliary and guided reproduce v1.16.0 outputs bitwise on CPU/x64."""

    def test_auxiliary_matches_pre_rewiring_values(self):
        initial, transition, log_obs, log_aux = _aux_model()
        posterior = smcx.auxiliary_filter(
            jr.key(21), initial, transition, log_obs, log_aux, EMISSIONS, 8
        )

        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik, dtype=np.float64),
            -1.7394860684888238,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess, dtype=np.float64),
            [
                6.396713581473555,
                5.41376740519584,
                4.108377258509512,
                7.554944040058368,
                6.495867893564239,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            [
                -0.5857624411535574,
                -0.3119479763077554,
                -0.3329898852515369,
                -0.3143980307753713,
                -0.19438773500060294,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -2.094168130848879,
                -1.8044431602557178,
                -1.6525589410444592,
                -3.8327280922543805,
                -3.395475561798124,
                -1.7005531971111998,
                -1.9868640550953853,
                -1.9253834093068876,
            ],
        )

    def test_guided_matches_pre_rewiring_values(self):
        initial, proposal, log_prop, log_trans, log_obs = _guided_model()
        posterior = smcx.guided_filter(
            jr.key(29),
            initial,
            proposal,
            log_prop,
            log_trans,
            log_obs,
            EMISSIONS,
            8,
        )

        np.testing.assert_array_equal(
            np.asarray(posterior.marginal_loglik, dtype=np.float64),
            -0.7156180161338204,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            [
                -0.2837211288005148,
                0.06362456791643989,
                -0.17111978682049123,
                0.07081450868941624,
                -0.3952161771186705,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -2.3342707309652493,
                -3.0271064963558043,
                -3.149824119353661,
                -1.3051532796418335,
                -11.259551063951381,
                -1.0724108278470375,
                -2.6230329472833587,
                -2.072308969364232,
            ],
        )


class TestRunSmcBoundary:
    """run_smc validates its boundary like the named filters (#285)."""

    def _model(self, log_lookahead=None):
        return smcx.StateSpaceModel(
            sample_initial=lambda key, params, i0: jr.normal(key, (1,)),
            sample_transition=lambda key, state, params, it: (
                state + 0.1 * jr.normal(key, (1,))
            ),
            log_observation=lambda emission, state, params, it: (
                -0.5 * (emission[0] - state[0]) ** 2
            ),
            log_lookahead=log_lookahead,
        )

    def _degenerate_fk(self):
        model = self._model(
            log_lookahead=lambda emission, state, params, it: jnp.asarray(
                -jnp.inf
            )
        )
        return smcx.auxiliary_fk(model, {}, jnp.zeros((5, 1)))

    def test_degenerate_lookahead_raises_like_auxiliary_filter(self):
        with pytest.raises(smcx.DegenerateWeightsError):
            run_smc(jr.key(0), self._degenerate_fk(), 64)

    def test_stage_gate_escape_hatch_still_returns(self):
        posterior = run_smc(
            jr.key(0),
            self._degenerate_fk(),
            64,
            gate_stage_normalizers=False,
        )
        assert np.isfinite(float(posterior.marginal_loglik))

    @pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_numeric_threshold(self, threshold):
        fk = smcx.bootstrap_fk(self._model(), {}, jnp.zeros((3, 1)))
        with pytest.raises(ValueError, match="resampling_threshold"):
            run_smc(jr.key(0), fk, 8, resampling_threshold=threshold)

    @pytest.mark.parametrize("num_particles", [0, -4])
    def test_rejects_nonpositive_particle_count(self, num_particles):
        fk = smcx.bootstrap_fk(self._model(), {}, jnp.zeros((3, 1)))
        with pytest.raises(ValueError, match="num_particles"):
            run_smc(jr.key(0), fk, num_particles)

    def test_rejects_mismatched_context_leaf_lengths(self):
        fk = smcx.bootstrap_fk(self._model(), {}, jnp.zeros((3, 1)))
        broken = fk._replace(contexts=(fk.contexts, jnp.zeros(2)))
        with pytest.raises(ValueError, match="context"):
            run_smc(jr.key(0), broken, 8)


class TestCompiledDegeneracyContract:
    """Under user JIT a zero-likelihood run returns exactly -inf (R5).

    Negative infinity is a valid zero-likelihood state that
    pseudo-marginal and outer SMC algorithms can treat as rejection;
    NaN would contaminate their acceptance arithmetic. The binding
    contract is AGENTS.md and architecture guide section 6.
    """

    def _degenerate_model(self, lookahead):
        return smcx.StateSpaceModel(
            sample_initial=lambda key, params, i0: jr.normal(key, (1,)),
            sample_transition=lambda key, state, params, it: state,
            log_observation=lambda emission, state, params, it: jnp.asarray(
                -jnp.inf
            ),
            log_lookahead=lookahead,
        )

    def test_jitted_bootstrap_zero_likelihood_is_minus_inf(self):
        model = self._degenerate_model(None)

        @jax.jit
        def run():
            fk = smcx.bootstrap_fk(model, {}, jnp.zeros((2, 1)))
            return run_smc(jr.key(0), fk, 8).marginal_loglik

        assert np.isneginf(float(run()))

    def test_jitted_auxiliary_zero_lookahead_is_minus_inf(self):
        model = self._degenerate_model(
            lambda emission, state, params, it: jnp.asarray(-jnp.inf)
        )
        model = model._replace(
            log_observation=lambda emission, state, params, it: (
                -0.5 * (emission[0] - state[0]) ** 2
            )
        )

        @jax.jit
        def run():
            fk = smcx.auxiliary_fk(model, {}, jnp.zeros((3, 1)))
            return run_smc(jr.key(0), fk, 8).marginal_loglik

        assert np.isneginf(float(run()))


class TestBoundarySweep:
    """Boolean particle counts and empty time axes are named errors (R9)."""

    def _fk(self):
        model = smcx.StateSpaceModel(
            sample_initial=lambda key, params, i0: jr.normal(key, (1,)),
            sample_transition=lambda key, state, params, it: state,
            log_observation=lambda emission, state, params, it: (
                -0.5 * (emission[0] - state[0]) ** 2
            ),
        )
        return smcx.bootstrap_fk(model, {}, jnp.zeros((3, 1)))

    def test_rejects_boolean_particle_count(self):
        with pytest.raises(ValueError, match="not a boolean"):
            run_smc(jr.key(0), self._fk(), True)

    def test_rejects_empty_time_axis(self):
        # The named-filter derivations already reject empty emissions;
        # a caller-constructed FeynmanKac must get the same named error
        # instead of a raw IndexError from ``leaf[0]``.
        broken = self._fk()._replace(contexts=(jnp.zeros((0, 1)),))
        with pytest.raises(ValueError, match="at least one time step"):
            run_smc(jr.key(0), broken, 8)
