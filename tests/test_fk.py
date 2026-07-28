# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Feynman-Kac core: fixed-key characterization and derivation tests.

The characterization values were captured from ``bootstrap_filter``
at v1.16.0 (commit 8aa3ac8, before the loop was extracted into
``smcx.fk``) and pin the rewiring bitwise on CPU: any change to the
key schedule, weight rule, or evidence accumulation fails here.
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
                6.253748789587073,
                5.702218383013827,
                5.6649161548844,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            [
                -0.26557714548728906,
                -0.19085703698798406,
                -0.2578773563344199,
                -0.1549142876501599,
                -0.1958651755189813,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -2.371662394403776,
                -2.1139233459107087,
                -2.4337907438084647,
                -1.366212813460103,
                -1.4336862444081755,
                -4.000778825077267,
                -1.9131331691718498,
                -3.249605525842992,
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
            -1.3814293076617907,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.ess, dtype=np.float64),
            [
                5.467540219699177,
                5.118317361827714,
                7.277171788942068,
                7.1869928944572665,
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
                5.413767405195838,
                4.1083772585095115,
                7.554944040058368,
                6.495867893564241,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            [
                -0.5857624411535574,
                -0.3119479763077553,
                -0.33298988525153717,
                -0.3143980307753711,
                -0.1943877350006029,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -2.094168130848879,
                -1.8044431602557178,
                -1.6525589410444594,
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
            -0.7156180161338206,
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.log_evidence_increments, dtype=np.float64),
            [
                -0.2837211288005148,
                0.06362456791643978,
                -0.17111978682049145,
                0.07081450868941652,
                -0.39521617711867063,
            ],
        )
        np.testing.assert_array_equal(
            np.asarray(posterior.filtered_log_weights, dtype=np.float64)[-1],
            [
                -2.33427073096525,
                -3.0271064963558034,
                -3.149824119353661,
                -1.3051532796418333,
                -11.259551063951383,
                -1.0724108278470375,
                -2.6230329472833596,
                -2.072308969364232,
            ],
        )
