# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Public entry points reject malformed structural inputs clearly."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import config as jaxtyping_config

import smcx


def _initial_sampler(key, num_particles):
    del key
    return jnp.zeros((num_particles, 1))


def _transition_sampler(key, state):
    del key
    return state


def _log_observation(emission, state):
    assert emission.ndim == 1
    del state
    return jnp.asarray(0.0)


def _run_bootstrap(emissions, num_particles, **kwargs):
    return smcx.bootstrap_filter(
        jr.key(0),
        _initial_sampler,
        _transition_sampler,
        _log_observation,
        emissions,
        num_particles,
        **kwargs,
    )


def _run_auxiliary(emissions, num_particles, **kwargs):
    return smcx.auxiliary_filter(
        jr.key(0),
        _initial_sampler,
        _transition_sampler,
        _log_observation,
        _log_observation,
        emissions,
        num_particles,
        **kwargs,
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


def _run_guided(emissions, num_particles, **kwargs):
    return smcx.guided_filter(
        jr.key(0),
        _initial_sampler,
        _proposal_sampler,
        _log_proposal,
        _log_transition,
        _log_observation,
        emissions,
        num_particles,
        **kwargs,
    )


def _param_initial_sampler(key, num_particles):
    del key
    return jnp.zeros((num_particles, 1))


def _log_quadratic(value):
    return -jnp.sum(value**2)


def _param_transition(key, state, params):
    del key, params
    return state


def _param_log_observation(emission, state, params):
    del emission, state, params
    return jnp.asarray(0.0)


def _run_liu_west(emissions, num_particles, *, shrinkage=0.95, **kwargs):
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
        **kwargs,
    )


_PARTICLE_FILTERS = (
    _run_bootstrap,
    _run_auxiliary,
    _run_guided,
    _run_liu_west,
)


def test_callback_receives_rank_one_integer_emissions_as_vectors():
    def log_observation(emission, state):
        assert emission.shape == (1,)
        assert emission.dtype == jnp.int32
        return 0.0 * state[0]

    posterior = smcx.bootstrap_filter(
        jr.key(152),
        _initial_sampler,
        _transition_sampler,
        log_observation,
        jnp.array([0, 1], dtype=jnp.int32),
        4,
    )

    assert posterior.log_evidence_increments.shape == (2,)


@pytest.mark.parametrize("run_filter", _PARTICLE_FILTERS)
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_])
def test_particle_filters_accept_scalar_discrete_emissions(run_filter, dtype):
    emissions = jnp.array([0, 1], dtype=dtype)

    posterior = run_filter(emissions, 4)

    assert posterior.log_evidence_increments.shape == (2,)


@pytest.mark.parametrize("run_filter", _PARTICLE_FILTERS)
@pytest.mark.parametrize(
    ("emissions", "num_particles", "message"),
    [
        (jnp.zeros((0, 1)), 4, "must contain at least one row"),
        (jnp.zeros((2, 1)), 0, "num_particles must be >= 1"),
        (jnp.zeros((2, 0)), 4, "emission_dim >= 1"),
        (jnp.zeros((2, 1, 1)), 4, r"shape \(T,\) or \(T, emission_dim\)"),
        ([0.0, 1.0], 4, "must be a JAX array"),
    ],
)
def test_particle_filters_validate_structure(
    run_filter,
    emissions,
    num_particles,
    message,
):
    with pytest.raises(ValueError, match=message):
        run_filter(emissions, num_particles)


def test_particle_filter_rejects_zero_width_inputs():
    with pytest.raises(ValueError, match="input_dim >= 1"):
        smcx.bootstrap_filter(
            jr.key(152),
            _initial_sampler,
            _transition_sampler,
            _log_observation,
            jnp.zeros((2, 1)),
            4,
            inputs=jnp.empty((2, 0)),
        )


def test_particle_filter_rejects_non_jax_inputs():
    """The shared sequence boundary owns the non-JAX structural error."""
    with pytest.raises(ValueError, match="inputs must be a JAX array"):
        smcx.bootstrap_filter(
            jr.key(152),
            _initial_sampler,
            _transition_sampler,
            _log_observation,
            jnp.zeros((2, 1)),
            4,
            inputs=[[0.0], [1.0]],  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("run_filter", _PARTICLE_FILTERS)
@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_particle_filters_reject_invalid_numeric_resampling_threshold(
    run_filter, threshold
):
    with pytest.raises(
        ValueError,
        match="resampling_threshold must be finite and nonnegative",
    ):
        run_filter(
            jnp.zeros((2, 1)),
            4,
            resampling_threshold=threshold,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (jnp.zeros(2), "output must have shape"),
        (jnp.asarray(0), "must have a floating dtype"),
    ],
)
def test_bootstrap_validates_log_observation(value, message):
    def invalid_log_observation(emission, state):
        del emission, state
        return value

    with pytest.raises(ValueError, match=message):
        smcx.bootstrap_filter(
            jr.key(0),
            _initial_sampler,
            _transition_sampler,
            invalid_log_observation,
            jnp.zeros((2, 1)),
            4,
        )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_bootstrap_rejects_low_precision_log_observation(dtype):
    def low_precision_log_observation(emission, state):
        del emission, state
        return jnp.asarray(0.0, dtype=dtype)

    with pytest.raises(ValueError, match="at least float32 precision"):
        smcx.bootstrap_filter(
            jr.key(0),
            _initial_sampler,
            _transition_sampler,
            low_precision_log_observation,
            jnp.zeros((2, 1)),
            512,
        )


def _valid_then(value):
    values = iter((jnp.asarray(0.0), value))

    def callback(*_args):
        return next(values)

    return callback


def _always(value):
    def callback(*_args):
        return value

    return callback


def _run_invalid_later_density(boundary, value):
    emissions = jnp.zeros((2, 1))
    if boundary == "bootstrap_observation":
        return smcx.bootstrap_filter(
            jr.key(0),
            _initial_sampler,
            _transition_sampler,
            _valid_then(value),
            emissions,
            4,
        )
    if boundary.startswith("auxiliary_"):
        observation = (
            _valid_then(value)
            if boundary == "auxiliary_observation"
            else _log_observation
        )
        auxiliary = (
            _always(value)
            if boundary == "auxiliary_auxiliary"
            else _log_observation
        )
        return smcx.auxiliary_filter(
            jr.key(0),
            _initial_sampler,
            _transition_sampler,
            observation,
            auxiliary,
            emissions,
            4,
        )
    if boundary.startswith("guided_"):
        observation = (
            _valid_then(value)
            if boundary == "guided_observation"
            else _log_observation
        )
        transition = (
            _always(value)
            if boundary == "guided_transition"
            else _log_transition
        )
        proposal = (
            _always(value) if boundary == "guided_proposal" else _log_proposal
        )
        return smcx.guided_filter(
            jr.key(0),
            _initial_sampler,
            _proposal_sampler,
            proposal,
            transition,
            observation,
            emissions,
            4,
        )
    observation = (
        _valid_then(value)
        if boundary == "liu_west_observation"
        else _param_log_observation
    )
    auxiliary = (
        _always(value)
        if boundary == "liu_west_auxiliary"
        else _param_log_observation
    )
    return smcx.liu_west_filter(
        jr.key(0),
        _initial_sampler,
        _param_transition,
        observation,
        auxiliary,
        _param_initial_sampler,
        emissions,
        4,
    )


@pytest.mark.parametrize(
    ("boundary", "callback_name"),
    [
        ("bootstrap_observation", "log_observation_fn"),
        ("auxiliary_observation", "log_observation_fn"),
        ("auxiliary_auxiliary", "log_auxiliary_fn"),
        ("guided_observation", "log_observation_fn"),
        ("guided_transition", "log_transition_fn"),
        ("guided_proposal", "log_proposal_fn"),
        ("liu_west_observation", "log_observation_fn"),
        ("liu_west_auxiliary", "log_auxiliary_fn"),
    ],
)
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (jnp.zeros(2), "output must have shape"),
        (jnp.asarray(0, dtype=jnp.int32), "output must have a floating dtype"),
    ],
)
def test_filters_validate_later_log_density_batches(
    boundary, callback_name, value, message
):
    with pytest.raises(ValueError, match=f"{callback_name} {message}"):
        _run_invalid_later_density(boundary, value)


@pytest.mark.parametrize(
    ("boundary", "callback_name"),
    [
        ("bootstrap_observation", "log_observation_fn"),
        ("auxiliary_observation", "log_observation_fn"),
        ("auxiliary_auxiliary", "log_auxiliary_fn"),
        ("guided_observation", "log_observation_fn"),
        ("guided_transition", "log_transition_fn"),
        ("guided_proposal", "log_proposal_fn"),
        ("liu_west_observation", "log_observation_fn"),
        ("liu_west_auxiliary", "log_auxiliary_fn"),
    ],
)
def test_filters_reject_later_low_precision_log_densities(
    boundary, callback_name
):
    value = jnp.asarray(0.0, dtype=jnp.bfloat16)

    with pytest.raises(
        ValueError,
        match=f"{callback_name} output must have at least float32 precision",
    ):
        _run_invalid_later_density(boundary, value)


@pytest.mark.parametrize("source", ["state", "parameter"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (jnp.zeros(4), r"shape \(num_particles, dimension\)"),
        (jnp.zeros((4, 0)), r"shape \(num_particles, dimension\)"),
        (jnp.zeros((4, 1), dtype=jnp.int32), "floating dtype"),
        (jnp.zeros((5, 1)), "leading dimension num_particles=4"),
        ([[0.0]] * 4, "must be a JAX array"),
    ],
)
def test_liu_west_validates_initial_dense_clouds(source, value, message):
    def invalid_initial(_key, _count):
        return value

    state_initial = invalid_initial if source == "state" else _initial_sampler
    parameter_initial = (
        invalid_initial if source == "parameter" else _param_initial_sampler
    )

    with pytest.raises(ValueError, match=message):
        smcx.liu_west_filter(
            jr.key(0),
            state_initial,
            _param_transition,
            _param_log_observation,
            _param_log_observation,
            parameter_initial,
            jnp.zeros((1, 1)),
            4,
        )


@pytest.mark.parametrize(
    ("transition_output", "message"),
    [
        ({"value": jnp.zeros(1)}, "PyTree structure"),
        (jnp.zeros(2), "preserve shape"),
        (jnp.zeros(1, dtype=jnp.int32), "preserve dtype"),
    ],
)
def test_liu_west_transition_preserves_state_contract(
    transition_output, message
):
    def invalid_transition(_key, _state, _params):
        return transition_output

    with pytest.raises(ValueError, match=message):
        smcx.liu_west_filter(
            jr.key(0),
            _initial_sampler,
            invalid_transition,
            _param_log_observation,
            _param_log_observation,
            _param_initial_sampler,
            jnp.zeros((2, 1)),
            4,
        )


def test_liu_west_validates_shrinkage_before_callbacks():
    with pytest.raises(ValueError, match="shrinkage must be in"):
        _run_liu_west(jnp.zeros((2, 1)), 1, shrinkage=1.1)


def test_simulate_requires_at_least_one_timestep():
    with pytest.raises(ValueError, match="num_timesteps must be >= 1"):
        smcx.simulate(
            jr.key(0),
            lambda _key: jnp.zeros(1),
            lambda _key, state: state,
            lambda _key, state: state,
            0,
        )


def _run_temper(initial_sampler=_initial_sampler, **kwargs):
    num_particles = kwargs.pop("num_particles", 4)
    return smcx.temper(
        jr.key(0),
        initial_sampler,
        _log_quadratic,
        _log_quadratic,
        num_particles,
        **kwargs,
    )


@pytest.mark.parametrize(
    "target_ess",
    [
        1.0,
        1.0 - float(jnp.finfo(jnp.float32).eps) / 2.0,
    ],
)
def test_temper_rejects_unsupported_upper_target_before_initialization(
    target_ess,
):
    def unexpected_sampler(_key, _count):
        pytest.fail("target_ess validation must precede model callbacks")

    with pytest.raises(ValueError, match="target_ess must be in"):
        _run_temper(unexpected_sampler, target_ess=target_ess)


@pytest.mark.parametrize(
    ("initial_sampler", "kwargs", "message"),
    [
        (_initial_sampler, {"num_particles": 0}, "num_particles must be >= 1"),
        (_initial_sampler, {"target_ess": 0.0}, "target_ess must be in"),
        (_initial_sampler, {"max_stages": 0}, "max_stages must be >= 1"),
        (
            lambda _key, count: jnp.zeros(count),
            {},
            r"must have shape \(N, d\)",
        ),
        (
            lambda _key, count: [[0.0]] * count,
            {},
            "must be a JAX array",
        ),
        (
            lambda _key, count: jnp.zeros((count, 1), dtype=jnp.int32),
            {},
            "initial_sampler output must have a floating dtype",
        ),
    ],
)
def test_temper_validates_public_structure(
    monkeypatch, initial_sampler, kwargs, message
):
    monkeypatch.setattr(jaxtyping_config, "jaxtyping_disable", True)
    with pytest.raises(ValueError, match=message):
        _run_temper(initial_sampler, **kwargs)


@pytest.mark.parametrize(
    ("callback_name", "log_prior_fn", "log_likelihood_fn"),
    [
        (
            "log_prior_fn",
            lambda _value: jnp.asarray(0.0, dtype=jnp.bfloat16),
            _log_quadratic,
        ),
        (
            "log_likelihood_fn",
            _log_quadratic,
            lambda _value: jnp.asarray(0.0, dtype=jnp.bfloat16),
        ),
    ],
)
def test_temper_rejects_low_precision_log_density(
    callback_name, log_prior_fn, log_likelihood_fn
):
    with pytest.raises(
        ValueError,
        match=f"{callback_name} output must have at least float32 precision",
    ):
        smcx.temper(
            jr.key(0),
            _initial_sampler,
            log_prior_fn,
            log_likelihood_fn,
            4,
        )


def _smc2_initial(key, count, params):
    del key, params
    return jnp.zeros((count, 1))


def _smc2_transition(key, state, params):
    del key, params
    return state


def _smc2_log_observation(emission, state, params):
    del emission, state, params
    return jnp.asarray(0.0)


def _run_smc2(emissions, num_theta=2, num_x=2, **kwargs):
    return smcx.smc2(
        jr.key(0),
        kwargs.pop("param_initial_sampler", _param_initial_sampler),
        _log_quadratic,
        _smc2_initial,
        _smc2_transition,
        _smc2_log_observation,
        emissions,
        num_theta,
        num_x,
        **kwargs,
    )


@pytest.mark.parametrize(
    (
        "emissions",
        "num_theta",
        "num_x",
        "kwargs",
        "message",
    ),
    [
        (jnp.zeros((0, 1)), 2, 2, {}, "must contain at least one row"),
        (jnp.zeros((2, 1)), 0, 2, {}, "num_theta must be >= 1"),
        (jnp.zeros((2, 1)), 2, 0, {}, "num_x must be >= 1"),
        (
            jnp.zeros((2, 1)),
            2,
            2,
            {"num_pmmh_steps": -1},
            "num_pmmh_steps must be >= 0",
        ),
        (
            jnp.zeros((2, 1)),
            2,
            2,
            {"param_initial_sampler": lambda _key, count: [[0.0]] * count},
            "must be a JAX array",
        ),
        (
            jnp.zeros((2, 1)),
            2,
            2,
            {
                "param_initial_sampler": lambda _key, count: jnp.zeros((
                    count,
                    0,
                ))
            },
            "param_dim >= 1",
        ),
    ],
)
def test_smc2_validates_public_structure(
    emissions,
    num_theta,
    num_x,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        _run_smc2(emissions, num_theta, num_x, **kwargs)


def test_smc2_accepts_scalar_emission_series():
    posterior = _run_smc2(
        jnp.zeros(2),
        ess_threshold=0.0,
        num_pmmh_steps=0,
    )

    assert posterior.filtered_params.shape == (2, 2, 1)


def test_smc2_rejects_low_precision_log_observation():
    def low_precision_log_observation(emission, state, params):
        del emission, state, params
        return jnp.asarray(0.0, dtype=jnp.bfloat16)

    with pytest.raises(
        ValueError,
        match="log_observation_fn output must have at least float32 precision",
    ):
        smcx.smc2(
            jr.key(0),
            _param_initial_sampler,
            _log_quadratic,
            _smc2_initial,
            _smc2_transition,
            low_precision_log_observation,
            jnp.zeros((1, 1)),
            2,
            2,
        )
