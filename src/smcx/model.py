# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

r"""State-space models as records of pure JAX callables.

A `StateSpaceModel` groups the per-particle callables that define one
state-space model, written once and reused by every derivation:
`bootstrap_fk`, `guided_fk`, and `auxiliary_fk` mechanically bind a
model, its parameters, and an observation sequence into a
`smcx.fk.FeynmanKac` for `smcx.fk.run_smc`. Nothing here is a model
class: the record carries no behavior, no distributions, and no
inheritance (ADR-0019; ADR-0023).

Two conventions replace the callback-era variation:

- ``params`` is an explicit PyTree argument threaded by the library,
  never closed over, so gradients with respect to parameters flow
  through filters and changing parameters cannot retrace.
- ``input_t`` is always the final argument and is ``None`` when the
  run has no exogenous inputs; a model that ignores inputs simply
  ignores the argument. One signature per role, no arity variants.

Optional capabilities are fields that default to ``None``; a
derivation that needs a missing capability raises a named error at
construction. smcx never inspects a model object's signatures.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import vmap

from smcx._utils import (
    _canonicalize_emissions,
    _canonicalize_inputs,
    _validate_log_density_batch,
)
from smcx.fk import CallbackNames, FeynmanKac
from smcx.types import (
    EmissionSequence,
    InputSequence,
    ModelEmissionSampler,
    ModelInitialSampler,
    ModelLogLookahead,
    ModelLogObservation,
    ModelLogProposal,
    ModelLogTransition,
    ModelProposalSampler,
    ModelTransitionSampler,
)


class StateSpaceModel(NamedTuple):
    """One state-space model as a record of per-particle callables.

    Every callable takes ``params`` and a trailing ``input_t`` (which
    is ``None`` when the run has no inputs). States may be arbitrary
    nonempty JAX PyTrees; emissions keep their model-owned dtype.

    Attributes:
        sample_initial: ``(key, params, input_0) -> state`` drawing one
            initial state; derivations vmap it over per-particle keys.
        sample_transition: ``(key, state, params, input_t) -> state``
            drawing one transition; must preserve the state PyTree
            structure, leaf shapes, and dtypes.
        log_observation: ``(emission, state, params, input_t) ->
            scalar`` observation log-density.
        log_transition: Optional ``(state, prev_state, params, input_t)
            -> scalar`` transition log-density (guided weighting,
            smoothing).
        sample_proposal: Optional ``(key, prev_state, emission, params,
            input_t) -> state`` guided proposal, which sees the current
            emission.
        log_proposal: Optional ``(emission, state, prev_state, params,
            input_t) -> scalar`` proposal log-density.
        log_lookahead: Optional ``(emission, state, params, input_t) ->
            scalar`` auxiliary-filter look-ahead evaluated at the
            pre-propagation state.
        sample_emission: Optional ``(key, state, params, input_t) ->
            emission``. Reserved capability: no smcx function consumes
            it today — `smcx.simulate` and the predictive functions
            take standalone callbacks — and its retention or removal
            is a compatibility decision, not implied usage.
    """

    sample_initial: ModelInitialSampler
    sample_transition: ModelTransitionSampler
    log_observation: ModelLogObservation
    log_transition: ModelLogTransition | None = None
    sample_proposal: ModelProposalSampler | None = None
    log_proposal: ModelLogProposal | None = None
    log_lookahead: ModelLogLookahead | None = None
    sample_emission: ModelEmissionSampler | None = None


def _prepare_sequences(
    emissions: EmissionSequence,
    inputs: InputSequence | None,
) -> tuple[Any, Callable[[Any], Any]]:
    """Canonicalize sequences into a context PyTree and input accessor."""
    emissions = _canonicalize_emissions(emissions)
    if inputs is None:
        return (emissions,), lambda context_t: None
    inputs_arr = _canonicalize_inputs(inputs, emissions.shape[0])
    return (emissions, inputs_arr), lambda context_t: context_t[1]


def _require(model: StateSpaceModel, field: str, derivation: str) -> None:
    """Raise a named error when a derivation needs a missing capability."""
    if getattr(model, field) is None:
        raise ValueError(
            f"{derivation} requires model.{field}; this StateSpaceModel "
            "does not define it"
        )


def bootstrap_fk(
    model: StateSpaceModel,
    params: Any,
    emissions: EmissionSequence,
    *,
    inputs: InputSequence | None = None,
) -> FeynmanKac:
    """Derive the bootstrap Feynman-Kac model.

    The mutation kernel is the model transition and the potential is
    the observation density.

    Args:
        model: State-space model record.
        params: Model parameter PyTree, passed to every callable.
        emissions: Observations shaped ``(T,)`` or ``(T, emission_dim)``;
            rank-one data become ``(T, 1)``; dtype is preserved.
        inputs: Optional exogenous inputs shaped ``(T,)`` or
            ``(T, input_dim)``; ``inputs[t]`` reaches the transition
            into ``t`` and the observation at ``t``; ``inputs[0]``
            reaches initialization.

    Returns:
        A `smcx.fk.FeynmanKac` ready for `smcx.fk.run_smc`.

    Raises:
        ValueError: The observation or input sequence is malformed.
    """
    contexts, get_input = _prepare_sequences(emissions, inputs)
    names = CallbackNames(
        m0="model.sample_initial output",
        m="model.sample_transition output",
        log_g="model.log_observation",
    )

    def m0(key, num_particles, context_t):
        keys = jr.split(key, num_particles)
        return vmap(
            lambda particle_key: model.sample_initial(
                particle_key, params, get_input(context_t)
            )
        )(keys)

    def m(key, parent, context_t):
        return model.sample_transition(
            key, parent, params, get_input(context_t)
        )

    def log_g(parent, state, context_t):
        del parent
        return model.log_observation(
            context_t[0], state, params, get_input(context_t)
        )

    return FeynmanKac(m0=m0, m=m, log_g=log_g, contexts=contexts, names=names)


def auxiliary_fk(
    model: StateSpaceModel,
    params: Any,
    emissions: EmissionSequence,
    *,
    inputs: InputSequence | None = None,
) -> FeynmanKac:
    """Derive the auxiliary Feynman-Kac model.

    Bootstrap dynamics plus the model's look-ahead twist.

    Args and canonicalization match `bootstrap_fk`.

    Raises:
        ValueError: ``model.log_lookahead`` is missing, or a sequence
            is malformed.
    """
    _require(model, "log_lookahead", "auxiliary_fk")
    base = bootstrap_fk(model, params, emissions, inputs=inputs)
    contexts = base.contexts
    get_input = (
        (lambda context_t: None)
        if len(contexts) == 1
        else (lambda context_t: context_t[1])
    )

    def log_eta(state, context_t):
        assert model.log_lookahead is not None
        return model.log_lookahead(
            context_t[0], state, params, get_input(context_t)
        )

    return base._replace(
        log_eta=log_eta,
        names=base.names._replace(log_eta="model.log_lookahead"),
    )


def guided_fk(
    model: StateSpaceModel,
    params: Any,
    emissions: EmissionSequence,
    *,
    inputs: InputSequence | None = None,
) -> FeynmanKac:
    """Derive the guided Feynman-Kac model.

    The mutation kernel is the model proposal and the step potential
    is the composite ``g * f / q``.
    Initialization weights by the observation density alone, exactly
    as `smcx.guided_filter` does.

    Args and canonicalization match `bootstrap_fk`.

    Raises:
        ValueError: ``model.sample_proposal``, ``model.log_proposal``,
            or ``model.log_transition`` is missing, or a sequence is
            malformed.
    """
    for field in ("sample_proposal", "log_proposal", "log_transition"):
        _require(model, field, "guided_fk")
    base = bootstrap_fk(model, params, emissions, inputs=inputs)
    contexts = base.contexts
    get_input = (
        (lambda context_t: None)
        if len(contexts) == 1
        else (lambda context_t: context_t[1])
    )
    sample_proposal = model.sample_proposal
    log_proposal = model.log_proposal
    log_transition = model.log_transition
    assert sample_proposal is not None
    assert log_proposal is not None
    assert log_transition is not None

    def m(key, parent, context_t):
        return sample_proposal(
            key, parent, context_t[0], params, get_input(context_t)
        )

    def log_g_batch(parents, propagated, context_t):
        emission_t = context_t[0]
        input_t = get_input(context_t)
        log_obs = jnp.asarray(
            vmap(
                lambda state: model.log_observation(
                    emission_t, state, params, input_t
                )
            )(propagated)
        )
        log_f = jnp.asarray(
            vmap(
                lambda state, prev: log_transition(state, prev, params, input_t)
            )(propagated, parents)
        )
        log_q = jnp.asarray(
            vmap(
                lambda state, prev: log_proposal(
                    emission_t, state, prev, params, input_t
                )
            )(propagated, parents)
        )
        num_particles = log_obs.shape[0]
        for name, values in (
            ("model.log_observation", log_obs),
            ("model.log_transition", log_f),
            ("model.log_proposal", log_q),
        ):
            _validate_log_density_batch(values, num_particles, name=name)
        return log_obs + log_f - log_q

    return base._replace(
        m=m,
        log_g_batch=log_g_batch,
        names=base.names._replace(m="model.sample_proposal output"),
    )
