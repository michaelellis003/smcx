# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: typed callback protocols, exogenous inputs,
# and structured-state aliases.

"""Shared aliases and callback protocols for smcx.

Core array and key aliases follow the conventions used by Dynamax
(``dynamax.types``); callback protocols describe smcx's public boundaries.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    Int32,
    PRNGKeyArray,
    PyTree,
    Shaped,
)

if TYPE_CHECKING:
    from smcx.containers import ParticleFilterRecord

PRNGKeyT = PRNGKeyArray
"""JAX PRNG key (handles both old and new JAX key formats)."""

Scalar = float | Float[Array, ""]
"""Python float or scalar JAX array with float dtype."""

StateTree: TypeAlias = PyTree[Shaped[Array, "..."]]
"""One latent state represented by a nonempty JAX PyTree of arrays."""

ParticleCloud: TypeAlias = PyTree[Shaped[Array, "num_particles ..."]]
"""Latent-state PyTree with a leading particle axis on every leaf."""

ParticleHistory: TypeAlias = PyTree[Shaped[Array, "ntime num_particles ..."]]
"""Latent-state PyTree with leading time and particle axes."""

StateHistory: TypeAlias = PyTree[Shaped[Array, "ntime ..."]]
"""Single-trajectory state PyTree with a leading time axis."""

FilterCarry: TypeAlias = PyTree[Shaped[Array, "..."]]
"""Caller-owned JAX PyTree carried by a particle-filter kernel."""


@runtime_checkable
class _ReplicatedLogMLFn(Protocol):
    """Run one filter replicate and return its log marginal likelihood."""

    def __call__(self, key: PRNGKeyT, /) -> Scalar: ...  # pragma: no cover


# Callback-driven models own the meaning and dtype of observations and
# exogenous inputs. Scalar events are canonicalized to length-one vectors
# before callbacks run. Runtime boundary aliases admit malformed values so
# plain-Python validators own the documented ValueError contract under the
# jaxtyping import hook.
if TYPE_CHECKING:
    Emission: TypeAlias = Shaped[Array, " emission_dim"]
    EmissionValue: TypeAlias = Shaped[Array, ""] | Emission
    EmissionSequence: TypeAlias = (
        Shaped[Array, " ntime"] | Shaped[Array, "ntime emission_dim"]
    )
    GaussianEmission: TypeAlias = Float[Array, " observation_dim"]
    GaussianEmissionSequence: TypeAlias = (
        Float[Array, " ntime"] | Float[Array, "ntime observation_dim"]
    )
    GaussianInput: TypeAlias = Float[Array, " input_dim"]
    GaussianInputSequence: TypeAlias = (
        Float[Array, " ntime"] | Float[Array, "ntime input_dim"]
    )
    ModelInput: TypeAlias = Shaped[Array, " input_dim"]
    InputValue: TypeAlias = Shaped[Array, ""] | ModelInput
    InputSequence: TypeAlias = (
        Shaped[Array, " ntime"] | Shaped[Array, "ntime input_dim"]
    )
else:
    Emission: TypeAlias = Shaped[Array, " emission_dim"]
    EmissionValue: TypeAlias = Any
    EmissionSequence: TypeAlias = Any
    GaussianEmission: TypeAlias = Shaped[Array, " observation_dim"]
    GaussianEmissionSequence: TypeAlias = Any
    GaussianInput: TypeAlias = Shaped[Array, " input_dim"]
    GaussianInputSequence: TypeAlias = Any
    ModelInput: TypeAlias = Shaped[Array, " input_dim"]
    InputValue: TypeAlias = Any
    InputSequence: TypeAlias = Any


@runtime_checkable
class InitialSampler(Protocol):
    """Draw an initial particle cloud."""

    def __call__(
        self, key: PRNGKeyT, num_particles: int, /
    ) -> ParticleCloud: ...


@runtime_checkable
class InitialSamplerWithInput(Protocol):
    """Draw an input-conditioned initial particle cloud."""

    def __call__(
        self,
        key: PRNGKeyT,
        num_particles: int,
        input_t: ModelInput,
        /,
    ) -> ParticleCloud: ...


@runtime_checkable
class DenseInitialSampler(Protocol):
    """Draw a dense initial cloud for Euclidean parameter algorithms."""

    def __call__(
        self, key: PRNGKeyT, num_particles: int, /
    ) -> Float[Array, "num_particles state_dim"]: ...


@runtime_checkable
class DenseInitialSamplerWithInput(Protocol):
    """Draw an input-conditioned dense initial particle cloud."""

    def __call__(
        self,
        key: PRNGKeyT,
        num_particles: int,
        input_t: ModelInput,
        /,
    ) -> Float[Array, "num_particles state_dim"]: ...


@runtime_checkable
class ParamInitialSampler(Protocol):
    """Draw an initial parameter cloud."""

    def __call__(
        self, key: PRNGKeyT, num_particles: int, /
    ) -> Float[Array, "num_particles param_dim"]: ...


@runtime_checkable
class ParamCloudInitialStateSampler(Protocol):
    """Draw initial states conditional on an aligned parameter cloud."""

    def __call__(
        self,
        key: PRNGKeyT,
        num_particles: int,
        params: Float[Array, "num_particles param_dim"],
        /,
    ) -> Float[Array, "num_particles state_dim"]: ...  # pragma: no cover


@runtime_checkable
class ParamCloudInitialStateSamplerWithInput(Protocol):
    """Draw input-aware initial states from an aligned parameter cloud."""

    def __call__(
        self,
        key: PRNGKeyT,
        num_particles: int,
        params: Float[Array, "num_particles param_dim"],
        input_t: ModelInput,
        /,
    ) -> Float[Array, "num_particles state_dim"]: ...  # pragma: no cover


@runtime_checkable
class ParamInitialStateSampler(Protocol):
    """Draw an initial state cloud for one parameter particle."""

    def __call__(
        self,
        key: PRNGKeyT,
        num_particles: int,
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, "num_particles state_dim"]: ...


@runtime_checkable
class StaticLogDensity(Protocol):
    """Evaluate one dense static-target log-density."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class IBISLogLikelihoodFn(Protocol):
    """Evaluate one deterministic conditional likelihood increment."""

    def __call__(
        self,
        emission_t: Emission,
        params: Float[Array, " param_dim"],
        input_t: ModelInput | None,
        /,
    ) -> Scalar: ...


@runtime_checkable
class TemperingMutationState(Protocol):
    """Expose the dense position carried by a tempering mutation kernel."""

    @property
    def position(self) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class TemperingMutationInfo(Protocol):
    """Expose one finite acceptance probability in the interval [0, 1]."""

    @property
    def acceptance_rate(self) -> Scalar: ...


@runtime_checkable
class TemperingScheduleFn(Protocol):
    """Choose the next temperature for one tempering stage."""

    def __call__(
        self,
        phi: float,
        normalized_log_weights: Float[Array, " num_particles"],
        log_likelihoods: Float[Array, " num_particles"],
        /,
    ) -> float: ...


@runtime_checkable
class TemperingMutationInitFn(Protocol):
    """Initialize caller-owned mutation state at the current target."""

    def __call__(
        self,
        position: Float[Array, " state_dim"],
        tempered_logdensity_fn: StaticLogDensity,
        /,
    ) -> TemperingMutationState: ...


@runtime_checkable
class TemperingMutationStepFn(Protocol):
    """Apply one keyed caller-owned mutation step."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: TemperingMutationState,
        tempered_logdensity_fn: StaticLogDensity,
        /,
    ) -> tuple[TemperingMutationState, TemperingMutationInfo]: ...


StaticMutationState: TypeAlias = TemperingMutationState
"""Generic static-target mutation state; alias of the tempering contract."""

StaticMutationInfo: TypeAlias = TemperingMutationInfo
"""Generic static-target mutation diagnostic; alias of tempering info."""

StaticMutationInitFn: TypeAlias = TemperingMutationInitFn
"""Generic static-target mutation initializer; alias of tempering init."""

StaticMutationStepFn: TypeAlias = TemperingMutationStepFn
"""Generic static-target mutation step; alias of the tempering step."""


class StaticMutation(NamedTuple):
    """Paired invariant-mutation callbacks for static-target samplers.

    One value carrying the initializer and step together, so the two
    cannot be supplied apart. `smcx.temper` and `smcx.ibis` accept it
    through their ``mutation`` argument as the primary form of the
    legacy ``mutation_init_fn``/``mutation_step_fn`` pair.

    Attributes:
        init: ``(position, logdensity_fn) -> state`` initializer; the
            state is a JAX PyTree with a dense ``position`` field.
        step: ``(key, state, logdensity_fn) -> (state, info)`` invariant
            step; info exposes a scalar floating ``acceptance_rate`` in
            ``[0, 1]``.
    """

    init: StaticMutationInitFn
    step: StaticMutationStepFn


@runtime_checkable
class TransitionSampler(Protocol):
    """Draw one particle from the transition distribution."""

    def __call__(self, key: PRNGKeyT, state: StateTree, /) -> StateTree: ...


@runtime_checkable
class TransitionSamplerWithInput(Protocol):
    """Draw one input-conditioned transition."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        input_t: ModelInput,
        /,
    ) -> StateTree: ...


@runtime_checkable
class SingleInitialSampler(Protocol):
    """Draw one initial state for forward simulation."""

    def __call__(self, key: PRNGKeyT, /) -> StateTree: ...


@runtime_checkable
class SingleInitialSamplerWithInput(Protocol):
    """Draw one input-conditioned initial state."""

    def __call__(
        self,
        key: PRNGKeyT,
        input_t: ModelInput,
        /,
    ) -> StateTree: ...


@runtime_checkable
class EmissionSampler(Protocol):
    """Draw one emission conditional on a state."""

    def __call__(self, key: PRNGKeyT, state: StateTree, /) -> EmissionValue: ...


@runtime_checkable
class EmissionSamplerWithInput(Protocol):
    """Draw one input-conditioned emission."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        input_t: ModelInput,
        /,
    ) -> EmissionValue: ...


@runtime_checkable
class LogObservationFn(Protocol):
    """Evaluate one particle's observation log-density."""

    def __call__(
        self,
        emission: Emission,
        state: StateTree,
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogObservationFnWithInput(Protocol):
    """Evaluate an input-conditioned observation log-density."""

    def __call__(
        self,
        emission: Emission,
        state: StateTree,
        input_t: ModelInput,
        /,
    ) -> Scalar: ...


@runtime_checkable
class ProposalSampler(Protocol):
    """Draw one particle from a guided proposal."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        emission: Emission,
        /,
    ) -> StateTree: ...


@runtime_checkable
class ProposalSamplerWithInput(Protocol):
    """Draw one particle from an input-conditioned proposal."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        emission: Emission,
        input_t: ModelInput,
        /,
    ) -> StateTree: ...


@runtime_checkable
class LogProposalFn(Protocol):
    """Evaluate one guided proposal log-density."""

    def __call__(
        self,
        emission: Emission,
        new_state: StateTree,
        old_state: StateTree,
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogProposalFnWithInput(Protocol):
    """Evaluate an input-conditioned proposal log-density."""

    def __call__(
        self,
        emission: Emission,
        new_state: StateTree,
        old_state: StateTree,
        input_t: ModelInput,
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogTransitionFn(Protocol):
    """Evaluate one transition log-density."""

    def __call__(
        self,
        new_state: StateTree,
        old_state: StateTree,
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogTransitionFnWithInput(Protocol):
    """Evaluate an input-conditioned transition log-density."""

    def __call__(
        self,
        new_state: StateTree,
        old_state: StateTree,
        input_t: ModelInput,
        /,
    ) -> Scalar: ...


@runtime_checkable
class TransitionMeanFn(Protocol):
    """Evaluate one nonlinear transition mean."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class TransitionMeanFnWithInput(Protocol):
    """Evaluate one input-conditioned nonlinear transition mean."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        input_t: GaussianInput,
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class TransitionJacobianFn(Protocol):
    """Evaluate a transition Jacobian with respect to state."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        /,
    ) -> Float[Array, "state_dim state_dim"]: ...


@runtime_checkable
class TransitionJacobianFnWithInput(Protocol):
    """Evaluate an input-conditioned transition state Jacobian."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        input_t: GaussianInput,
        /,
    ) -> Float[Array, "state_dim state_dim"]: ...


@runtime_checkable
class ObservationMeanFn(Protocol):
    """Evaluate one nonlinear observation mean."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        /,
    ) -> Float[Array, " observation_dim"]: ...


@runtime_checkable
class ObservationMeanFnWithInput(Protocol):
    """Evaluate one input-conditioned nonlinear observation mean."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        input_t: GaussianInput,
        /,
    ) -> Float[Array, " observation_dim"]: ...


@runtime_checkable
class ObservationJacobianFn(Protocol):
    """Evaluate an observation Jacobian with respect to state."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        /,
    ) -> Float[Array, "observation_dim state_dim"]: ...


@runtime_checkable
class ObservationJacobianFnWithInput(Protocol):
    """Evaluate an input-conditioned observation state Jacobian."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
        input_t: GaussianInput,
        /,
    ) -> Float[Array, "observation_dim state_dim"]: ...


@runtime_checkable
class ParamTransitionSampler(Protocol):
    """Draw one parameter-conditioned transition."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class ParamTransitionSamplerWithInput(Protocol):
    """Draw one parameter- and input-conditioned transition."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        input_t: ModelInput,
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class ParamEmissionSampler(Protocol):
    """Draw one emission conditional on a state and static parameters."""

    def __call__(  # pragma: no cover
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        /,
    ) -> EmissionValue: ...


@runtime_checkable
class ParamEmissionSamplerWithInput(Protocol):
    """Draw one parameter- and input-conditioned emission."""

    def __call__(  # pragma: no cover
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        input_t: ModelInput,
        /,
    ) -> EmissionValue: ...


@runtime_checkable
class ParamLogObservationFn(Protocol):
    """Evaluate one parameter-conditioned observation log-density."""

    def __call__(
        self,
        emission: Emission,
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class ParamLogObservationFnWithInput(Protocol):
    """Evaluate a parameter- and input-conditioned log-density."""

    def __call__(
        self,
        emission: Emission,
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        input_t: ModelInput,
        /,
    ) -> Scalar: ...


@runtime_checkable
class ResamplingFn(Protocol):
    """Draw ancestor indices from normalized particle weights.

    The result must be a JAX array with shape ``(num_samples,)``, dtype
    ``int32``, and entries in ``[0, len(weights))``.
    """

    def __call__(
        self,
        key: PRNGKeyT,
        weights: Float[Array, " num_particles"],
        num_samples: int,
        /,
    ) -> Int32[Array, " num_samples"]: ...


@runtime_checkable
class ResamplingCriterion(Protocol):
    """Decide whether to resample one normalized particle cloud."""

    def __call__(
        self,
        normalized_log_weights: Float[Array, " num_particles"],
        current_ess: Float[Array, ""],
        time_index: Int[Array, ""],
        /,
    ) -> bool | Bool[Array, ""]: ...


@runtime_checkable
class ParticleFilterInitFn(Protocol):
    """Initialize a caller-owned particle-filter kernel."""

    def __call__(
        self,
        time_index: Int[Array, ""],
        emission_t: Emission,
        key_t: PRNGKeyT,
        /,
    ) -> "tuple[FilterCarry, ParticleFilterRecord]": ...


@runtime_checkable
class ParticleFilterInitFnWithInput(Protocol):
    """Initialize an input-aware caller-owned particle-filter kernel."""

    def __call__(
        self,
        time_index: Int[Array, ""],
        emission_t: Emission,
        input_t: ModelInput,
        key_t: PRNGKeyT,
        /,
    ) -> "tuple[FilterCarry, ParticleFilterRecord]": ...


@runtime_checkable
class ParticleFilterStepFn(Protocol):
    """Advance a caller-owned particle-filter kernel."""

    def __call__(
        self,
        carry: FilterCarry,
        time_index: Int[Array, ""],
        emission_t: Emission,
        key_t: PRNGKeyT,
        /,
    ) -> "tuple[FilterCarry, ParticleFilterRecord]": ...


@runtime_checkable
class ParticleFilterStepFnWithInput(Protocol):
    """Advance an input-aware caller-owned particle-filter kernel."""

    def __call__(
        self,
        carry: FilterCarry,
        time_index: Int[Array, ""],
        emission_t: Emission,
        input_t: ModelInput,
        key_t: PRNGKeyT,
        /,
    ) -> "tuple[FilterCarry, ParticleFilterRecord]": ...


@runtime_checkable
class ModelInitialSampler(Protocol):
    """Draw one initial state of a `smcx.model.StateSpaceModel`."""

    def __call__(
        self,
        key: PRNGKeyT,
        params: Any,
        input_0: ModelInput | None,
        /,
    ) -> StateTree: ...


@runtime_checkable
class ModelTransitionSampler(Protocol):
    """Draw one transition of a model record."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> StateTree: ...


@runtime_checkable
class ModelLogObservation(Protocol):
    """Observation log-density of a model record."""

    def __call__(
        self,
        emission: Emission,
        state: StateTree,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> Scalar: ...


@runtime_checkable
class ModelLogTransition(Protocol):
    """Transition log-density of a model record."""

    def __call__(
        self,
        state: StateTree,
        prev_state: StateTree,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> Scalar: ...


@runtime_checkable
class ModelProposalSampler(Protocol):
    """Draw one guided proposal, which sees the current emission."""

    def __call__(
        self,
        key: PRNGKeyT,
        prev_state: StateTree,
        emission: Emission,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> StateTree: ...


@runtime_checkable
class ModelLogProposal(Protocol):
    """Proposal log-density of a model record."""

    def __call__(
        self,
        emission: Emission,
        state: StateTree,
        prev_state: StateTree,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> Scalar: ...


@runtime_checkable
class ModelLogLookahead(Protocol):
    """Auxiliary look-ahead evaluated at the pre-propagation state."""

    def __call__(
        self,
        emission: Emission,
        state: StateTree,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> Scalar: ...


@runtime_checkable
class ModelEmissionSampler(Protocol):
    """Draw one emission for simulation and posterior prediction."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        params: Any,
        input_t: ModelInput | None,
        /,
    ) -> EmissionValue: ...


@runtime_checkable
class FamilyMomentMatch(Protocol):
    """Match conjugate prior parameters to two predictor moments."""

    def __call__(
        self,
        forecast_mean: Scalar,
        forecast_variance: Scalar,
        /,
    ) -> tuple[Scalar, Scalar]: ...


@runtime_checkable
class FamilyLogForecast(Protocol):
    """Exact one-step forecast log density of a conjugate family."""

    def __call__(
        self,
        emission: Scalar,
        alpha: Scalar,
        beta: Scalar,
        /,
    ) -> Scalar: ...


@runtime_checkable
class FamilyEmissionSampler(Protocol):
    """Draw one emission given a realized linear predictor.

    The family's link commitment for path simulation: the callable
    maps the linear predictor through the family's link and samples
    one emission from the implied observation distribution.
    """

    def __call__(
        self,
        key: PRNGKeyT,
        linear_predictor: Scalar,
        /,
    ) -> Scalar: ...


@runtime_checkable
class FamilyConjugateUpdate(Protocol):
    """Exact conjugate posterior parameters of a family."""

    def __call__(
        self,
        emission: Scalar,
        alpha: Scalar,
        beta: Scalar,
        /,
    ) -> tuple[Scalar, Scalar]: ...


@runtime_checkable
class FamilyPosteriorMoments(Protocol):
    """Posterior predictor moments fed back by linear Bayes."""

    def __call__(
        self,
        alpha: Scalar,
        beta: Scalar,
        /,
    ) -> tuple[Scalar, Scalar]: ...
