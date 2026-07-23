# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

# Descends from smcjax@e93d527 (https://github.com/michaelellis003/smcjax),
# Apache-2.0. Modified: typed callback protocols, exogenous inputs,
# and structured-state aliases.

"""Shared aliases and callback protocols for smcx.

Matches the conventions used by Dynamax (``dynamax.types``).
"""

from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from jaxtyping import Array, Float, Int, Int32, PRNGKeyArray, PyTree, Shaped

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

# Static checkers see the accepted rank-one/rank-two contract. At runtime,
# beartype must admit any rank so the public plain-Python validator can raise
# the documented ValueError instead of a wrapper-specific type-check error.
if TYPE_CHECKING:
    EmissionSequence: TypeAlias = Float[Array, "ntime emission_dim"]
    InputSequence: TypeAlias = (
        Float[Array, " ntime"] | Float[Array, "ntime input_dim"]
    )
else:
    EmissionSequence: TypeAlias = Float[Array, "*emission_shape"]
    InputSequence: TypeAlias = Float[Array, "*input_shape"]


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
        input_t: Float[Array, " input_dim"],
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
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, "num_particles state_dim"]: ...


@runtime_checkable
class ParamInitialSampler(Protocol):
    """Draw an initial parameter cloud."""

    def __call__(
        self, key: PRNGKeyT, num_particles: int, /
    ) -> Float[Array, "num_particles param_dim"]: ...


@runtime_checkable
class ParamInitialStateSampler(Protocol):
    """Draw a parameter-conditioned initial state-particle cloud."""

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
        input_t: Float[Array, " input_dim"],
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
        input_t: Float[Array, " input_dim"],
        /,
    ) -> StateTree: ...


@runtime_checkable
class EmissionSampler(Protocol):
    """Draw one emission conditional on a state."""

    def __call__(
        self, key: PRNGKeyT, state: StateTree, /
    ) -> Float[Array, " emission_dim"]: ...


@runtime_checkable
class EmissionSamplerWithInput(Protocol):
    """Draw one input-conditioned emission."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " emission_dim"]: ...


@runtime_checkable
class LogObservationFn(Protocol):
    """Evaluate one particle's observation log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        state: StateTree,
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogObservationFnWithInput(Protocol):
    """Evaluate an input-conditioned observation log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        state: StateTree,
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class ProposalSampler(Protocol):
    """Draw one particle from a guided proposal."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        emission: Float[Array, " emission_dim"],
        /,
    ) -> StateTree: ...


@runtime_checkable
class ProposalSamplerWithInput(Protocol):
    """Draw one particle from an input-conditioned proposal."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: StateTree,
        emission: Float[Array, " emission_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> StateTree: ...


@runtime_checkable
class LogProposalFn(Protocol):
    """Evaluate one guided proposal log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        new_state: StateTree,
        old_state: StateTree,
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogProposalFnWithInput(Protocol):
    """Evaluate an input-conditioned proposal log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        new_state: StateTree,
        old_state: StateTree,
        input_t: Float[Array, " input_dim"],
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
        input_t: Float[Array, " input_dim"],
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
class TransitionJacobianFn(Protocol):
    """Evaluate a transition Jacobian with respect to state."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
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
class ObservationJacobianFn(Protocol):
    """Evaluate an observation Jacobian with respect to state."""

    def __call__(
        self,
        state: Float[Array, " state_dim"],
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
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class ParamLogObservationFn(Protocol):
    """Evaluate one parameter-conditioned observation log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class ParamLogObservationFnWithInput(Protocol):
    """Evaluate a parameter- and input-conditioned log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        state: Float[Array, " state_dim"],
        params: Float[Array, " param_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class ResamplingFn(Protocol):
    """Draw ancestor indices from normalized particle weights."""

    def __call__(
        self,
        key: PRNGKeyT,
        weights: Float[Array, " num_particles"],
        num_samples: int,
        /,
    ) -> Int32[Array, " num_samples"]: ...


@runtime_checkable
class ParticleFilterInitFn(Protocol):
    """Initialize a caller-owned particle-filter kernel."""

    def __call__(
        self,
        time_index: Int[Array, ""],
        emission_t: Float[Array, " emission_dim"],
        key_t: PRNGKeyT,
        /,
    ) -> "tuple[FilterCarry, ParticleFilterRecord]": ...


@runtime_checkable
class ParticleFilterInitFnWithInput(Protocol):
    """Initialize an input-aware caller-owned particle-filter kernel."""

    def __call__(
        self,
        time_index: Int[Array, ""],
        emission_t: Float[Array, " emission_dim"],
        input_t: Float[Array, " input_dim"],
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
        emission_t: Float[Array, " emission_dim"],
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
        emission_t: Float[Array, " emission_dim"],
        input_t: Float[Array, " input_dim"],
        key_t: PRNGKeyT,
        /,
    ) -> "tuple[FilterCarry, ParticleFilterRecord]": ...
