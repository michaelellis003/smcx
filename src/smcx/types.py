# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Shared aliases and callback protocols for smcx.

Matches the conventions used by Dynamax (``dynamax.types``).
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, Float, Int, PRNGKeyArray

PRNGKeyT = PRNGKeyArray
"""JAX PRNG key (handles both old and new JAX key formats)."""

Scalar = float | Float[Array, ""]
"""Python float or scalar JAX array with float dtype."""

InputSequence = Float[Array, "*input_shape"]
"""Candidate input sequence; public entry points validate rank one or two."""


@runtime_checkable
class InitialSampler(Protocol):
    """Draw an initial particle cloud."""

    def __call__(
        self, key: PRNGKeyT, num_particles: int, /
    ) -> Float[Array, "num_particles state_dim"]: ...


@runtime_checkable
class InitialSamplerWithInput(Protocol):
    """Draw an input-conditioned initial particle cloud."""

    def __call__(
        self,
        key: PRNGKeyT,
        num_particles: int,
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, "num_particles state_dim"]: ...


@runtime_checkable
class TransitionSampler(Protocol):
    """Draw one particle from the transition distribution."""

    def __call__(
        self, key: PRNGKeyT, state: Float[Array, " state_dim"], /
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class TransitionSamplerWithInput(Protocol):
    """Draw one input-conditioned transition."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class LogObservationFn(Protocol):
    """Evaluate one particle's observation log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        state: Float[Array, " state_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogObservationFnWithInput(Protocol):
    """Evaluate an input-conditioned observation log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        state: Float[Array, " state_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class ProposalSampler(Protocol):
    """Draw one particle from a guided proposal."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        emission: Float[Array, " emission_dim"],
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class ProposalSamplerWithInput(Protocol):
    """Draw one particle from an input-conditioned proposal."""

    def __call__(
        self,
        key: PRNGKeyT,
        state: Float[Array, " state_dim"],
        emission: Float[Array, " emission_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Float[Array, " state_dim"]: ...


@runtime_checkable
class LogProposalFn(Protocol):
    """Evaluate one guided proposal log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        new_state: Float[Array, " state_dim"],
        old_state: Float[Array, " state_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogProposalFnWithInput(Protocol):
    """Evaluate an input-conditioned proposal log-density."""

    def __call__(
        self,
        emission: Float[Array, " emission_dim"],
        new_state: Float[Array, " state_dim"],
        old_state: Float[Array, " state_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogTransitionFn(Protocol):
    """Evaluate one transition log-density."""

    def __call__(
        self,
        new_state: Float[Array, " state_dim"],
        old_state: Float[Array, " state_dim"],
        /,
    ) -> Scalar: ...


@runtime_checkable
class LogTransitionFnWithInput(Protocol):
    """Evaluate an input-conditioned transition log-density."""

    def __call__(
        self,
        new_state: Float[Array, " state_dim"],
        old_state: Float[Array, " state_dim"],
        input_t: Float[Array, " input_dim"],
        /,
    ) -> Scalar: ...


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
    ) -> Int[Array, " num_samples"]: ...
