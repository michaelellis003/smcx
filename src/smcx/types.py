# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases for smcx.

Callback Protocols (ADR-0008 forms) are added here as the modules
that consume them land.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx
from jaxtyping import Float, UInt32

# Splittable RNG key produced by ``mx.random.key`` / ``mx.random.split``
# (ADR-0005: every stochastic function takes one explicitly).
KeyT = UInt32[mx.array, " 2"]

# A Python float or a zero-dimensional MLX array. Matches smcjax's
# (Dynamax-convention) ``Scalar`` alias.
Scalar = float | Float[mx.array, ""]


# --- Callback Protocols (ADR-0008) -----------------------------------
# Structural, so plain closures/lambdas satisfy them with no imports.
# When an `inputs` array is supplied to a filter, every per-step
# callback receives a trailing `input_t` argument (the *WithInput
# forms); the two-argument parity forms remain valid otherwise.


@runtime_checkable
class InitialSampler(Protocol):
    """Draws the whole initial particle cloud: ``(key, n) -> (n, d)``."""

    def __call__(self, key: mx.array, num_particles: int, /) -> mx.array: ...


@runtime_checkable
class TransitionSampler(Protocol):
    """Per-particle transition draw: ``(key, state) -> state``."""

    def __call__(self, key: mx.array, state: mx.array, /) -> mx.array: ...


@runtime_checkable
class TransitionSamplerWithInput(Protocol):
    """Input-driven transition: ``(key, state, input_t) -> state``."""

    def __call__(
        self, key: mx.array, state: mx.array, input_t: mx.array, /
    ) -> mx.array: ...


@runtime_checkable
class LogObservationFn(Protocol):
    """Per-particle observation log-density: ``(emission, state)``."""

    def __call__(self, emission: mx.array, state: mx.array, /) -> mx.array: ...


@runtime_checkable
class LogObservationFnWithInput(Protocol):
    """Input-driven observation log-density."""

    def __call__(
        self, emission: mx.array, state: mx.array, input_t: mx.array, /
    ) -> mx.array: ...


@runtime_checkable
class EmissionSampler(Protocol):
    """Per-particle emission draw: ``(key, state) -> emission``."""

    def __call__(self, key: mx.array, state: mx.array, /) -> mx.array: ...


@runtime_checkable
class EmissionSamplerWithInput(Protocol):
    """Input-driven emission draw: ``(key, state, input_t)``."""

    def __call__(
        self, key: mx.array, state: mx.array, input_t: mx.array, /
    ) -> mx.array: ...
