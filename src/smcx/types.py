# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases for smcx.

Callback Protocols (ADR-0008 forms) are added here as the modules
that consume them land.
"""

import mlx.core as mx
from jaxtyping import Float, UInt32

# Splittable RNG key produced by ``mx.random.key`` / ``mx.random.split``
# (ADR-0005: every stochastic function takes one explicitly).
KeyT = UInt32[mx.array, " 2"]

# A Python float or a zero-dimensional MLX array. Matches smcjax's
# (Dynamax-convention) ``Scalar`` alias.
Scalar = float | Float[mx.array, ""]
