# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Small numerical primitives shared across inference algorithms."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def _neumaier_add(
    total: Float[Array, "..."],
    correction: Float[Array, "..."],
    value: float | Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Add one value while retaining a Neumaier correction."""
    updated = total + value
    correction = correction + jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, correction
