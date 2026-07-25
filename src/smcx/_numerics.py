# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Small numerical primitives shared across inference algorithms."""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


def _neumaier_add(
    total: Float[Array, "..."],
    correction: Float[Array, "..."],
    value: float | Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Add one value while retaining a Neumaier correction.

    References:
        Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur
        Summation endlicher Summen. https://doi.org/10.1002/zamm.19740540106
    """
    updated = total + value
    correction = correction + jnp.where(
        jnp.abs(total) >= jnp.abs(value),
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, correction


def _neumaier_prefix_sum(
    values: Float[Array, " nvalues"],
) -> Float[Array, " nvalues"]:
    """Return every prefix from sequential Neumaier summation."""

    def add(
        carry: tuple[Float[Array, ""], Float[Array, ""]],
        value: Float[Array, ""],
    ) -> tuple[
        tuple[Float[Array, ""], Float[Array, ""]],
        Float[Array, ""],
    ]:
        total, correction = _neumaier_add(*carry, value)
        return (total, correction), total + correction

    zero = jnp.zeros((), dtype=values.dtype)
    _, prefixes = lax.scan(add, (zero, zero), values)
    return prefixes
