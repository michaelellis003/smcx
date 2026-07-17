# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised by smcx."""

__all__ = ["DegenerateWeightsError"]


class DegenerateWeightsError(ValueError):
    """All particle weights collapsed to zero (log-weights all -inf).

    Raised by the filter entry points after the scan completes, when
    the marginal log-likelihood comes back ``-inf`` or NaN. Detection
    is host-side, so it fires only in eager execution: inside a
    user ``jax.jit`` the marginal simply carries the ``-inf``/NaN
    through (pure functions signal degeneracy; they cannot raise).
    Catch this in pseudo-marginal outer loops (e.g. PMMH) to reject
    the proposal that caused it.
    """
