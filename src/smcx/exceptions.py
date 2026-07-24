# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised by smcx."""

__all__ = ["DegenerateWeightsError"]


class DegenerateWeightsError(ValueError):
    """A checked particle-weight or evidence state is ``-inf`` or NaN.

    Public algorithms raise at their eager shell boundaries, which may be
    initialization, an intermediate stage, or the end of a scan. Inside a
    user ``jax.jit``, traced values propagate instead because transformed
    pure functions cannot raise from data-dependent checks.
    Catch this in pseudo-marginal outer loops (e.g. PMMH) to reject
    the proposal that caused it.
    """
