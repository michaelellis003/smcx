# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Exceptions raised by smcx."""


class DegenerateWeightsError(ValueError):
    """All particle weights collapsed to zero (log-weights all -inf).

    Raised by the filter loop shell at an eval boundary (detection
    latency up to the eval lag; ADR-0003, design §6) — pure functions
    like ``log_normalize`` only *signal* degeneracy (``-inf``
    normalizer, NaN ESS). Catch this in pseudo-marginal outer loops
    (e.g. PMMH) to reject the proposal that caused it.
    """
