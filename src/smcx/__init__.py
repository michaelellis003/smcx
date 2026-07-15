# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""smcx.

Sequential Monte Carlo for Apple silicon, built on MLX
"""

from smcx.auxiliary import auxiliary_filter
from smcx.bootstrap import bootstrap_filter
from smcx.containers import (
    ParticleFilterPosterior,
    ParticleFilterResult,
    ParticleState,
)
from smcx.exceptions import DegenerateWeightsError
from smcx.resampling import multinomial, residual, stratified, systematic
from smcx.simulate import simulate
from smcx.weights import ess, log_ess, log_normalize, normalize

__version__ = "0.1.0"

__all__ = [
    "DegenerateWeightsError",
    "ParticleFilterPosterior",
    "ParticleFilterResult",
    "ParticleState",
    "__version__",
    "auxiliary_filter",
    "bootstrap_filter",
    "ess",
    "log_ess",
    "log_normalize",
    "multinomial",
    "normalize",
    "residual",
    "simulate",
    "stratified",
    "systematic",
]
