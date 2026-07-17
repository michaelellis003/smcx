# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Sequential Monte Carlo and particle filtering in JAX."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from smcx.auxiliary import auxiliary_filter
from smcx.bootstrap import bootstrap_filter
from smcx.containers import (
    LiuWestPosterior,
    ParticleFilterPosterior,
    ParticleFilterResult,
    ParticleState,
)
from smcx.diagnostics import (
    crps,
    cumulative_log_score,
    diagnose,
    log_bayes_factor,
    log_ml_increments,
    param_weighted_mean,
    param_weighted_quantile,
    pareto_k_diagnostic,
    particle_diversity,
    posterior_predictive_sample,
    replicated_log_ml,
    tail_ess,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)
from smcx.exceptions import DegenerateWeightsError
from smcx.guided import guided_filter
from smcx.liu_west import liu_west_filter
from smcx.resampling import (
    multinomial,
    residual,
    stratified,
    systematic,
)
from smcx.simulate import simulate
from smcx.weights import ess, log_ess, log_normalize, normalize

try:
    __version__ = _version("smcx")
except _PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "DegenerateWeightsError",
    "LiuWestPosterior",
    "ParticleFilterPosterior",
    "ParticleFilterResult",
    "ParticleState",
    "__version__",
    "auxiliary_filter",
    "bootstrap_filter",
    "crps",
    "cumulative_log_score",
    "diagnose",
    "ess",
    "guided_filter",
    "liu_west_filter",
    "log_bayes_factor",
    "log_ess",
    "log_ml_increments",
    "log_normalize",
    "multinomial",
    "normalize",
    "param_weighted_mean",
    "param_weighted_quantile",
    "pareto_k_diagnostic",
    "particle_diversity",
    "posterior_predictive_sample",
    "replicated_log_ml",
    "residual",
    "simulate",
    "stratified",
    "systematic",
    "tail_ess",
    "weighted_mean",
    "weighted_quantile",
    "weighted_variance",
]
