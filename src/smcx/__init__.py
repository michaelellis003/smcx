# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Sequential Monte Carlo and particle filtering in JAX."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from smcx.auxiliary import auxiliary_filter
from smcx.bootstrap import (
    bootstrap_filter,
    bootstrap_init,
    bootstrap_step,
    bootstrap_update,
)
from smcx.containers import (
    BootstrapCheckpoint,
    BootstrapStepInfo,
    GaussianFilterPosterior,
    LiuWestPosterior,
    ParticleFilterPosterior,
    ParticleFilterResult,
    ParticleState,
    SMC2Posterior,
    TemperedPosterior,
)
from smcx.diagnostics import (
    crps,
    cumulative_log_score,
    diagnose,
    log_bayes_factor,
    log_ml_increments,
    log_ml_variance,
    param_weighted_mean,
    param_weighted_quantile,
    pareto_k_diagnostic,
    particle_diversity,
    posterior_predictive_sample,
    reconstruct_trajectories,
    replicated_log_ml,
    tail_ess,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)
from smcx.exceptions import DegenerateWeightsError
from smcx.guided import guided_filter
from smcx.kalman import kalman_filter
from smcx.liu_west import liu_west_filter
from smcx.reporting import to_arviz
from smcx.resampling import (
    multinomial,
    residual,
    stratified,
    systematic,
)
from smcx.simulate import simulate
from smcx.smc2 import smc2
from smcx.tempering import temper
from smcx.weights import ess, log_ess, log_normalize, normalize

try:
    __version__ = _version("smcx")
except _PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BootstrapCheckpoint",
    "BootstrapStepInfo",
    "DegenerateWeightsError",
    "GaussianFilterPosterior",
    "LiuWestPosterior",
    "ParticleFilterPosterior",
    "ParticleFilterResult",
    "ParticleState",
    "SMC2Posterior",
    "TemperedPosterior",
    "__version__",
    "auxiliary_filter",
    "bootstrap_filter",
    "bootstrap_init",
    "bootstrap_step",
    "bootstrap_update",
    "crps",
    "cumulative_log_score",
    "diagnose",
    "ess",
    "guided_filter",
    "kalman_filter",
    "liu_west_filter",
    "log_bayes_factor",
    "log_ess",
    "log_ml_increments",
    "log_ml_variance",
    "log_normalize",
    "multinomial",
    "normalize",
    "param_weighted_mean",
    "param_weighted_quantile",
    "pareto_k_diagnostic",
    "particle_diversity",
    "posterior_predictive_sample",
    "reconstruct_trajectories",
    "replicated_log_ml",
    "residual",
    "simulate",
    "smc2",
    "stratified",
    "systematic",
    "tail_ess",
    "temper",
    "to_arviz",
    "weighted_mean",
    "weighted_quantile",
    "weighted_variance",
]
