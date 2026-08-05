# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""State-space inference with Gaussian and Monte Carlo methods."""

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
    DGLMFilterPosterior,
    DGLMForecast,
    DGLMSmootherPosterior,
    DLMFilterPosterior,
    DLMForecast,
    DLMSmootherPosterior,
    GaussianFilterPosterior,
    GaussianForecast,
    GaussianSmootherPosterior,
    IBISPosterior,
    LiuWestPosterior,
    ParameterFilterResult,
    ParticleFilterPosterior,
    ParticleFilterRecord,
    ParticleFilterResult,
    ParticleSmootherPosterior,
    ParticleState,
    SMC2Posterior,
    TemperedPosterior,
)
from smcx.dglm import (
    DGLMFamily,
    bernoulli,
    binomial,
    dglm_filter,
    dglm_forecast,
    dglm_smoother,
    poisson,
)
from smcx.diagnostics import (
    crps,
    cumulative_log_score,
    diagnose,
    log_bayes_factor,
    log_ml_increments,
    log_ml_variance,
    param_posterior_predictive_sample,
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
from smcx.dlm import dlm_filter, dlm_forecast, dlm_smoother
from smcx.exceptions import DegenerateWeightsError
from smcx.fk import CallbackNames, FeynmanKac, run_smc
from smcx.guided import guided_filter
from smcx.ibis import ibis
from smcx.kalman import (
    extended_kalman_filter,
    gaussian_filter,
    gaussian_smoother,
    kalman_filter,
    kalman_forecast,
    posterior_sample,
    rts_smoother,
    smoothed_cross_covariances,
    taylor_order1,
    unscented,
    unscented_kalman_filter,
)
from smcx.liu_west import liu_west_filter
from smcx.model import (
    LinearGaussianModel,
    StateSpaceModel,
    auxiliary_fk,
    bootstrap_fk,
    guided_fk,
)
from smcx.reporting import to_arviz
from smcx.resampling import (
    multinomial,
    residual,
    stratified,
    systematic,
)
from smcx.runner import run_particle_filter
from smcx.simulate import simulate
from smcx.smc2 import smc2
from smcx.smoothing import backward_simulation
from smcx.tempering import temper
from smcx.types import StaticMutation
from smcx.weights import ess, log_ess, log_normalize, normalize

try:
    __version__ = _version("smcx")
except _PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BootstrapCheckpoint",
    "BootstrapStepInfo",
    "CallbackNames",
    "DGLMFamily",
    "DGLMFilterPosterior",
    "DGLMForecast",
    "DGLMSmootherPosterior",
    "DLMFilterPosterior",
    "DLMForecast",
    "DLMSmootherPosterior",
    "DegenerateWeightsError",
    "FeynmanKac",
    "GaussianFilterPosterior",
    "GaussianForecast",
    "GaussianSmootherPosterior",
    "IBISPosterior",
    "LinearGaussianModel",
    "LiuWestPosterior",
    "ParameterFilterResult",
    "ParticleFilterPosterior",
    "ParticleFilterRecord",
    "ParticleFilterResult",
    "ParticleSmootherPosterior",
    "ParticleState",
    "SMC2Posterior",
    "StateSpaceModel",
    "StaticMutation",
    "TemperedPosterior",
    "__version__",
    "auxiliary_filter",
    "auxiliary_fk",
    "backward_simulation",
    "bernoulli",
    "binomial",
    "bootstrap_filter",
    "bootstrap_fk",
    "bootstrap_init",
    "bootstrap_step",
    "bootstrap_update",
    "crps",
    "cumulative_log_score",
    "dglm_filter",
    "dglm_forecast",
    "dglm_smoother",
    "diagnose",
    "dlm_filter",
    "dlm_forecast",
    "dlm_smoother",
    "ess",
    "extended_kalman_filter",
    "gaussian_filter",
    "gaussian_smoother",
    "guided_filter",
    "guided_fk",
    "ibis",
    "kalman_filter",
    "kalman_forecast",
    "liu_west_filter",
    "log_bayes_factor",
    "log_ess",
    "log_ml_increments",
    "log_ml_variance",
    "log_normalize",
    "multinomial",
    "normalize",
    "param_posterior_predictive_sample",
    "param_weighted_mean",
    "param_weighted_quantile",
    "pareto_k_diagnostic",
    "particle_diversity",
    "poisson",
    "posterior_predictive_sample",
    "posterior_sample",
    "reconstruct_trajectories",
    "replicated_log_ml",
    "residual",
    "rts_smoother",
    "run_particle_filter",
    "run_smc",
    "simulate",
    "smc2",
    "smoothed_cross_covariances",
    "stratified",
    "systematic",
    "tail_ess",
    "taylor_order1",
    "temper",
    "to_arviz",
    "unscented",
    "unscented_kalman_filter",
    "weighted_mean",
    "weighted_quantile",
    "weighted_variance",
]
