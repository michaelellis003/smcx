# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Sequential Monte Carlo and particle filtering in JAX."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from blackjax.smc.ess import ess, log_ess
from blackjax.smc.resampling import (
    multinomial,
    residual,
    stratified,
    systematic,
)

from smcjax.auxiliary import auxiliary_filter
from smcjax.bootstrap import bootstrap_filter
from smcjax.containers import (
    LiuWestPosterior,
    ParticleFilterPosterior,
    ParticleFilterResult,
    ParticleState,
)
from smcjax.diagnostics import (
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
from smcjax.liu_west import liu_west_filter
from smcjax.simulate import simulate
from smcjax.weights import log_normalize, normalize

try:
    __version__ = _version('smcjax')
except _PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'LiuWestPosterior',
    'ParticleFilterPosterior',
    'ParticleFilterResult',
    'ParticleState',
    '__version__',
    'auxiliary_filter',
    'bootstrap_filter',
    'crps',
    'cumulative_log_score',
    'diagnose',
    'ess',
    'liu_west_filter',
    'log_bayes_factor',
    'log_ess',
    'log_ml_increments',
    'log_normalize',
    'multinomial',
    'normalize',
    'param_weighted_mean',
    'param_weighted_quantile',
    'pareto_k_diagnostic',
    'particle_diversity',
    'posterior_predictive_sample',
    'replicated_log_ml',
    'residual',
    'simulate',
    'stratified',
    'systematic',
    'tail_ess',
    'weighted_mean',
    'weighted_quantile',
    'weighted_variance',
]
