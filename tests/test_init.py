# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
"""Tests for the package-level API."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from smcjax import __version__


def test_version_is_accessible():
    """Test that __version__ is a non-empty string."""
    assert isinstance(__version__, str)
    assert __version__ != ''


def test_public_api_exports_all_expected_names(package):
    """Test that __all__ contains exactly the expected public API."""
    expected = [
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
    assert sorted(package.__all__) == sorted(expected)


def test_version_fallback_when_package_not_found():
    """Test that __version__ falls back to '0.0.0' when not installed."""
    import importlib

    import smcjax

    with patch(
        'importlib.metadata.version',
        side_effect=PackageNotFoundError,
    ):
        importlib.reload(smcjax)
        assert smcjax.__version__ == '0.0.0'

    # Restore the real version
    importlib.reload(smcjax)
