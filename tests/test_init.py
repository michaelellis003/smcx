# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Tests for the package-level API."""

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from smcx import __version__


def test_version_is_accessible():
    """Test that __version__ is a non-empty string."""
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_public_api_exports_all_expected_names(package):
    """Test that __all__ contains exactly the expected public API."""
    expected = [
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
        "guided_filter",
        "kalman_filter",
        "crps",
        "cumulative_log_score",
        "diagnose",
        "ess",
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
        "reconstruct_trajectories",
        "particle_diversity",
        "posterior_predictive_sample",
        "replicated_log_ml",
        "residual",
        "simulate",
        "smc2",
        "stratified",
        "systematic",
        "temper",
        "tail_ess",
        "to_arviz",
        "weighted_mean",
        "weighted_quantile",
        "weighted_variance",
    ]
    assert sorted(package.__all__) == sorted(expected)


def test_version_fallback_when_package_not_found():
    """Test that __version__ falls back to '0.0.0' when not installed."""
    import importlib

    import smcx

    with patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError,
    ):
        importlib.reload(smcx)
        assert smcx.__version__ == "0.0.0"

    # Restore the real version
    importlib.reload(smcx)
