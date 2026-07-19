# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Integrity checks for promoted external numerical references."""

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import tomllib

from tests._kalman import kalman_1d
from tests._lgssm_reference import (
    DATA_SHA256,
    EMISSIONS,
    EXACT_LOG_LIKELIHOOD,
    FILTERED_MEANS,
    FILTERED_VARIANCES,
    REFERENCE_TIMES,
    STATES,
)

_VALIDATION_PACKAGES = {
    "blackjax",
    "dynamax",
    "nimblesmc",
    "particles",
    "tensorflow_probability",
}

# Exact preimage used by the one-time input-aware filter comparison. Keeping
# the reviewed case bytes here makes the reported SHA and exact target
# reconstructible without retaining any comparator adapter or dependency.
_FILTER_CAMPAIGN_CASE = b"""{
  "case_id": "scalar_lgssm_inputs_v1",
  "generation": {
    "generator": "numpy.random.default_rng",
    "numpy_seed": 20260718
  },
  "model": {
    "initial_mean": -0.2,
    "initial_variance": 0.81,
    "transition_coefficient": 0.72,
    "transition_input_coefficient": 0.4,
    "transition_variance": 0.12249999999999998,
    "emission_coefficient": 1.15,
    "emission_input_coefficient": -0.25,
    "emission_variance": 0.30250000000000005
  },
  "inputs": [
    0.14999999999999997, 0.6371769798568134, 0.8809944365682281,
    0.8476737687789735, 1.1133677577570993, 1.067403745101067,
    0.7137496588129473, 0.6705042077136512, 0.3674799147848894,
    -0.15779427963076692, -0.26511027022784267, -0.5169766353138886,
    -0.8836433363803227, -0.7470757985689723, -0.7033411328637246,
    -0.7630841969033664, -0.35002974160294215, -0.09771406148922522,
    -0.044889197368562866, 0.3697776022687522, 0.5129910856099306,
    0.36138561933438423, 0.5047357554603695, 0.34738462617973914,
    -0.09230369690121709, -0.18439424421701536, -0.4914024172834511,
    -0.9733425915466924, -0.9931210476440747, -1.1215435520177763
  ],
  "emissions": [
    1.6148513272032408, 0.544570639891619, -0.29712293597845063,
    0.605840301887772, 1.6299896578353026, 0.005289045037596529,
    1.3939902512045688, 1.3161532178020445, 1.854560506882229,
    1.3371024588089853, 0.37094024725988756, -0.288454511615057,
    0.7598077941694681, -0.6533416889119478, -0.15413657878391662,
    -1.2511031630229048, 0.04678416885119474, -1.0932343197184284,
    -0.2875910637292114, 1.3683668799096294, -0.3726921564537323,
    0.10631582049182943, 0.7800785933703176, -0.6436336160757017,
    -0.3958602196636053, 0.3950730311050927, 0.6159561135311202,
    -0.015654723351526012, 0.2673726359092178, -0.21716668985420096
  ]
}
"""
_FILTER_CAMPAIGN_SHA256 = (
    "d59064d711ba96f3d61da207c79b8f1b4526eade25620dd40ec10f6ed47d2689"
)
_FILTER_CAMPAIGN_EXACT_LOGZ = -34.14084035418424
_FILTER_CAMPAIGN_FINAL_MEAN = -0.5786147401295777
_FILTER_CAMPAIGN_FINAL_VARIANCE = 0.0987191304917186


def _campaign_package(name: str) -> bool:
    """Return whether an import/dependency name is a campaign validator."""
    root = re.split(r"[.\s\[<>=!~;]", name, maxsplit=1)[0]
    root = root.lower().replace("-", "_")
    return root in _VALIDATION_PACKAGES


def test_lgssm_reference_hash_is_stable():
    """Frozen observations and states retain their reviewed byte content."""
    payload = STATES.astype("<f8").tobytes() + EMISSIONS.astype("<f8").tobytes()
    assert hashlib.sha256(payload).hexdigest() == DATA_SHA256


def test_dynamax_reference_matches_independent_kalman_recurrence():
    """A local equation-derived oracle reproduces the frozen outside values."""
    log_likelihood, means, variances = kalman_1d(
        EMISSIONS[:, 0], 0.9, 0.25, 1.0, 0.0, 1.0
    )
    # Dynamax adds a 1e-9 covariance jitter, explaining the few-e-9
    # difference from the direct float64 recurrence.
    np.testing.assert_allclose(log_likelihood, EXACT_LOG_LIKELIHOOD, atol=3e-9)
    np.testing.assert_allclose(
        means[REFERENCE_TIMES], FILTERED_MEANS, atol=5e-10
    )
    np.testing.assert_allclose(
        variances[REFERENCE_TIMES], FILTERED_VARIANCES, atol=6e-10
    )


def test_filter_campaign_case_and_exact_target_are_reconstructible():
    """The promoted input-aware case reproduces its hash and Kalman target.

    The deleted adapters ran this case in float64 on CPU with T=30,
    N=4096, and R=128. smcx and particles used systematic resampling at
    ESS/N < .5 and seeds 20261000--20261127; TFP used its 0.25.0 defaults
    and seeds 20261200--20261327. Guided proposals were the transition
    prior, the exact Gaussian conditional, and a Gaussian with mean
    ``prior + .25 * (adjusted_y / h - prior)`` and variance ``.18``.
    """
    assert hashlib.sha256(_FILTER_CAMPAIGN_CASE).hexdigest() == (
        _FILTER_CAMPAIGN_SHA256
    )
    case = json.loads(_FILTER_CAMPAIGN_CASE)
    model = case["model"]
    inputs = np.asarray(case["inputs"], dtype=np.float64)
    emissions = np.asarray(case["emissions"], dtype=np.float64)

    mean = float(model["initial_mean"])
    variance = float(model["initial_variance"])
    logz = 0.0
    for time, emission in enumerate(emissions):
        if time:
            mean = (
                model["transition_coefficient"] * mean
                + model["transition_input_coefficient"] * inputs[time]
            )
            variance = (
                model["transition_coefficient"] ** 2 * variance
                + model["transition_variance"]
            )
        predicted_emission = (
            model["emission_coefficient"] * mean
            + model["emission_input_coefficient"] * inputs[time]
        )
        innovation_variance = (
            model["emission_coefficient"] ** 2 * variance
            + model["emission_variance"]
        )
        residual = emission - predicted_emission
        logz -= 0.5 * (
            math.log(2.0 * math.pi * innovation_variance)
            + residual**2 / innovation_variance
        )
        gain = variance * model["emission_coefficient"] / innovation_variance
        mean += gain * residual
        variance -= gain * model["emission_coefficient"] * variance

    assert logz == pytest.approx(_FILTER_CAMPAIGN_EXACT_LOGZ, abs=1e-13)
    assert mean == pytest.approx(_FILTER_CAMPAIGN_FINAL_MEAN, abs=1e-14)
    assert variance == pytest.approx(_FILTER_CAMPAIGN_FINAL_VARIANCE, abs=1e-14)


def test_unit_tests_do_not_import_external_smc_validators():
    """Outside SMC implementations stay out of the permanent test graph."""
    tests_dir = Path(__file__).parent
    violations = []
    for path in tests_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if _campaign_package(name):
                    line = getattr(node, "lineno", 0)
                    violations.append(f"{path.name}:{line}: {name}")
            if not isinstance(node, ast.Call) or not node.args:
                continue
            # Catch common dynamic import forms as well as static imports.
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            first = node.args[0]
            if (
                func_name in {"__import__", "import_module", "importorskip"}
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and _campaign_package(first.value)
            ):
                violations.append(f"{path.name}:{node.lineno}: {first.value}")
    assert not violations, violations


def test_dev_group_excludes_external_smc_validators():
    """Ordinary test installs do not resolve one-time comparators."""
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    dev = data["dependency-groups"]["dev"]
    violations = [item for item in dev if _campaign_package(item)]
    assert not violations, violations
