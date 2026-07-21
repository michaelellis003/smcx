# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Host-side statistical gates for the issue #30 accuracy campaign."""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from benchmarks.tempering_accuracy.core import GaussianTarget

_REPLICATES = 32
_FLOORS = {"cpu_f64": 1e-10, "mps_f32": 5e-5}


class ReplicateEstimate(NamedTuple):
    """Host summaries retained from one committed inference key."""

    mean: np.ndarray
    covariance: np.ndarray
    evidence_ratio: float
    structural_passed: bool
    stages: int
    pair_evaluations: int


class CenteringGate(NamedTuple):
    """One registered scalar centering decision."""

    family: str
    index: int
    estimate: float
    oracle: float
    error: float
    standard_deviation: float
    estimator_se: float
    tolerance: float
    passed: bool


class AccuracyAnalysis(NamedTuple):
    """Hard-gate result for one mathematical campaign cell."""

    mean_gates: tuple[CenteringGate, ...]
    covariance_gates: tuple[CenteringGate, ...]
    evidence_gate: CenteringGate
    evidence_resolution_width: float
    status: str
    correctness_eligible: bool


def dct_directions(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the registered DCT-II frequencies and unit directions."""
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    count = min(16, dimension)
    frequencies = np.asarray([
        index * (dimension - 1) // (count - 1) for index in range(count)
    ])
    coordinates = np.arange(dimension, dtype=np.float64) + 0.5
    directions = np.cos(
        math.pi * frequencies[:, None] * coordinates[None, :] / dimension
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return frequencies, directions


def summarize_replicate(
    particles: np.ndarray,
    log_evidence: float,
    target: GaussianTarget,
    *,
    structural_passed: bool,
    stages: int,
    pair_evaluations: int,
) -> ReplicateEstimate:
    """Reduce one particle cloud to the registered float64 summaries."""
    values = np.asarray(particles, dtype=np.float64)
    expected = (target.dimension,)
    if values.ndim != 2 or values.shape[1:] != expected or values.shape[0] < 2:
        raise ValueError("particles must have shape (n>=2, target.dimension)")
    with np.errstate(over="ignore", invalid="ignore"):
        evidence_ratio = float(np.exp(log_evidence - target.log_evidence))
    return ReplicateEstimate(
        np.mean(values, axis=0),
        np.cov(values, rowvar=False, ddof=1),
        evidence_ratio,
        structural_passed,
        stages,
        pair_evaluations,
    )


def _centering_gate(
    family: str,
    index: int,
    values: np.ndarray,
    oracle: float,
    floor: float,
) -> CenteringGate:
    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
    estimator_se = standard_deviation / math.sqrt(_REPLICATES)
    tolerance = max(6 * estimator_se, floor)
    error = estimate - oracle
    finite = bool(np.all(np.isfinite(values)) and math.isfinite(oracle))
    passed = finite and bool(abs(error) <= np.nextafter(tolerance, math.inf))
    return CenteringGate(
        family,
        index,
        estimate,
        oracle,
        error,
        standard_deviation,
        estimator_se,
        tolerance,
        passed,
    )


def analyze_accuracy(
    estimates: Sequence[ReplicateEstimate],
    target: GaussianTarget,
    lane: str,
) -> AccuracyAnalysis:
    """Apply structural, finite, six-SE, and evidence-resolution gates."""
    if len(estimates) != _REPLICATES:
        raise ValueError("analysis requires exactly 32 replicates")
    if lane not in _FLOORS:
        raise ValueError(f"unknown lane: {lane}")
    floor = _FLOORS[lane]
    means = np.stack([estimate.mean for estimate in estimates])
    covariances = np.stack([estimate.covariance for estimate in estimates])
    evidence_ratios = np.asarray([
        estimate.evidence_ratio for estimate in estimates
    ])
    _, directions = dct_directions(target.dimension)
    projected = np.einsum("qi,rij,qj->rq", directions, covariances, directions)
    projected_oracle = np.einsum(
        "qi,ij,qj->q", directions, target.posterior_covariance, directions
    )
    mean_gates = tuple(
        _centering_gate("mean", index, means[:, index], oracle, floor)
        for index, oracle in enumerate(target.posterior_mean)
    )
    covariance_gates = tuple(
        _centering_gate(
            "projected_covariance", index, projected[:, index], oracle, floor
        )
        for index, oracle in enumerate(projected_oracle)
    )
    evidence_gate = _centering_gate(
        "evidence_ratio", 0, evidence_ratios, 1.0, floor
    )
    with np.errstate(over="ignore", invalid="ignore"):
        evidence_resolution_width = (
            6 * float(np.std(evidence_ratios, ddof=1)) / math.sqrt(_REPLICATES)
        )
    finite = bool(
        np.all(np.isfinite(means))
        and np.all(np.isfinite(covariances))
        and np.all(np.isfinite(evidence_ratios))
    )
    if not all(estimate.structural_passed for estimate in estimates):
        status = "failed_structural"
    elif not finite:
        status = "failed_nonfinite"
    elif evidence_resolution_width > 0.10:
        status = "indeterminate_evidence"
    elif not all(
        gate.passed for gate in (*mean_gates, *covariance_gates, evidence_gate)
    ):
        status = "failed_accuracy"
    else:
        status = "eligible"
    return AccuracyAnalysis(
        mean_gates,
        covariance_gates,
        evidence_gate,
        evidence_resolution_width,
        status,
        status == "eligible",
    )
