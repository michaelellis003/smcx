# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Host-side statistical gates for the issue #30 accuracy campaign."""

import math
from collections.abc import Sequence
from numbers import Integral
from typing import NamedTuple

import numpy as np

from benchmarks.tempering_accuracy.core import GaussianTarget

_REPLICATES = 32
_FLOORS = {"cpu_f64": 1e-10, "mps_f32": 5e-5}
_T_31_975 = 2.0395134464


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


class LossSummary(NamedTuple):
    """One accuracy loss with separate uncertainty and work normalizers."""

    family: str
    replicate_losses: tuple[float, ...]
    mse: float
    rmse: float
    mse_standard_error: float
    rmse_standard_error: float
    mse_interval_low: float
    mse_interval_high: float
    median_steady_seconds: float | None
    median_pair_evaluations: float
    fixed_key_time_normalized_loss: float | None
    evaluation_normalized_loss: float


class AccuracyAnalysis(NamedTuple):
    """Hard-gate result for one mathematical campaign cell."""

    mean_gates: tuple[CenteringGate, ...]
    covariance_gates: tuple[CenteringGate, ...]
    evidence_gate: CenteringGate
    evidence_resolution_width: float
    mean_loss: LossSummary
    covariance_loss: LossSummary
    evidence_loss: LossSummary
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
    finite = bool(
        np.all(np.isfinite(values))
        and math.isfinite(oracle)
        and math.isfinite(estimate)
        and math.isfinite(error)
        and math.isfinite(standard_deviation)
        and math.isfinite(estimator_se)
        and math.isfinite(tolerance)
    )
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


def _loss_summary(
    family: str,
    losses: np.ndarray,
    pair_evaluations: np.ndarray,
    median_steady_seconds: float | None,
) -> LossSummary:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mse = float(np.mean(losses))
        rmse = float(np.sqrt(mse))
        mse_standard_error = float(
            np.std(losses, ddof=1) / math.sqrt(_REPLICATES)
        )
        if rmse == 0:
            mse_standard_error = 0.0
            rmse_standard_error = 0.0
        else:
            rmse_standard_error = mse_standard_error / (2 * rmse)
        interval_radius = _T_31_975 * mse_standard_error
        interval_low = max(0.0, mse - interval_radius)
        interval_high = mse + interval_radius
        median_pairs = float(np.median(pair_evaluations))
        time_normalized = (
            None
            if median_steady_seconds is None
            else mse * median_steady_seconds
        )
        evaluation_normalized = mse * median_pairs
    return LossSummary(
        family,
        tuple(float(value) for value in losses),
        mse,
        rmse,
        mse_standard_error,
        rmse_standard_error,
        interval_low,
        interval_high,
        median_steady_seconds,
        median_pairs,
        time_normalized,
        evaluation_normalized,
    )


def _accuracy_loss_is_finite(summary: LossSummary) -> bool:
    scalars = (
        summary.mse,
        summary.rmse,
        summary.mse_standard_error,
        summary.rmse_standard_error,
        summary.mse_interval_low,
        summary.mse_interval_high,
    )
    return bool(
        np.all(np.isfinite(summary.replicate_losses))
        and all(math.isfinite(value) for value in scalars)
    )


def analyze_accuracy(
    estimates: Sequence[ReplicateEstimate],
    target: GaussianTarget,
    lane: str,
    *,
    steady_block_median_seconds: Sequence[float] | None = None,
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
    pair_counts = tuple(estimate.pair_evaluations for estimate in estimates)
    if any(
        isinstance(count, bool) or not isinstance(count, Integral) or count <= 0
        for count in pair_counts
    ):
        raise ValueError("pair_evaluations must be positive integers")
    pair_evaluations = np.asarray(pair_counts, dtype=np.float64)
    if steady_block_median_seconds is None:
        median_steady_seconds = None
    else:
        steady = np.asarray(steady_block_median_seconds, dtype=np.float64)
        if steady.shape != (5,) or not np.all(np.isfinite(steady)):
            raise ValueError(
                "steady_block_median_seconds must contain five finite values"
            )
        if np.any(steady <= 0):
            raise ValueError("steady_block_median_seconds must be positive")
        median_steady_seconds = float(np.median(steady))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean_losses = np.mean(
            (means - target.posterior_mean) ** 2
            / np.diag(target.posterior_covariance),
            axis=1,
        )
        covariance_delta = covariances - target.posterior_covariance
        covariance_losses = np.sum(covariance_delta**2, axis=(1, 2)) / np.sum(
            target.posterior_covariance**2
        )
        evidence_losses = (evidence_ratios - 1) ** 2
    mean_loss = _loss_summary(
        "mean", mean_losses, pair_evaluations, median_steady_seconds
    )
    covariance_loss = _loss_summary(
        "covariance", covariance_losses, pair_evaluations, median_steady_seconds
    )
    evidence_loss = _loss_summary(
        "evidence", evidence_losses, pair_evaluations, median_steady_seconds
    )
    frequencies, directions = dct_directions(target.dimension)
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
            "projected_covariance",
            int(frequency),
            projected[:, ordinal],
            oracle,
            floor,
        )
        for ordinal, (frequency, oracle) in enumerate(
            zip(frequencies, projected_oracle, strict=True)
        )
    )
    evidence_gate = _centering_gate(
        "evidence_ratio", 0, evidence_ratios, 1.0, floor
    )
    with np.errstate(over="ignore", invalid="ignore"):
        evidence_resolution_width = (
            6 * float(np.std(evidence_ratios, ddof=1)) / math.sqrt(_REPLICATES)
        )
    gates = (*mean_gates, *covariance_gates, evidence_gate)
    losses = (mean_loss, covariance_loss, evidence_loss)
    finite = bool(
        np.all(np.isfinite(means))
        and np.all(np.isfinite(covariances))
        and np.all(np.isfinite(evidence_ratios))
        and math.isfinite(evidence_resolution_width)
        and all(np.all(np.isfinite(gate[2:8])) for gate in gates)
        and all(_accuracy_loss_is_finite(loss) for loss in losses)
    )
    if not all(estimate.structural_passed for estimate in estimates):
        status = "failed_structural"
    elif not finite:
        status = "failed_nonfinite"
    elif evidence_resolution_width > 0.10:
        status = "indeterminate_evidence"
    elif not all(gate.passed for gate in gates):
        status = "failed_accuracy"
    else:
        status = "eligible"
    return AccuracyAnalysis(
        mean_gates,
        covariance_gates,
        evidence_gate,
        evidence_resolution_width,
        mean_loss,
        covariance_loss,
        evidence_loss,
        status,
        status == "eligible",
    )
