# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen statistical-analysis contracts for issue #30."""

import math

import numpy as np
import pytest

from benchmarks.tempering_accuracy.analysis import (
    ReplicateEstimate,
    analyze_accuracy,
    dct_directions,
    summarize_replicate,
)
from benchmarks.tempering_accuracy.core import build_target


def _estimates(
    *,
    mean_offsets=None,
    covariances=None,
    evidence_ratios=None,
    structural=None,
):
    target = build_target("G0", 4, np.float64)
    zeros = np.zeros((32, 4)) if mean_offsets is None else mean_offsets
    covs = (
        np.repeat(target.posterior_covariance[None, :, :], 32, axis=0)
        if covariances is None
        else covariances
    )
    ratios = np.ones(32) if evidence_ratios is None else evidence_ratios
    structural_flags = [True] * 32 if structural is None else structural
    return target, tuple(
        ReplicateEstimate(
            target.posterior_mean + zeros[index],
            covs[index],
            float(ratios[index]),
            structural_flags[index],
            3,
            1_000 + index,
        )
        for index in range(32)
    )


@pytest.mark.parametrize(
    ("dimension", "expected"),
    (
        (4, (0, 1, 2, 3)),
        (32, tuple((*range(0, 30, 2), 31))),
        (
            128,
            (0, 8, 16, 25, 33, 42, 50, 59, 67, 76, 84, 93, 101, 110, 118, 127),
        ),
    ),
)
def test_dct_directions_freeze_frequencies_and_orthonormality(
    dimension, expected
):
    frequencies, directions = dct_directions(dimension)

    assert tuple(frequencies) == expected
    np.testing.assert_allclose(
        directions @ directions.T,
        np.eye(min(16, dimension)),
        atol=2e-14,
    )


def test_replicate_summary_uses_ddof_one_and_evidence_ratio():
    target = build_target("G0", 4, np.float64)
    particles = np.array([
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0, 7.0],
    ])
    estimate = summarize_replicate(
        particles,
        target.log_evidence + math.log(1.25),
        target,
        structural_passed=True,
        stages=7,
        pair_evaluations=123,
    )

    np.testing.assert_allclose(estimate.mean, [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(estimate.covariance, np.full((4, 4), 4.0))
    assert estimate.evidence_ratio == pytest.approx(1.25)
    assert estimate.stages == 7
    assert estimate.pair_evaluations == 123


def test_exact_oracle_estimates_are_eligible_at_lane_floor():
    target, estimates = _estimates()
    analysis = analyze_accuracy(estimates, target, "cpu_f64")

    assert analysis.status == "eligible"
    assert analysis.correctness_eligible
    assert all(gate.passed for gate in analysis.mean_gates)
    assert all(gate.passed for gate in analysis.covariance_gates)
    assert analysis.evidence_gate.passed
    assert analysis.evidence_resolution_width == pytest.approx(0.0)

    target_32 = build_target("G0", 32, np.float64)
    exact = ReplicateEstimate(
        target_32.posterior_mean,
        target_32.posterior_covariance,
        1.0,
        True,
        1,
        1,
    )
    gates = analyze_accuracy((exact,) * 32, target_32, "cpu_f64")
    assert tuple(gate.index for gate in gates.covariance_gates) == tuple(
        dct_directions(32)[0]
    )


@pytest.mark.parametrize(
    ("lane", "offset"), (("cpu_f64", 2e-10), ("mps_f32", 6e-5))
)
def test_zero_variance_bias_above_lane_floor_fails(lane, offset):
    offsets = np.zeros((32, 4))
    offsets[:, 0] = offset
    target, estimates = _estimates(mean_offsets=offsets)
    analysis = analyze_accuracy(estimates, target, lane)

    assert analysis.status == "failed_accuracy"
    assert not analysis.mean_gates[0].passed


def test_six_se_boundary_passes_and_larger_centering_error_fails():
    values = np.arange(32, dtype=np.float64)
    values = (values - values.mean()) / values.std(ddof=1)
    boundary = 6 / math.sqrt(32)
    offsets = np.zeros((32, 4))
    offsets[:, 0] = values + boundary
    target, estimates = _estimates(mean_offsets=offsets)
    gate = analyze_accuracy(estimates, target, "cpu_f64").mean_gates[0]

    assert gate.tolerance == pytest.approx(boundary)
    assert gate.passed

    offsets[:, 0] += boundary * 1e-6
    _, outside = _estimates(mean_offsets=offsets)
    assert not analyze_accuracy(outside, target, "cpu_f64").mean_gates[0].passed


def test_centered_high_variance_evidence_is_indeterminate():
    ratios = np.tile([0.5, 1.5], 16)
    target, estimates = _estimates(evidence_ratios=ratios)
    analysis = analyze_accuracy(estimates, target, "cpu_f64")

    assert analysis.evidence_gate.passed
    assert analysis.evidence_resolution_width > 0.10
    assert analysis.status == "indeterminate_evidence"
    assert not analysis.correctness_eligible


def test_nonfinite_and_structural_failures_are_retained_with_precedence():
    ratios = np.ones(32)
    ratios[0] = np.inf
    target, estimates = _estimates(evidence_ratios=ratios)
    assert analyze_accuracy(estimates, target, "cpu_f64").status == (
        "failed_nonfinite"
    )

    structural = [True] * 32
    structural[0] = False
    _, estimates = _estimates(evidence_ratios=ratios, structural=structural)
    assert analyze_accuracy(estimates, target, "cpu_f64").status == (
        "failed_structural"
    )


def test_analysis_requires_exactly_32_replicates():
    target, estimates = _estimates()

    with pytest.raises(ValueError, match="exactly 32"):
        analyze_accuracy(estimates[:-1], target, "cpu_f64")
