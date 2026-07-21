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
    assert (estimate.stages, estimate.pair_evaluations) == (7, 123)

    overflow = summarize_replicate(
        particles,
        target.log_evidence + 1_000,
        target,
        structural_passed=True,
        stages=1,
        pair_evaluations=1,
    )
    assert math.isinf(overflow.evidence_ratio)


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

    offsets = np.full((32, 4), 1e308)
    _, estimates = _estimates(mean_offsets=offsets)
    analysis = analyze_accuracy(estimates, target, "cpu_f64")
    assert analysis.status == "failed_nonfinite"
    assert not analysis.mean_gates[0].passed


def test_analysis_requires_exactly_32_replicates():
    target, estimates = _estimates()

    with pytest.raises(ValueError, match="exactly 32"):
        analyze_accuracy(estimates[:-1], target, "cpu_f64")


def test_loss_summaries_freeze_uncertainty_and_efficiency_formulas():
    target = build_target("G0", 4, np.float64)
    signed = np.linspace(-1e-4, 1e-4, 32)
    offsets = np.zeros((32, 4))
    offsets[:, 0] = signed * math.sqrt(target.posterior_covariance[0, 0])
    covariances = np.asarray([
        (1 + value) * target.posterior_covariance for value in signed
    ])
    target, estimates = _estimates(
        mean_offsets=offsets,
        covariances=covariances,
        evidence_ratios=1 + signed,
    )

    analysis = analyze_accuracy(
        estimates,
        target,
        "cpu_f64",
        steady_block_median_seconds=(0.5, 0.1, 0.3, 0.2, 0.4),
    )

    expected_losses = {
        "mean": signed**2 / target.dimension,
        "covariance": signed**2,
        "evidence": signed**2,
    }
    for summary in (
        analysis.mean_loss,
        analysis.covariance_loss,
        analysis.evidence_loss,
    ):
        losses = expected_losses[summary.family]
        mse = float(np.mean(losses))
        mse_se = float(np.std(losses, ddof=1) / math.sqrt(32))
        rmse = math.sqrt(mse)
        np.testing.assert_allclose(summary.replicate_losses, losses)
        assert summary.mse == pytest.approx(mse)
        assert summary.rmse == pytest.approx(rmse)
        assert summary.mse_standard_error == pytest.approx(mse_se)
        assert summary.rmse_standard_error == pytest.approx(mse_se / (2 * rmse))
        assert summary.mse_interval_low == pytest.approx(
            max(0.0, mse - 2.0395134464 * mse_se)
        )
        assert summary.mse_interval_high == pytest.approx(
            mse + 2.0395134464 * mse_se
        )
        assert summary.median_steady_seconds == pytest.approx(0.3)
        assert summary.median_pair_evaluations == pytest.approx(1_015.5)
        assert summary.fixed_key_time_normalized_loss == pytest.approx(
            mse * 0.3
        )
        assert summary.evaluation_normalized_loss == pytest.approx(
            mse * 1_015.5
        )


def test_zero_losses_have_zero_uncertainty_and_normalized_losses():
    target, estimates = _estimates()
    analysis = analyze_accuracy(
        estimates,
        target,
        "cpu_f64",
        steady_block_median_seconds=(0.1, 0.2, 0.3, 0.4, 0.5),
    )

    for summary in (
        analysis.mean_loss,
        analysis.covariance_loss,
        analysis.evidence_loss,
    ):
        assert summary.mse == 0
        assert summary.rmse == 0
        assert summary.mse_standard_error == 0
        assert summary.rmse_standard_error == 0
        assert summary.mse_interval_low == 0
        assert summary.mse_interval_high == 0
        assert summary.fixed_key_time_normalized_loss == 0
        assert summary.evaluation_normalized_loss == 0


def test_loss_interval_is_clipped_below_at_zero():
    target = build_target("G0", 4, np.float64)
    offsets = np.zeros((32, 4))
    offsets[0, 0] = math.sqrt(4e-4 * target.posterior_covariance[0, 0])
    target, estimates = _estimates(mean_offsets=offsets)

    summary = analyze_accuracy(estimates, target, "cpu_f64").mean_loss

    assert summary.mse_interval_low == 0
    assert summary.mse_interval_high > summary.mse


def test_derived_loss_overflow_is_failed_nonfinite():
    target, estimates = _estimates(evidence_ratios=np.full(32, 1e200))

    analysis = analyze_accuracy(estimates, target, "cpu_f64")

    assert math.isinf(analysis.evidence_loss.mse)
    assert analysis.status == "failed_nonfinite"
    assert not analysis.correctness_eligible


def test_efficiency_overflow_does_not_change_correctness_status():
    target = build_target("G0", 4, np.float64)
    signs = np.tile([-1.0, 1.0], 16)
    offsets = signs[:, None] * 2 * np.sqrt(np.diag(target.posterior_covariance))
    target, estimates = _estimates(mean_offsets=offsets)
    untimed = analyze_accuracy(estimates, target, "cpu_f64")

    timed = analyze_accuracy(
        estimates,
        target,
        "cpu_f64",
        steady_block_median_seconds=(1e308,) * 5,
    )

    assert untimed.status == "eligible"
    assert timed.status == untimed.status
    normalized = timed.mean_loss.fixed_key_time_normalized_loss
    assert normalized is not None
    assert math.isinf(normalized)


@pytest.mark.parametrize(
    "seconds",
    (
        (0.1, 0.2, 0.3, 0.4),
        (0.1, 0.2, 0.3, 0.4, np.inf),
        (0.1, 0.2, 0.0, 0.4, 0.5),
    ),
)
def test_timing_normalizer_requires_five_finite_positive_block_medians(
    seconds,
):
    target, estimates = _estimates()

    with pytest.raises(ValueError, match="steady_block_median_seconds"):
        analyze_accuracy(
            estimates,
            target,
            "cpu_f64",
            steady_block_median_seconds=seconds,
        )


@pytest.mark.parametrize("pair_evaluations", (0, -1, 1.5, True))
def test_evaluation_normalizer_requires_positive_integral_pair_counts(
    pair_evaluations,
):
    target, estimates = _estimates()
    malformed = estimates[0]._replace(pair_evaluations=pair_evaluations)

    with pytest.raises(ValueError, match="pair_evaluations"):
        analyze_accuracy((malformed, *estimates[1:]), target, "cpu_f64")
