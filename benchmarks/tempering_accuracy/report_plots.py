# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic public figures for tempering-accuracy evidence."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from benchmarks.tempering_accuracy.report_markdown import _validate


def _fields(names: str) -> set[str]:
    return set(names.split())


_GATE = _fields(
    "family index estimate oracle error standard_deviation estimator_se "
    "tolerance passed"
)
_LOSS = _fields(
    "family replicate_losses mse rmse mse_standard_error "
    "rmse_standard_error mse_interval_low mse_interval_high "
    "median_steady_seconds median_pair_evaluations "
    "fixed_key_time_normalized_loss evaluation_normalized_loss"
)
_WORK = _fields(
    "stages mean_acceptance min_ess_fraction total_pairs resampling_events "
    "ancestor_draws"
)
_SUMMARY = _fields("values median q1 q3 iqr minimum maximum")
_LANES = ("cpu_f64", "mps_f32")
_DIMENSIONS = (4, 32, 128)
_SWEEPS = (5, 20, 50)
_LOSSES = ("mean", "covariance", "evidence")


class PlotSummary(NamedTuple):
    """Counts behind the two figures."""

    evaluated_gate_cells: int
    unavailable_gate_cells: int
    eligible_cost_cells: int
    unavailable_cost_cells: int


class _PlotCell(NamedTuple):
    cell: Mapping[str, Any]
    gate_ratio: float | None
    cost: tuple[float, float, float, float] | None


def _bad(detail: str) -> ValueError:
    return ValueError(f"plot evidence {detail}")


def _map(value: object, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _bad("has an invalid mapping schema")
    return cast(Mapping[str, Any], value)


def _seq(value: object) -> Sequence[Any]:
    if not isinstance(value, list | tuple):
        raise _bad("has an invalid sequence")
    return value


def _number(value: Any, *, positive: bool = False) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not math.isfinite(value):
        raise _bad("has an invalid number")
    result = float(value)
    if positive and result <= 0:
        raise _bad("has a non-positive number")
    return result


def _gate_ratio(accuracy: Mapping[str, Any], dimension: int) -> float | None:
    mean = _seq(accuracy["mean_gates"])
    covariance = _seq(accuracy["covariance_gates"])
    expected_covariance = tuple(
        index * (dimension - 1) // (min(16, dimension) - 1)
        for index in range(min(16, dimension))
    )
    expected = (
        *(("mean", index) for index in range(dimension)),
        *(("projected_covariance", index) for index in expected_covariance),
        ("evidence_ratio", 0),
    )
    gates = (*mean, *covariance, accuracy["evidence_gate"])
    if len(gates) != len(expected):
        raise _bad("has the wrong registered gates")
    ratios = []
    for value, (family, index) in zip(gates, expected, strict=True):
        gate = _map(value, _GATE)
        if (
            gate["family"] != family
            or gate["index"] != index
            or type(gate["index"]) is not int
            or type(gate["passed"]) is not bool
        ):
            raise _bad("has an unregistered gate")
        error = _number(gate["error"])
        tolerance = _number(gate["tolerance"], positive=True)
        if error is None or tolerance is None:
            return None
        ratios.append(abs(error) / tolerance)
    resolution = _number(accuracy["evidence_resolution_width"])
    if resolution is None:
        return None
    if resolution < 0:
        raise _bad("has a negative resolution width")
    ratios.append(resolution / 0.10)
    return max(ratios)


def _cost(
    accuracy: Mapping[str, Any], work: object
) -> tuple[float, float, float, float]:
    if (
        not isinstance(work, Mapping)
        or not {"total_pairs"} <= set(work) <= _WORK
    ):
        raise _bad("has invalid work evidence")
    summaries = cast(Mapping[str, Any], work)
    pairs = _map(summaries["total_pairs"], _SUMMARY)
    values = _seq(pairs["values"])
    median_pairs = _number(pairs["median"], positive=True)
    if not values or median_pairs is None:
        raise _bad("has invalid work evidence")
    losses = []
    for name in _LOSSES:
        loss = _map(accuracy[f"{name}_loss"], _LOSS)
        rmse = _number(loss["rmse"])
        if loss["family"] != name or rmse is None or rmse < 0:
            raise _bad("has invalid loss evidence")
        losses.append(rmse)
    return median_pairs, losses[0], losses[1], losses[2]


def _plot_cells(evidence: Mapping[str, Any]) -> tuple[_PlotCell, ...]:
    rows = []
    for item in _validate(evidence)[:72]:
        cell = cast(Mapping[str, Any], item["cell"])
        accuracy = item["accuracy"]
        if accuracy is None:
            rows.append(_PlotCell(cell, None, None))
            continue
        accuracy = cast(Mapping[str, Any], accuracy)
        eligible = item["status"] == "eligible"
        if accuracy["correctness_eligible"] is not eligible:
            raise _bad("has an inconsistent eligibility decision")
        ratio = _gate_ratio(accuracy, cell["dimension"])
        cost = _cost(accuracy, item["work"]) if eligible else None
        rows.append(_PlotCell(cell, ratio, cost))
    return tuple(rows)


def _save(figure: Figure, path: Path) -> None:
    figure.savefig(
        path,
        format="png",
        dpi=144,
        facecolor="white",
        metadata={"Software": "smcx"},
    )


def _gate_figure(rows: Sequence[_PlotCell], path: Path) -> None:
    figure = Figure(figsize=(9.2, 6.8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, squeeze=False)
    lookup = {
        (
            row.cell["lane"],
            row.cell["geometry"],
            row.cell["dimension"],
            row.cell["reference_particles"],
            row.cell["sweeps"],
        ): row
        for row in rows
    }
    colors = ("#e5e7eb", "#dbeafe", "#fed7aa")
    try:
        for lane_index, lane in enumerate(_LANES):
            for geometry_index, geometry in enumerate(("G0", "G1")):
                axis = axes[lane_index][geometry_index]
                states = []
                labels = []
                for dimension in _DIMENSIONS:
                    for particles in (1_000, 10_000):
                        row_states, row_labels = [], []
                        for sweeps in _SWEEPS:
                            plot_cell = lookup[
                                lane, geometry, dimension, particles, sweeps
                            ]
                            ratio = plot_cell.gate_ratio
                            row_states.append(
                                0
                                if ratio is None
                                else 1
                                + int(plot_cell.cost is None or ratio > 1)
                            )
                            row_labels.append(
                                "—" if ratio is None else f"{ratio:.2g}x"
                            )
                        states.append(row_states)
                        labels.append(row_labels)
                axis.imshow(
                    states,
                    cmap=ListedColormap(colors),
                    vmin=0,
                    vmax=2,
                    aspect="auto",
                )
                for row_index, labels_row in enumerate(labels):
                    for column, label in enumerate(labels_row):
                        axis.text(
                            column,
                            row_index,
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
                axis.set_title(
                    f"{geometry} · "
                    f"{'CPU float64' if lane == 'cpu_f64' else 'MPS float32'}"
                )
                axis.set_xticks(range(3), _SWEEPS)
                axis.set_yticks(
                    range(6),
                    [
                        f"d={dimension}, N={particles // 1000}k"
                        for dimension in _DIMENSIONS
                        for particles in (1_000, 10_000)
                    ],
                )
                axis.tick_params(labelleft=geometry_index == 0)
                axis.set_xlabel("RWM sweeps")
        figure.suptitle("Current systematic RWM: worst gate / threshold")
        figure.text(
            0.5,
            0.035,
            "Blue: passes ≤1x · orange: ineligible · grey/—: unavailable",
            ha="center",
        )
        figure.subplots_adjust(
            left=0.14, right=0.98, top=0.90, bottom=0.12, hspace=0.40
        )
        _save(figure, path)
    finally:
        figure.clear()


def _cost_figure(rows: Sequence[_PlotCell], path: Path) -> None:
    figure = Figure(figsize=(10.2, 6.4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, squeeze=False)
    geometry_colors = {"G0": "#2563eb", "G1": "#ea580c"}
    markers = {4: "o", 32: "s", 128: "^"}
    try:
        for lane_index, lane in enumerate(_LANES):
            lane_rows = [row for row in rows if row.cell["lane"] == lane]
            eligible = [row for row in lane_rows if row.cost is not None]
            for loss_index, loss in enumerate(_LOSSES):
                axis = axes[lane_index][loss_index]
                for row in eligible:
                    assert row.cost is not None
                    particles = row.cell["reference_particles"]
                    axis.scatter(
                        row.cost[0],
                        row.cost[loss_index + 1],
                        s=24 if particles == 1_000 else 58,
                        marker=markers[row.cell["dimension"]],
                        color=geometry_colors[row.cell["geometry"]],
                    )
                axis.set_xscale("log")
                axis.set_yscale("symlog", linthresh=1e-6)
                axis.set_title(loss.capitalize())
                axis.set_xlabel("Target-pair evaluations")
                axis.set_ylabel(
                    ("CPU float64" if lane == "cpu_f64" else "MPS float32")
                    + "\nRMSE"
                )
                axis.text(
                    0.98,
                    0.97,
                    f"eligible {len(eligible)}/{len(lane_rows)}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                )
        figure.suptitle(
            "Eligible accuracy against logical work (execution lanes separate)"
        )
        figure.text(
            0.5,
            0.055,
            "Blue: G0; orange: G1. Marker: d=4 ○, d=32 □, d=128 △; "
            "size: N. No timing ratio is used.",
            ha="center",
        )
        figure.tight_layout(rect=(0, 0.10, 1, 0.93))
        _save(figure, path)
    finally:
        figure.clear()


def render_plots(
    evidence: Mapping[str, Any], gate_path: Path, cost_path: Path
) -> PlotSummary:
    """Write deterministic gate and within-lane cost PNG figures."""
    paths = (gate_path, cost_path)
    if any(
        not isinstance(path, Path) or path.suffix.lower() != ".png"
        for path in paths
    ):
        raise ValueError("plot destinations must be explicit PNG paths")
    if gate_path.absolute() == cost_path.absolute():
        raise ValueError("plot destinations must be distinct")
    rows = _plot_cells(evidence)
    gate_count = sum(row.gate_ratio is not None for row in rows)
    cost_count = sum(row.cost is not None for row in rows)
    summary = PlotSummary(
        gate_count, len(rows) - gate_count, cost_count, len(rows) - cost_count
    )
    _gate_figure(rows, gate_path)
    _cost_figure(rows, cost_path)
    return summary
