# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic public figures for tempering-accuracy evidence."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

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


def _gate_ratio(accuracy: Mapping[str, Any]) -> float | None:
    gates = (
        *accuracy["mean_gates"],
        *accuracy["covariance_gates"],
        accuracy["evidence_gate"],
    )
    if accuracy["evidence_resolution_width"] is None or any(
        gate["error"] is None or gate["tolerance"] is None for gate in gates
    ):
        return None
    ratios = [
        abs(float(gate["error"])) / float(gate["tolerance"]) for gate in gates
    ]
    ratios.append(float(accuracy["evidence_resolution_width"]) / 0.10)
    return max(ratios)


def _cost(
    accuracy: Mapping[str, Any], work: object
) -> tuple[float, float, float, float]:
    summaries = cast(Mapping[str, Any], work)
    losses = [float(accuracy[f"{name}_loss"]["rmse"]) for name in _LOSSES]
    return (
        float(summaries["total_pairs"]["median"]),
        losses[0],
        losses[1],
        losses[2],
    )


def _plot_cells(evidence: Mapping[str, Any]) -> tuple[_PlotCell, ...]:
    rows = []
    cells = cast(Sequence[Mapping[str, Any]], evidence["cells"])
    for item in cells[:72]:
        cell = cast(Mapping[str, Any], item["cell"])
        accuracy = item["accuracy"]
        if accuracy is None:
            rows.append(_PlotCell(cell, None, None))
            continue
        accuracy = cast(Mapping[str, Any], accuracy)
        eligible = (
            item["status"] == "eligible"
            and accuracy["correctness_eligible"] is True
        )
        ratio = _gate_ratio(accuracy)
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
                                else 1 + int(plot_cell.cost is None)
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
    sweep_alphas = {5: 0.45, 20: 0.7, 50: 1.0}
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
                        alpha=sweep_alphas[row.cell["sweeps"]],
                    )
                axis.set_xscale("log")
                axis.set_yscale("log")
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
            "Current systematic RWM: eligible accuracy against logical work"
        )
        figure.text(
            0.5,
            0.055,
            "Blue: G0; orange: G1. Marker: d=4 ○, d=32 □, d=128 △; "
            "size: N; opacity: 5/20/50 sweeps. No timing ratio is used.",
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
    rows = _plot_cells(evidence)
    gate_count = sum(row.gate_ratio is not None for row in rows)
    cost_count = sum(row.cost is not None for row in rows)
    summary = PlotSummary(
        gate_count, len(rows) - gate_count, cost_count, len(rows) - cost_count
    )
    _gate_figure(rows, gate_path)
    _cost_figure(rows, cost_path)
    return summary
