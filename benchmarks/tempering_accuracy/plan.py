# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic campaign plan and work accounting for issue #30."""

import itertools
import math
import random
from collections.abc import Sequence
from typing import NamedTuple

ORDER_SEED = 20_260_719
TIMING_BLOCK_COUNT = 5

_GEOMETRIES = ("G0", "G1")
_DIMENSIONS = (4, 32, 128)
_PARTICLE_COUNTS = (1_000, 10_000)
_SWEEP_COUNTS = (5, 20, 50)
_LANES = ("cpu_f64", "mps_f32")
_CHALLENGE_CELLS = ((32, 1_000), (128, 1_000), (128, 10_000))


class CampaignCell(NamedTuple):
    """One mathematical/backend cell in the registered campaign."""

    arm: str
    geometry: str
    dimension: int
    reference_particles: int
    lane: str
    resampler: str
    sweeps: int
    chains: int | None
    chain_length: int | None


class WorkCount(NamedTuple):
    """Logical target work and resampling work for one realized run."""

    initial_pairs: int
    proposal_pairs: int
    total_pairs: int
    prior_components: int
    likelihood_components: int
    resampling_events: int
    ancestor_draws: int
    retained_states_per_stage: int


def _standard_cell(
    arm: str,
    geometry: str,
    dimension: int,
    particles: int,
    lane: str,
    resampler: str,
    sweeps: int,
) -> CampaignCell:
    return CampaignCell(
        arm,
        geometry,
        dimension,
        particles,
        lane,
        resampler,
        sweeps,
        None,
        None,
    )


def current_cells() -> tuple[CampaignCell, ...]:
    """Return all 72 mandatory systematic-RWM cells."""
    return tuple(
        _standard_cell(
            "current_systematic",
            geometry,
            dimension,
            particles,
            lane,
            "systematic",
            sweeps,
        )
        for geometry, dimension, particles, sweeps, lane in itertools.product(
            _GEOMETRIES,
            _DIMENSIONS,
            _PARTICLE_COUNTS,
            _SWEEP_COUNTS,
            _LANES,
        )
    )


def matched_cells() -> tuple[CampaignCell, ...]:
    """Return the 12 standard-RWM multinomial challenge cells."""
    return tuple(
        _standard_cell(
            "matched_multinomial",
            geometry,
            dimension,
            particles,
            lane,
            "multinomial",
            20,
        )
        for geometry, (dimension, particles), lane in itertools.product(
            _GEOMETRIES, _CHALLENGE_CELLS, _LANES
        )
    )


def waste_free_cells() -> tuple[CampaignCell, ...]:
    """Return the 12 preregistered waste-free challenge cells."""
    cells = []
    for geometry, (dimension, particles), lane in itertools.product(
        _GEOMETRIES, _CHALLENGE_CELLS, _LANES
    ):
        chains = 100
        chain_length = 1 + 20 * particles // chains
        cells.append(
            CampaignCell(
                "waste_free_multinomial",
                geometry,
                dimension,
                particles,
                lane,
                "multinomial",
                20,
                chains,
                chain_length,
            )
        )
    return tuple(cells)


def waste_free_smoke_cells() -> tuple[CampaignCell, ...]:
    """Return the two lane executions of the waste-free smoke target."""
    return tuple(
        cell._replace(dimension=4, sweeps=5, chain_length=51)
        for cell in waste_free_cells()[:2]
    )


def current_smoke_cells() -> tuple[CampaignCell, ...]:
    """Return the four registered current-RWM structural smoke cells."""
    return tuple(
        cell
        for cell in current_cells()
        if cell.dimension == 4
        and cell.reference_particles == 1_000
        and cell.sweeps == 5
    )


def retained_particles(cell: CampaignCell) -> int:
    """Return the equal-weight cloud size retained after one stage."""
    if cell.arm != "waste_free_multinomial":
        return cell.reference_particles
    if cell.chains is None or cell.chain_length is None:
        raise ValueError("waste-free cells require chains and chain_length")
    return cell.chains * cell.chain_length


def work_count(cell: CampaignCell, stages: int) -> WorkCount:
    """Count logical component pairs for one realized stage count."""
    if stages < 0:
        raise ValueError("stages must be nonnegative")
    initial = retained_particles(cell)
    if cell.arm == "waste_free_multinomial":
        assert cell.chains is not None and cell.chain_length is not None
        proposals_per_stage = cell.chains * (cell.chain_length - 1)
        ancestors_per_stage = cell.chains
    else:
        proposals_per_stage = cell.reference_particles * cell.sweeps
        ancestors_per_stage = cell.reference_particles
    proposals = stages * proposals_per_stage
    total = initial + proposals
    return WorkCount(
        initial,
        proposals,
        total,
        total,
        total,
        stages,
        stages * ancestors_per_stage,
        retained_particles(cell),
    )


def centering_summary_count(cell: CampaignCell) -> int:
    """Return mean, projected-covariance, and evidence gate count."""
    return cell.dimension + min(16, cell.dimension) + 1


def timing_blocks(
    cells: Sequence[CampaignCell],
) -> tuple[tuple[CampaignCell, ...], ...]:
    """Return five seeded Latin rotations of a complete cell sequence."""
    first = list(cells)
    if not first:
        return ()
    random.Random(ORDER_SEED).shuffle(first)
    if len(first) == 1:
        return tuple((first[0],) for _ in range(TIMING_BLOCK_COUNT))
    step = math.ceil(len(first) / TIMING_BLOCK_COUNT)
    while math.gcd(step, len(first)) != 1:
        step += 1
    return tuple(
        tuple(first[offset:] + first[:offset])
        for offset in (
            (block * step) % len(first) for block in range(TIMING_BLOCK_COUNT)
        )
    )


def cell_id(cell: CampaignCell) -> str:
    """Return a stable filename-safe identifier for one campaign cell."""
    identifier = (
        f"{cell.arm}-{cell.geometry.lower()}-d{cell.dimension}"
        f"-n{cell.reference_particles}-{cell.lane}-{cell.resampler}"
        f"-s{cell.sweeps}"
    )
    if cell.chains is not None and cell.chain_length is not None:
        identifier += f"-m{cell.chains}-p{cell.chain_length}"
    return identifier
