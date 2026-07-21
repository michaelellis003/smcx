# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen campaign-plan contracts for issue #30."""

import hashlib
import itertools

from benchmarks.tempering_accuracy.plan import (
    CampaignCell,
    cell_id,
    centering_summary_count,
    current_cells,
    current_smoke_cells,
    matched_cells,
    retained_particles,
    timing_blocks,
    waste_free_cells,
    waste_free_smoke_cells,
    work_count,
)


def test_current_plan_is_exact_registered_cartesian_product():
    cells = current_cells()
    observed = {
        (
            cell.geometry,
            cell.dimension,
            cell.reference_particles,
            cell.sweeps,
            cell.lane,
        )
        for cell in cells
    }
    expected = set(
        itertools.product(
            ("G0", "G1"),
            (4, 32, 128),
            (1_000, 10_000),
            (5, 20, 50),
            ("cpu_f64", "mps_f32"),
        )
    )

    assert len(cells) == len(set(cells)) == 72
    assert observed == expected
    assert {cell.arm for cell in cells} == {"current_systematic"}
    assert {cell.resampler for cell in cells} == {"systematic"}


def test_comparator_plans_cover_the_exact_challenge_cells():
    challenge = {(32, 1_000), (128, 1_000), (128, 10_000)}
    expected = set(
        itertools.product(("G0", "G1"), challenge, ("cpu_f64", "mps_f32"))
    )

    for cells, arm in (
        (matched_cells(), "matched_multinomial"),
        (waste_free_cells(), "waste_free_multinomial"),
    ):
        assert len(cells) == len(set(cells)) == 12
        assert {
            (
                cell.geometry,
                (cell.dimension, cell.reference_particles),
                cell.lane,
            )
            for cell in cells
        } == expected
        assert {cell.arm for cell in cells} == {arm}
        assert {cell.resampler for cell in cells} == {"multinomial"}
        assert {cell.sweeps for cell in cells} == {20}


def test_current_smoke_crosses_geometry_and_lane_only():
    cells = current_smoke_cells()

    assert len(cells) == 4
    assert {
        (
            cell.geometry,
            cell.dimension,
            cell.reference_particles,
            cell.sweeps,
            cell.lane,
        )
        for cell in cells
    } == {
        (geometry, 4, 1_000, 5, lane)
        for geometry in ("G0", "G1")
        for lane in ("cpu_f64", "mps_f32")
    }

    assert tuple(map(cell_id, waste_free_smoke_cells())) == (
        "waste_free_multinomial-g0-d4-n1000-cpu_f64-multinomial-s5-m100-p51",
        "waste_free_multinomial-g0-d4-n1000-mps_f32-multinomial-s5-m100-p51",
    )


def test_waste_free_plan_freezes_chain_sizes_and_matched_proposals():
    for cell in waste_free_cells():
        expected_length = 201 if cell.reference_particles == 1_000 else 2_001
        assert cell.chains == 100
        assert cell.chain_length == expected_length
        assert retained_particles(cell) == 100 * expected_length

        assert work_count(cell, 1).proposal_pairs == 20 * (
            cell.reference_particles
        )


def test_work_count_includes_initialization_and_realized_stages():
    standard = current_cells()[0]
    standard_count = work_count(standard, 3)
    assert standard_count.initial_pairs == 1_000
    assert standard_count.proposal_pairs == 15_000
    assert standard_count[2:5] == (16_000,) * 3
    assert standard_count.resampling_events == 3
    assert standard_count.ancestor_draws == 3_000
    assert standard_count.retained_states_per_stage == 1_000

    waste_free = waste_free_cells()[0]
    waste_free_count = work_count(waste_free, 3)
    assert waste_free_count.initial_pairs == 20_100
    assert waste_free_count.proposal_pairs == 60_000
    assert waste_free_count.total_pairs == 80_100
    assert waste_free_count.resampling_events == 3
    assert waste_free_count.ancestor_draws == 300
    assert waste_free_count.retained_states_per_stage == 20_100

    assert work_count(standard, 0).proposal_pairs == 0


def test_centering_gate_counts_match_preregistration_amendment():
    current = sum(centering_summary_count(cell) for cell in current_cells())
    matched = sum(centering_summary_count(cell) for cell in matched_cells())
    waste_free = sum(
        centering_summary_count(cell) for cell in waste_free_cells()
    )

    assert current == 4_872
    assert matched == waste_free == 1_356
    assert current + matched + waste_free == 7_584


def test_timing_blocks_freeze_balanced_order_and_worker_count():
    cells = current_cells()
    blocks = timing_blocks(cells)

    assert len(blocks) == 5
    assert all(set(block) == set(cells) for block in blocks)
    assert sum(map(len, blocks)) == 360
    assert all(
        len({block[index] for block in blocks}) == 5 for index in range(72)
    )
    payload = "\n".join(cell_id(cell) for block in blocks for cell in block)
    assert hashlib.sha256(payload.encode()).hexdigest() == (
        "286ea5901bf34291a2c0b30eda309e0b95253cd5c5d3e75b5efdfe2072d4f63b"
    )


def test_cell_ids_are_stable_and_unique_across_all_arms():
    cells: tuple[CampaignCell, ...] = (
        *current_cells(),
        *matched_cells(),
        *waste_free_cells(),
    )
    identifiers = tuple(map(cell_id, cells))

    assert len(identifiers) == len(set(identifiers)) == 96
    assert cell_id(waste_free_cells()[0]).endswith("-m100-p201")
