"""Scenario tests for `DistanceRecycling` on a 10x10x10 Ni FCC (~4000 atoms).

Three vacancies are placed so that two events are close together (A & B, ~8 Å
apart) and a third is far (C, ~20 Å on a diagonal). Executing the event at
vacancy A should:
  - recycle the event at C (didn't move, far away), and
  - discard the event at B (didn't move, but close enough that its neighborhood
    could have been perturbed).
"""

from __future__ import annotations

import numpy as np

from pykmc.event_recycling import DistanceRecycling

from .conftest import make_active_table, row


def _recycler(
    movement_thr: float = 0.02, distance_thr: float = 10.0
) -> DistanceRecycling:
    return DistanceRecycling(movement_thr=movement_thr, distance_thr=distance_thr)


def test_close_discarded_far_recycled(ni_fcc_3vacancies) -> None:
    """8-Å vacancy must NOT be recycled; 20-Å vacancy MUST be recycled."""
    system, central = ni_fcc_3vacancies
    assert len(system.positions) == 4000 - 3
    positions_pre = system.positions.copy()
    # Simulate execution at vacancy A: shift its central atom by 0.3 Å.
    system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
    active = make_active_table([row(c) for c in central])
    recycled = _recycler().select_recyclable(
        active,
        executed_idx=0,
        system=system,
        positions_pre=positions_pre,
    )
    assert len(recycled) == 1
    assert int(recycled.iloc[0]["atom_index"]) == central[2]


def test_distance_threshold_boundary(ni_fcc_3vacancies) -> None:
    """Widen distance_thr past C's ~20 Å → C is no longer 'far' → 0 recycled."""
    system, central = ni_fcc_3vacancies
    positions_pre = system.positions.copy()
    system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
    active = make_active_table([row(c) for c in central])
    recycled = _recycler(distance_thr=25.0).select_recyclable(
        active,
        executed_idx=0,
        system=system,
        positions_pre=positions_pre,
    )
    assert len(recycled) == 0


def test_movement_check_overrides_distance(ni_fcc_3vacancies) -> None:
    """Move C's central atom by 0.05 Å (> movement_thr) → not recycled despite being far."""
    system, central = ni_fcc_3vacancies
    positions_pre = system.positions.copy()
    system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
    system.positions[central[2]] = positions_pre[central[2]] + np.array(
        [0.05, 0.0, 0.0]
    )
    active = make_active_table([row(c) for c in central])
    recycled = _recycler().select_recyclable(
        active,
        executed_idx=0,
        system=system,
        positions_pre=positions_pre,
    )
    assert len(recycled) == 0


def test_self_excluded(ni_fcc_3vacancies) -> None:
    """The executed-event row never appears in the recycled output."""
    system, central = ni_fcc_3vacancies
    positions_pre = system.positions.copy()
    system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
    active = make_active_table([row(c) for c in central])
    recycled = _recycler().select_recyclable(
        active,
        executed_idx=0,
        system=system,
        positions_pre=positions_pre,
    )
    assert central[0] not in [int(a) for a in recycled["atom_index"].tolist()]


def test_pbc_wrap_close(ni_fcc_4vacancies) -> None:
    """A 4th vacancy across the periodic wrap must be identified as CLOSE.

    With L=35.2, D at A + (33,0,0) wraps to A − (2.2,0,0) → PBC distance ≈ 2.2 Å.
    Only the 20-Å vacancy C should be recycled.
    """
    system, central = ni_fcc_4vacancies
    positions_pre = system.positions.copy()
    system.positions[central[0]] = positions_pre[central[0]] + np.array([0.3, 0.0, 0.0])
    active = make_active_table([row(c) for c in central])
    recycled = _recycler().select_recyclable(
        active,
        executed_idx=0,
        system=system,
        positions_pre=positions_pre,
    )
    recycled_idx = [int(a) for a in recycled["atom_index"].tolist()]
    assert recycled_idx == [central[2]]
