"""Shared fixtures for event-recycling tests.

The 10x10x10 Ni FCC supercell helper and the 3-vacancy / 4-vacancy fixtures
live here so future Recycling implementations (besides `DistanceRecycling`)
can reuse the same geometry.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.event_table import ActiveEventTable


# 3-vacancy scenario:
#   A at box center (17.6, 17.6, 17.6)
#   B at A + (8, 0, 0)   → ~8 Å from A   (must NOT be recycled)
#   C at A + (14, 14, 5) → ~20.4 Å from A (must be recycled)
#
# Box L = 35.2 Å limits single-axis PBC distance to L/2 = 17.6 Å, so "far"
# must use a diagonal placement.
_A_TARGET: tuple[float, float, float] = (17.6, 17.6, 17.6)
_B_TARGET: tuple[float, float, float] = (25.6, 17.6, 17.6)
_C_TARGET: tuple[float, float, float] = (31.6, 31.6, 22.6)


def make_active_table(rows: list[dict]) -> ActiveEventTable:
    """Build an ActiveEventTable directly from a list of row dicts."""
    table = pd.DataFrame(rows)
    return ActiveEventTable(Mock(), event_dataframe=table)


def row(atom_index: int) -> dict:
    """Build a stub ActiveEventTable row keyed by `atom_index`."""
    return {
        "atom_index": atom_index,
        "saddle_positions": np.zeros((1, 3)),
        "final_positions": np.zeros((1, 3)),
        "energy_barrier": 0.5,
        "k": 1.0,
        "num_reference_event": 0,
        "refined": "T",
    }


def make_ni_fcc_with_vacancies(
    vacancy_targets: list[tuple[float, float, float]],
) -> tuple[System, list[int], list[int]]:
    """Build a 10x10x10 Ni FCC supercell with vacancies at the given target positions.

    Each target is snapped to the closest FCC site (PBC-aware). For each removed
    atom, the closest surviving FCC neighbor is recorded as the "central atom"
    of a candidate event at that vacancy.

    Returns
    -------
    (system, vacancy_post_indices, central_atom_indices)
        - system : 4000 - len(vacancy_targets) atoms, box L = 35.2 Å, cubic.
        - vacancy_post_indices : the post-removal indices of the removed atoms
          (always -1 since they no longer exist; included for symmetry of API).
        - central_atom_indices : post-removal indices of one surviving neighbor
          per vacancy, suitable as the central_atom_index of a candidate event.

    """
    a = 3.52
    repeats = 10
    basis = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ]
        )
        * a
    )

    positions = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                shift = np.array([i, j, k]) * a
                for atom in basis:
                    positions.append(atom + shift)
    positions = np.array(positions)
    L = repeats * a

    # Snap each target to the closest FCC site (PBC-aware).
    removed_pre = []
    for target in vacancy_targets:
        diff = positions - np.asarray(target)
        diff -= L * np.round(diff / L)
        idx = int(np.argmin(np.linalg.norm(diff, axis=1)))
        removed_pre.append(idx)

    # Pick one nearest neighbor per vacancy (must not itself be a vacancy or already chosen).
    central_pre = []
    for vac_idx in removed_pre:
        diff = positions - positions[vac_idx]
        diff -= L * np.round(diff / L)
        order = np.argsort(np.linalg.norm(diff, axis=1))
        for j in order:
            j_int = int(j)
            if j_int == vac_idx or j_int in removed_pre or j_int in central_pre:
                continue
            central_pre.append(j_int)
            break

    # Remove vacancies and compute pre→post index map.
    keep_mask = np.ones(len(positions), dtype=bool)
    for idx in removed_pre:
        keep_mask[idx] = False
    survivor_positions = positions[keep_mask]
    pre_to_post = {int(pre): post for post, pre in enumerate(np.where(keep_mask)[0])}

    system = System()
    system.positions = survivor_positions
    system.types = ["Ni"] * len(survivor_positions)
    system.cell = np.diag([L, L, L])
    system.pbc = np.array([True, True, True])
    system.index = np.arange(len(survivor_positions))

    central_post = [pre_to_post[c] for c in central_pre]
    vacancy_post = [pre_to_post.get(r, -1) for r in removed_pre]
    return system, vacancy_post, central_post


@pytest.fixture
def ni_fcc_3vacancies() -> tuple[System, list[int]]:
    """Ni FCC supercell with vacancies at A (center), B (~8 Å), C (~20 Å)."""
    system, _vac, central = make_ni_fcc_with_vacancies(
        [_A_TARGET, _B_TARGET, _C_TARGET]
    )
    return system, central


@pytest.fixture
def ni_fcc_4vacancies() -> tuple[System, list[int]]:
    """3-vacancy scenario plus a 4th vacancy across the periodic wrap from A.

    D at A + (33, 0, 0) → wraps to A − (2.2, 0, 0) under PBC, so its
    minimum-image distance from A is ~2.2 Å. Confirms PBC awareness.
    """
    d_target = (_A_TARGET[0] + 33.0, _A_TARGET[1], _A_TARGET[2])
    system, _vac, central = make_ni_fcc_with_vacancies(
        [_A_TARGET, _B_TARGET, _C_TARGET, d_target]
    )
    return system, central
