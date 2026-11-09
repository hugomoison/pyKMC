"""``remove_duplicates`` must compare a carried row on the atoms it was refined on.

A row's ``saddle_positions`` follow the rcut neighbor order of the step it was
refined in (its ``neighbors`` column). The list is rebuilt every step, so once
every atom has moved a little (active volume off) a carried row holds the same
ids in another order. Read through the current order its coordinates land on
the wrong atoms, the duplicate check never matches, and the fresh copy of the
same hop stays in the table. The rows are built by hand on a vacancy in FCC Ni.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc import System
from pykmc.event_table import ActiveEventTable
from pykmc.neighbors_list import NeighborsList

from .conftest import make_ni_fcc_with_vacancies

_VACANCY = np.array([17.6, 17.6, 17.6])  # an FCC site of the 10x10x10 cell
_HALF = 1.76  # a / 2
_RNEI, _RCUT = 3.0, 6.5

# system, neighbors list, centers c1 and c2, hopping atoms m, m2, m3
Cell = tuple[System, NeighborsList, int, int, int, int, int]


def _atom_at(system: System, position: np.ndarray) -> int:
    diff = system.positions - np.asarray(position)
    diff -= np.diag(system.cell) * np.round(diff / np.diag(system.cell))
    return int(np.argmin(np.linalg.norm(diff, axis=1)))


@pytest.fixture(scope="module")
def vacancy_cell() -> Cell:
    """FCC Ni with one vacancy, its neighbors list and five of its first neighbors."""
    system, _vac, _central = make_ni_fcc_with_vacancies([tuple(_VACANCY)])
    nl = NeighborsList(system, _RNEI, _RCUT)
    # first neighbors of the vacancy: two central atoms, three hopping atoms
    c1 = _atom_at(system, _VACANCY + [-_HALF, _HALF, 0])
    c2 = _atom_at(system, _VACANCY + [0, _HALF, _HALF])
    m = _atom_at(system, _VACANCY + [_HALF, _HALF, 0])
    m2 = _atom_at(system, _VACANCY + [-_HALF, 0, _HALF])
    m3 = _atom_at(system, _VACANCY + [_HALF, 0, -_HALF])
    return system, nl, c1, c2, m, m2, m3


def _row(
    system: System, center: int, ids: np.ndarray, mover: int, ref: int = 0
) -> dict:
    """Row for `mover` hopping halfway to the vacancy, stored in the order `ids`."""
    ids = np.asarray(ids, dtype=int)
    saddle = system.positions[ids].copy()
    saddle[ids == mover] = 0.5 * (system.positions[mover] + _VACANCY)
    return {
        "atom_index": center,
        "saddle_positions": saddle,
        "final_positions": saddle.copy(),
        "neighbors": ids,
        "energy_barrier": 1.1878,
        "k": 1.0,
        "num_reference_event": ref,
        "refined": "T",
    }


def _table(rows: list[dict]) -> ActiveEventTable:
    return ActiveEventTable(
        SimpleNamespace(psr=SimpleNamespace(matching_score_thr=0.4)),
        event_dataframe=pd.DataFrame(rows),
    )


def _old_order(nl: NeighborsList, atom: int) -> np.ndarray:
    """Return the atom's current rcut neighbors in another order (an earlier step's)."""
    ids = np.asarray(nl.get_neighbors("rcut", atom), dtype=int)
    return np.random.default_rng(20261109).permutation(ids)


def _symmetric_rows(vacancy_cell: Cell, carried_order: np.ndarray) -> list[dict]:
    """Two carried rows on c1, then fresh rows on c2: one copies hop m, one is new."""
    system, nl, c1, c2, m, m2, m3 = vacancy_cell
    fresh = nl.get_neighbors("rcut", c2)
    return [
        _row(system, c1, carried_order, m),
        _row(system, c1, carried_order, m2),
        _row(system, c2, fresh, m),  # same hop as row 0, found from c2
        _row(system, c2, fresh, m3),
    ]


@pytest.mark.parametrize("reordered", [True, False], ids=["old-order", "same-order"])
def test_symmetric_copy_of_carried_row_is_removed(
    vacancy_cell: Cell, reordered: bool
) -> None:
    """Part 2: the fresh copy of a carried hop on another central atom is a duplicate.

    With the order unchanged (active volume on) this already worked.
    """
    system, nl, c1, c2, m, m2, m3 = vacancy_cell
    carried = _old_order(nl, c1) if reordered else nl.get_neighbors("rcut", c1)
    table = _table(_symmetric_rows(vacancy_cell, carried))

    table.remove_duplicates(system.cell, nl)

    # the carried copy (earlier in the table) is kept, the fresh c2 copy goes
    assert table.table["atom_index"].tolist() == [c1, c1, c2]
    kept_c2 = table.table.iloc[2]
    expected = _row(system, c2, kept_c2["neighbors"], m3)["saddle_positions"]
    assert np.allclose(kept_c2["saddle_positions"], expected)


def test_same_atom_copy_of_carried_row_is_removed(vacancy_cell: Cell) -> None:
    """Part 1: same central atom, carried row in an old order, fresh row in the current one."""
    system, nl, c1, c2, m, m2, m3 = vacancy_cell
    table = _table(
        [
            _row(system, c1, _old_order(nl, c1), m, ref=0),
            _row(system, c1, nl.get_neighbors("rcut", c1), m, ref=1),
        ]
    )

    table.remove_duplicates(system.cell, nl)

    assert len(table.table) == 1


def test_distinct_hops_on_carried_and_fresh_rows_are_kept(vacancy_cell: Cell) -> None:
    """Aligning must not merge different hops."""
    system, nl, c1, c2, m, m2, m3 = vacancy_cell
    table = _table(
        [
            _row(system, c1, _old_order(nl, c1), m, ref=0),
            _row(system, c1, _old_order(nl, c1), m2, ref=0),
            _row(system, c1, nl.get_neighbors("rcut", c1), m3, ref=1),
            _row(system, c2, nl.get_neighbors("rcut", c2), m3, ref=0),
        ]
    )

    table.remove_duplicates(system.cell, nl)

    # same atom (part 1) and same ref (part 2) pairs are all distinct hops; the
    # c2/ref 0 copy of m3 is not compared with the c1/ref 1 one (other ref)
    assert len(table.table) == 4
