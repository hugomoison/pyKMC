"""Tests that active-table dedup aligns rows by their stored neighbour ids.

``ActiveEventTable.remove_duplicates`` compares two rows' ``saddle_positions``
to decide whether they encode the same physical event. Each row's positions are
ordered positionally by the row's own ``neighbors`` integer-id array (row ``k``
belongs to absolute atom ``neighbors[k]``). Recycled rows keep the neighbour
ordering captured at event time while fresh rows are built from the current
:class:`NeighborsList`, so two rows describing the same event can carry the same
atoms in a **different order** (and, after the system moved, even a different
atom **set**).

Pre-fix, part 1 (same central atom) compared the two arrays element-wise --
crashing outright on different-sized environments -- and part 2 (symmetric
events, different central atoms) re-derived the alignment map from a fresh
``get_neighbors('rcut', ...)`` -- both assuming an ordering that does not hold
for recycled rows. This suite locks in the alignment-by-stored-id behaviour:
permuted-but-identical events now dedup, distinct events that would alias under
the wrong ordering stay distinct, identical-ordering rows behave exactly as
before, and None/length-mismatched neighbours keep both rows.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc.event_table import ActiveEventTable

_CELL = np.diag([20.0, 20.0, 20.0])


def _config() -> Mock:
    cfg = Mock()
    cfg.rateconstant.style = "constant"
    cfg.rateconstant.T = 300.0
    cfg.rateconstant.k0 = 10.0
    cfg.psr.matching_score_thr = 0.1
    return cfg


def _row(
    atom_index: int,
    saddle_positions: np.ndarray,
    neighbors: "np.ndarray | None",
    energy_barrier: float = 0.5,
    num_reference_event: int = 0,
) -> dict:
    """Build one active-event row dict for a test DataFrame."""
    return {
        "atom_index": atom_index,
        "saddle_positions": np.asarray(saddle_positions, dtype=float),
        "final_positions": np.asarray(saddle_positions, dtype=float),
        "neighbors": (None if neighbors is None else np.asarray(neighbors, dtype=int)),
        "energy_barrier": energy_barrier,
        "k": 1.0,
        "num_reference_event": num_reference_event,
        "refined": "T",
    }


def _table(rows: list[dict]) -> ActiveEventTable:
    return ActiveEventTable(_config(), event_dataframe=pd.DataFrame(rows))


# A canonical 4-atom saddle geometry, ordered by atoms [0, 1, 2, 3].
_SADDLE = np.array(
    [
        [10.0, 10.0, 10.0],
        [12.5, 10.0, 10.0],
        [10.0, 12.0, 10.0],
        [10.0, 10.0, 12.0],
    ]
)


def test_size_mismatched_environments_no_crash_both_kept() -> None:
    """A recycled row whose stored environment size differs must not crash.

    This is the loud recycling failure mode: a recycled row keeps the
    environment captured at event time while a fresh row on the same central
    atom reflects the current neighbour list, and after membership drift the
    two ``saddle_positions`` arrays have different lengths. Pre-fix, part 1
    fed both straight into ``compute_delr``, which raises a numpy broadcast
    ``ValueError``. Differing neighbour sets on the same central atom mean a
    different local environment, so both rows must simply be kept.
    """
    table = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            _row(atom_index=1, saddle_positions=_SADDLE[:3], neighbors=[0, 1, 2]),
        ]
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_permuted_ordering_same_event_is_deduplicated() -> None:
    """Same physical event with permuted neighbour ordering -> one duplicate.

    Row A stores atoms [0, 1, 2, 3]; row B stores the SAME atoms in order
    [3, 2, 1, 0] with its ``saddle_positions`` permuted to match. Physically
    identical, so exactly one row must survive.

    Pre-fix this FAILED: part 1 compared the two arrays element-wise, so
    ``saddle[0]`` of A (atom 0) was differenced against ``saddle[0]`` of B
    (atom 3). That yields a large ``delr`` and the true duplicate was KEPT
    (its rate double-counted in the rejection-free selection).
    """
    perm = [3, 2, 1, 0]
    table = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            _row(
                atom_index=1,
                saddle_positions=_SADDLE[perm],
                neighbors=[3, 2, 1, 0],
            ),
        ]
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 1


def test_permuted_ordering_distinct_events_stay_distinct() -> None:
    """Two distinct events that would alias under a raw permuted compare.

    Row A (atoms [0, 1, 2, 3]) moves atom 1. Row B stores neighbours
    [1, 0, 2, 3] but is a genuinely different saddle (a different atom moved).
    A naive element-wise compare of the permuted arrays could coincidentally
    line up the moved coordinates; the id-aligned compare must keep both.
    """
    saddle_b = _SADDLE.copy()
    saddle_b[0] = [10.0, 10.0, 10.0]  # atom 0 at rest
    saddle_b[1] = [12.0, 10.0, 10.0]  # atom 1 at rest (no +0.5 A hop)
    saddle_b[2] = [10.0, 13.0, 10.0]  # atom 2 moved instead
    # Store row B permuted: neighbours [1, 0, 2, 3], positions permuted to match.
    perm = [1, 0, 2, 3]
    table = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            _row(
                atom_index=1,
                saddle_positions=saddle_b[perm],
                neighbors=perm,
            ),
        ]
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_identical_ordering_matches_prefix_behaviour() -> None:
    """Identical-ordering duplicate/non-duplicate outcomes are unchanged.

    With identical neighbour orderings the aligned compare reduces to the
    original element-wise ``compute_delr``, so the accept/reject verdict must
    match the pre-fix code exactly.
    """
    # (a) identical geometry, identical ordering -> duplicate removed.
    dup = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
        ]
    )
    dup.remove_duplicates(_CELL)
    assert len(dup.table) == 1

    # (b) different geometry, identical ordering -> both kept.
    saddle_far = _SADDLE.copy()
    saddle_far[1] = [15.0, 10.0, 10.0]  # well beyond matching_score_thr
    distinct = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            _row(atom_index=1, saddle_positions=saddle_far, neighbors=[0, 1, 2, 3]),
        ]
    )
    distinct.remove_duplicates(_CELL)
    assert len(distinct.table) == 2


def test_none_neighbors_keeps_both_rows() -> None:
    """A row with ``neighbors=None`` is not comparable -> both rows kept."""
    table = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=None),
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
        ]
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_length_mismatched_neighbors_keeps_both_rows() -> None:
    """A neighbours array shorter than its positions is not comparable."""
    table = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            # neighbours length 3 but 4 saddle rows -> cannot trust the ordering.
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2]),
        ]
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_disjoint_sets_same_central_atom_kept() -> None:
    """Same central atom but disjoint neighbour sets -> different environment.

    A recycled row whose membership drifted after the system moved must not be
    treated as a duplicate of a fresh row on the same central atom.
    """
    table = _table(
        [
            _row(atom_index=1, saddle_positions=_SADDLE, neighbors=[0, 1, 2, 3]),
            _row(
                atom_index=1,
                saddle_positions=_SADDLE,
                neighbors=[4, 5, 6, 7],
            ),
        ]
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_symmetric_events_permuted_ordering_deduplicated() -> None:
    """Part-2: symmetric events on different central atoms, permuted ordering.

    The symmetric pass only fires for a ``num_reference_event`` that appears
    more than once on some single central atom, so this scenario has two rows
    on atom 1 (with distinct barriers, keeping part 1 from touching them) plus
    a cross-atom duplicate on atom 2. Over their shared neighbour shell the
    atom-2 row matches the first atom-1 row; the stored orderings differ, so the
    atom-2 row is only detected as a duplicate once alignment uses the stored
    ``neighbors`` column.

    Pre-fix, part 2 built its alignment map from a fresh
    ``get_neighbors('rcut', ...)`` ordering rather than the stored ``neighbors``
    column, scattering coordinates onto the wrong atoms for recycled rows.
    """
    # Shared atoms {1, 2, 3} carry identical positions in the matching rows; the
    # non-shared atom differs. central_atom1 (=1) is a member of the atom-2 shell.
    shared_pos = {
        1: [12.0, 10.0, 10.0],
        2: [10.0, 12.0, 10.0],
        3: [10.0, 10.0, 12.0],
    }
    # Row A0 on atom 1: neighbours [1, 2, 3, 0], extra atom 0.
    nbrs_a0 = [1, 2, 3, 0]
    pos_a0 = np.array([shared_pos[1], shared_pos[2], shared_pos[3], [8.0, 8.0, 8.0]])
    # Row A1 on atom 1: a genuinely different saddle + different barrier so part
    # 1 leaves it alone; only present to make num_ref=7 fire the symmetric pass.
    pos_a1 = pos_a0.copy()
    pos_a1[0] = [15.0, 10.0, 10.0]
    # Row B on atom 2: neighbours [3, 2, 1, 4] (permuted), extra atom 4; matches
    # A0 over the shared atoms {1, 2, 3}.
    nbrs_b = [3, 2, 1, 4]
    pos_b = np.array([shared_pos[3], shared_pos[2], shared_pos[1], [15.0, 15.0, 15.0]])

    # A truthy neighbours_list enables the symmetric pass; its return value is
    # no longer consumed for alignment.
    neighbors_list = Mock()
    neighbors_list.get_neighbors.return_value = np.asarray([1, 2, 3, 0], dtype=int)

    table = _table(
        [
            _row(
                atom_index=1,
                saddle_positions=pos_a0,
                neighbors=nbrs_a0,
                num_reference_event=7,
            ),
            _row(
                atom_index=1,
                saddle_positions=pos_a1,
                neighbors=nbrs_a0,
                energy_barrier=0.9,
                num_reference_event=7,
            ),
            _row(
                atom_index=2,
                saddle_positions=pos_b,
                neighbors=nbrs_b,
                num_reference_event=7,
            ),
        ]
    )
    table.remove_duplicates(_CELL, neighbors_list=neighbors_list)
    # A0 and A1 survive part 1 (distinct barriers, distinct geometry); B is the
    # symmetric duplicate of A0 removed by part 2.
    assert len(table.table) == 2
    assert set(table.table["atom_index"]) == {1}
