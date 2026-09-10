"""Tests that reconstruction consumes the stored neighbour ordering.

A refined active event stores ``saddle_positions``/``final_positions`` ordered
by the in-rcut neighbour list captured at refinement time. The per-step
neighbour list is rebuilt every KMC step and ``cKDTree.query_ball_point`` may
return the same absolute atoms in a different order. If reconstruction
re-derives the ordering from the current list (the pre-fix bug), the stored
row-i coordinates scatter onto the wrong absolute atoms for recycled events,
producing a LAMMPS "Lost atoms" crash. The fix persists the row's own
neighbour index list and consumes it in ``_reconstruction_active_event``; this
suite locks that behaviour in.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd

from pykmc import System
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.result import ErrorType, EventRefinementOutput

# A 4-atom cubic 20 A PBC system. Absolute atom 1 is the moving atom: its saddle
# row carries a +0.5 A displacement in x relative to its minimum, every other
# row sits at its minimum.
_CELL = np.diag([20.0, 20.0, 20.0])
_MIN_POSITIONS = np.array(
    [
        [10.0, 10.0, 10.0],
        [12.0, 10.0, 10.0],
        [10.0, 12.0, 10.0],
        [10.0, 10.0, 12.0],
    ]
)
# Saddle/final ordered to the canonical neighbour list [0, 1, 2, 3] with the
# moving displacement on ROW INDEX 1 (=> absolute atom 1).
_SADDLE_ROWS = _MIN_POSITIONS.copy()
_SADDLE_ROWS[1] = [12.5, 10.0, 10.0]
_FINAL_ROWS = _MIN_POSITIONS.copy()
_FINAL_ROWS[1] = [13.0, 10.0, 10.0]


def _toy_system() -> System:
    return System(
        positions=_MIN_POSITIONS.copy(),
        types=np.array(["Ni", "Ni", "Ni", "Ni"]),
        cell=_CELL.copy(),
        pbc=np.array([True, True, True]),
        index=np.array([0, 1, 2, 3]),
    )


def _config() -> Mock:
    cfg = Mock()
    cfg.control.recycle = False
    cfg.reconstruction.push_fraction = 0.15
    cfg.reconstruction.n_movers = 3
    cfg.reconstruction.containment_margin = 1.0
    cfg.reconstruction.shell_tolerance = 1.0
    cfg.atomicenvironment.rcut = 6.5
    cfg.psr.matching_score_thr = 0.1
    return cfg


def _active_table(neighbors: np.ndarray) -> ActiveEventTable:
    """Build a one-row ActiveEventTable carrying an explicit neighbours column."""
    cfg = Mock()
    cfg.rateconstant.style = "constant"
    cfg.rateconstant.T = 300.0
    cfg.rateconstant.k0 = 10.0
    table = pd.DataFrame(
        [
            {
                "atom_index": 0,
                "saddle_positions": _SADDLE_ROWS.copy(),
                "final_positions": _FINAL_ROWS.copy(),
                "neighbors": np.asarray(neighbors, dtype=int),
                "energy_barrier": 0.5,
                "k": 1.0,
                "num_reference_event": 0,
                "refined": "T",
            }
        ]
    )
    return ActiveEventTable(cfg, event_dataframe=table)


def _echo_manager() -> tuple[Mock, list[np.ndarray]]:
    """Build a manager whose minimize echoes its input and records each call."""
    captured: list[np.ndarray] = []

    def _echo(config=None, positions=None, types=None):  # noqa: ANN001, ANN202
        captured.append(np.asarray(positions).copy())
        return positions.copy(), -1.0

    manager = Mock()
    manager.group_minimize_with_results.side_effect = _echo
    return manager, captured


def _make_kmc(neighbors_list_return: list[int]) -> tuple[KMC, list[np.ndarray]]:
    kmc = KMC(config=_config())
    kmc.system = _toy_system()
    manager, captured = _echo_manager()
    kmc.manager = manager
    # The fix must IGNORE this stub: the stored neighbours column is authoritative.
    neighbors_list = Mock()
    neighbors_list.get_neighbors.return_value = np.asarray(
        neighbors_list_return, dtype=int
    )
    kmc.neighbors_list = neighbors_list
    return kmc, captured


def test_case_a_canonical_order_moves_absolute_atom_1() -> None:
    """Canonical: stored neighbours == per-step order. Displacement on atom 1."""
    kmc, captured = _make_kmc(neighbors_list_return=[0, 1, 2, 3])
    active_table = _active_table(np.array([0, 1, 2, 3]))

    kmc._reconstruction_active_event(0, active_table)

    min1_input = captured[0]
    # Absolute atom 1 carries the moving (pushed) displacement in x.
    assert not np.allclose(min1_input[1], _MIN_POSITIONS[1])
    assert min1_input[1, 0] > _MIN_POSITIONS[1, 0]
    # Absolute atoms 2 and 3 are untouched (their rows were static minimum rows).
    assert np.allclose(min1_input[2], _MIN_POSITIONS[2])
    assert np.allclose(min1_input[3], _MIN_POSITIONS[3])


def test_case_b_reordered_neighbour_list_still_moves_absolute_atom_1() -> None:
    """Recycle reality: the per-step neighbour list comes back reordered.

    With the fix the SAME assertions as Case A hold because the row's stored
    ``neighbors`` column [0, 1, 2, 3] is authoritative. (On pre-fix code this
    case scattered the displacement onto absolute atom 2 -- the faithful repro
    of the silent recycling corruption.)
    """
    kmc, captured = _make_kmc(neighbors_list_return=[0, 2, 1, 3])
    active_table = _active_table(np.array([0, 1, 2, 3]))

    kmc._reconstruction_active_event(0, active_table)

    min1_input = captured[0]
    assert not np.allclose(min1_input[1], _MIN_POSITIONS[1])
    assert min1_input[1, 0] > _MIN_POSITIONS[1, 0]
    assert np.allclose(min1_input[2], _MIN_POSITIONS[2])
    assert np.allclose(min1_input[3], _MIN_POSITIONS[3])


def test_missing_neighbors_column_fails_fast() -> None:
    """A row whose neighbours column is None returns a clear Err, never scatters."""
    kmc, _captured = _make_kmc(neighbors_list_return=[0, 1, 2, 3])
    active_table = _active_table(np.array([0, 1, 2, 3]))
    active_table.table.at[0, "neighbors"] = None

    result = kmc._reconstruction_active_event(0, active_table)

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA


def test_length_mismatched_neighbors_column_fails_fast() -> None:
    """A neighbours column shorter than the stored positions returns an Err."""
    kmc, _captured = _make_kmc(neighbors_list_return=[0, 1, 2, 3])
    active_table = _active_table(np.array([0, 1, 2, 3]))
    active_table.table.at[0, "neighbors"] = np.array([0, 1, 2])

    result = kmc._reconstruction_active_event(0, active_table)

    assert not result.is_ok()
    assert result.err_value().type == ErrorType.RECONSTRUCTION_INVALID_EVENT_DATA


def test_basin_tmp_event_carries_neighbors_through_build_event_series() -> None:
    """An EventRefinementOutput.neighbors flows into the row and is consumed.

    Mirrors the basin exit path: the basin builds a tmp EventRefinementOutput
    with its own neighbours ordering; build_event_series must surface it as a
    column so _reconstruction_active_event consumes the basin ordering.
    """
    cfg = Mock()
    cfg.rateconstant.style = "constant"
    cfg.rateconstant.T = 300.0
    cfg.rateconstant.k0 = 10.0
    table = ActiveEventTable(cfg)

    basin_neighbors = np.array([3, 1, 0, 2])
    tmp_event = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=_SADDLE_ROWS.copy(),
        E_saddle=-1.0,
        min2_positions=_FINAL_ROWS.copy(),
        dE_forward=0.5,
        num_reference_event=0,
        neighbors=basin_neighbors,
    )
    table.add_events(tmp_event)

    assert "neighbors" in table.table.columns
    np.testing.assert_array_equal(table.table.at[0, "neighbors"], basin_neighbors)

    # And it is the array consumed by reconstruction (not a re-derived one).
    kmc = KMC(config=_config())
    kmc.system = _toy_system()
    manager, captured = _echo_manager()
    kmc.manager = manager
    nl = Mock()
    nl.get_neighbors.return_value = np.array([0, 1, 2, 3])
    kmc.neighbors_list = nl

    kmc._reconstruction_active_event(0, table)

    # neighbours [3, 1, 0, 2]: saddle row 1 (the mover) -> absolute atom 1.
    min1_input = captured[0]
    assert not np.allclose(min1_input[1], _MIN_POSITIONS[1])
