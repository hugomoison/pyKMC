"""Unit tests for event-table robustness around dealloying and mixed PBC."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pykmc.event_table import ActiveEventTable


def _minimal_config() -> SimpleNamespace:
    return SimpleNamespace(psr=SimpleNamespace(matching_score_thr=0.1))


def _migration_rows_without_event_type() -> pd.DataFrame:
    # Mimics a table loaded from an older pickle: no "event_type" column.
    return pd.DataFrame(
        {
            "atom_index": [0, 1],
            "energy_barrier": [0.5, 0.9],
            "saddle_positions": [
                np.zeros((2, 3)),
                np.ones((2, 3)),
            ],
            "num_reference_event": [0, 1],
        }
    )


def test_remove_duplicates_without_event_type_column() -> None:
    """Tables without the event_type column (old pickles) must not raise."""
    table = ActiveEventTable(
        config=_minimal_config(), event_dataframe=_migration_rows_without_event_type()
    )

    table.remove_duplicates(cell=np.diag([10.0, 10.0, 10.0]))

    assert len(table.table) == 2


def test_remove_duplicates_skips_dealloying_rows() -> None:
    """A dealloying row identical to a migration row must not be deduplicated.

    The two rows are genuine duplicates (same atom, same barrier, identical
    saddle positions): if dealloying rows were mistakenly compared, one row
    would be removed and the count would drop to 1.
    """
    df = pd.DataFrame(
        {
            "atom_index": [0, 0],
            "energy_barrier": [0.5, 0.5],
            "saddle_positions": [np.zeros((2, 3)), np.zeros((2, 3))],
            "num_reference_event": [0, 0],
            "event_type": ["migration", "dealloying"],
        }
    )
    table = ActiveEventTable(config=_minimal_config(), event_dataframe=df)

    table.remove_duplicates(cell=np.diag([10.0, 10.0, 10.0]))

    assert len(table.table) == 2


def _chain_config() -> SimpleNamespace:
    return SimpleNamespace(
        control=SimpleNamespace(reference_table=None),
        atomicenvironment=SimpleNamespace(
            atom_coloring_mode="grey", rnei=1.2, rcut=2.5, neighbors_add=0.0
        ),
        eventsearch=SimpleNamespace(
            emin_event=0.0,
            emax_event=100.0,
            backward_emin_event=0.0,
            energy_asymmetry=100.0,
        ),
        ira=SimpleNamespace(sym_thr=0.1),
    )


def test_add_events_threads_pbc_through_to_temp_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """add_events must carry the source pbc into all three temp Systems.

    Drives the public chain add_events -> is_valid_new_event ->
    _build_event_series with the heavy callees (NeighborsList, graph,
    unique_symmetries, rate) stubbed. Records the pbc each temp System
    carries, and runs _build_event_series to completion so the
    move_atom_idx lookup is exercised with the Python lists NeighborsList
    produces (the np.asarray coercion regression guard).
    """
    import pykmc.event_table as et
    from pykmc.event_table import ReferenceEventTable

    n_atoms = 4
    seen_pbc = []

    class _StubNeighborsList:
        def __init__(self, system: object, rnei: float, rcut: float) -> None:
            seen_pbc.append(system.pbc)
            everyone = list(range(n_atoms))  # Python lists, like query_ball_point
            self.neighbors_list = {
                "rnei": [[j for j in everyone if j != i] for i in range(n_atoms)],
                "rcut": [everyone for _ in range(n_atoms)],
            }

    monkeypatch.setattr(et, "NeighborsList", _StubNeighborsList)
    monkeypatch.setattr(et, "graph", lambda *args, **kwargs: ["stub-id"])
    monkeypatch.setattr(
        et,
        "unique_symmetries",
        lambda *args, **kwargs: ([np.eye(3)], [np.arange(n_atoms)]),
    )
    monkeypatch.setattr(et, "compute_rate_Eyring", lambda dE, config: 1.0)

    table = ReferenceEventTable(config=_chain_config())

    pbc = np.array([True, True, False])
    rng = np.random.default_rng(0)
    move_atom = 2
    event = SimpleNamespace(
        min1_positions=rng.random((n_atoms, 3)) * 5.0,
        saddle_positions=rng.random((n_atoms, 3)) * 5.0,
        min2_positions=rng.random((n_atoms, 3)) * 5.0,
        move_atom_index=move_atom,
        dE_forward=0.5,
        dE_backward=0.5,
        cell=np.diag([5.0, 5.0, 5.0]),
        types=None,
    )

    results = table.add_events([event], pbc=pbc)

    # The event survived validation and the full series build (including the
    # np.where move-atom lookup on list-backed neighbor environments).
    assert len(results) == 1
    assert results[0].is_ok()
    assert len(table.table) == 1
    assert table.table.iloc[0]["move_atom_idx"] == move_atom  # rcut env is 0..n-1

    # All three temp Systems (min1, saddle, min2) carried the source pbc.
    assert len(seen_pbc) == 3
    for recorded in seen_pbc:
        assert np.array_equal(recorded, pbc)
