"""Unit tests for full-colour typing in the reference event table."""

import numpy as np
import pandas as pd

from pykmc import NeighborsList, System
from pykmc.environments.graph_nauty import graph
from pykmc.event_table import ReferenceEventTable


def _build_trivial_series(config, system):
    """Run ``_build_event_series`` for a trivial (min1==saddle==min2) event.

    This exercises the graph/symmetry/type-storage path without going through
    barrier validation, which is unrelated to colouring.
    """
    table = ReferenceEventTable(config)
    pos = system.positions
    return table._build_event_series(
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.5,
        cell=system.cell,
        types=list(system.types),
    )


class TestReferenceTableTypes:
    """The ``types`` column is always populated, so the schema is mode-independent."""

    def test_types_stored_in_both_modes(
        self, system_binary_fcc, config_system_single_type
    ):
        """Both grey and full store the real forward/backward neighbour element types."""
        config = config_system_single_type

        # Expected forward types = element of each atom in atom 0's rcut shell.
        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]
        expected = list(np.array(system_binary_fcc.types)[forward_neighbors])

        for mode in ("grey", "full"):
            config.atomicenvironment.atom_coloring_mode = mode
            fwd, bwd = _build_trivial_series(config, system_binary_fcc)

            assert fwd["types"] is not None, mode
            assert list(fwd["types"]) == expected, mode
            # A genuine multi-element environment carries both species.
            assert set(fwd["types"]) == {"Ni", "Fe"}, mode
            assert bwd["types"] is not None, mode

    def test_event_id_differs_by_mode_same_types(
        self, system_binary_fcc, config_system_single_type
    ):
        """Full colouring changes the reference hash (event_id), but not the stored types."""
        config = config_system_single_type

        config.atomicenvironment.atom_coloring_mode = "grey"
        grey_fwd, _ = _build_trivial_series(config, system_binary_fcc)

        config.atomicenvironment.atom_coloring_mode = "full"
        full_fwd, _ = _build_trivial_series(config, system_binary_fcc)

        # Colouring changes the hash...
        assert grey_fwd["event_id"] != full_fwd["event_id"]
        # ...but the stored types are identical (storage is mode-independent).
        assert list(grey_fwd["types"]) == list(full_fwd["types"])

    def test_types_aligned_with_positions(
        self, system_binary_fcc, config_system_single_type
    ):
        """``types`` is element-wise aligned with the position slices (the ordering invariant).

        Equal length alone is tautological (everything is sliced by the same index
        array), so we tie both the type label *and* the position row of each entry
        back to the same global atom index. For this trivial event
        (min1==saddle==min2==system.positions) every position slice reduces to
        ``system.positions[neighbour_list]``.
        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]

        fwd, _ = _build_trivial_series(config, system_binary_fcc)

        n = len(forward_neighbors)
        assert len(fwd["types"]) == n
        # Each entry's type and its three position rows all map to the same atom.
        for k, atom in enumerate(forward_neighbors):
            assert fwd["types"][k] == system_binary_fcc.types[atom]
            for key in ("initial_positions", "saddle_positions", "final_positions"):
                assert np.array_equal(fwd[key][k], system_binary_fcc.positions[atom]), (
                    key,
                    k,
                )


class TestGreyDedupSpeciesGating:
    """Specify the coloring-mode gating of de-dup in ``is_new_event``.

    ``is_new_event`` mirrors the PSR / classification paths: in "full" mode it
    feeds the REAL element types to IRA, so geometrically-identical but
    species-swapped saddles stay DISTINCT; in "grey" mode every atom is greyed to a
    single dummy label (``'X'``), so the same swapped saddles de-duplicate as one
    event (grey-alloy approximation). These tests pin both sides of that gate.
    """

    def _insert_forward_row(self, table: ReferenceEventTable, fwd: pd.Series) -> None:
        """Append a single forward event ``Series`` as a row of ``table.table``.

        Parameters
        ----------
        table : ReferenceEventTable
            The reference table whose ``table`` DataFrame is appended to.
        fwd : pd.Series
            The forward event series to insert as a single row.

        """
        fwd = fwd.copy()
        fwd["idx_ref"] = 0
        fwd["idx_backward"] = 0
        table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)

    def test_grey_swapped_species_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """Grey mode merges swapped-species, geometrically-identical duplicates.

        A trivial (min1==saddle==min2) event is stored, then an identical-geometry
        event with every Ni<->Fe swapped is queried. The grey ``event_id`` hash is
        species-blind so both share an id and reach the IRA de-dup path; that path
        greys every atom to ``'X'``, so the geometries match and ``is_new_event``
        reports the swapped event as a duplicate.

        Parameters
        ----------
        system_binary_fcc : object
            Binary Ni/Fe FCC ``System`` fixture (alternating species).
        config_system_single_type : object
            Loaded ``Config`` fixture; its coloring mode is set to ``grey`` here.

        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"

        pos = system_binary_fcc.positions
        cell = system_binary_fcc.cell
        types = list(system_binary_fcc.types)
        swapped = ["Fe" if t == "Ni" else "Ni" for t in types]

        table = ReferenceEventTable(config)
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=types,
        )
        self._insert_forward_row(table, fwd)

        fwd_swapped, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=swapped,
        )

        # Grey hash is species-blind: identical geometry -> identical event_id,
        # so the swapped event reaches the (grey-gated) IRA de-dup path.
        assert fwd_swapped["event_id"] == fwd["event_id"]
        # ...the stored/queried element types still genuinely differ (storage is
        # mode-independent; greying to 'X' happens inside is_new_event).
        assert list(fwd_swapped["types"]) != list(fwd["types"])

        # Grey-gated de-dup: types are greyed to 'X', so the swapped event MERGES.
        assert table.is_new_event(fwd_swapped) is False

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False

    def test_full_swapped_species_not_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """Full mode keeps swapped-species saddles distinct (species-aware de-dup).

        Same geometry as the grey case, but in "full" mode ``is_new_event`` feeds
        the real (swapped) element types to IRA, so the match is rejected and the
        swapped event is reported as new.

        Parameters
        ----------
        system_binary_fcc : object
            Binary Ni/Fe FCC ``System`` fixture (alternating species).
        config_system_single_type : object
            Loaded ``Config`` fixture; its coloring mode is set to ``full`` here.

        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

        pos = system_binary_fcc.positions
        cell = system_binary_fcc.cell
        types = list(system_binary_fcc.types)
        swapped = ["Fe" if t == "Ni" else "Ni" for t in types]

        table = ReferenceEventTable(config)
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=types,
        )
        self._insert_forward_row(table, fwd)

        fwd_swapped, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=swapped,
        )

        # Full mode compares real element types (species-aware event_id and/or
        # type-aware IRA), so the swapped saddle is not merged -> reported as new.
        assert table.is_new_event(fwd_swapped) is True

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False


def test_event_series_matches_full_system_neighbors_list(
    system_binary_fcc: object, config_system_single_type: object
) -> None:
    """Ids and crops, atom order included, are those of full-system neighbor lists.

    Atom 0, at the box corner, moves in the saddle and in min2 and every other atom
    is jittered, so the three environments differ and wrap across the box. The
    reference builds a ``NeighborsList`` of the whole system per configuration.

    Parameters
    ----------
    system_binary_fcc : object
        Binary Ni/Fe FCC ``System`` fixture (alternating species).
    config_system_single_type : object
        Loaded ``Config`` fixture; its coloring mode is set to ``full`` here.

    """
    config = config_system_single_type
    config.atomicenvironment.atom_coloring_mode = "full"
    rnei = config.atomicenvironment.rnei
    rcut = config.atomicenvironment.rcut
    cell = system_binary_fcc.cell
    types = list(system_binary_fcc.types)
    rng = np.random.default_rng(0)
    min1 = system_binary_fcc.positions
    saddle, min2 = (
        np.mod(min1 + rng.normal(0.0, 0.05, min1.shape), np.diag(cell))
        for _ in range(2)
    )
    saddle[0] = min1[0] + 0.4
    min2[0] = min1[0] + 0.8

    lists = [
        NeighborsList(System(positions=p, cell=cell), rnei, rcut).neighbors_list
        for p in (min1, saddle, min2)
    ]
    ids = [graph(nl["rnei"], nl["rcut"], atom_idx=[0], types=types)[0] for nl in lists]
    forward, backward = lists[0]["rcut"][0], lists[2]["rcut"][0]
    assert len(set(ids)) == 3
    assert sorted(forward) != sorted(backward)

    fwd, bwd = ReferenceEventTable(config)._build_event_series(
        min1_positions=min1,
        saddle_positions=saddle,
        min2_positions=min2,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.3,
        cell=cell,
        types=types,
    )

    assert [fwd["event_id"], fwd["id_saddle"], fwd["id_final"]] == ids
    assert [bwd["event_id"], bwd["id_saddle"], bwd["id_final"]] == ids[::-1]
    assert np.array_equal(fwd["initial_positions"], min1[forward])
    assert np.array_equal(bwd["initial_positions"], min2[backward])
