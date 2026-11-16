"""Unit tests for full-colour typing in the reference event table."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pykmc import NeighborsList, constants
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.rate_constant import RateConstantError, RateConstantOutput, RateRequest
from pykmc.result import (
    Err,
    ErrorInfo,
    ErrorType,
    EventRefinementOutput,
    EventSearchOutput,
    Ok,
)


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


class _SpyRateConstant:
    """Record each ``compute`` call; fail on the barriers listed in ``fail``."""

    def __init__(self, fail: tuple[float, ...] = ()) -> None:
        self.calls = []
        self.fail = fail

    def compute(self, rate_requests: list[RateRequest]) -> list:
        self.calls.append(list(rate_requests))
        results = []
        for request in rate_requests:
            if request.dE in self.fail:
                backup = RateConstantOutput(k=-request.dE, prefactor=1.0)
                results.append(
                    Err(
                        ErrorInfo(
                            type=RateConstantError.COMPUTATION_FAILED,
                            message="spy failure",
                            variables={"backup": backup},
                        )
                    )
                )
            else:
                results.append(
                    Ok(RateConstantOutput(k=10.0 * request.dE, prefactor=10.0))
                )
        return results


class TestAddEventsRateConstant:
    """``add_events`` computes the rate constants of all new events in one call.

    ``is_valid_new_event`` is replaced by prepared results: event A keeps forward
    and backward (2 rows), event B is rejected, event C keeps only forward (1 row).
    """

    @staticmethod
    def _event(dE_forward: float, dE_backward: float) -> EventSearchOutput:
        positions = np.zeros((2, 3))
        return EventSearchOutput(
            central_atom_index=0,
            min1_positions=positions,
            saddle_positions=positions + 1.0,
            min2_positions=positions + 2.0,
            dE_forward=dE_forward,
            dE_backward=dE_backward,
            move_atom_index=1,
        )

    @staticmethod
    def _rows(n: int) -> Ok:
        return Ok(pd.DataFrame({"energy_barrier": [0.0] * n, "k": [np.nan] * n}))

    def _add(
        self,
        config: object,
        monkeypatch: pytest.MonkeyPatch,
        rate_constant: _SpyRateConstant,
        loggers: object = None,
    ) -> tuple[ReferenceEventTable, list[EventSearchOutput]]:
        events = [self._event(0.1, 0.2), self._event(0.3, 0.4), self._event(0.5, 0.6)]
        validations = iter(
            [
                self._rows(2),
                Err(ErrorInfo(type=ErrorType.EVENT_NOT_NEW, message="")),
                self._rows(1),
            ]
        )
        table = ReferenceEventTable(
            config, rate_constant=rate_constant, loggers=loggers
        )
        monkeypatch.setattr(
            table, "is_valid_new_event", lambda **kwargs: next(validations)
        )
        table.add_events(events)
        return table, events

    def test_one_call_with_a_request_per_added_row(
        self, config_system_single_type: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One ``compute`` call; forward from min1, backward from min2, same saddle."""
        spy = _SpyRateConstant()
        table, (a, _, c) = self._add(config_system_single_type, monkeypatch, spy)

        assert len(spy.calls) == 1
        requests = spy.calls[0]
        assert [r.dE for r in requests] == [0.1, 0.2, 0.5]
        assert requests[0].initial_positions is a.min1_positions
        assert requests[1].initial_positions is a.min2_positions
        assert requests[2].initial_positions is c.min1_positions
        assert requests[0].saddle_positions is requests[1].saddle_positions
        assert requests[2].saddle_positions is c.saddle_positions
        assert all(r.central_atom == 1 for r in requests)

        assert list(table.table["k"]) == pytest.approx([1.0, 2.0, 5.0])
        assert list(table.table["k_pref"]) == pytest.approx([10.0, 10.0, 10.0])

    def test_failed_rate_uses_the_backup(
        self, config_system_single_type: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row whose rate fails gets the backup of the error, with a warning."""
        spy = _SpyRateConstant(fail=(0.2,))
        loggers = MagicMock()
        table, _ = self._add(config_system_single_type, monkeypatch, spy, loggers)

        assert list(table.table["k"]) == pytest.approx([1.0, -0.2, 5.0])
        assert list(table.table["k_pref"]) == pytest.approx([10.0, 1.0, 10.0])
        # the backward row of event A (idx_ref 1) is the one that failed
        loggers.warning.assert_called_once_with(
            "log",
            "Rate constant compute failed on reference event 1 (spy failure), "
            "use backup value k0 = 1 ps^-1",
        )


class TestActiveAddEventsRateConstant:
    """``ActiveEventTable.add_events`` and the rate constant of its events."""

    def test_reference_prefactor_with_active_barrier(
        self, config_system_single_type: object
    ) -> None:
        """k = k_pref of the reference event x Boltzmann factor of the active barrier."""
        config = config_system_single_type
        reference_table = ReferenceEventTable(config)
        reference_table.table = pd.DataFrame({"idx_ref": [0, 1], "k_pref": [2.0, 4.0]})
        events = [
            EventRefinementOutput(
                central_atom_index=3,
                saddle_positions=np.zeros((2, 3)),
                E_saddle=0.0,
                dE_forward=dE,
                num_reference_event=num_reference_event,
                refined="T",
            )
            for dE, num_reference_event in ((0.5, 1), (0.3, 0))
        ]

        active_table = ActiveEventTable(config)
        active_table.add_events(events, reference_table)

        kT = constants.KB * config.rateconstant.T
        assert list(active_table.table["k_pref"]) == pytest.approx([4.0, 2.0])
        assert list(active_table.table["k"]) == pytest.approx(
            [4.0 * np.exp(-0.5 / kT), 2.0 * np.exp(-0.3 / kT)]
        )
