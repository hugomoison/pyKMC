"""A recycled row must not outlive its reference event.

A failed reconstruction purges reference events (``ReferenceEventTable.remove``)
while rows recycled on other sites still point at them; the next step then
raised ``KeyError`` in ``info_active_events`` or ``IndexError`` in
``KMC.reconstruction``. ``KMC.run`` now evicts such rows, after the step log and
the recycling prune have read the executed row.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from ase.geometry import find_mic

from pykmc.basins.detection import DetectorThreshold
from pykmc.config import EventRecyclingConfig
from pykmc.event_table import ReferenceEventTable
from pykmc.kmc import KMC
from pykmc.result import Err, ErrorInfo, ErrorType, EventRefinementOutput, Ok

from ..test_kmc_step_logging import (
    _FakeEventSearch,
    _FakeManager,
    _FakeRefinement,
    _RecordingLoggers,
)
from .conftest import row


def _ref_row(idx_ref: int, idx_backward: int, event_id: str) -> dict:
    """Build a minimal reference row with the columns the lookups consume."""
    return {
        "idx_ref": idx_ref,
        "event_id": event_id,
        "idx_backward": idx_backward,
        "id_final": f"final_{event_id}",
        "energy_barrier": 0.5,
        "dra": 1.0,
    }


def test_detect_is_false_when_executed_ref_was_purged() -> None:
    """Basin detection on an executed row whose ref was purged this step."""
    executed = pd.Series({**row(3), "num_reference_event": 5, "energy_barrier": 0.1})
    reference = pd.DataFrame([_ref_row(0, 0, "topo0")])  # ref 5 purged

    assert not DetectorThreshold().detect(executed, reference, 0.5, True)


def test_run_evicts_recycled_orphan_after_prune(
    config_Cu: Any,
    system_Cu: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Two recycled steps: a purge must not orphan rows or break the step log.

    Step 1: B (ref 0) fails INVALID_MIN1, so ref 0 and its backward 1 are
    purged; A, which shares ref 0, is executed; C (ref 0) is far and unmoved,
    so the recycler carries it over. Step 2 adds D (ref 2) and executes it.
    """
    config = config_Cu
    config.control.n_steps = 2
    config.control.basin = False
    config.control.recycle = True
    config.control.reference_table = None

    system = system_Cu
    positions = np.array(system.positions, copy=True)
    _, dist = find_mic(positions - positions[0], system.cell, pbc=True)
    a = 0
    c = int(np.argmax(dist))
    b, d = (int(i) for i in np.argsort(dist)[1:3])
    config.eventrecycling = EventRecyclingConfig(
        style="displacement", movement_thr=0.02, distance_thr=0.5 * dist[c]
    )

    sim = KMC(config)
    sim.system = system
    sim.manager = _FakeManager()
    sim.loggers = _RecordingLoggers()
    sim.visited_environments = set()
    sim.reference_table = ReferenceEventTable(config)
    sim.reference_table.table = pd.DataFrame(
        [
            _ref_row(0, 1, "topo0"),
            _ref_row(1, 0, "topo1"),
            _ref_row(2, 3, "topo2"),
            _ref_row(3, 2, "topo3"),
        ]
    )

    def _event(atom: int, ref: int) -> EventRefinementOutput:
        return EventRefinementOutput(
            central_atom_index=atom,
            saddle_positions=positions.copy(),
            E_saddle=1.0,
            min2_positions=positions.copy(),
            dE_forward=0.5,
            num_reference_event=ref,
            refined="T",
        )

    refinements = iter(
        [
            [_event(b, 0), _event(a, 0), _event(c, 0)],  # step 1
            [_event(d, 2)],  # step 2
        ]
    )

    def _select_atom(atoms: list[int]) -> Callable[[Any], tuple[int, float, float]]:
        order = iter(atoms)

        def _select(active_table: Any) -> tuple[int, float, float]:
            atom = next(order)
            idx = int(np.flatnonzero(active_table.table["atom_index"] == atom)[0])
            return idx, 1.0, 1.0

        return _select

    def _reconstruct(idx: int, active_table: Any) -> Any:
        atom = int(active_table.table.loc[idx].at["atom_index"])
        if atom == b:
            return Err(
                ErrorInfo(
                    type=ErrorType.RECONSTRUCTION_INVALID_MIN1,
                    message="stub",
                    variables={},
                )
            )
        moved = np.array(sim.system.positions, copy=True)
        moved[atom] += np.array([0.3, 0.0, 0.0])
        return Ok(SimpleNamespace(min2_positions=moved, min2_etot=-1.0))

    seen_pairs: list[set[tuple[int, int]]] = []

    def _refine(subset: Any, existing_pairs: Any = None) -> _FakeRefinement:
        seen_pairs.append(set(existing_pairs or ()))
        return _FakeRefinement(next(refinements))

    monkeypatch.setattr(sim, "minimize_system", lambda: None)
    monkeypatch.setattr(sim, "execute_event_searches", lambda atoms: _FakeEventSearch())
    monkeypatch.setattr(sim, "add_reference_events", lambda results: [])
    monkeypatch.setattr(sim, "execute_refinements", _refine)
    monkeypatch.setattr(sim, "_select_event", _select_atom([b, a, d]))
    monkeypatch.setattr(sim, "_reconstruction_active_event", _reconstruct)
    monkeypatch.setattr(sim, "_save", lambda: None)
    monkeypatch.setattr(sim, "_append_snapshot_to_trajectory", lambda: None)
    monkeypatch.setattr(sim, "_save_restart_file", lambda *args: None)
    monkeypatch.setattr(sim, "_close", lambda: None)
    monkeypatch.chdir(tmp_path)

    sim.run()

    # Step 1 logged the executed row A although its ref 0 was purged that step.
    logged = [
        (s["step"], int(s["num_reference_event"])) for s in sim.loggers.step_lines
    ]
    assert logged == [(1, 0), (2, 2)]
    # Ref 0 went with its backward sibling 1.
    assert set(sim.reference_table.table["idx_ref"]) == {2, 3}
    # The orphan C (ref 0) was evicted after the prune: step 2 never saw it.
    assert (c, 0) not in seen_pairs[1]
