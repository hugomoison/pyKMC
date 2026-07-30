"""Regression test for the end-of-step ordering in ``KMC.run``.

The step-output log reads the executed event's row from the active table
(``active_table.table.loc[idx_selected_event]``). Pruning the table for
recycling must therefore happen AFTER that read: with no recycler attached
(``recycle = False``, the default) ``prune_for_recycling`` clears the whole
table, and with a recycler attached the executed row is always dropped, so a
prune placed before the log raises ``KeyError`` on the first executed step.

The test drives one real ``KMC.run`` step with the engine-, search- and
refinement-stages stubbed out, and asserts the step line was written from the
still-live row and that the prune still ran afterwards.
"""

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

import pykmc.kmc as kmc_module
from pykmc.kmc import KMC
from pykmc.result import EventRefinementOutput


class _FakeManager:
    """Engine-pool stand-in exposing the calls ``KMC.run`` makes."""

    def initialize_sessions(self, config: Any, system: Any) -> None:
        """Pretend to boot the session pool."""

    def use_local(self) -> None:
        """Pretend to switch engines to local mode."""

    def use_global(self) -> None:
        """Pretend to switch engines to global mode."""

    def set_all_positions(self, positions: Optional[np.ndarray] = None) -> None:
        """Pretend to broadcast positions to the engines."""

    def close_all(self) -> None:
        """Pretend to shut the pool down."""


class _RecordingLoggers:
    """No-op logger that records the step-output line arguments.

    ``KMC.run`` evaluates ``active_table.table.loc[idx_selected_event]``
    while building the arguments for ``table_line_info_kmc`` — the exact
    expression that raises ``KeyError`` if the table was pruned too early —
    so recording the received values proves the row was still live.
    """

    def __init__(self) -> None:
        self.step_lines: list[dict[str, Any]] = []

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Swallow info messages."""

    def error(self, *args: Any, **kwargs: Any) -> None:
        """Swallow error messages."""

    def events_file_step_first_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file header lines."""

    def events_applicable_info_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file applicable-event lines."""

    def events_basin_info_line(self, *args: Any, **kwargs: Any) -> None:
        """Swallow events-file basin lines."""

    def table_line_info_kmc(
        self,
        name: str,
        step: int,
        delta_t: float,
        total_time: float,
        num_reference_event: Any,
        energy_barrier: Any,
        k_event: Any,
        k_tot: Any,
        total_energy: Any,
        cpu_time: float,
        wall_time: float,
    ) -> None:
        """Record the step line exactly as ``KMC.run`` evaluated it."""
        self.step_lines.append(
            {
                "step": step,
                "num_reference_event": num_reference_event,
                "energy_barrier": energy_barrier,
                "k": k_event,
            }
        )


class _FakeOkResult:
    """Minimal Ok-result wrapper matching the interface ``run`` uses."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def is_ok(self) -> bool:
        """Mirror the result protocol."""
        return True

    def ok_value(self) -> Any:
        """Return the wrapped reconstruction output."""
        return self._value


class _FakeEventSearch:
    """Event-search output with no successful searches."""

    results: list[Any] = []

    def get_successes_results(self) -> list[Any]:
        """No new generic events this step."""
        return []


class _FakeRefinement:
    """Refinement output carrying one pre-built successful event."""

    def __init__(self, outputs: list[EventRefinementOutput]) -> None:
        self._outputs = outputs
        self.results: list[Any] = []

    def get_successes_results(self) -> list[EventRefinementOutput]:
        """Return the canned refined events."""
        return self._outputs


def test_step_log_reads_executed_row_before_prune(
    config_Cu: Any,
    system_Cu: Any,
    reference_table_Cu_fake: Any,
    visited_environments_Cu: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """One executed step must log the event's row, then prune the table."""
    config = config_Cu
    config.control.n_steps = 1
    config.control.basin = False
    config.control.recycle = False

    sim = KMC(config)
    sim.system = system_Cu
    sim.manager = _FakeManager()
    sim.loggers = _RecordingLoggers()
    sim.reference_table = reference_table_Cu_fake
    sim.visited_environments = visited_environments_Cu

    ref_idx = int(reference_table_Cu_fake.table.index[0])
    positions = np.array(system_Cu.positions, copy=True)
    refined_event = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=positions.copy(),
        E_saddle=1.0,
        min2_positions=positions.copy(),
        dE_forward=0.5,
        num_reference_event=ref_idx,
        refined="T",
    )
    reconstruction_output = SimpleNamespace(
        min2_positions=positions.copy(), min2_etot=-1.0
    )

    monkeypatch.setattr(sim, "minimize_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(sim, "execute_event_searches", lambda atoms: _FakeEventSearch())
    monkeypatch.setattr(sim, "add_reference_events", lambda results: [])
    monkeypatch.setattr(
        sim,
        "execute_refinements",
        lambda subset, existing_pairs=None: _FakeRefinement([refined_event]),
    )
    monkeypatch.setattr(
        sim,
        "reconstruction",
        lambda table: (_FakeOkResult(reconstruction_output), 1.0, 1.0, 0, [], []),
    )
    monkeypatch.setattr(sim, "_save", lambda: None)
    monkeypatch.setattr(sim, "_append_snapshot_to_trajectory", lambda: None)
    monkeypatch.setattr(sim, "_save_restart_file", lambda *args: None)
    monkeypatch.setattr(sim, "_close", lambda: None)
    monkeypatch.setattr(
        kmc_module,
        "info_active_events",
        lambda *args, **kwargs: SimpleNamespace(output_msg=lambda: ""),
    )

    monkeypatch.chdir(tmp_path)
    sim.run()

    # The step line was written from the still-live executed row.
    assert len(sim.loggers.step_lines) == 1
    line = sim.loggers.step_lines[0]
    assert line["num_reference_event"] == ref_idx
    assert line["energy_barrier"] == pytest.approx(0.5)

    # The prune still ran after the log: with no recycler attached the
    # persistent active table is cleared for the next step.
    assert len(sim.active_table.table) == 0
