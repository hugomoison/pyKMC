"""Tests for the prototype State coordination object (issue #63).

Uses the shared conftest fixtures ``system_single_type_fcc`` (256-atom Ni FCC)
and ``mock_config`` (cna/graph, rnei=3.01, rcut=6.5).
"""

from unittest.mock import Mock

import numpy as np

from pykmc import AtomicEnvironment, NeighborsList, System
from pykmc.state import State


def _manual_environment(system: System, cfg: Mock) -> AtomicEnvironment:
    """Reproduce the current manual kmc.py / StateData build for comparison."""
    ae = cfg.atomicenvironment
    nl = NeighborsList(system, ae.rnei, ae.rcut)
    return AtomicEnvironment(
        ae.style,
        nl.neighbors_list["rnei"],
        nl.neighbors_list.get("rcut"),
        ae.neighbors_add or 0,
    )


class TestState:
    """Behaviour of the prototype State coordination object."""

    def test_matches_manual_build(
        self, system_single_type_fcc: System, mock_config: Mock
    ) -> None:
        """State must produce the same environment IDs as the manual path."""
        state = State(system_single_type_fcc, mock_config)
        manual = _manual_environment(system_single_type_fcc, mock_config)
        assert (
            list(state.atomic_environment.atomic_environment_list)
            == list(manual.atomic_environment_list)
        )

    def test_lazy_invalidation(
        self, system_single_type_fcc: System, mock_config: Mock
    ) -> None:
        """sync=False marks stale; next access rebuilds and re-syncs."""
        state = State(system_single_type_fcc, mock_config)
        _ = state.atomic_environment  # build
        assert state.is_synced()

        state.set_positions(system_single_type_fcc.positions.copy(), sync=False)
        assert not state.is_synced()

        _ = state.neighbors_list  # access rebuilds
        assert state.is_synced()

    def test_eager_sync(
        self, system_single_type_fcc: System, mock_config: Mock
    ) -> None:
        """sync=True rebuilds immediately."""
        state = State(system_single_type_fcc, mock_config)
        state.set_positions(system_single_type_fcc.positions.copy(), sync=True)
        assert state.is_synced()

    def test_release_heavy_objects(
        self, system_single_type_fcc: System, mock_config: Mock
    ) -> None:
        """release_heavy_objects drops derived data and re-dirties."""
        state = State(system_single_type_fcc, mock_config)
        _ = state.atomic_environment
        assert state.is_synced()

        state.release_heavy_objects()
        assert not state.is_synced()
        assert state.atomic_environment is not None  # rebuilds on access
        assert state.is_synced()

    def test_set_positions_updates_system(
        self, system_single_type_fcc: System, mock_config: Mock
    ) -> None:
        """Moving an atom updates System and keeps State synced."""
        state = State(system_single_type_fcc, mock_config)
        new = system_single_type_fcc.positions.copy()
        new[0] = new[0] + np.array([0.1, 0.0, 0.0])
        state.set_positions(new)
        assert state.is_synced()
        np.testing.assert_allclose(state.system.positions[0], new[0], atol=1e-8)

    def test_invalidate(
        self, system_single_type_fcc: System, mock_config: Mock
    ) -> None:
        """invalidate() forces a rebuild on next access."""
        state = State(system_single_type_fcc, mock_config)
        _ = state.neighbors_list
        assert state.is_synced()
        state.invalidate()
        assert not state.is_synced()
        _ = state.neighbors_list
        assert state.is_synced()
