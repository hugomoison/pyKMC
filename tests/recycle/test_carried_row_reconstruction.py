"""A recycled active row must be reconstructed on the atoms it was refined on.

A refined row stores its saddle and final positions in the order of the central
atom's ``rcut`` neighbour list at refinement. The list is rebuilt every step and
``cKDTree.query_ball_point`` returns unsorted ids, so a carried row placed on
the rebuilt list lands on the wrong atoms (``ERROR: Lost atoms`` on a multi-rank
engine). The test runs ``Refinement.execute`` -> ``KMC.add_active_events`` ->
``KMC._reconstruction_active_event`` with the PSR match and the engine stubbed,
and changes only the neighbour list between the two steps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import pykmc.kmc as kmc_module
import pykmc.refinement as refinement_module
from pykmc import System
from pykmc.event_table import ActiveEventTable
from pykmc.kmc import KMC
from pykmc.refinement import Refinement
from pykmc.psr import PSROutput
from pykmc.result import Err, ErrorInfo, ErrorType, Ok

_CELL = np.diag([20.0, 20.0, 20.0])
# Positions indexed by absolute atom id; atom 0 is the central atom, atom 1 hops.
_MIN = np.array(
    [
        [10.0, 10.0, 10.0],
        [12.5, 10.0, 10.0],
        [10.0, 12.5, 10.0],
        [10.0, 10.0, 12.5],
        [7.5, 10.0, 10.0],
    ]
)
_SADDLE = _MIN.copy()
_SADDLE[1] = [13.75, 10.0, 10.0]
_FINAL = _MIN.copy()
_FINAL[1] = [15.0, 10.0, 10.0]
_REFINED_ORDER = [0, 1, 2, 3, 4]
_TOTAL_ENERGY = -10.0
_BARRIER = 0.5


class _NeighborsList:
    """Neighbour list returning a fixed `rcut` list for the central atom."""

    def __init__(self, order: list[int]) -> None:
        self.order = order

    def get_neighbors(self, cutoff_type: str, idx: int) -> list[int]:
        """Return the `rcut` list of the central atom."""
        return list(self.order)


class _IdentityPSR:
    """Point-set registration mapping the reference event onto itself."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> "_IdentityPSR":
        """Mirror the strategy factory of `PointSetRegistration`."""
        return cls()

    def match_and_check(self, *args: Any, **kwargs: Any) -> Ok[PSROutput]:
        """Return the identity transformation."""
        return Ok(
            PSROutput(
                matching_score=0.0,
                rotation_matrix=np.eye(3),
                translation_matrix=np.zeros(3),
                permutation_matrix=np.arange(len(_MIN)),
            )
        )


class _RecordingReconstruction:
    """Stand-in for `Reconstruction` recording what it is asked to rebuild."""

    calls: list[dict[str, np.ndarray]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def reconstruct(
        self,
        supposed_min1_positions: np.ndarray,
        supposed_min2_positions: np.ndarray,
        saddle_positions: np.ndarray,
        cell: np.ndarray,
        delr_thr: float,
        neighbors: Any = None,
    ) -> Err[ErrorInfo]:
        """Record the system handed to the group engine and fail."""
        self.calls.append(
            {
                "system": np.array(saddle_positions),
                "min2": np.array(supposed_min2_positions),
                "neighbors": np.array(neighbors, dtype=int),
            }
        )
        return Err(
            ErrorInfo(type=ErrorType.RECONSTRUCTION_INVALID_MIN1, message="stub")
        )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        control=SimpleNamespace(recycle=False, active_volume=False, refine_thr=0.5),
        eventsearch=SimpleNamespace(refined_energy_thr=0.1),
        psr=SimpleNamespace(style="ira", matching_score_thr=0.4),
        ira=SimpleNamespace(),
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full"),
        rateconstant=SimpleNamespace(style="constant", k0=10.0, T=300.0),
    )


def _reference_events() -> pd.DataFrame:
    """One reference event, stored in the refinement-time neighbour order."""
    return pd.DataFrame(
        [
            {
                "event_id": "env",
                "idx_ref": 0,
                "initial_positions": _MIN[_REFINED_ORDER].copy(),
                "types": ["Ni"] * len(_MIN),
                "move_atom_idx": 1,
                "saddle_positions": _SADDLE[_REFINED_ORDER].copy(),
                "final_positions": _FINAL[_REFINED_ORDER].copy(),
                "sym_matrix": [np.eye(3)],
                "sym_perm": [np.arange(len(_MIN))],
                "energy_barrier": _BARRIER,
                "k": 1.0,
            }
        ]
    )


@pytest.mark.parametrize(
    "rebuilt_order",
    [
        pytest.param([0, 1, 2, 3, 4], id="same-order"),
        pytest.param([0, 3, 1, 4, 2], id="reordered"),
        pytest.param([0, 1, 2, 3], id="member-left-rcut"),
    ],
)
def test_carried_row_is_reconstructed_on_its_own_atoms(
    rebuilt_order: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The system sent to the group engine holds the refined saddle, atom for atom."""
    monkeypatch.setattr(refinement_module, "PointSetRegistration", _IdentityPSR)
    monkeypatch.setattr(kmc_module, "Reconstruction", _RecordingReconstruction)
    _RecordingReconstruction.calls = []

    config = _config()
    system = System(
        types=np.array(["Ni"] * len(_MIN)),
        positions=_MIN.copy(),
        cell=_CELL.copy(),
        pbc=np.array([True, True, True]),
        index=np.arange(len(_MIN)),
    )
    atomic_environment = Mock()
    atomic_environment.get_atoms_with_id.return_value = [0]

    # Step N: refine the event on central atom 0 and store the active row.
    refinement = Refinement(
        config,
        Mock(),
        system,
        _NeighborsList(_REFINED_ORDER),
        atomic_environment,
        Mock(),
    )
    refinement.execute(_reference_events(), _TOTAL_ENERGY)
    kmc = KMC(config)
    kmc.system = system
    kmc.active_table = ActiveEventTable(config)
    kmc.add_active_events(refinement.get_successes_results())
    assert len(kmc.active_table.table) == 1

    # Step N+1: the row is carried over and the neighbour list was rebuilt.
    kmc.neighbors_list = _NeighborsList(rebuilt_order)
    kmc._reconstruction_active_event(0, kmc.active_table)

    (call,) = _RecordingReconstruction.calls
    np.testing.assert_allclose(call["system"], _SADDLE)
    np.testing.assert_allclose(call["min2"], _FINAL[call["neighbors"]])
    np.testing.assert_allclose(system.positions, _MIN)
