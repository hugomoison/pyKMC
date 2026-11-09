"""The move atom of a searched event is picked from minimum-image displacements.

pARTn returns wrapped coordinates, so an atom that hops across a periodic
boundary shows a raw displacement of about one box length. Picking the move atom
from raw displacements discarded that atom and stored the event around a
relaxing neighbour, a second description of a hop already in the table.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lammps")
pytest.importorskip("pypARTn")

from pykmc.activevolume import active_volume as av  # noqa: E402

_CELL = np.diag([10.0, 10.0, 10.0])
_CONFIG = SimpleNamespace(atomicenvironment=SimpleNamespace(rcut=6.5))

# Local (active-volume) indices
_HOPPER, _NEIGHBOUR, _ON_PLANE, _STILL = 0, 1, 2, 3


class _FakeArtn:
    """Serve pARTn's ``tau_*`` arrays from a dict."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def extract(self, name: str) -> np.ndarray:
        return self._arrays[name]


def _boundary_hop() -> dict[str, np.ndarray]:
    """One atom hops 2 Angstrom along y, through the y = 0 face of the box."""
    min1 = np.array(
        [
            [5.0, 9.6, 5.0],  # hopper: 1.0 from the saddle through the boundary
            [5.0, 2.0, 5.0],  # neighbour: relaxes by 0.17
            [0.0, 5.0, 5.0],  # sits on the x = 0 face, written as x = 0
            [7.0, 7.0, 7.0],  # does not move
        ]
    )
    saddle = min1.copy()
    saddle[_HOPPER] = [5.0, 0.6, 5.0]
    saddle[_NEIGHBOUR] = [5.0, 2.17, 5.0]
    saddle[_ON_PLANE] = [10.0, 5.0, 5.0]  # same site, written as x = L
    min2 = saddle.copy()
    min2[_HOPPER] = [5.0, 1.6, 5.0]
    return {"tau_min1": min1, "tau_sad": saddle, "tau_min2": min2}


def test_move_atom_is_the_atom_that_hops_across_the_boundary() -> None:
    """The hopping atom is found although its raw displacement is 9 Angstrom."""
    atom_map = np.array([7, 3, 5, 1])
    positions = np.zeros((8, 3))

    *_, index_move = av.position_results_AV(
        _CONFIG, _FakeArtn(_boundary_hop()), atom_map, positions, _CELL
    )

    assert index_move == atom_map[_HOPPER]


def test_move_atom_ignores_an_atom_rewrapped_onto_the_opposite_face() -> None:
    """A site written as 0 in one array and L in the other has not moved."""
    arrays = _boundary_hop()
    arrays["tau_sad"][_HOPPER] = arrays["tau_min1"][_HOPPER]  # nobody hops
    atom_map = np.array([7, 3, 5, 1])

    *_, index_move = av.position_results_AV(
        _CONFIG, _FakeArtn(arrays), atom_map, np.zeros((8, 3)), _CELL
    )

    assert index_move == atom_map[_NEIGHBOUR]
