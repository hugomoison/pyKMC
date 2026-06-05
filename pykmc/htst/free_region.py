"""PBC-aware selection of free (movable) atoms around an event center.

Assumes an orthogonal simulation cell (matching the orthogonal-cell assumption
elsewhere in pyKMC, e.g. eventsearch position centering). For triclinic cells,
swap in ase.geometry.find_mic.
"""

from __future__ import annotations

import numpy as np


def select_free_indices(
    positions: np.ndarray,
    center_index: int,
    radius: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Return sorted indices of atoms within ``radius`` of the center atom.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) Cartesian positions.
    center_index : int
        Index of the atom at the center of the free region.
    radius : float
        Cutoff radius in Angstrom (inclusive).
    cell : np.ndarray
        (3, 3) simulation cell (orthogonal).
    pbc : np.ndarray
        (3,) booleans for periodicity per axis.

    Returns
    -------
    np.ndarray
        Sorted int array of free-atom indices (always includes ``center_index``).

    """
    delta = positions - positions[center_index]
    lengths = np.array([cell[0, 0], cell[1, 1], cell[2, 2]], dtype=float)
    for axis in range(3):
        if pbc[axis] and lengths[axis] > 0:
            delta[:, axis] -= lengths[axis] * np.round(delta[:, axis] / lengths[axis])
    dist = np.linalg.norm(delta, axis=1)
    return np.sort(np.where(dist <= radius)[0].astype(int))
