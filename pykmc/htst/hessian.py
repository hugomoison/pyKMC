"""Finite-difference mass-weighted partial Hessian from a forces callable.

Engine-agnostic: the caller supplies ``forces_fn(positions) -> (N, 3)`` forces.
The LAMMPS binding lives in lammps_operations.compute_event_prefactors.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def mass_weighted_partial_hessian(
    forces_fn: Callable[[np.ndarray], np.ndarray],
    positions: np.ndarray,
    masses: np.ndarray,
    free_indices: np.ndarray,
    dx: float,
) -> np.ndarray:
    """Assemble the (3F, 3F) mass-weighted partial Hessian by central differences.

    H_{ab} = -(F_a(+dx_b) - F_a(-dx_b)) / (2 dx) / sqrt(m_a m_b),
    over the free atoms only (boundary atoms held fixed).

    Parameters
    ----------
    forces_fn : Callable
        Maps full (N, 3) positions to full (N, 3) forces.
    positions : np.ndarray
        (N, 3) reference (relaxed) positions.
    masses : np.ndarray
        (N,) atomic masses (amu), per atom.
    free_indices : np.ndarray
        Indices of free atoms (length F).
    dx : float
        Finite-difference step (Angstrom).

    Returns
    -------
    np.ndarray
        (3F, 3F) symmetric mass-weighted Hessian.

    """
    free = np.asarray(free_indices, dtype=int)
    n_free = len(free)
    hessian = np.zeros((3 * n_free, 3 * n_free), dtype=float)
    sqrt_m = np.sqrt(masses[free])

    for j_local, j_atom in enumerate(free):
        for jc in range(3):
            col = 3 * j_local + jc
            pos_p = positions.copy()
            pos_m = positions.copy()
            pos_p[j_atom, jc] += dx
            pos_m[j_atom, jc] -= dx
            f_p = forces_fn(pos_p)[free].reshape(-1)
            f_m = forces_fn(pos_m)[free].reshape(-1)
            hessian[:, col] = -(f_p - f_m) / (2.0 * dx)

    # mass-weight: H_ab / sqrt(m_a m_b)
    inv = np.repeat(1.0 / sqrt_m, 3)
    hessian = hessian * np.outer(inv, inv)
    # symmetrize to clean up finite-difference asymmetry
    return 0.5 * (hessian + hessian.T)
