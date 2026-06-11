"""Module for detecting unique symmetry of an atomic environment based on atomic displacements."""

import ira_mod
import numpy as np


def unique_symmetries(
    initial_positions: np.ndarray, final_positions: np.ndarray, sym_thr: float,
    types: list = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Identify the unique symmetry operations of an event based on atomic displacements.

    This function computes all the symmetry operations of the initial configuration using `ira_mod`,
    then filters out equivalent operations by comparing the associated atomic displacements after applying the symmetries.

    Parameters
    ----------
    initial_positions : np.ndarray
        Initial atomic positions (N, 3).
    final_positions : np.ndarray
        Final atomic positions (N, 3).
    sym_thr : float
        Symmetry tolerance threshold for the `ira_mod` symmetry detection.
    types : list, optional
        Element types for each atom. When provided, symmetry detection respects
        element types (fewer symmetries for multi-element systems).

    Returns
    -------
    sym_matrix : list[np.ndarray]
        Arrays of unique 3,3 symmetry rotation matrices, including the identity. Shape: (M, 3, 3).
    sym_perm : list[np.ndarray]
        Arrays of corresponding atom index permutations for each symmetry. Shape: (M, N),
        where M is the number of unique symmetries and N the number of atoms.

    """
    # Find all symmetries of initial_positions
    nat = len(initial_positions)
    typ = list(types) if types is not None else nat * [1]

    sofi = ira_mod.SOFI()
    sym = sofi.compute(nat, typ, initial_positions, sym_thr)  # sym data ira object

    # Find unique symmetries
    # Displacment event matrix
    displacements = initial_positions - final_positions

    unique_displacements = [displacements]
    unique_sym_index = []

    for i in range(len(sym.matrix)):  # Loop over all symmetries
        is_duplicated = False
        # Apply symmetry to displacements event matrix
        new_displacements = displacements @ sym.matrix[i].T
        new_displacements = new_displacements[sym.perm[i]]

        for disp in unique_displacements:  # Check if alreay in unique_displacements
            if np.allclose(disp, new_displacements, atol=1e-2, rtol=0):
                is_duplicated = True
                break

        if not is_duplicated:  # if new unique symmetry
            unique_sym_index.append(i)  # add symmtry to unique
            unique_displacements.append(new_displacements)

    # unique symetries and add identity :
    sym_matrix = np.concatenate(
        [[np.eye(3)]] + [[sym.matrix[i]] for i in unique_sym_index], axis=0
    )
    # associated permutation :
    sym_perm = np.array([np.arange(nat)] + [sym.perm[i] for i in unique_sym_index])
    return sym_matrix, sym_perm
