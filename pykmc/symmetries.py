"""Module for detecting unique symmetry of an atomic environment based on atomic displacements."""

import ira_mod
import numpy as np
from .system import Configuration


def unique_symmetries(
    initial_configuration: Configuration,
    final_configuration: Configuration,
    sym_thr: float,
    full: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Identify the unique symmetry operations of an event based on atomic displacements.

    This function computes all the symmetry operations of the initial configuration using `ira_mod`,
    then filters out equivalent operations by comparing the associated atomic displacements after applying the symmetries.

    Parameters
    ----------
    initial_configuration : Configuration
        Initial atomic types/positions/cell.
    final_configuration : Configuration
        Final atomic types/positions/cell.
    sym_thr : float
        Symmetry tolerance threshold for the `ira_mod` symmetry detection.
    full : bool, optional
        When True, symmetry detection respects `initial_configuration.types`
        (fewer symmetries for multi-element systems); by default False (grey,
        species-blind detection).

    Returns
    -------
    sym_matrix : list[np.ndarray]
        Arrays of unique 3,3 symmetry rotation matrices, including the identity. Shape: (M, 3, 3).
    sym_perm : list[np.ndarray]
        Arrays of corresponding atom index permutations for each symmetry. Shape: (M, N),
        where M is the number of unique symmetries and N the number of atoms.

    """
    initial_positions = initial_configuration.positions
    final_positions = final_configuration.positions

    # Find all symmetries of initial_positions
    nat = len(initial_configuration)
    typ = list(initial_configuration.types) if full else nat * [1]

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
