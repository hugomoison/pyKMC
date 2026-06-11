"""Module containing function to apply geometric transformations."""

__all__ = [
    "transform_positions",
    "translate",
    "push_towards",
    "compute_delr",
    "per_atom_displacement",
    "minimum_image_distance",
]
import ase.geometry
import numpy as np


def transform_positions(
    positions: np.ndarray,
    transformation_matrix: np.ndarray,
    translation_matrix: np.ndarray,
    permutation_matrix: np.ndarray,
) -> np.ndarray:
    """Apply rotation, translation and permutation to all positions.

    Parameters
    ----------
    positions : np.ndarray
        positions to transform.
    transformation_matrix : np.ndarray
        transformation matrix (e.g. rotation).
    translation_matrix : np.ndarray
        translation matrix
    permutation_matrix : np.ndarray
        permutation matrix.

    Returns
    -------
    np.ndarray
        The transformed positions.

    """
    transform_positions = positions @ transformation_matrix.T + translation_matrix
    return transform_positions[permutation_matrix]


def translate(
    positions: np.ndarray, displacement: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Translate atomic positions by a displacement vector and apply periodic wrapping.

    Parameters
    ----------
    positions : np.ndarray
        Array of atomic positions with shape (N, 3), where N is the number of atoms.
    displacement : np.ndarray
        Displacement vector of shape (3,) to be added to each position.
    cell : np.ndarray
        Simulation cell (3x3 matrix) defining the periodic boundaries.

    Returns
    -------
    np.ndarray
        Translated and wrapped atomic positions, same shape as the input `positions`.

    """
    positions += displacement
    positions = ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=True)
    positions[positions < 0] = 0
    return positions


def push_towards(current_positions, target_positions, fraction=0.1, cell=None):
    displacement = target_positions - current_positions

    if cell is not None:
        box = np.diag(cell)
        displacement -= np.round(displacement / box) * box
        # unwrap target
        target_positions_unwrapped = current_positions + displacement
    else:
        target_positions_unwrapped = target_positions

    new_positions = current_positions + fraction * (
        target_positions_unwrapped - current_positions
    )

    if cell is not None:
        new_positions = ase.geometry.wrap_positions(
            positions=new_positions, cell=cell, pbc=[True, True, True]
        )
    return new_positions


def compute_delr(positions_1, positions_2, cell=None):
    displacements = positions_2 - positions_1

    if cell is not None:
        cell_lengths = np.linalg.norm(cell, axis=1)

        # apply pbc

        for i in range(3):
            displacements[:, i] -= cell_lengths[i] * np.round(
                displacements[:, i] / cell_lengths[i]
            )

    # Calcul des normes des déplacements
    distances = np.linalg.norm(displacements, axis=1)

    # Retour du déplacement maximum
    delr = np.max(distances)

    return delr


def per_atom_displacement(
    positions_pre: np.ndarray,
    positions_post: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Per-atom PBC-aware displacement magnitude (orthorhombic minimum-image).

    Same minimum-image trick as `compute_delr`, but returns the full per-atom
    array of Euclidean distances instead of just the maximum.

    Parameters
    ----------
    positions_pre : np.ndarray
        Shape (N, 3) positions before the displacement.
    positions_post : np.ndarray
        Shape (N, 3) positions after the displacement.
    cell : np.ndarray
        3x3 simulation cell (orthorhombic; row-wise lattice vectors).

    Returns
    -------
    np.ndarray
        Shape (N,) of per-atom displacement magnitudes in Angstroms.

    """
    disp = positions_post - positions_pre
    cell_lengths = np.linalg.norm(cell, axis=1)
    for i in range(3):
        disp[:, i] -= cell_lengths[i] * np.round(disp[:, i] / cell_lengths[i])
    return np.linalg.norm(disp, axis=1)


def minimum_image_distance(
    position_a: np.ndarray,
    position_b: np.ndarray,
    cell: np.ndarray,
) -> float:
    """PBC minimum-image Euclidean distance between two positions (orthorhombic).

    Single-pair counterpart of `per_atom_displacement`: applies the same
    per-axis minimum-image wrap to the separation vector and returns its norm.

    Parameters
    ----------
    position_a : np.ndarray
        Shape (3,) first position.
    position_b : np.ndarray
        Shape (3,) second position.
    cell : np.ndarray
        3x3 simulation cell (orthorhombic; row-wise lattice vectors).

    Returns
    -------
    float
        Minimum-image distance in Angstroms.

    """
    dvec = position_b - position_a
    cell_lengths = np.linalg.norm(cell, axis=1)
    for i in range(3):
        dvec[i] -= cell_lengths[i] * np.round(dvec[i] / cell_lengths[i])
    return float(np.linalg.norm(dvec))
