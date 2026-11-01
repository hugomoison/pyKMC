"""Module containing function to apply geometric transformations."""

__all__ = [
    "transform_positions",
    "translate",
    "push_towards",
    "compute_delr",
    "per_atom_displacement",
    "minimum_image_distance",
    "unwrap_relative_to",
    "av_membership",
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


def unwrap_relative_to(
    positions: np.ndarray,
    reference_position: np.ndarray,
    cell: np.ndarray,
    pbc: bool = True,
) -> np.ndarray:
    """Unwrap positions into the periodic image closest to a reference position.

    A single-snapshot, reference-relative unwrap (not the trajectory unwrap
    that reconstructs continuous paths across many frames). Thin wrapper
    around `ase.geometry.find_mic`, so it works for orthorhombic and
    triclinic cells alike.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) positions to unwrap.
    reference_position : np.ndarray
        Shape (3,) position each unwrapped position is placed closest to.
    cell : np.ndarray
        3x3 simulation cell (row-wise lattice vectors).
    pbc : bool
        Periodic boundary conditions, forwarded to `find_mic`.

    Returns
    -------
    np.ndarray
        Shape (N, 3) unwrapped positions.

    """
    vectors, _ = ase.geometry.find_mic(positions - reference_position, cell, pbc=pbc)
    return reference_position + vectors


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


def av_membership(
    positions: np.ndarray,
    centre: int,
    r_inner: float,
    r_outer: float,
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the atoms around one atom into an inner sphere and a buffer shell.

    This is the active-volume membership rule: distances to atom `centre` are
    minimum-image distances from `ase.geometry.find_mic`, periodic on every axis.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 3) atomic positions.
    centre : int
        Index of the atom distances are measured from.
    r_inner : float
        Atoms within this distance are in the inner sphere.
    r_outer : float
        Atoms farther than `r_inner` and within this distance are in the buffer.
    cell : np.ndarray
        3x3 simulation cell (row-wise lattice vectors).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted indices of the inner atoms, and of the buffer atoms.

    Notes
    -----
    Each distance is the one `find_mic` gives for that atom alone, as the active
    volume always measured it. On a non-orthorhombic cell `find_mic` picks its
    algorithm from all the vectors it is given, so a call on all atoms at once
    can give a different, shorter image: there each atom is measured alone. On
    an orthorhombic cell both algorithms give the minimum image, so all atoms
    are measured at once and only the atoms within 1e-6 A of a radius, where
    rounding could move them across it, are measured again one at a time.

    """
    positions = np.asarray(positions, dtype=float)
    center = positions[centre]
    if not ase.geometry.is_orthorhombic(cell):
        distances = np.array(
            [
                np.abs(ase.geometry.find_mic(pos - center, cell, pbc=True)[1])
                for pos in positions
            ]
        )
    else:
        _, distances = ase.geometry.find_mic(positions - center, cell, pbc=True)
        distances = np.abs(distances)
        at_radius = (np.abs(distances - r_inner) < 1e-6) | (
            np.abs(distances - r_outer) < 1e-6
        )
        for i in np.flatnonzero(at_radius):
            distances[i] = np.abs(
                ase.geometry.find_mic(positions[i] - center, cell, pbc=True)[1]
            )
    inner = np.flatnonzero(distances <= r_inner).astype(np.int64)
    buffer = np.flatnonzero((distances > r_inner) & (distances <= r_outer)).astype(
        np.int64
    )
    return inner, buffer
