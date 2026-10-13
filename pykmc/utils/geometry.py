"""Module containing function to apply geometric transformations."""

__all__ = [
    "wrap_configuration",
    "transform_positions",
    "translate",
    "push_towards",
    "compute_distances",
    "count_moved_atoms",
    "compute_delr_max",
    "compute_delr_l2",
    "per_atom_displacement",
    "minimum_image_distance",
    "minimum_image_vector",
    "image_shift",
    "unwrap_around",
]
import ase.geometry
import numpy as np
from ..system import Configuration


def wrap_configuration(configuration: Configuration) -> Configuration:
    """Return `configuration` with positions wrapped into its own cell (orthorhombic, PBC)."""
    positions = ase.geometry.wrap_positions(
        positions=configuration.positions, cell=configuration.cell, pbc=True
    )
    positions[positions < 0] = 0
    return Configuration(positions=positions, types=configuration.types, cell=configuration.cell)


def transform_positions(
    configuration: Configuration,
    transformation_matrix: np.ndarray,
    translation_matrix: np.ndarray,
    permutation_matrix: np.ndarray,
    wrap: bool = True,
) -> Configuration:
    """Apply rotation, translation and permutation to a `Configuration`.

    Parameters
    ----------
    configuration : Configuration
        Configuration to transform.
    transformation_matrix : np.ndarray
        transformation matrix (e.g. rotation).
    translation_matrix : np.ndarray
        translation matrix
    permutation_matrix : np.ndarray
        permutation matrix.
    wrap : bool, optional
        Wrap the transformed positions back into `configuration.cell` (orthorhombic,
        PBC), by default True.

    Returns
    -------
    Configuration
        The transformed configuration; `types` is permuted the same way as
        `positions` so the two stay aligned.

    """
    positions = configuration.positions @ transformation_matrix.T + translation_matrix
    positions = positions[permutation_matrix]
    types = np.asarray(configuration.types)[permutation_matrix]
    result = Configuration(positions=positions, types=types, cell=configuration.cell)
    return wrap_configuration(result) if wrap else result


def translate(
    configuration: Configuration, displacement: np.ndarray, wrap: bool = True
) -> Configuration:
    """Translate a `Configuration`'s positions by a displacement vector.

    Parameters
    ----------
    configuration : Configuration
        Configuration to translate.
    displacement : np.ndarray
        Displacement vector of shape (3,) to be added to each position.
    wrap : bool, optional
        Wrap the translated positions back into `configuration.cell` (orthorhombic,
        PBC), by default True.

    Returns
    -------
    Configuration
        The translated configuration, same `types`/`cell` as the input.

    """
    result = configuration + displacement
    return wrap_configuration(result) if wrap else result


def push_towards(
    current: Configuration, target: Configuration, fraction: float = 0.1, wrap: bool = True
) -> Configuration:
    """Move `current` a `fraction` of the way towards `target` (PBC-aware).

    Parameters
    ----------
    current : Configuration
        Starting configuration; its `cell` is used for the PBC-aware displacement
        and (if `wrap`) the final wrap.
    target : Configuration
        Target configuration to move towards.
    fraction : float, optional
        Fraction of the (PBC-unwrapped) displacement to apply, by default 0.1.
    wrap : bool, optional
        Wrap the new positions back into `current.cell`, by default True.

    Returns
    -------
    Configuration
        `current`, moved `fraction` of the way towards `target`.

    """
    cell = current.cell
    box = np.diag(cell)
    displacement = (target - current).positions
    displacement -= np.round(displacement / box) * box

    result = current + fraction * displacement

    return wrap_configuration(result) if wrap else result


def compute_distances(
    configuration_1: Configuration, configuration_2: Configuration, wrap: bool = True
) -> np.ndarray:
    """Return per-atom distances between two configurations.

    `wrap` controls whether the periodic minimum-image convention is used
    (via `configuration_1.cell`) when computing distances, by default True.
    """
    displacements = configuration_2.positions - configuration_1.positions

    if wrap:
        _wrapped_displacements, distances = ase.geometry.find_mic(
            displacements, cell=configuration_1.cell, pbc=True
        )
        return np.asarray(distances)

    return np.linalg.norm(displacements, axis=1)


def compute_delr(
    configuration_1: Configuration, configuration_2: Configuration, wrap: bool = True
):
    return compute_delr_max(configuration_1, configuration_2, wrap=wrap)


def count_moved_atoms(
    configuration_1: Configuration, configuration_2: Configuration, threshold: float, wrap: bool = True
) -> int:
    """Return the number of atoms displaced by more than ``threshold``."""
    distances = compute_distances(configuration_1, configuration_2, wrap=wrap)
    return int(np.count_nonzero(distances > threshold))


def compute_delr_max(
    configuration_1: Configuration, configuration_2: Configuration, wrap: bool = True
) -> float:
    distances = compute_distances(configuration_1, configuration_2, wrap=wrap)
    if distances.size == 0:
        return 0.0
    return float(np.max(distances))


def compute_delr_l2(
    configuration_1: Configuration, configuration_2: Configuration, wrap: bool = True
) -> float:
    distances = compute_distances(configuration_1, configuration_2, wrap=wrap)
    return float(np.linalg.norm(distances))


def per_atom_displacement(
    configuration_pre: Configuration, configuration_post: Configuration, wrap: bool = True
) -> np.ndarray:
    """Per-atom displacement magnitude, minimum-image PBC-aware by default (orthorhombic).

    `wrap` controls whether each component is minimum-image corrected via
    `configuration_pre.cell`, by default True.
    """
    disp = configuration_post.positions - configuration_pre.positions
    if wrap:
        cell_lengths = np.linalg.norm(configuration_pre.cell, axis=1)
        for i in range(3):
            disp[:, i] -= cell_lengths[i] * np.round(disp[:, i] / cell_lengths[i])
    return np.linalg.norm(disp, axis=1)


def minimum_image_vector(
    position_a: np.ndarray, position_b: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """PBC minimum-image displacement vector position_b - position_a (orthorhombic)."""
    dvec = position_b - position_a
    cell_lengths = np.linalg.norm(cell, axis=1)
    for i in range(3):
        dvec[i] -= cell_lengths[i] * np.round(dvec[i] / cell_lengths[i])
    return dvec


def minimum_image_distance(
    position_a: np.ndarray, position_b: np.ndarray, cell: np.ndarray
) -> float:
    """PBC minimum-image Euclidean distance between two positions (orthorhombic)."""
    return float(np.linalg.norm(minimum_image_vector(position_a, position_b, cell)))


def image_shift(
    positions: np.ndarray, center: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """Per-atom translation bringing positions into the periodic image closest to center.

    Orthorhombic cells only. Returned separately from :func:`unwrap_around` so that
    several configurations of the same cluster can share one shift, which keeps the
    displacements between them true vectors.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions with shape (N, 3).
    center : np.ndarray
        Reference position of shape (3,), typically the cluster's central atom.
    cell : np.ndarray
        Simulation box cell (3x3).

    Returns
    -------
    np.ndarray
        Translations of shape (N, 3), each a whole number of cell vectors.

    """
    box = np.diag(cell)
    return np.round((center - positions) / box) * box


def unwrap_around(
    configuration: Configuration, center: np.ndarray | Configuration
) -> Configuration:
    """Return `configuration` with positions in the periodic image closest to center (orthorhombic).

    Parameters
    ----------
    configuration : Configuration
        The configuration to unwrap; its `cell` is used for the periodic image search.
    center : np.ndarray | Configuration
        Reference position(s) to unwrap around: either a single point of shape
        (3,), typically the cluster's central atom, or a per-atom reference
        array of shape (N, 3) -- e.g. a previously-unwrapped cluster, so each
        atom unwraps relative to its own counterpart instead of one shared
        centre. Passing a `Configuration` here is equivalent to passing its
        `.positions`.

    Returns
    -------
    Configuration
        The unwrapped configuration, same `types`/`cell` as the input.

    """
    center_positions = center.positions if isinstance(center, Configuration) else center
    positions = configuration.positions
    return Configuration(
        positions=positions + image_shift(positions, center_positions, configuration.cell),
        types=configuration.types,
        cell=configuration.cell,
    )
