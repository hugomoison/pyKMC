"""Atoms present and displaced in an HTST Hessian calculation."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

from pykmc.utils.geometry import av_membership


@dataclass(frozen=True)
class HessianRegion:
    """Atom sets of one Hessian calculation.

    Attributes
    ----------
    present : np.ndarray
        Sorted global indices of the atoms built into the scratch engine.
    displaced : np.ndarray
        Sorted positions within `present` of the atoms the Hessian displaces. Their
        global indices are `present[displaced]`. The other present atoms stay fixed.
    pbc : tuple[bool, bool, bool]
        Periodic boundary conditions of the scratch box.
    n_zero_modes : int
        3 (the translations) when every atom of the system is displaced, else 0.

    """

    present: np.ndarray
    displaced: np.ndarray
    pbc: tuple[bool, bool, bool]
    n_zero_modes: int


def hessian_mode(config: Any) -> Literal["av", "reduced", "full"]:
    """Return the Hessian region mode selected by the configuration.

    'av' when the active volume is on (the Hessian region is the active volume),
    'full' when `free_radius` is 'full' (whole system), else 'reduced' (a sphere of
    radius `free_radius` and its frozen shell).
    """
    if config.control.active_volume:
        return "av"
    if config.rateconstant.free_radius == "full":
        return "full"
    return "reduced"


def _as_pbc(pbc: Any) -> tuple[bool, bool, bool]:
    return (bool(pbc[0]), bool(pbc[1]), bool(pbc[2]))


def _minimum_image(delta: np.ndarray, cell: np.ndarray, pbc: Any) -> np.ndarray:
    """Apply the minimum image convention to `delta` in place (orthogonal cell)."""
    for axis in range(3):
        if pbc[axis]:
            length = cell[axis, axis]
            delta[:, axis] -= length * np.round(delta[:, axis] / length)
    return delta


def build_region(
    mode: Literal["av", "reduced", "full"],
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Any,
    center_index: Optional[int] = None,
    r_displaced: Optional[float] = None,
    r_present: Optional[float] = None,
) -> HessianRegion:
    """Select the atoms present and displaced in a Hessian calculation.

    Parameters
    ----------
    mode : {'av', 'reduced', 'full'}
        'av': the active volume around `center_index`, by `av_membership` (the
        rule of `define_AV`, always fully periodic). 'reduced': a sphere around
        `center_index`. 'full': every atom.
    positions : np.ndarray
        Atomic positions, shape (N, 3). Only N is used in 'full' mode.
    cell : np.ndarray
        Simulation cell (orthogonal).
    pbc : sequence of bool
        Periodic boundary conditions of the system.
    center_index : int, optional
        Atom the distances are measured from.
    r_displaced : float, optional
        Atoms within this distance of the centre are displaced.
    r_present : float, optional
        Atoms within this distance of the centre are present.

    Returns
    -------
    HessianRegion
        The atom sets of the calculation.

    """
    n_atoms = len(positions)
    if mode == "full":
        everything = np.arange(n_atoms, dtype=np.int64)
        return HessianRegion(everything, everything.copy(), _as_pbc(pbc), 3)

    positions = np.asarray(positions, dtype=float)
    if mode == "av":
        if center_index is None or r_displaced is None or r_present is None:
            raise ValueError("'av' mode needs center_index, r_displaced and r_present")
        # the active volume itself: the rule of define_AV, periodic on every axis
        inner, buffer = av_membership(
            positions, center_index, r_displaced, r_present, cell
        )
        present = np.union1d(inner, buffer)
        displaced = np.searchsorted(present, inner).astype(np.int64)
        return HessianRegion(
            present,
            displaced,
            (True, True, True),
            3 if displaced.size == n_atoms else 0,
        )

    pbc = _as_pbc(pbc)
    delta = positions - positions[center_index]
    distances = np.linalg.norm(_minimum_image(delta, np.asarray(cell), pbc), axis=1)
    present = np.flatnonzero(distances <= r_present).astype(np.int64)
    displaced = np.flatnonzero(distances[present] <= r_displaced).astype(np.int64)
    return HessianRegion(present, displaced, pbc, 3 if displaced.size == n_atoms else 0)


def moving_atom(
    min_positions: np.ndarray, saddle_positions: np.ndarray, cell: np.ndarray, pbc: Any
) -> int:
    """Return the index of the atom that moves most from the minimum to the saddle.

    Displacements follow the minimum image convention on the periodic axes.
    """
    delta = np.asarray(saddle_positions, dtype=float) - np.asarray(
        min_positions, dtype=float
    )
    distances = np.linalg.norm(_minimum_image(delta, np.asarray(cell), pbc), axis=1)
    return int(np.argmax(distances))


def wrapped_on_open_axis(
    region: HessianRegion,
    geometries: Any,
    start_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Any,
) -> bool:
    """Tell whether a geometry was wrapped across a non-periodic axis.

    The geometries of a reference event are recentred with a periodic wrap on every
    axis. Along a non-periodic axis, the present atoms that were wrapped sit one
    cell length away from the others, at positions the system never had.

    Parameters
    ----------
    region : HessianRegion
        The region of the calculation.
    geometries : sequence of np.ndarray
        Full-size geometries of the event (saddle and minima).
    start_positions : np.ndarray
        Positions of the system the event was searched from, shape (N, 3).
    cell : np.ndarray
        Simulation cell (orthogonal).
    pbc : sequence of bool
        Periodic boundary conditions of the system.

    """
    cell = np.asarray(cell)
    for axis in range(3):
        if pbc[axis]:
            continue
        for geometry in geometries:
            shift = (
                geometry[region.present, axis] - start_positions[region.present, axis]
            )
            if shift.max() - shift.min() > cell[axis, axis] / 2:
                return True
    return False


def frozen_overlap(region: HessianRegion, frozen_ids: np.ndarray) -> int:
    """Count the frozen atoms among the displaced atoms of `region`.

    Frozen atoms that are only present (never displaced) do not count.
    """
    return int(np.intersect1d(region.present[region.displaced], frozen_ids).size)


def full_hessian_mb(n_atoms: int) -> float:
    """Return the size (MB = 1e6 bytes) of the Hessian of a whole system."""
    return 8 * (3 * n_atoms) ** 2 / 1e6


def region_record(config: Any, shell_width: Optional[float]) -> str:
    """Return the region settings as stored with each nu0.

    One of 'av rmov=<rmov> ract=<ract>',
    'reduced free_radius=<free_radius> shell=<shell_width>' and 'full'.
    """
    mode = hessian_mode(config)
    if mode == "av":
        return f"av rmov={config.activevolume.rmov:g} ract={config.activevolume.ract:g}"
    if mode == "reduced":
        return (
            f"reduced free_radius={config.rateconstant.free_radius:g} "
            f"shell={shell_width:g}"
        )
    return "full"
