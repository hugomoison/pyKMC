"""Spherical region."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from ase.geometry import find_mic

from .base import RegionStrategy


class SphereConfig(Protocol):
    """Parameters of a sphere.

    Exactly one of `center` and `center_atom` must be set. An attribute that
    is not defined counts as None.

    Attributes
    ----------
    center : Sequence[float] or None
        Fixed center [x, y, z] (Å).
    center_atom : int or None
        0-based index of the atom used as center. Its position is read each
        time the region is evaluated, so the sphere follows the atom.
    radius : float
        Radius (Å).

    """

    center: Sequence[float] | None
    center_atom: int | None
    radius: float


class Sphere(RegionStrategy):
    """Positions at a distance ``<= radius`` from the center."""

    name = "sphere"

    def __init__(self, config: SphereConfig) -> None:
        center = getattr(config, "center", None)
        center_atom = getattr(config, "center_atom", None)
        if (center is None) == (center_atom is None):
            raise ValueError(
                f"{self.name} region requires exactly one of center and center_atom."
            )
        self.config = config
        self._center = None if center is None else np.asarray(center, dtype=float)
        self._center_atom = center_atom

    def center(self, positions: np.ndarray) -> np.ndarray:
        """Return the center: the fixed one, or the position of `center_atom`."""
        if self._center is not None:
            return self._center
        if not 0 <= self._center_atom < len(positions):
            raise IndexError(
                f"center_atom={self._center_atom} is out of range for "
                f"{len(positions)} atoms."
            )
        return positions[self._center_atom]

    def distances(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Distance of each position to the center, minimum image if `cell` is given."""
        positions = np.asarray(positions, dtype=float)
        diff = positions - self.center(positions)
        if cell is None:
            return np.linalg.norm(diff, axis=1)
        _, dists = find_mic(diff, cell, pbc=True)
        return dists

    def contains(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Return True for positions within `radius` of the center."""
        return self.distances(positions, cell) <= self.config.radius
