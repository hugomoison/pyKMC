"""Axis-aligned box region."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from ase.geometry import wrap_positions

from .base import RegionStrategy


class BoxConfig(Protocol):
    """Parameters of an axis-aligned box.

    Attributes
    ----------
    lo : Sequence[float]
        Lower corner [xlo, ylo, zlo] (Å).
    hi : Sequence[float]
        Upper corner [xhi, yhi, zhi] (Å).

    """

    lo: Sequence[float]
    hi: Sequence[float]


class Box(RegionStrategy):
    """Positions with ``lo <= x, y, z <= hi`` (bounds included).

    When a cell is given, positions are first wrapped back into the cell, so
    the box is expected to lie inside it.
    """

    name = "box"

    def __init__(self, config: BoxConfig) -> None:
        self.config = config

    def contains(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Return True for positions inside the box."""
        positions = np.asarray(positions, dtype=float)
        if cell is not None:
            positions = wrap_positions(positions, cell, pbc=True)
        lo = np.asarray(self.config.lo, dtype=float)
        hi = np.asarray(self.config.hi, dtype=float)
        return np.all((positions >= lo) & (positions <= hi), axis=1)
