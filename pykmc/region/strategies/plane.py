"""Half-space region bounded by a plane normal to x, y or z."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from ase.geometry import wrap_positions

from .base import RegionStrategy

_AXIS = {"x": 0, "y": 1, "z": 2}


class PlaneConfig(Protocol):
    """Parameters of a half-space.

    Attributes
    ----------
    normal : {"x", "y", "z"}
        Axis normal to the plane.
    threshold : float
        Position of the plane along `normal` (Å).
    side : {"above", "below"}
        Kept half-space. Both include the plane itself.

    """

    normal: Literal["x", "y", "z"]
    threshold: float
    side: Literal["above", "below"]


class Plane(RegionStrategy):
    """Positions on the `side` of the plane, the plane itself included.

    When a cell is given, positions are first wrapped back into the cell.
    """

    name = "plane"

    def __init__(self, config: PlaneConfig) -> None:
        if config.side not in ("above", "below"):
            raise ValueError(
                f"plane region requires side 'above' or 'below', got {config.side!r}"
            )
        self.config = config

    def contains(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Return True for positions on `side` of the plane."""
        positions = np.asarray(positions, dtype=float)
        if cell is not None:
            positions = wrap_positions(positions, cell, pbc=True)
        coord = positions[:, _AXIS[self.config.normal]]
        if self.config.side == "above":
            return coord >= self.config.threshold
        return coord <= self.config.threshold
