"""Axis-aligned box region."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, kw_only=True)
class BoxParams:
    """Concrete :class:`BoxConfig`, to build a box from code.

    Example: ``BoxParams(lo=(0.0, 0.0, 0.0), hi=(5.0, 5.0, 5.0))``.
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
        lo = getattr(config, "lo", None)
        hi = getattr(config, "hi", None)
        if lo is None or hi is None or len(lo) != 3 or len(hi) != 3:
            raise ValueError(
                f"box region requires lo and hi with 3 values each, got lo={lo!r} "
                f"and hi={hi!r}."
            )
        if np.any(np.asarray(lo, dtype=float) > np.asarray(hi, dtype=float)):
            raise ValueError(f"box region requires lo <= hi, got lo={lo} and hi={hi}.")
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
