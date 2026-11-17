"""Spherical shell region."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .sphere import Sphere, SphereConfig


class ShellConfig(SphereConfig, Protocol):
    """Parameters of a spherical shell.

    Same as :class:`SphereConfig` (`center` or `center_atom`, `radius` as the
    outer radius), plus:

    Attributes
    ----------
    inner_radius : float
        Inner radius (Å).

    """

    inner_radius: float


class Shell(Sphere):
    """Positions at a distance in ``(inner_radius, radius]`` from the center.

    The inner bound is excluded, so a ``Sphere`` of radius ``r1`` and a
    ``Shell`` of radii ``(r1, r2)`` split a ``Sphere`` of radius ``r2`` without
    overlap.
    """

    name = "shell"

    def __init__(self, config: ShellConfig) -> None:
        super().__init__(config)

    def contains(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Return True for positions with ``inner_radius < d <= radius``."""
        dists = self.distances(positions, cell)
        return (dists > self.config.inner_radius) & (dists <= self.config.radius)
