"""Spherical shell region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .sphere import Sphere, SphereConfig, SphereParams


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


@dataclass(frozen=True, kw_only=True)
class ShellParams(SphereParams):
    """Concrete :class:`ShellConfig`, to build a shell from code.

    Example: ``ShellParams(center_atom=12, radius=6.0, inner_radius=3.0)``.
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
        inner_radius = getattr(config, "inner_radius", None)
        if inner_radius is None or not 0 <= inner_radius < config.radius:
            raise ValueError(
                f"shell region requires 0 <= inner_radius < radius, got "
                f"inner_radius={inner_radius!r} and radius={config.radius!r}."
            )

    def contains(
        self, positions: np.ndarray, cell: np.ndarray | None = None
    ) -> np.ndarray:
        """Return True for positions with ``inner_radius < d <= radius``."""
        dists = self.distances(positions, cell)
        return (dists > self.config.inner_radius) & (dists <= self.config.radius)
