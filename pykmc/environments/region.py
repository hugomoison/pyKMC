"""RegionParameters-based atomic environment classification."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pykmc.parameters import RegionParameters
    from pykmc.system import Configuration


def region(r: RegionParameters, configuration: Configuration) -> list[str]:
    """Classify each atom as ``'in'`` or ``'out'``.

    Union semantics: an atom is ``'in'`` if it matches any of
    ``r.types``, ``r.indices``, or falls inside the geometric region.

    Parameters
    ----------
    r : RegionParameters
        Selection criteria (geometry, types, indices).
    configuration : Configuration
        Current atom types/positions/cell.

    Returns
    -------
    list[str]
        ``'in'`` or ``'out'`` classification for each atom.
    """
    positions = configuration.positions
    n = len(positions)
    selected: set[int] = set()

    if r.types:
        type_set = set(r.types)
        selected.update(i for i, t in enumerate(configuration.types) if t in type_set)

    if r.indices:
        selected.update(r.indices)

    if r.region_type is not None:
        selected.update(_resolve_geometric(r, positions))

    return ["in" if i in selected else "out" for i in range(n)]


def _resolve_geometric(r: RegionParameters, positions: np.ndarray) -> set[int]:
    """Return the set of atom indices that fall inside/outside the geometric region."""
    n = len(positions)

    if r.region_type == "sphere":
        c = np.array(r.center)
        dists = np.linalg.norm(positions - c, axis=1)
        base = set(np.where(dists <= r.radius)[0].tolist())

    elif r.region_type == "shell":
        c = np.array(r.center)
        dists = np.linalg.norm(positions - c, axis=1)
        base = set(
            np.where((dists >= r.inner_radius) & (dists <= r.radius))[0].tolist()
        )

    elif r.region_type == "box":
        lo = r.lo
        hi = r.hi
        mask = (
            (positions[:, 0] >= lo[0])
            & (positions[:, 0] <= hi[0])
            & (positions[:, 1] >= lo[1])
            & (positions[:, 1] <= hi[1])
            & (positions[:, 2] >= lo[2])
            & (positions[:, 2] <= hi[2])
        )
        base = set(np.where(mask)[0].tolist())

    elif r.region_type == "plane":
        ax = {"x": 0, "y": 1, "z": 2}[r.normal]
        if r.side == "above":
            return set(np.where(positions[:, ax] >= r.threshold)[0].tolist())
        else:  # "below"
            return set(np.where(positions[:, ax] <= r.threshold)[0].tolist())

    if r.side == "outside":
        return set(range(n)) - base
    return base
