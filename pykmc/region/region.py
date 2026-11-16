"""Selection of atoms by type, index and geometric region."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from .strategies import RegionStrategy

if TYPE_CHECKING:
    from pykmc.config import RegionConfig


class Region:
    """Select atoms by type, index or geometric shape, with union semantics.

    An atom is selected if its type is in `types`, OR its index is in
    `indices`, OR it lies in the shape (or outside of it when `invert` is True).

    Parameters
    ----------
    shape : RegionStrategy, optional
        Geometric shape. When None, only `types` and `indices` select atoms.
    types : Iterable[str], optional
        Chemical symbols to select.
    indices : Iterable[int], optional
        0-based atom indices to select. Indices outside ``[0, N)`` are ignored.
    invert : bool, optional
        Select the atoms outside the shape instead of inside. It does not
        apply to `types` and `indices`.

    """

    def __init__(
        self,
        shape: RegionStrategy | None = None,
        types: Iterable[str] = (),
        indices: Iterable[int] = (),
        invert: bool = False,
    ) -> None:
        self.shape = shape
        self.types = frozenset(types)
        self.indices = tuple(indices)
        self.invert = invert

    def mask(
        self,
        positions: np.ndarray,
        atom_types: list[str] | None = None,
        cell: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return which atoms are selected.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 3)
            Cartesian positions.
        atom_types : list[str], optional
            Chemical symbol of each atom. Required when `types` is not empty.
        cell : np.ndarray, shape (3, 3), optional
            Simulation cell. When given, the shape is evaluated with periodic
            boundary conditions.

        Returns
        -------
        np.ndarray, shape (N,)
            Boolean mask, True for selected atoms.

        Raises
        ------
        ValueError
            If `types` is not empty and `atom_types` is None.

        """
        n = len(positions)
        selected = np.zeros(n, dtype=bool)

        if self.types:
            if atom_types is None:
                raise ValueError("Selecting atoms by type requires atom_types.")
            selected |= np.fromiter(
                (t in self.types for t in atom_types), dtype=bool, count=n
            )

        if self.indices:
            idx = np.asarray(self.indices, dtype=int)
            selected[idx[(idx >= 0) & (idx < n)]] = True

        if self.shape is not None:
            inside = self.shape.contains(positions, cell)
            selected |= ~inside if self.invert else inside

        return selected

    def select(
        self,
        positions: np.ndarray,
        atom_types: list[str] | None = None,
        cell: np.ndarray | None = None,
    ) -> list[int]:
        """Return the sorted indices of the selected atoms. See :meth:`mask`."""
        return np.flatnonzero(self.mask(positions, atom_types, cell)).tolist()

    @classmethod
    def create(
        cls,
        shape: str,
        *,
        types: Iterable[str] = (),
        indices: Iterable[int] = (),
        invert: bool = False,
        **kwargs,
    ) -> "Region":
        """Build a Region wired with the requested shape.

        Parameters
        ----------
        shape : str
            Name of the registered shape (e.g. "sphere").
        types, indices, invert
            See :class:`Region`.
        **kwargs
            Forwarded to the shape's constructor (e.g. ``config=...``).

        """
        return cls(
            shape=RegionStrategy.create(shape, **kwargs),
            types=types,
            indices=indices,
            invert=invert,
        )

    @classmethod
    def from_config(cls, config: "RegionConfig") -> "Region":
        """Build a Region from an ``[inactive_atoms]``/``[frozen_atoms]`` section.

        ``side='outside'`` inverts sphere, shell and box shapes. For a plane,
        ``side`` ('above'/'below') is read by the shape itself.
        """
        shape = (
            None
            if config.region_type is None
            else RegionStrategy.create(config.region_type, config=config)
        )
        return cls(
            shape=shape,
            types=config.types,
            indices=config.indices,
            invert=config.side == "outside",
        )
