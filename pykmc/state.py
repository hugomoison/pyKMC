"""Coordinates a :class:`System` with its position-derived representations.

Prototype for issue #63. A single ``State`` object owns the atomic positions
together with the data derived from them (neighbour list, atomic-environment
IDs) and keeps them consistent, so callers stop manually re-instantiating
``NeighborsList`` + ``AtomicEnvironment`` after every move. Today that manual
rebuild is duplicated in the KMC main loop (``kmc.py`` "Update variables" block)
and in ``basins.basin.StateData.ensure_full_state``; this object is intended to
absorb and replace ``StateData``.

Consistency model: *lazy invalidation*. Mutating positions through
:meth:`State.set_positions` marks the derived data dirty; the neighbour list and
atomic environment are rebuilt on next access, or eagerly when ``sync=True``.
This replaces the per-call ``update_state`` boolean sketched in #63 with a dirty
flag, which naturally batches repeated small intermediate moves into a single
rebuild rather than recomputing on every call.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .atomic_environment import AtomicEnvironment
from .config import Config
from .neighbors_list import NeighborsList
from .system import System

NeighborSource = Literal["scipy", "lammps"]


class State:
    """Single owner of a :class:`System` and its derived representations.

    Attributes
    ----------
    system : System
        The atomic system. Positions should be mutated via
        :meth:`set_positions` (or a direct mutation followed by
        :meth:`invalidate`) so the derived data stays consistent.
    config : Config
        Supplies ``config.atomicenvironment`` (``style``, ``rnei``, ``rcut``,
        ``neighbors_add``) used to (re)build the derived representations.
    neighbor_source : {"scipy", "lammps"}
        Backend used to build the neighbour list. Only ``"scipy"`` (the current
        ``NeighborsList`` / cKDTree implementation) is wired up; ``"lammps"`` is
        a seam for the LAMMPS neighbour-list backend (#63 / #65) and raises
        :class:`NotImplementedError` until that lands.

    """

    def __init__(
        self,
        system: System,
        config: Config,
        *,
        neighbor_source: NeighborSource = "scipy",
    ) -> None:
        self._system = system
        self._config = config
        self._neighbor_source: NeighborSource = neighbor_source
        self._neighbors_list: NeighborsList | None = None
        self._atomic_environment: AtomicEnvironment | None = None
        self._dirty = True

    # -- access ----------------------------------------------------------------

    @property
    def system(self) -> System:
        """Return the underlying :class:`System` (always current)."""
        return self._system

    @property
    def neighbors_list(self) -> NeighborsList:
        """Return the neighbour list, rebuilding lazily if positions changed."""
        self.sync()
        assert self._neighbors_list is not None  # established by _rebuild()
        return self._neighbors_list

    @property
    def atomic_environment(self) -> AtomicEnvironment:
        """Return the atomic-environment IDs, rebuilding lazily if needed."""
        self.sync()
        assert self._atomic_environment is not None  # established by _rebuild()
        return self._atomic_environment

    # -- mutation --------------------------------------------------------------

    def set_positions(
        self,
        positions: np.ndarray,
        atom_idx: np.ndarray | None = None,
        *,
        sync: bool = True,
    ) -> None:
        """Update positions and (re)synchronise the derived data.

        Parameters
        ----------
        positions : np.ndarray of float
            New positions, forwarded to :meth:`System.update_positions`.
        atom_idx : np.ndarray of int, optional
            Subset of atoms to update, forwarded to
            :meth:`System.update_positions`.
        sync : bool, default True
            If ``True``, rebuild the neighbour list and atomic environment now.
            If ``False``, only mark them stale; they rebuild on next access. Use
            ``sync=False`` for batches of small intermediate moves to avoid
            redundant rebuilds (the dirty-flag analogue of #63's ``update_state``
            flag).

        """
        self._system.update_positions(positions, atom_idx)
        self._dirty = True
        if sync:
            self._rebuild()

    def invalidate(self) -> None:
        """Mark the derived data stale (call after mutating positions directly)."""
        self._dirty = True

    def sync(self) -> None:
        """Rebuild the derived data if it is stale; otherwise do nothing."""
        if (
            self._dirty
            or self._neighbors_list is None
            or self._atomic_environment is None
        ):
            self._rebuild()

    def is_synced(self) -> bool:
        """Return ``True`` if the derived data matches the current positions."""
        return not self._dirty

    def release_heavy_objects(self) -> None:
        """Drop the derived objects to free memory (absorbs ``StateData``).

        The next access to :attr:`neighbors_list` or :attr:`atomic_environment`
        rebuilds them on demand.
        """
        self._neighbors_list = None
        self._atomic_environment = None
        self._dirty = True

    # -- internal --------------------------------------------------------------

    def _rebuild(self) -> None:
        """Rebuild the neighbour list and atomic environment from positions.

        Mirrors the manual block in the KMC loop and
        ``StateData.ensure_full_state`` so swapping this object in is
        behaviour-preserving.
        """
        if self._neighbor_source != "scipy":
            raise NotImplementedError(
                f"neighbor_source={self._neighbor_source!r} is not implemented; "
                "only 'scipy' (NeighborsList / cKDTree) is available so far."
            )
        ae = self._config.atomicenvironment
        self._neighbors_list = NeighborsList(self._system, ae.rnei, ae.rcut)
        self._atomic_environment = AtomicEnvironment(
            ae.style,
            self._neighbors_list.neighbors_list["rnei"],
            self._neighbors_list.neighbors_list.get("rcut"),
            ae.neighbors_add or 0,
        )
        self._dirty = False
