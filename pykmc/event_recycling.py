"""Event recycling: carry events with unmoved, distant central atoms into the next KMC step.

The module exposes an abstract ``Recycling`` interface and one concrete
strategy, ``DistanceRecycling``. Future strategies should subclass
``Recycling`` and implement ``select_recyclable``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .event_table import ActiveEventTable
from .system import System
from .utils.geometry import minimum_image_distance, per_atom_displacement


class Recycling(ABC):
    """Abstract interface for event recycling strategies.

    A recycling strategy decides, given the active event table from step N
    together with system position snapshots taken before and after the
    executed event, which rows can be reused at step N+1 (i.e. carried
    over without a fresh search + refinement).
    """

    @abstractmethod
    def select_recyclable(
        self,
        active_table: ActiveEventTable,
        executed_idx: int,
        system: System,
        positions_pre: np.ndarray,
    ) -> pd.DataFrame:
        """Return the subset of `active_table.table` that can be reused next step.

        Parameters
        ----------
        active_table : ActiveEventTable
            The active event table built at step N (post-refinement).
        executed_idx : int
            Row index in `active_table.table` of the event that was executed.
        system : System
            The atomic system, positions already advanced to post-execution.
        positions_pre : np.ndarray
            Snapshot of `system.positions` taken just before the event was
            applied (same atom ordering as `system.positions`).

        Returns
        -------
        pd.DataFrame
            Subset of `active_table.table` to carry into step N+1. Empty
            DataFrame if nothing is recyclable.

        """


class DistanceRecycling(Recycling):
    """Recycle events whose central atom (a) did not move and (b) is far from the executed event.

    A candidate event survives iff BOTH:

      1. its central atom's pre→post displacement (PBC minimum-image) is below
         ``movement_thr``, AND
      2. its central atom's distance to the executed event's central atom
         (PBC-aware minimum-image) is above ``distance_thr``.

    Otherwise the event is dropped and step N+1 will re-search / re-refine it.
    Both thresholds are in Angstroms.
    """

    def __init__(self, movement_thr: float, distance_thr: float) -> None:
        self.movement_thr = movement_thr
        self.distance_thr = distance_thr

    def select_recyclable(
        self,
        active_table: ActiveEventTable,
        executed_idx: int,
        system: System,
        positions_pre: np.ndarray,
    ) -> pd.DataFrame:
        """Apply the two-fold check; see class docstring."""
        table = active_table.table

        # Trivial case: only the executed event in the table — nothing to recycle.
        if executed_idx not in table.index or len(table) <= 1:
            return table.iloc[0:0].copy()

        # Per-atom displacement magnitudes pre → post (PBC minimum-image).
        # Vectorized over the whole system; we'll index by atom_index below.
        disp = per_atom_displacement(positions_pre, system.positions, system.cell)

        # Reference point for the distance check: the just-executed event's
        # central atom (post-execution position).
        executed_atom = int(table.loc[executed_idx, "atom_index"])
        executed_pos = system.positions[executed_atom]

        keep_rows = []
        for i in table.index:
            # Never recycle the just-executed event itself.
            if i == executed_idx:
                continue
            atom_idx = int(table.loc[i, "atom_index"])

            # (1) Movement check: this central atom must NOT have moved.
            if disp[atom_idx] >= self.movement_thr:
                continue

            # (2) Distance check: this central atom must be FAR from the
            # executed event (PBC minimum-image).
            if (
                minimum_image_distance(
                    executed_pos, system.positions[atom_idx], system.cell
                )
                <= self.distance_thr
            ):
                continue

            keep_rows.append(i)

        if not keep_rows:
            return table.iloc[0:0].copy()
        return table.loc[keep_rows].reset_index(drop=True).copy()
