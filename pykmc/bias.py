"""Event selection bias for KMC simulations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .event_table import ActiveEventTable, ReferenceEventTable
    from .system import System
    from .atomic_environment import AtomicEnvironment


def _moving_atom_displacement(
    event: pd.Series,
    system: System,
    reference_table: ReferenceEventTable,
) -> np.ndarray:
    """Return the displacement of the moving atom for a candidate event.

    Parameters
    ----------
    event : pd.Series
        One row of the active event table.
    system : System
        Current atomic configuration.
    reference_table : ReferenceEventTable
        Reference table used to look up ``move_atom_idx``.

    Returns
    -------
    np.ndarray, shape (3,)
        Displacement vector (final − initial) of the moving atom.
    """
    atom_idx = int(event["atom_index"])
    num_ref = int(event["num_reference_event"])
    ref_row = reference_table.table[reference_table.table["idx_ref"] == num_ref].iloc[0]
    move_atom_idx = int(ref_row["move_atom_idx"])
    final_pos = event["final_positions"][move_atom_idx]
    current_pos = system.positions[atom_idx]
    return final_pos - current_pos


class Bias(ABC):
    """Abstract base class for event selection bias.

    Supports two modes controlled by the *mode* parameter:

    - ``"filter"`` (default): rejection-loop mode.  Candidates are drawn one
      by one from a shrinking pool; events that fail :meth:`accept` are removed
      before the next draw.  ``delta_t`` and ``ktot`` reflect the effective
      total rate at the moment of acceptance.
    - ``"boost"``: rate-boost mode.  Events that pass :meth:`accept` have their
      rates multiplied by a dynamic factor α so they fire with probability
      *bias_weight* at each step, while all other events remain in the pool and
      compete at their natural rates.  ``delta_t`` is corrected to the true
      total rate.

    Subclasses must implement :meth:`accept`.  Subclasses that need to cache
    per-step context (e.g. topology lookups) should override :meth:`_prepare`.

    Parameters
    ----------
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Target probability ∈ (0, 1) that a desired event is selected at each
        step.  Only used in ``"boost"`` mode.  Default is 0.5.
    pass_unlisted : bool
        Return value of :meth:`accept` for atoms that are **not** in the
        ``atom_indices`` whitelist.  ``False`` (default) treats non-listed
        atoms as rejected/undesired.  ``True`` lets them pass unconditionally;
        only valid in ``"filter"`` mode when ``atom_indices`` is also set.
    """

    def __init__(
        self,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
    ) -> None:
        if mode == "boost" and pass_unlisted:
            raise ValueError(
                "pass_unlisted=True is incompatible with mode='boost': "
                "non-desired atoms would be incorrectly boosted. "
                "Set pass_unlisted=False in boost mode."
            )
        self.enabled: bool = True
        self.mode = mode
        self.bias_weight = bias_weight
        self.pass_unlisted = pass_unlisted

    @abstractmethod
    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> bool:
        """Return True if the event is accepted (filter) or desired (boost).

        Parameters
        ----------
        event : pd.Series
            One row of the active event table representing the candidate event.
        system : System
            Current atomic configuration used to read atom positions.
        reference_table : ReferenceEventTable
            Reference event table used to retrieve the moving-atom index
            within the event's local neighbourhood array.

        Returns
        -------
        bool
            True if the event satisfies the bias condition, False otherwise.
        """
        pass

    def _prepare(
        self,
        system: System,
        reference_table: ReferenceEventTable,
        atomic_environment: AtomicEnvironment,
    ) -> None:
        """Pre-loop hook called once per :meth:`select` invocation.

        Override in subclasses that need to cache per-step context (e.g. look
        up which atoms currently carry a given topology).  The base
        implementation is a no-op.

        Parameters
        ----------
        system : System
            Current atomic configuration.
        reference_table : ReferenceEventTable
            Reference event table.
        atomic_environment : AtomicEnvironment
            Current atomic environment (topology IDs per atom).
        """
        pass

    def select(
        self,
        selection_algorithm: callable,
        l_k: np.ndarray,
        active_table: ActiveEventTable,
        system: System,
        reference_table: ReferenceEventTable,
        atomic_environment: AtomicEnvironment | None = None,
    ) -> tuple[int, float, float]:
        """Select an event using the configured bias mode.

        Parameters
        ----------
        selection_algorithm : callable
            Function with signature ``(rates) -> (index, delta_t, ktot)``.
        l_k : np.ndarray
            Rate constants for all active events.
        active_table : ActiveEventTable
            Active event table providing event metadata.
        system : System
            Current atomic configuration.
        reference_table : ReferenceEventTable
            Reference event table.
        atomic_environment : AtomicEnvironment or None, optional
            Current atomic environment; forwarded to :meth:`_prepare`.

        Returns
        -------
        tuple[int, float, float]
            - int: index of the selected event in the active table.
            - float: time increment.
            - float: total rate constant.
        """
        if not self.enabled:
            return selection_algorithm(l_k)
        self._prepare(system, reference_table, atomic_environment)
        match self.mode:
            case "filter":
                return self._select_filter(
                    selection_algorithm, l_k, active_table, system, reference_table
                )
            case "boost":
                return self._select_boost(
                    selection_algorithm, l_k, active_table, system, reference_table
                )

    def _select_filter(
        self,
        selection_algorithm: callable,
        l_k: np.ndarray,
        active_table: ActiveEventTable,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> tuple[int, float, float]:
        """Rejection-loop selection: remove failing events one by one."""
        candidate_events = list(range(len(l_k)))
        while candidate_events:
            idx_in_candidates, delta_t, ktot = selection_algorithm(
                l_k[candidate_events]
            )
            idx = candidate_events[idx_in_candidates]
            if self.accept(active_table.table.loc[idx], system, reference_table):
                return idx, delta_t, ktot
            candidate_events.remove(idx)
        return selection_algorithm(l_k)

    def _select_boost(
        self,
        selection_algorithm: callable,
        l_k: np.ndarray,
        active_table: ActiveEventTable,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> tuple[int, float, float]:
        """Rate-boost selection: multiply desired event rates by dynamic α."""
        desired_mask = np.array(
            [
                self.accept(row, system, reference_table)
                for _, row in active_table.table.iterrows()
            ]
        )
        k_boost = l_k[desired_mask].sum()
        k_free = l_k[~desired_mask].sum()
        if k_boost == 0 or k_free == 0:
            return selection_algorithm(l_k)
        alpha = self.bias_weight * k_free / ((1 - self.bias_weight) * k_boost)
        l_k_boosted = l_k.copy()
        l_k_boosted[desired_mask] *= alpha
        idx, delta_t_boosted, ktot_boosted = selection_algorithm(l_k_boosted)
        k_total_true = k_boost + k_free
        delta_t = delta_t_boosted * ktot_boosted / k_total_true
        return idx, delta_t, k_total_true

    def _moving_atom_displacement(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> np.ndarray:
        """Return the displacement of the moving atom for a candidate event."""
        return _moving_atom_displacement(event, system, reference_table)


class DirectionBias(Bias):
    """Bias events where the moving atom's displacement projects onto a direction.

    An event is accepted when the projection of the moving atom's displacement
    onto *direction* is greater than or equal to *threshold*.

    Parameters
    ----------
    direction : array-like, shape (3,)
        Desired direction vector (normalised internally).
    atom_indices : list[int] or None, optional
        Global indices of atoms to bias.  When *None* (default) all atoms
        are subject to the bias.
    threshold : float, optional
        Minimum required projection onto *direction*.  Default is 0.
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Target probability of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-listed atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    """

    def __init__(
        self,
        direction: np.ndarray,
        atom_indices: list[int] | None = None,
        threshold: float = 0.0,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
    ) -> None:
        super().__init__(
            mode=mode, bias_weight=bias_weight, pass_unlisted=pass_unlisted
        )
        d = np.asarray(direction, dtype=float)
        self._direction = d / np.linalg.norm(d)
        self._atom_set = set(atom_indices) if atom_indices is not None else None
        self._threshold = threshold

    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> bool:
        if (
            self._atom_set is not None
            and int(event["atom_index"]) not in self._atom_set
        ):
            return self.pass_unlisted
        displacement = self._moving_atom_displacement(event, system, reference_table)
        return float(np.dot(displacement, self._direction)) >= self._threshold


class PointBias(Bias):
    """Bias events where the moving atom moves toward a target point.

    For each candidate event the local direction toward *target_point* is
    computed from the atom's current position.  The event is accepted when
    the projection of the displacement onto that direction is greater than
    or equal to *threshold*.

    Parameters
    ----------
    target_point : array-like, shape (3,)
        Reference point in Cartesian coordinates.
    atom_indices : list[int] or None, optional
        Global indices of atoms to bias.  When *None* (default) all atoms
        are subject to the bias.
    threshold : float, optional
        Minimum required projection onto the direction toward *target_point*.
        Default is 0.
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Target probability of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-listed atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    """

    def __init__(
        self,
        target_point: np.ndarray,
        atom_indices: list[int] | None = None,
        threshold: float = 0.0,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
    ) -> None:
        super().__init__(
            mode=mode, bias_weight=bias_weight, pass_unlisted=pass_unlisted
        )
        self._target = np.asarray(target_point, dtype=float)
        self._atom_set = set(atom_indices) if atom_indices is not None else None
        self._threshold = threshold

    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
    ) -> bool:
        atom_idx = int(event["atom_index"])
        if self._atom_set is not None and atom_idx not in self._atom_set:
            return self.pass_unlisted
        current_pos = system.positions[atom_idx]
        to_target = self._target - current_pos
        dist = np.linalg.norm(to_target)
        if dist < 1e-10:
            return True
        local_direction = to_target / dist
        displacement = self._moving_atom_displacement(event, system, reference_table)
        return float(np.dot(displacement, local_direction)) >= self._threshold


class TopoBias(Bias):
    """Bias events that reduce the distance between two topology defects.

    On each KMC step :meth:`_prepare` locates all atoms carrying
    ``topo_source`` and ``topo_target`` in the current atomic environment.
    :meth:`accept` then accepts a candidate event only if the moving atom
    belongs to the source topology and its displacement brings it closer to the
    nearest target-topology atom.  If either topology is absent the bias is
    inactive for that step.

    Parameters
    ----------
    topo_source : str | bytes
        Topology ID of the defect to move (e.g. vacancy graph ID).
    topo_target : str | bytes
        Topology ID of the defect to approach (e.g. interstitial graph ID).
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Target probability of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-source atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    """

    def __init__(
        self,
        topo_source: str | bytes,
        topo_target: str | bytes,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
    ) -> None:
        super().__init__(
            mode=mode, bias_weight=bias_weight, pass_unlisted=pass_unlisted
        )
        self._topo_source = topo_source
        self._topo_target = topo_target
        self._source_positions = None
        self._target_positions = None

    def _prepare(self, system, reference_table, atomic_environment) -> None:
        source_atoms = atomic_environment.get_atoms_with_id(self._topo_source)
        target_atoms = atomic_environment.get_atoms_with_id(self._topo_target)
        self._source_positions = (
            system.positions[source_atoms] if source_atoms else None
        )
        self._target_positions = (
            system.positions[target_atoms] if target_atoms else None
        )

    def accept(self, event, system, reference_table) -> bool:
        if self._source_positions is None or self._target_positions is None:
            return True
        current_pos = system.positions[int(event["atom_index"])]
        if not any(np.allclose(current_pos, s) for s in self._source_positions):
            return self.pass_unlisted
        displacement = self._moving_atom_displacement(event, system, reference_table)
        final_pos = current_pos + displacement
        current_min_dist = np.min(
            np.linalg.norm(self._target_positions - current_pos, axis=1)
        )
        final_min_dist = np.min(
            np.linalg.norm(self._target_positions - final_pos, axis=1)
        )
        return final_min_dist < current_min_dist
