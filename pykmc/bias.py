"""Event selection bias for KMC simulations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal
import numpy as np
import pandas as pd
from .utils.geometry import minimum_image_vector

if TYPE_CHECKING:
    from .event_table import ActiveEventTable, ReferenceEventTable
    from .system import System
    from .atomic_environment import AtomicEnvironment
    from .neighbors_list import NeighborsList


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
      compete at their natural rates.  α is floored at 1: if the desired events
      already fire with probability ≥ *bias_weight* under their true rates, no
      boosting is applied and the true rates are used unmodified.  ``delta_t``
      is corrected to the true total rate.

    Subclasses must implement :meth:`accept`.  Subclasses that need to cache
    per-step context (e.g. topology lookups) should override :meth:`_prepare`.

    Parameters
    ----------
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Probability ∈ (0, 1) that a desired event is selected at each step.
        Only used in ``"boost"`` mode, and only enforced as a floor: if desired
        events already fire with probability ≥ *bias_weight* under their true
        rates, they are left unboosted.  Default is 0.5.
    pass_unlisted : bool
        Return value of :meth:`accept` for atoms that are **not** in the
        ``atom_indices`` whitelist.  ``False`` (default) treats non-listed
        atoms as rejected/undesired.  ``True`` lets them pass unconditionally;
        only valid in ``"filter"`` mode when ``atom_indices`` is also set.
    require_center : bool
        When True and ``atom_indices`` is set, only the event's central atom is
        tested against the bias predicate.  When False (default), the predicate
        is satisfied by any listed atom found in the event neighbourhood.
    thr_boost : float or None, optional
        Only used in ``"boost"`` mode.  Desired events whose barrier
        (``energy_barrier``) exceeds ``thr_boost`` are excluded from the boost:
        their rate is left unmodified instead of being multiplied by α.
        ``None`` (default) disables this cutoff.
    """

    def __init__(
        self,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
        require_center: bool = False,
        thr_boost: float | None = None,
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
        self.require_center = require_center
        self.thr_boost = thr_boost

    @abstractmethod
    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
        neighbors_list: NeighborsList | None = None,
    ) -> bool:
        """Return True if the event is accepted (filter) or desired (boost).

        Parameters
        ----------
        event : pd.Series
            One row of the active event table representing the candidate event.
        system : System
            Current atomic configuration used to read atom positions.
        reference_table : ReferenceEventTable
            Reference event table.
        neighbors_list : NeighborsList or None, optional
            Neighbour list of the current system, used to locate an atom within
            the event's ``final_positions`` array.

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
        neighbors_list: NeighborsList | None = None,
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
        neighbors_list : NeighborsList or None, optional
            Neighbour list of the current system.
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
        neighbors_list: NeighborsList | None = None,
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
        neighbors_list : NeighborsList or None, optional
            Neighbour list of the current system; forwarded to :meth:`accept`.

        Returns
        -------
        tuple[int, float, float]
            - int: index of the selected event in the active table.
            - float: time increment.
            - float: total rate constant.
        """
        if not self.enabled:
            return selection_algorithm(l_k)
        self._prepare(system, reference_table, atomic_environment, neighbors_list)
        match self.mode:
            case "filter":
                return self._select_filter(
                    selection_algorithm,
                    l_k,
                    active_table,
                    system,
                    reference_table,
                    neighbors_list,
                )
            case "boost":
                return self._select_boost(
                    selection_algorithm,
                    l_k,
                    active_table,
                    system,
                    reference_table,
                    neighbors_list,
                )

    def _select_filter(
        self,
        selection_algorithm: callable,
        l_k: np.ndarray,
        active_table: ActiveEventTable,
        system: System,
        reference_table: ReferenceEventTable,
        neighbors_list: NeighborsList | None = None,
    ) -> tuple[int, float, float]:
        """Rejection-loop selection: remove failing events one by one."""
        candidate_events = list(range(len(l_k)))
        while candidate_events:
            idx_in_candidates, delta_t, ktot = selection_algorithm(
                l_k[candidate_events]
            )
            idx = candidate_events[idx_in_candidates]
            event = active_table.table.loc[idx]
            if self.accept(event, system, reference_table, neighbors_list):
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
        neighbors_list: NeighborsList | None = None,
    ) -> tuple[int, float, float]:
        """Rate-boost selection: multiply desired event rates by dynamic α."""
        desired_mask = np.array(
            [
                self.accept(row, system, reference_table, neighbors_list)
                and (
                    self.thr_boost is None
                    or float(row["energy_barrier"]) <= self.thr_boost
                )
                for _, row in active_table.table.iterrows()
            ]
        )
        k_boost = l_k[desired_mask].sum()
        k_free = l_k[~desired_mask].sum()
        if k_boost == 0 or k_free == 0:
            return selection_algorithm(l_k)
        alpha = max(1.0, self.bias_weight * k_free / ((1 - self.bias_weight) * k_boost))

        l_k_boosted = l_k.copy()
        l_k_boosted[desired_mask] *= alpha
        idx, delta_t_boosted, ktot_boosted = selection_algorithm(l_k_boosted)
        k_total_true = k_boost + k_free
        delta_t = delta_t_boosted * ktot_boosted / k_total_true
        return idx, delta_t, k_total_true

    def _get_displacement(
        self,
        event: pd.Series,
        system: System,
        neighbors_list: NeighborsList | None,
        atom_idx: int,
    ) -> np.ndarray:
        """Return displacement of atom_idx in this event via neighbourhood lookup."""
        if neighbors_list is None:
            final_positions = np.asarray(event["final_positions"], dtype=float)
            if final_positions.ndim == 1:
                target_position = final_positions
            elif len(final_positions) == len(system.positions):
                target_position = final_positions[atom_idx]
            else:
                target_position = final_positions[0]
            return minimum_image_vector(
                system.positions[atom_idx], target_position, system.cell
            )
        neighborhood = np.asarray(
            neighbors_list.get_neighbors("rcut", int(event["atom_index"]))
        )
        k = int(np.where(neighborhood == atom_idx)[0][0])
        return minimum_image_vector(
            system.positions[atom_idx], event["final_positions"][k], system.cell
        )

    def _biased_atom_displacements(
        self,
        event: pd.Series,
        system: System,
        neighbors_list: NeighborsList | None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (atom_idx, displacement) for each biased atom in the neighbourhood."""
        if neighbors_list is None:
            atom_idx = int(event["atom_index"])
            if atom_idx in self._atom_set:
                yield (
                    atom_idx,
                    self._get_displacement(event, system, neighbors_list, atom_idx),
                )
            return
        neighborhood = np.asarray(
            neighbors_list.get_neighbors("rcut", int(event["atom_index"]))
        )
        for atom_idx in self._atom_set:
            if atom_idx not in neighborhood:
                continue
            yield (
                atom_idx,
                self._get_displacement(event, system, neighbors_list, atom_idx),
            )


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
        Probability floor of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-listed atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    require_center : bool
        When True and ``atom_indices`` is set, only the event's central atom is
        tested.  When False (default), the condition is satisfied by any atom in
        ``atom_indices`` found in the event neighbourhood.
    thr_boost : float or None, optional
        Only used in ``"boost"`` mode.  Desired events whose barrier exceeds
        this threshold are excluded from the boost.  Default is ``None``.
    """

    def __init__(
        self,
        direction: np.ndarray,
        atom_indices: list[int] | None = None,
        threshold: float = 0.0,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
        require_center: bool = False,
        thr_boost: float | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            bias_weight=bias_weight,
            pass_unlisted=pass_unlisted,
            require_center=require_center,
            thr_boost=thr_boost,
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
        neighbors_list: NeighborsList | None = None,
    ) -> bool:
        atom_idx = int(event["atom_index"])

        if self._atom_set is None or self.require_center:
            if self._atom_set is not None and atom_idx not in self._atom_set:
                return self.pass_unlisted
            displacement = self._get_displacement(
                event, system, neighbors_list, atom_idx
            )
            return float(np.dot(displacement, self._direction)) >= self._threshold

        for _, displacement in self._biased_atom_displacements(
            event, system, neighbors_list
        ):
            if float(np.dot(displacement, self._direction)) >= self._threshold:
                return True
        return self.pass_unlisted


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
        Probability floor of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-listed atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    require_center : bool
        When True and ``atom_indices`` is set, only the event's central atom is
        tested.  When False (default), the condition is satisfied by any atom in
        ``atom_indices`` found in the event neighbourhood.
    thr_boost : float or None, optional
        Only used in ``"boost"`` mode.  Desired events whose barrier exceeds
        this threshold are excluded from the boost.  Default is ``None``.
    """

    def __init__(
        self,
        target_point: np.ndarray,
        atom_indices: list[int] | None = None,
        threshold: float = 0.0,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
        require_center: bool = False,
        thr_boost: float | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            bias_weight=bias_weight,
            pass_unlisted=pass_unlisted,
            require_center=require_center,
            thr_boost=thr_boost,
        )
        self._target = np.asarray(target_point, dtype=float)
        self._atom_set = set(atom_indices) if atom_indices is not None else None
        self._threshold = threshold

    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
        neighbors_list: NeighborsList | None = None,
    ) -> bool:
        atom_idx = int(event["atom_index"])

        if self._atom_set is None or self.require_center:
            if self._atom_set is not None and atom_idx not in self._atom_set:
                return self.pass_unlisted
            current_pos = system.positions[atom_idx]
            to_target = minimum_image_vector(current_pos, self._target, system.cell)
            dist = np.linalg.norm(to_target)
            if dist < 1e-10:
                return True
            local_direction = to_target / dist
            displacement = self._get_displacement(
                event, system, neighbors_list, atom_idx
            )
            return float(np.dot(displacement, local_direction)) >= self._threshold

        for atom_idx, displacement in self._biased_atom_displacements(
            event, system, neighbors_list
        ):
            current_pos = system.positions[atom_idx]
            to_target = minimum_image_vector(current_pos, self._target, system.cell)
            dist = np.linalg.norm(to_target)
            if dist < 1e-10:
                return True
            local_direction = to_target / dist
            if float(np.dot(displacement, local_direction)) >= self._threshold:
                return True
        return self.pass_unlisted


class TopoBias(Bias):
    """Bias events that reduce the distance between two topology defects.

    On each KMC step :meth:`_prepare` locates all atoms carrying
    ``topo_source`` and ``topo_target`` in the current atomic environment.
    :meth:`accept` then accepts a candidate event only if the moving atom
    belongs to the source topology and its displacement brings it closer to the
    nearest target-topology atom.  If the target topology is absent the bias is
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
        Probability floor of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-source atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    thr_boost : float or None, optional
        Only used in ``"boost"`` mode.  Desired events whose barrier exceeds
        this threshold are excluded from the boost.  Default is ``None``.
    """

    def __init__(
        self,
        topo_source: str | bytes,
        topo_target: str | bytes,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
        thr_boost: float | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            bias_weight=bias_weight,
            pass_unlisted=pass_unlisted,
            thr_boost=thr_boost,
        )
        self._topo_source = topo_source
        self._topo_target = topo_target
        self._source_atoms: set[int] = set()
        self._target_positions = None

    def _prepare(
        self,
        system,
        reference_table,
        atomic_environment,
        neighbors_list=None,
    ) -> None:
        self._source_atoms = set(
            atomic_environment.get_atoms_with_id(self._topo_source)
        )
        target_atoms = atomic_environment.get_atoms_with_id(self._topo_target)
        self._target_positions = (
            system.positions[target_atoms] if target_atoms else None
        )

    def accept(self, event, system, reference_table, neighbors_list=None) -> bool:
        atom_idx = int(event["atom_index"])
        if atom_idx not in self._source_atoms:
            return self.pass_unlisted
        if self._target_positions is None:
            return True

        current_pos = system.positions[atom_idx]
        displacement = self._get_displacement(event, system, neighbors_list, atom_idx)
        final_pos = current_pos + displacement

        current_min_dist = min(
            float(
                np.linalg.norm(minimum_image_vector(current_pos, target, system.cell))
            )
            for target in self._target_positions
        )
        final_min_dist = min(
            float(np.linalg.norm(minimum_image_vector(final_pos, target, system.cell)))
            for target in self._target_positions
        )
        return final_min_dist < current_min_dist
