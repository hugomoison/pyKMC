"""Event selection bias for KMC simulations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
import logging
import numpy as np
import pandas as pd
from .log import fmt_hash
from .utils.geometry import minimum_image_vector

if TYPE_CHECKING:
    from .event_table import ActiveEventTable, ReferenceEventTable
    from .system import System
    from .atomic_environment import AtomicEnvironment
    from .neighbors_list import NeighborsList


_LOGGER = logging.getLogger("log")


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
      boosting is applied and the true rates are used unmodified (the bias
      never suppresses an already-favoured event below its natural rate).
      ``delta_t`` is corrected to the true total rate.

    Subclasses must implement :meth:`accept`.  Subclasses that need to cache
    per-step context (e.g. topology lookups) should override :meth:`_prepare`.

    Parameters
    ----------
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Target probability ∈ (0, 1) that a desired event is selected at each
        step.  Only used in ``"boost"`` mode, and only enforced as a floor:
        if desired events already fire with probability ≥ *bias_weight*
        under their true rates, they are left unboosted.  Default is 0.5.
    pass_unlisted : bool
        Return value of :meth:`accept` for atoms that are **not** in the
        ``atom_indices`` whitelist.  ``False`` (default) treats non-listed
        atoms as rejected/undesired.  ``True`` lets them pass unconditionally;
        only valid in ``"filter"`` mode when ``atom_indices`` is also set.
    thr_boost : float or None, optional
        Only used in ``"boost"`` mode.  Desired events whose barrier
        (``dE_forward``) exceeds ``thr_boost`` are excluded from the boost:
        their rate is left unmodified instead of being multiplied by α.
        ``None`` (default) disables this cutoff.
    """

    def __init__(
        self,
        mode: Literal["filter", "boost"] = "filter",
        bias_weight: float = 0.5,
        pass_unlisted: bool = False,
        require_central: bool = False,
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
        self.require_central = require_central
        self.thr_boost = thr_boost
        self.current_step = 0
        self._step_is_active = True

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
            Reference event table used to retrieve the moving-atom index
            within the event's local neighbourhood array.
        neighbors_list : NeighborsList or None, optional
            Neighbour list of the current system.  Required when
            ``atom_indices`` is set so that the bias can locate a specific
            atom within ``event["final_configuration"]``.

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

        Returns
        -------
        tuple[int, float, float]
            - int: index of the selected event in the active table.
            - float: time increment.
            - float: total rate constant.
        """
        if not self.enabled:
            _LOGGER.debug("\t :=> Bias disabled, using unbiased selection")
            return selection_algorithm(l_k)
        self.current_step += 1
        _LOGGER.debug(
            f"\n\t :=> Bias select: mode={self.mode},"
            f" events={len(l_k)}, k_total={float(np.sum(l_k)):.6e}"
        )
        self._prepare(system, reference_table, atomic_environment, neighbors_list)
        if not self._step_is_active:
            _LOGGER.debug("\t :=> Bias inactive on this step, using unbiased selection")
            return selection_algorithm(l_k)
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
            num_ref = event.get("num_reference_event", None)
            ref_rows = reference_table.table[
                reference_table.table["idx_ref"] == num_ref
            ]
            event_id = (
                fmt_hash(ref_rows["event_id"].values[0])
                if len(ref_rows) > 0 and "event_id" in ref_rows.columns
                else "?"
            )
            if self.accept(event, system, reference_table, neighbors_list):
                _LOGGER.debug(
                    f"\t\t :=> Filter: accepted event {idx}"
                    f" (atom={event.get('atom_index', '?')},"
                    f" ref={num_ref},"
                    f" event_id={event_id},"
                    f" Ea={event.get('dE_forward', float('nan')):.6f} eV,"
                    f" k={float(l_k[idx]):.6e})"
                )
                return idx, delta_t, ktot
            _LOGGER.debug(
                f"\t\t :=> Filter: rejected event {idx}"
                f" (atom={event.get('atom_index', '?')},"
                f" ref={num_ref},"
                f" event_id={event_id},"
                f" Ea={event.get('dE_forward', float('nan')):.6f} eV)"
            )
            candidate_events.remove(idx)
        _LOGGER.debug(
            "\t :=> Filter: all candidates rejected, using unbiased selection"
        )
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
                    or float(row.get("dE_forward", float("inf"))) <= self.thr_boost
                )
                for _, row in active_table.table.iterrows()
            ]
        )
        k_boost = l_k[desired_mask].sum()
        k_free = l_k[~desired_mask].sum()
        if k_boost == 0 or k_free == 0:
            _LOGGER.debug(
                f"\t :=> Boost: degenerate split"
                f" (k_boost={float(k_boost):.6e}, k_free={float(k_free):.6e}),"
                f" using unbiased selection"
            )
            return selection_algorithm(l_k)
        alpha = max(
            1.0, self.bias_weight * k_free / ((1 - self.bias_weight) * k_boost)
        )
        _LOGGER.debug(
            f"\t :=> Boost: desired={int(np.count_nonzero(desired_mask))},"
            f" undesired={int(np.count_nonzero(~desired_mask))},"
            f" k_boost={float(k_boost):.6e}, k_free={float(k_free):.6e},"
            f" bias_weight={float(self.bias_weight):.3f}, alpha={float(alpha):.6e}"
            + (" (floored, desired already dominant)" if alpha == 1.0 else "")
        )
        for i in np.nonzero(desired_mask)[0]:
            row = active_table.table.loc[i]
            num_ref = row.get("num_reference_event", None)
            event_id = fmt_hash(
                reference_table.table[reference_table.table["idx_ref"] == num_ref][
                    "event_id"
                ].values[0]
            )
            _LOGGER.debug(
                f"\t\t :=> Desired event: idx={int(i)},"
                f" atom_index={row.get('atom_index', '?')},"
                f" reference_event={num_ref},"
                f" Ea={row.get('dE_forward', float('nan')):.6f} eV,"
                f" event_id={event_id}"
            )
        l_k_boosted = l_k.copy()
        l_k_boosted[desired_mask] *= alpha
        idx, delta_t_boosted, ktot_boosted = selection_algorithm(l_k_boosted)
        k_total_true = k_boost + k_free
        delta_t = delta_t_boosted * ktot_boosted / k_total_true
        selected_event = active_table.table.loc[idx]
        num_ref = selected_event.get("num_reference_event", None)
        ref_rows = reference_table.table[reference_table.table["idx_ref"] == num_ref]
        event_id = (
            fmt_hash(ref_rows["event_id"].values[0])
            if len(ref_rows) > 0 and "event_id" in ref_rows.columns
            else "?"
        )
        _LOGGER.debug(
            f"\t :=> Boost: selected event {idx},"
            f" event_id={event_id},"
            f" Ea={selected_event.get('dE_forward', float('nan')):.6f} eV,"
            f" corrected_delta_t={float(delta_t):.6e},"
            f" true_k_total={float(k_total_true):.6e}"
        )
        return idx, delta_t, k_total_true

    def _get_displacement(
        self,
        event: pd.Series,
        system: System,
        neighbors_list: NeighborsList,
        atom_idx: int,
    ) -> np.ndarray:
        """Return displacement of atom_idx in this event via neighbourhood lookup."""
        if neighbors_list is None:
            final_positions = np.asarray(event["final_configuration"].positions, dtype=float)
            if final_positions.ndim == 1:
                target_position = final_positions
            elif len(final_positions) == len(system):
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
            system.positions[atom_idx], event["final_configuration"].positions[k], system.cell
        )

    def _biased_atom_displacements(
        self,
        event: pd.Series,
        system: System,
        neighbors_list: NeighborsList,
    ):
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
            where = np.where(neighborhood == atom_idx)[0]
            if len(where) == 0:
                continue
            k = int(where[0])
            yield atom_idx, minimum_image_vector(
                system.positions[atom_idx], event["final_configuration"].positions[k], system.cell
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
        Target probability of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-listed atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    require_central : bool
        When True and ``atom_indices`` is set, only the most-moving atom
        (``event["atom_index"]``) is checked.  If it is not in
        ``atom_indices``, ``pass_unlisted`` is returned.  When False
        (default), the condition is satisfied by any atom in ``atom_indices``
        found in the event neighbourhood.
    step_interval : int or None, optional
        Apply the bias only every ``step_interval`` KMC steps. ``None`` or ``1``
        means the bias is active at every step.
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
        require_central: bool = False,
        step_interval: int | None = None,
        thr_boost: float | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            bias_weight=bias_weight,
            pass_unlisted=pass_unlisted,
            require_central=require_central,
            thr_boost=thr_boost,
        )
        d = np.asarray(direction, dtype=float)
        self._direction = d / np.linalg.norm(d)
        self._atom_set = set(atom_indices) if atom_indices is not None else None
        self._threshold = threshold
        if step_interval is not None and step_interval < 1:
            raise ValueError("step_interval must be >= 1")
        self._step_interval = step_interval

    def _prepare(
        self,
        system: System,
        reference_table: ReferenceEventTable,
        atomic_environment: AtomicEnvironment,
        neighbors_list: NeighborsList | None = None,
    ) -> None:
        self._step_is_active = (
            self._step_interval is None
            or self._step_interval == 1
            or self.current_step is None
            or self.current_step % self._step_interval == 0
        )

    def accept(
        self,
        event: pd.Series,
        system: System,
        reference_table: ReferenceEventTable,
        neighbors_list: NeighborsList | None = None,
    ) -> bool:
        atom_idx = int(event["atom_index"])
        if self._atom_set is None or self.require_central:
            if self._atom_set is not None and atom_idx not in self._atom_set:
                return self.pass_unlisted
            displacement = self._get_displacement(
                event, system, neighbors_list, atom_idx
            )
            projection = float(np.dot(displacement, self._direction))
            accepted = projection >= self._threshold
            _LOGGER.debug(
                f"\t\t :=> Direction bias: atom {atom_idx},"
                f" projection={projection:+.6e}, threshold={float(self._threshold):.6e},"
                f" accepted={accepted}"
            )
            return accepted
        for atom_idx, displacement in self._biased_atom_displacements(
            event, system, neighbors_list
        ):
            projection = float(np.dot(displacement, self._direction))
            _LOGGER.debug(
                f"\t\t :=> Direction bias: atom {atom_idx},"
                f" projection={projection:+.6e}, threshold={float(self._threshold):.6e}"
            )
            if projection >= self._threshold:
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
        Target probability of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-listed atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    require_central : bool
        When True and ``atom_indices`` is set, only the most-moving atom
        (``event["atom_index"]``) is checked.  If it is not in
        ``atom_indices``, ``pass_unlisted`` is returned.  When False
        (default), the condition is satisfied by any atom in ``atom_indices``
        found in the event neighbourhood.
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
        require_central: bool = False,
        thr_boost: float | None = None,
    ) -> None:
        super().__init__(
            mode=mode,
            bias_weight=bias_weight,
            pass_unlisted=pass_unlisted,
            require_central=require_central,
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
        if self._atom_set is None or self.require_central:
            atom_idx = int(event["atom_index"])
            if self._atom_set is not None and atom_idx not in self._atom_set:
                return self.pass_unlisted
            current_pos = system.positions[atom_idx]
            to_target = minimum_image_vector(current_pos, self._target, system.cell)
            dist = np.linalg.norm(to_target)
            if dist < 1e-10:
                return True
            displacement = self._get_displacement(
                event, system, neighbors_list, atom_idx
            )
            projection = float(np.dot(displacement, to_target / dist))
            _LOGGER.debug(
                f"\t\t :=> Point bias: atom {atom_idx},"
                f" projection={projection:+.6e}, threshold={float(self._threshold):.6e},"
                f" accepted={projection >= self._threshold}"
            )
            return projection >= self._threshold
        for atom_idx, displacement in self._biased_atom_displacements(
            event, system, neighbors_list
        ):
            current_pos = system.positions[atom_idx]
            to_target = minimum_image_vector(current_pos, self._target, system.cell)
            dist = np.linalg.norm(to_target)
            if dist < 1e-10:
                return True
            projection = float(np.dot(displacement, to_target / dist))
            _LOGGER.debug(
                f"\t\t :=> Point bias: atom {atom_idx},"
                f" projection={projection:+.6e}, threshold={float(self._threshold):.6e}"
            )
            if projection >= self._threshold:
                return True
        return self.pass_unlisted


class TopoBias(Bias):
    """Bias events that reduce the distance between two topology defects.

    The topology IDs of the source and (optional) target atoms are resolved
    once at construction from ``atom_source_idx`` / ``atom_target_idx``.
    On each KMC step :meth:`_prepare` locates all atoms that currently carry
    those fixed topology IDs.  :meth:`accept` then accepts a candidate event
    only if the most-moving atom belongs to the source topology and either:

    - moves toward the nearest target-topology atom (two-index mode), or
    - its displacement projects onto *direction* above *threshold* (one-index mode).

    Parameters
    ----------
    atom_source_idx : int
        Index of the atom whose topology ID at construction time is used as
        the source topology.
    atomic_environment : AtomicEnvironment
        Atomic environment at construction time, used for the one-time
        topology-ID lookup.
    atom_target_idx : int or None, optional
        Index of the atom whose topology ID is used as the target.  When
        *None* (default), one-index (direction) mode is active.
    direction : array-like, shape (3,) or None, optional
        Required in one-index mode.  Desired displacement direction.
    threshold : float, optional
        Minimum projection onto *direction* in one-index mode.  Default is 0.
    mode : {"filter", "boost"}
        Selection mode.  Default is ``"filter"``.
    bias_weight : float
        Target probability of desired event selection in boost mode.
    pass_unlisted : bool
        Return value of :meth:`accept` for non-source atoms.  Default is
        ``False``.  Setting ``True`` is only valid in ``"filter"`` mode.
    thr_boost : float or None, optional
        Only used in ``"boost"`` mode.  Desired events whose barrier exceeds
        this threshold are excluded from the boost.  Default is ``None``.
    """

    def __init__(
        self,
        atom_source_idx: int,
        atomic_environment,
        atom_target_idx: int | None = None,
        direction: np.ndarray | None = None,
        threshold: float = 0.0,
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
        self._topo_source = atomic_environment.atomic_environment_list[atom_source_idx]
        self._topo_target = (
            atomic_environment.atomic_environment_list[atom_target_idx]
            if atom_target_idx is not None
            else None
        )
        if self._topo_target is None:
            if direction is None:
                raise ValueError(
                    "direction is required when atom_target_idx is not given"
                )
            d = np.asarray(direction, dtype=float)
            self._direction = d / np.linalg.norm(d)
        else:
            self._direction = None
        self._threshold = threshold
        self._source_atoms: set[int] = set()
        self._target_positions = None

    def _prepare(
        self,
        system,
        reference_table,
        atomic_environment,
        neighbors_list=None,
    ) -> None:
        source_atoms = atomic_environment.get_atoms_with_id(self._topo_source)
        self._source_atoms = set(source_atoms)
        if self._topo_target is not None:
            target_atoms = atomic_environment.get_atoms_with_id(self._topo_target)
            self._target_positions = (
                system.positions[target_atoms] if target_atoms else None
            )
        n_target = (
            len(atomic_environment.get_atoms_with_id(self._topo_target))
            if self._topo_target
            else 0
        )
        _LOGGER.debug(
            f"\t :=> Topo bias prepare: source_atoms={len(source_atoms)}, target_atoms={n_target}"
        )

    def accept(self, event, system, reference_table, neighbors_list=None) -> bool:
        atom_idx = int(event["atom_index"])
        if atom_idx not in self._source_atoms:
            _LOGGER.debug(
                f"\t\t :=> Topo bias: atom {atom_idx} not in source topology,"
                f" pass_unlisted={self.pass_unlisted}"
            )
            return self.pass_unlisted
        displacement = self._get_displacement(event, system, neighbors_list, atom_idx)
        if self._direction is not None:
            projection = float(np.dot(displacement, self._direction))
            accepted = projection >= self._threshold
            _LOGGER.debug(
                f"\t\t :=> Topo bias: atom {atom_idx},"
                f" projection={projection:+.6e}, threshold={float(self._threshold):.6e},"
                f" accepted={accepted}"
            )
            return accepted
        if self._target_positions is None:
            _LOGGER.debug(
                "\t :=> Topo bias: no target-topology atoms this step, inactive (accepted=True)"
            )
            return True
        current_pos = system.positions[atom_idx]
        final_pos = current_pos + displacement
        current_min_dist = min(
            np.linalg.norm(minimum_image_vector(current_pos, target, system.cell))
            for target in self._target_positions
        )
        final_min_dist = min(
            np.linalg.norm(minimum_image_vector(final_pos, target, system.cell))
            for target in self._target_positions
        )
        accepted = final_min_dist < current_min_dist
        _LOGGER.debug(
            f"\t\t :=> Topo bias: atom {atom_idx},"
            f" current_min_dist={current_min_dist:.6e}, final_min_dist={final_min_dist:.6e},"
            f" accepted={accepted}"
        )
        return accepted
