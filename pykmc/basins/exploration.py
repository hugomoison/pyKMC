from __future__ import annotations
from abc import ABC, abstractmethod
from pykmc import Parameters, ReferenceEventTable
from typing import TYPE_CHECKING
from .connectivity import StatesConnectivity, BasinStatesConnectivity
from .detection import DetectorThreshold
from ..result import ShapeID
import pandas as pd

if TYPE_CHECKING:
    from .basin import StateData

# TODO: BaseExplorer : use it, for the moment it is here in case we implemente other exploration method
# TODO: Posibility to use different detector (with builder) when other detector will be implemented
# TODO: From Parameters we only use params.basin.energy_thr, maybe just have a energy_thr parameter, but could be problematic if implement other detector.
# NOTE: In the future, KMC will use a State object encapsulating System, AtomicEnvironment, NeighbolorsList, so the StateDate will no longer be necessary, will need to adjust at that point.


class Explorer(ABC):
    """Abstract class for basin exploration algorithms."""

    @abstractmethod
    def explore(self) -> bool:
        """Explore the basins."""
        pass


class BasinGenericEventExplorer(Explorer):
    """
    Explorer that constructs a `StateConnectivity` object for one state using only the
    generic events from a reference event table.

    This explorer inspects a state, identifies all applicable
    generic events, and records the corresponding transitions in a connectivity table.

    Parameters
    ----------
    params : Parameters
        `Parameters` object with simulation parameters.
    reference_table : ReferenceEventTable

    Attributes
    ----------
    params : Parameters
        `Parameters` object with simulation parameters.
    reference_table : ReferenceEventTable
        `ReferenceEventTable` object containing all generic KMC events currently known.
    connectivity_table : StatesConnectivity
        Object that store the connectivity of the current state to other states. It is the object that we want to build when using the Explorer.
    detector : DetectorThreshold
        Detector object used to decide if a discovered state is transient or absorbing.
    """

    def __init__(self, params: Parameters, reference_table: ReferenceEventTable) -> None:
        self.params = params
        self.reference_table: ReferenceEventTable = reference_table
        self.connectivity_table: StatesConnectivity = BasinStatesConnectivity()
        self.detector = DetectorThreshold()

    def explore(
        self,
        state: "StateData",
        atom_shapes: dict[int, ShapeID],
        state_index: int = 0,
        start_index: int = 1,
    ) -> None:
        """
        Explore the given state and populate a connectivity table with
        transition information derived from generic events.

        For each event applicable to the current atomic environment:
            - Determine whether the resulting state is transient or absorbing.
            - For each atom on which the event can occur, and for each of its
              symmetry variants, record a connectivity entry.

        Parameters
        ----------
        state : StateData
            Current atomic configuration to explore.
        atom_shapes : dict[int, ShapeID]
            Every non-crystal atom's already-resolved `ShapeID` in `state`,
            from `BasinsGenericEvents.is_states_has_unknown_environments()`
            (classify-only -- exploration never mints a new persistent shape
            from a hypothetical state) -- passed straight through to
            `live_events()` so it can look shapes up instead of reclassifying.
        state_index : int, optional
            Index of the current state in the global basin state list.
            By default 0 (first state).
        start_index : int, optional
            Starting index to assign to newly discovered states.
            This increments as transitions are added.

        Returns
        -------
        None
        """

        # Loop over every (atom, event) pair whose ShapeID actually matches;
        # per-event info (detect()'s real PSR/IRA backward check included)
        # is computed once per event, cached across every atom sharing it.
        count = 0
        row_info_cache: dict[int, tuple] = {}
        for at, df_event in self.reference_table.live_events(atom_shapes):
            ref_event = int(df_event.at["idx_ref"])
            if ref_event not in row_info_cache:
                is_transient = self.detector.detect(
                    df_event,
                    self.reference_table.table,
                    self.params.basin.energy_thr,
                )
                backward_idx = df_event.at["idx_backward"]
                dE_backward = self.reference_table.table[
                    self.reference_table.table["idx_ref"] == backward_idx
                ]["dE_forward"].values[0]
                k_backward = self.reference_table.table[
                    self.reference_table.table["idx_ref"] == backward_idx
                ]["k"].values[0]
                row_info_cache[ref_event] = (is_transient, dE_backward, k_backward)
            is_transient, dE_backward, k_backward = row_info_cache[ref_event]

            # Loop over symmetries :
            for i in range(len(df_event.at["sym_matrix"])):
                # for each symmetries add connectivity in table
                self.connectivity_table.add_connectivity(
                    state=state_index,
                    state_connexion=start_index + count,
                    event_connexion=ref_event,
                    central_atom=at,
                    sym=i,
                    transient=is_transient,
                    dE_forward=df_event["dE_forward"],
                    k_forward=df_event["k"],
                    dE_backward=dE_backward,
                    k_backward=k_backward,
                )

                # update count
                count += 1

    def get_connectivity_table(self) -> pd.DataFrame:
        """
        Return the connectivity table DataFrame.

        Returns
        -------
        pd.DataFrame
            Tabular representation of all discovered transitions.
        """
        return self.connectivity_table.get_table()

    def clear(self) -> None:
        """
        Clear the stored connectivity table.

        Returns
        -------
        None
        """
        self.connectivity_table.clear()
