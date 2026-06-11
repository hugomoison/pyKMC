from __future__ import annotations
from abc import ABC, abstractmethod
from pykmc import Config, ReferenceEventTable
from typing import TYPE_CHECKING
from .connectivity import StatesConnectivity, BasinStatesConnectivity
from .detection import DetectorThreshold
import pandas as pd

if TYPE_CHECKING:
    from .basin import StateData

# TODO: BaseExplorer : use it, for the moment it is here in case we implemente other exploration method
# TODO: Posibility to use different detector (with builder) when other detector will be implemented
# TODO: From Config we only use config.basin.energy_thr, maybe just have a energy_thr parameter, but could be problematic if implement other detector.
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
    config : Config
        `Config` object with simulation parameters.
    reference_table : ReferenceEventTable

    Attributes
    ----------
    config : Config
        `Config` object with simulation parameters.
    reference_table : ReferenceEventTable
        `ReferenceEventTable` object containing all generic KMC events currently known.
    connectivity_table : StatesConnectivity
        Object that store the connectivity of the current state to other states. It is the object that we want to build when using the Explorer.
    detector : DetectorThreshold
        Detector object used to decide if a discovered state is transient or absorbing.
    """

    def __init__(self, config: Config, reference_table: ReferenceEventTable) -> None:
        self.config = config
        self.reference_table: ReferenceEventTable = reference_table
        self.connectivity_table: StatesConnectivity = BasinStatesConnectivity()
        self.detector = DetectorThreshold()

    def explore(
        self, state: "StateData", state_index: int = 0, start_index: int = 1
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

        # Find all applicable events on the state
        df_applicable_events = self.reference_table.has_id_subset_table(
            state.environment.atomic_environment_list
        )

        # Loop over all applicable events :
        count = 0
        for (
            idx,
            df_event,
        ) in (
            df_applicable_events.iterrows()
        ):  # Note : idx is the original index of the self.reference_table.table
            # check if df_event leads to transient state
            is_transient = self.detector.detect(
                df_event, self.reference_table.table, self.config.basin.energy_thr
            )
            # All atoms on which we can apply the event :
            l_atoms = state.environment.get_atoms_with_id(df_event["event_id"])
            # Find backward info
            backward_idx = self.reference_table.table.loc[idx].at["idx_backward"]
            dE_backward = self.reference_table.table[
                self.reference_table.table["idx_ref"] == backward_idx
            ]["energy_barrier"].values[0]
            #            dE_backward = self.reference_table.table.loc[backward_idx].at["energy_barrier"]
            k_backward = self.reference_table.table[
                self.reference_table.table["idx_ref"] == backward_idx
            ]["k"].values[0]
            # k_backward = self.reference_table.table.loc[backward_idx].at["k"]
            ref_event = self.reference_table.table.loc[idx].at["idx_ref"]

            # Loop over all atoms on which we can apply the event :
            for at in l_atoms:
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
                        dE_forward=df_event["energy_barrier"],
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
