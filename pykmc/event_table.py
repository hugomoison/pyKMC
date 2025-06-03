"""Module implementing Classes to manage reference events and active events."""

import pandas as pd
from .rate_constant import compute_rate_Eyring
from .config import Config
import numpy as np
from .environments.graph_nauty import graph
from .system import System
from .neighbors_list import NeighborsList
from .symmetries import unique_symmetries
from .result import (
    Result,
    ErrorInfo,
    Ok,
    Err,
    ErrorType,
    EventSearchOutput,
    EventRefinementOutput,
)


class ReferenceEventTable:
    """Store reference events and manage them.

    Parameters
    ----------
    config : Config
        The atomic simulations configuration.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._initialize_table()

    def add_events(
        self, events: list[EventSearchOutput]
    ) -> Result[pd.DataFrame, ErrorInfo]:
        """Events events to the table dataframe.

        Parameters
        ----------
        events : list[EventSearchOutput]
            list of EventSearchOutput dataclass with events to be added to the table dataframe.

        Returns
        -------
        Result[pd.DataFrame, ErrorInfo]
            The results of the operation.

        """
        results_is_valid_events = []
        # Check if the event is valid based on is_valid_new_event conditions
        for ev in events:
            results_is_valid_events.append(
                self.is_valid_new_event(
                    min1_positions=ev.min1_positions,
                    saddle_positions=ev.saddle_positions,
                    min2_positions=ev.min2_positions,
                    move_atom_idx=ev.move_atom_index,
                    dE_forward=ev.dE_forward,
                    dE_backward=ev.dE_backward,
                    cell=ev.cell,
                )
            )
        df_valid_events = self.get_valid_events(results_is_valid_events)
        for df in df_valid_events:
            self.add(df)

        return results_is_valid_events

    def is_valid_new_event(
        self,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        move_atom_idx: int,
        dE_forward: float,
        dE_backward: float,
        cell: np.ndarray,
    ) -> Result[pd.DataFrame, ErrorInfo]:
        """Check if the event has the required conditions to be added to the table DataFrame based on the configuration's parameters.

        Parameters
        ----------
        min1_positions : np.ndarray
            event's positions of the first minimum.
        saddle_positions : np.ndarray
            event's positions of the saddle point.
        min2_positions : np.ndarray
            event's positions of the second minimum.
        move_atom_idx : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.
        cell : np.ndarray
            Simulation box cell.

        Returns
        -------
        Result[pd.DataFrame, ErrorInfo]
            The results of the operation.

        """
        # Energy bounds
        emin = self.config.eventsearch.emin_event
        emax = self.config.eventsearch.emax_event
        backward_emin = self.config.eventsearch.backward_emin_event
        energy_asymmetry = self.config.eventsearch.energy_asymmetry

        if dE_forward > emax:  # barrier energy too high, reject the event
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_ENERGY_HIGHER_THAN_THRESHOLD,
                    message="Energy barrier of the event higher than emax_event",
                    details="Energy barrier = {}, energy max threshold = {}".format(
                        dE_forward, emax
                    ),
                )
            )

        elif dE_forward < emin:  # barrier energy too low, reject the event
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_ENERGY_LOWER_THAN_THRESHOLD,
                    message="Energy barrier of the event lower than emin_event",
                    details="Energy barrier = {}, energy min threshold = {}".format(
                        dE_forward, emin
                    ),
                )
            )

        elif (
            dE_backward < emin
        ):  # backard reaction energy barrier too low, reject the event
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD,
                    message="Backward energy barrier of the event lower than emin_event",
                    details="Backward Energy barrier = {}, energy min threshold = {}".format(
                        dE_backward, emin
                    ),
                )
            )

        # TODO Maybe REMOVE THIS, IT SHOULD NOT HAPPEN
        elif (dE_forward > energy_asymmetry * backward_emin) and (
            dE_backward < backward_emin
        ):  # Asymmetric event, reject
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_ASYMMETRIC,
                    message="Found event is highly asymmetric",
                    details="Foward barrier eneryg > {} and backward barrier energy < {}".format(
                        energy_asymmetry * backward_emin, backward_emin
                    ),
                )
            )

        else:  # Event is valid, construct event Series
            dfevent_forward, dfevent_backward = self._build_event_series(
                min1_positions=min1_positions,
                saddle_positions=saddle_positions,
                min2_positions=min2_positions,
                index_move=move_atom_idx,
                dE_forward=dE_forward,
                dE_backward=dE_backward,
                cell=cell,
            )
            if self.is_new_event(
                dfevent=dfevent_forward
            ):  # check if event not already in the catalog
                if (
                    dfevent_forward["event_id"] == dfevent_forward["id_final"]
                ):  # backward reaction same as forward
                    return Ok(dfevent_forward.to_frame().T)  # return only forward event
                else:
                    dfevent = pd.concat(
                        [dfevent_forward.to_frame().T, dfevent_backward.to_frame().T],
                        ignore_index=True,
                    )
                    return Ok(dfevent)  # return foward and backward event

            else:
                return Err(
                    ErrorInfo(
                        type=ErrorType.EVENT_NOT_NEW,
                        message="Found event already in reference table",
                        details="Same topology",
                    )
                )

    def is_new_event(self, dfevent: pd.Series) -> bool:
        """Check if the constructed event Series is already in the table.

        Parameters
        ----------
        dfevent : pd.Series
            the event's Serie.

        Returns
        -------
        bool
            if the event is in the table.

        """
        # Only select rows with same event_id as dfenvent :
        subset = self.table[self.table["event_id"] == dfevent["event_id"]]
        # subset of subset with rows with the same saddle_id :
        subset = subset[subset["id_saddle"] == dfevent["id_saddle"]]
        # subset of subset of subset with rows with the same final_id :
        subset = subset[subset["id_final"] == dfevent["id_final"]]
        if len(subset) == 0:
            return True
        else:
            return False

    def get_valid_events(
        self, results_is_valid_event: list[Result[pd.Series, ErrorInfo]]
    ) -> list[pd.Series]:
        """Return the list of successful Result.

        Parameters
        ----------
        results_is_valid_event : list[Result[pd.Series, ErrorInfo]]
            list of Result containing event to be added to the table, or ErrorInfo.

        Returns
        -------
        list[pd.Series]
            list of successful Result.

        """
        return [e.ok_value() for e in results_is_valid_event if e.is_ok()]

    def add(self, dfevent: pd.Series) -> None:
        """Add on event series to the table.

        Parameters
        ----------
        dfevent : pd.Series
            The event series.

        """
        self.table = pd.concat([self.table, dfevent], ignore_index=True)

    def has_id_subset_table(self, ids: list[str | bytes]) -> pd.DataFrame:
        """Return subset table with event having id in ids.

        Parameters
        ----------
        ids : list[str | bytes]
            list of IDs.

        Returns
        -------
        pd.DataFrame
            Subset of the reference table dataframe with only event having IDs in ids.

        """
        return self.table[self.table["event_id"].isin(ids)]

    def _build_event_series(
        self,
        min1_positions: np.ndarray,
        saddle_positions: np.ndarray,
        min2_positions: np.ndarray,
        index_move: int,
        dE_forward: float,
        dE_backward: float,
        cell: np.ndarray,
    ) -> tuple[pd.Series, pd.Series]:
        """Build foward and backward events Series.

        Parameters
        ----------
        min1_positions : np.ndarray
            event's positions of the first minimum.
        saddle_positions : np.ndarray
            event's positions of the saddle point.
        min2_positions : np.ndarray
            event's positions of the second minimum.
        index_move : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.
        cell : np.ndarray
            Simulation box cell.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            tuple containing :
            - a pd.Series of the foward reaction.
            - a pd.Series of the backward reaction.

        """
        # compute neighbors list for initial, saddle and final positions -> to compute graphs
        min1system = System()
        min1system.positions = min1_positions
        min1system.cell = cell
        min1neighbors_list = NeighborsList(
            min1system,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        saddlesystem = System()
        saddlesystem.positions = saddle_positions
        saddlesystem.cell = cell
        saddleneighbors_list = NeighborsList(
            saddlesystem,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        min2system = System()
        min2system.positions = min2_positions
        min2system.cell = cell
        min2neighbors_list = NeighborsList(
            min2system,
            self.config.atomicenvironment.rnei,
            self.config.atomicenvironment.rcut,
        )

        # TODO need to see how to deal with different style for atomic environment ID
        # Compute all needed topology ID :
        id_min1 = graph(
            min1neighbors_list.neighbors_list["rnei"],
            min1neighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
        )[0]
        id_saddle = graph(
            saddleneighbors_list.neighbors_list["rnei"],
            saddleneighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
        )[0]
        id_min2 = graph(
            min2neighbors_list.neighbors_list["rnei"],
            min2neighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
        )[0]

        neighbor_list_forwward = min1neighbors_list.neighbors_list["rcut"][index_move]
        neighbor_list_backward = min2neighbors_list.neighbors_list["rcut"][index_move]

        # Symmetries :
        sym_matrix, sym_perm = unique_symmetries(
            min1_positions[neighbor_list_forwward],
            min2_positions[neighbor_list_forwward],
            self.config.ira.sym_thr,
        )
        dfevent_forward = pd.Series(
            {
                "event_id": id_min1,
                "initial_positions": min1_positions[neighbor_list_forwward],
                "saddle_positions": saddle_positions[neighbor_list_forwward],
                "final_positions": min2_positions[neighbor_list_forwward],
                "energy_barrier": dE_forward,
                "k": compute_rate_Eyring(dE_forward, self.config),
                "id_saddle": id_saddle,
                "id_final": id_min2,
                "move_atom_idx": np.where(neighbor_list_forwward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                'backward_event_id' : -1,                                          
                'backward_energy_barrier' : dE_backward
            }
        )

        sym_matrix, sym_perm = unique_symmetries(
            min2_positions[neighbor_list_backward],
            min1_positions[neighbor_list_backward],
            self.config.ira.sym_thr,
        )
        dfevent_backward = pd.Series(
            {
                "event_id": id_min2,
                "initial_positions": min2_positions[neighbor_list_backward],
                "saddle_positions": saddle_positions[neighbor_list_backward],
                "final_positions": min1_positions[neighbor_list_backward],
                "energy_barrier": dE_backward,
                "k": compute_rate_Eyring(dE_backward, self.config),
                "id_saddle": id_saddle,
                "id_final": id_min1,
                "move_atom_idx": np.where(neighbor_list_backward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                'backward_event_id' : -1,                                         
                'backward_energy_barrier' : dE_forward
            }
        )

        return dfevent_forward, dfevent_backward

    def _initialize_table(self) -> None:
        """Initialize the reference event table.

        If a path to a reference table is in the configurations it reads it, otherwise initialize an empty dataframe.
        """
        if self.config.control.reference_table is not None:
            self.table = pd.read_pickle(self.config.control.reference_table)
        else:
            self.table = pd.DataFrame(
                columns=[
                    "event_id",
                    "initial_positions",
                    "saddle_positions",
                    "final_positions",
                    "energy_barrier",
                    "k",
                    "id_saddle",
                    "id_final",
                    "move_atom_idx",
                    "sym_matrix",
                    "sym_perm",
                    'backward_event',           
                    'backward_energy_barrier'
                ]
            )

    def save(self, outfile: str = "reference_table.pickle") -> None:
        """Save the reference event table to a pickle file.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'reference_table.pickle'.

        """
        self.table.to_pickle(outfile)


class ActiveEventTable:
    """Store active events and manage them.

    Parameters
    ----------
    config : Config
        The atomic simulations configuration.
    event_dataframe : pd.DataFrame, optional
        An table with active event use to initialize the table. by default 'None'.

    """

    def __init__(self, config: Config, event_dataframe: pd.DataFrame = None):
        self.config = config

        if event_dataframe is not None:
            if not isinstance(event_dataframe, pd.DataFrame):
                raise TypeError("event_dataframe must be a pandas DataFrame or None.")
            self.table = event_dataframe
        else:
            columns = {
                "atom_index": pd.Series(dtype="int64"),
                "final_positions": pd.Series(dtype="object"),
                "energy_barrier": pd.Series(dtype="float64"),
                "k": pd.Series(dtype="float64"),
                "num_reference_event": pd.Series(dtype="int64"),
                "backward_energy_barrier": pd.Series(dtype="float64")
            }
            self.table = pd.DataFrame(columns)

    def add_events(self, events: EventRefinementOutput | list[EventRefinementOutput]) -> None:
        """Add active events to the table.

        Parameters
        ----------
        events : EventRefinementOutput | list[EventRefinementOutput]
            An EventRefinementOuput dataclass, or a list of it, with active event to be added to the table.

        Raises
        ------
        TypeError
            if events is not a EventRefinementOuput dataclass or a list of it.

        """
        if isinstance(events, list):
            dfactive = []
            for e in events:
                dfactive.append(self.build_event_series(e))
        elif isinstance(events, EventRefinementOutput):
            dfactive = self.build_event_series(events)
        else:
            raise TypeError(
                "Input 'events' must be an EventRefinementOutput dataclass or a list of it."
            )
        self.add(dfactive)

    def add(self, dfevents: pd.Series | list[pd.Series]) -> None:
        """Add a pd.Series of the active events.

        Parameters
        ----------
        dfevents : pd.Series | list[pd.Series]
            a pd.Series of an event to be added to the table, or a list of it.

        Raises
        ------
        TypeError
            if dfevents is not a pd.Series.

        """
        if isinstance(dfevents, pd.Series):
            df_to_add = dfevents.to_frame().T
        elif isinstance(dfevents, list):
            if not all(isinstance(s, pd.Series) for s in dfevents):
                raise TypeError("All elements in the input list must be pandas Series.")
            df_to_add = pd.DataFrame(dfevents)
        else:
            raise TypeError(
                "Input 'dfevents' must be a pandas Series or a list of pandas Series."
            )

        self.table = pd.concat([self.table, df_to_add], ignore_index=True)

    def build_event_series(
        self, event_search_output: EventRefinementOutput
    ) -> pd.Series:
        """Build an event Series based on the EventRefinementOuput dataclass.

        Parameters
        ----------
        event_search_output : EventRefinementOutput
            The dataclass with the active event informations.

        Returns
        -------
        pd.Series
            The pd.Series of the event.

        """
        dfactive = pd.Series(
            {
                "atom_index": event_search_output.central_atom_index,
                "final_positions": event_search_output.min2_positions,
                "energy_barrier": event_search_output.dE_forward,
                "k": compute_rate_Eyring(event_search_output.dE_forward, self.config),
                "num_reference_event": event_search_output.num_reference_event,
                "backward_energy_barrier": event_search_output.dE_backward
            }
        )
        return dfactive
