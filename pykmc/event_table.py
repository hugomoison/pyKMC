"""Module implementing Classes to manage reference events and active events."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from .point_set_registration import simple_ira, check_match
from .utils.geometry import align_positions_by_neighbors, compute_delr

if TYPE_CHECKING:
    from .event_recycling import Recycling


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
            res = self.is_valid_new_event(
                min1_positions=ev.min1_positions,
                saddle_positions=ev.saddle_positions,
                min2_positions=ev.min2_positions,
                move_atom_idx=ev.move_atom_index,
                dE_forward=ev.dE_forward,
                dE_backward=ev.dE_backward,
                cell=ev.cell,
                types=ev.types,
            )
            results_is_valid_events.append(res)
            if res.is_ok():
                self.add(res.ok_value())
        # df_valid_events = self.get_valid_events(results_is_valid_events)

        # Check if events in results are not the same :

        # for df in df_valid_events:
        #    self.add(df)

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
        types: list[str] = None,
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
        types : list[str]
            Event's atom types.

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
                types=types,
            )
            if self.is_new_event(
                dfevent=dfevent_forward
            ):  # check if event not already in the catalog
                if (
                    dfevent_forward["event_id"] == dfevent_forward["id_final"]
                ):  # We are sure that the backward reaction same as forward
                    # dfevent_forward["idx_backward"] = len(self.table)
                    return Ok(dfevent_forward.to_frame().T)  # return only forward event

                # TODO : this is the same logic as is_new_event(), it is a quick fix but need to unify this
                # TODO : will be easier when refacto ReferenceTable with Event dataclass
                # backward event could still be the same as the forward one :

                elif (
                    dfevent_forward["event_id"] == dfevent_backward["event_id"]
                ):  # same topo
                    if (
                        abs(
                            dfevent_forward["energy_barrier"]
                            - dfevent_backward["energy_barrier"]
                        )
                        < 0.25
                    ):  # maybe same event so IRA check
                        ref_saddle = dfevent_forward["saddle_positions"].copy()
                        nat_ref = len(ref_saddle)
                        typ_event = nat_ref * ["X"]
                        typ_ref = typ_event
                        result = simple_ira(
                            nat_ref,
                            typ_event,
                            dfevent_backward["saddle_positions"].copy(),
                            nat_ref,
                            typ_ref,
                            ref_saddle,
                            self.config.ira.kmax_factor,
                        )

                        # if match
                        if result.is_ok():
                            # if matching score
                            result = check_match(
                                result, self.config.psr.matching_score_thr
                            )
                            if result.is_ok():  # same backward and forward event
                                return Ok(dfevent_forward.to_frame().T)
                        else:
                            if self.is_new_event(dfevent=dfevent_backward):
                                dfevent = pd.concat(
                                    [
                                        dfevent_forward.to_frame().T,
                                        dfevent_backward.to_frame().T,
                                    ],
                                    ignore_index=True,
                                )
                                return Ok(dfevent)  # return both
                            else:
                                return Ok(dfevent_forward.to_frame().T)

                # we know they are different
                else:
                    # to the atomic environment of the forward event
                    if self.is_new_event(dfevent=dfevent_backward):
                        # backward is also new
                        dfevent = pd.concat(
                            [
                                dfevent_forward.to_frame().T,
                                dfevent_backward.to_frame().T,
                            ],
                            ignore_index=True,
                        )
                        return Ok(dfevent)  # return foward and backward event
                    else:
                        # backard is already known
                        return Ok(dfevent_forward.to_frame().T)  # return only forward

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
        if len(subset) == 0:
            return True

        # if same  id, chekc if same dE
        tol = 0.25
        dE = dfevent["energy_barrier"]
        subset = subset[(subset["energy_barrier"] - dE).abs() <= tol]
        if len(subset) == 0:
            return True

        # if all same, check PSR  saddle_initial
        event_saddle = dfevent["saddle_positions"]
        nat_event = len(event_saddle)
        # Mirror the PSR / classification paths: only feed real element types to IRA
        # in "full" coloring mode. In "grey" mode every atom is greyed to a single
        # dummy label ('X'), so geometrically-identical species-swapped saddles
        # de-duplicate as one event (grey-alloy approximation). The None-"types"
        # fallback keeps list(None) from raising when a row stores no types.
        full = self.config.atomicenvironment.atom_coloring_mode == "full"
        typ_event = (
            list(dfevent["types"])
            if full and dfevent["types"] is not None
            else nat_event * ["X"]
        )

        for _, ev in subset.iterrows():
            ref_saddle = ev["saddle_positions"]
            nat_ref = len(ref_saddle)
            typ_ref = (
                list(ev["types"])
                if full and ev["types"] is not None
                else nat_ref * ["X"]
            )

            result = simple_ira(
                nat_event,
                typ_event,
                event_saddle,
                nat_ref,
                typ_ref,
                ref_saddle,
                self.config.ira.kmax_factor,
            )

            if not result.is_ok():  # no match
                continue

            result = check_match(result, self.config.psr.matching_score_thr)
            if not result.is_ok():  # matching score > thr
                continue

            return False
        return True

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
        # Check if only one or two events (if event is its own backard or not)
        ref = self.max_idx_ref()
        if len(dfevent) == 1:
            dfevent["idx_ref"] = ref
            dfevent["idx_backward"] = ref
        else:
            dfevent.loc[0, "idx_ref"] = ref
            dfevent.loc[0, "idx_backward"] = ref + 1
            dfevent.loc[1, "idx_ref"] = ref + 1
            dfevent.loc[1, "idx_backward"] = ref

        self.table = pd.concat([self.table, dfevent], ignore_index=True)

    def has_id_subset_table(self, ids: list[str]) -> pd.DataFrame:
        """Return subset table with event having id in ids.

        Parameters
        ----------
        ids : list[str]
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
        types: list[str] = None,
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
        types : list[str], optional
            Element type of each atom. When provided, the per-event local types are
            always stored in the ``types`` column (both coloring modes, so the schema
            is mode-independent). Colouring is only *applied* to graph
            hashing/symmetry detection when the configured coloring mode is 'full'.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            tuple containing :
            - a pd.Series of the foward reaction.
            - a pd.Series of the backward reaction.

        """
        full = self.config.atomicenvironment.atom_coloring_mode == "full"
        # Only use element types for graph/symmetry computation in full coloring mode
        graph_types = types if full else None

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
            types=graph_types,
        )[0]
        id_saddle = graph(
            saddleneighbors_list.neighbors_list["rnei"],
            saddleneighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
            types=graph_types,
        )[0]
        id_min2 = graph(
            min2neighbors_list.neighbors_list["rnei"],
            min2neighbors_list.neighbors_list["rcut"],
            atom_idx=[index_move],
            types=graph_types,
        )[0]

        # query_ball_point can hand back Python lists; coerce to arrays so the
        # element-wise comparisons (np.where) and type indexing below behave.
        neighbor_list_forward = np.asarray(
            min1neighbors_list.neighbors_list["rcut"][index_move]
        )
        neighbor_list_backward = np.asarray(
            min2neighbors_list.neighbors_list["rcut"][index_move]
        )

        # Element types of each neighbor. These are ALWAYS stored (both modes) so
        # the reference-table schema is mode-independent; colouring is only *applied*
        # in matching/symmetry, which is gated below.
        local_types_forward = (
            list(np.array(types)[neighbor_list_forward]) if types is not None else None
        )
        local_types_backward = (
            list(np.array(types)[neighbor_list_backward]) if types is not None else None
        )

        # Colour symmetry detection only in full coloring mode (usage gate).
        sym_types_forward = local_types_forward if full else None
        sym_types_backward = local_types_backward if full else None

        # Symmetries :
        sym_matrix, sym_perm = unique_symmetries(
            min1_positions[neighbor_list_forward],
            min2_positions[neighbor_list_forward],
            self.config.ira.sym_thr,
            types=sym_types_forward,
        )

        # dr :
        move_atom_idx_forward = np.where(neighbor_list_forward == index_move)[0][0]
        dra_forward = np.linalg.norm(
            min1_positions[neighbor_list_forward][move_atom_idx_forward]
            - saddle_positions[neighbor_list_forward][move_atom_idx_forward]
        )
        move_atom_idx_backward = np.where(neighbor_list_backward == index_move)[0][0]
        dra_backward = np.linalg.norm(
            min2_positions[neighbor_list_backward][move_atom_idx_backward]
            - saddle_positions[neighbor_list_backward][move_atom_idx_backward]
        )

        dfevent_forward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "event_id": id_min1,
                "initial_positions": min1_positions[neighbor_list_forward],
                "saddle_positions": saddle_positions[neighbor_list_forward],
                "final_positions": min2_positions[neighbor_list_forward],
                "types": local_types_forward,
                "energy_barrier": dE_forward,
                "k": compute_rate_Eyring(dE_forward, self.config),
                "id_saddle": id_saddle,
                "id_final": id_min2,
                "move_atom_idx": np.where(neighbor_list_forward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                "idx_backward": -1,  # unknown yet,
                "dra": dra_forward,
            }
        )

        sym_matrix, sym_perm = unique_symmetries(
            min2_positions[neighbor_list_backward],
            min1_positions[neighbor_list_backward],
            self.config.ira.sym_thr,
            types=sym_types_backward,
        )
        dfevent_backward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "event_id": id_min2,
                "initial_positions": min2_positions[neighbor_list_backward],
                "saddle_positions": saddle_positions[neighbor_list_backward],
                "final_positions": min1_positions[neighbor_list_backward],
                "types": local_types_backward,
                "energy_barrier": dE_backward,
                "k": compute_rate_Eyring(dE_backward, self.config),
                "id_saddle": id_saddle,
                "id_final": id_min1,
                "move_atom_idx": np.where(neighbor_list_backward == index_move)[0][0],
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                "idx_backward": -1,  # unknown yet
                "dra": dra_backward,
            }
        )

        return dfevent_forward, dfevent_backward

    def max_idx_ref(self) -> int:
        """Return max value of idx_ref"""
        if len(self.table) == 0:
            return 0
        else:
            return int(self.table["idx_ref"].max()) + 1

    def _initialize_table(self) -> None:
        """Initialize the reference event table.

        If a path to a reference table is in the configurations it reads it, otherwise initialize an empty dataframe.
        """
        if self.config.control.reference_table is not None:
            self.table = pd.read_pickle(self.config.control.reference_table)
        else:
            self.table = pd.DataFrame(
                {
                    "idx_ref": pd.Series(dtype="int64"),
                    "event_id": pd.Series(dtype="str"),
                    "initial_positions": pd.Series(dtype="object"),
                    "saddle_positions": pd.Series(dtype="object"),
                    "final_positions": pd.Series(dtype="object"),
                    "types": pd.Series(dtype="object"),
                    "energy_barrier": pd.Series(dtype="float64"),
                    "k": pd.Series(dtype="float64"),
                    "id_saddle": pd.Series(dtype="str"),
                    "id_final": pd.Series(dtype="str"),
                    "move_atom_idx": pd.Series(dtype="int64"),
                    "sym_matrix": pd.Series(dtype="object"),
                    "sym_perm": pd.Series(dtype="object"),
                    "idx_backward": pd.Series(dtype="int64"),
                    "dra": pd.Series(dtype="float64"),
                }
            )

    def remove(self, idx_refs: list[int]) -> None:
        """Remove events with ind == idx_ref as well as its backward event

        Parameters
        ----------
        ind : int
            index of the event to be removed
        """

        idx_refs = set(idx_refs)  # make a set if there are doublons

        backward_refs = set(
            self.table.loc[self.table["idx_ref"].isin(idx_refs), "idx_backward"].astype(
                int
            )
        )  # find set idx backwards

        all_refs = idx_refs | backward_refs  # all ref to remove

        self.table = self.table[~self.table["idx_ref"].isin(all_refs)].reset_index(
            drop=True
        )  # keep event not (~) in all refs

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

    def __init__(
        self,
        config: Config,
        event_dataframe: pd.DataFrame = None,
        recycler: "Recycling | None" = None,
    ):
        self.config = config
        # Optional recycling plugin. If attached, `prune_for_recycling` keeps
        # the rows the recycler selects between KMC steps. If None, the table
        # is cleared at the end of each step (matching prior behavior).
        self.recycler = recycler

        if event_dataframe is not None:
            if not isinstance(event_dataframe, pd.DataFrame):
                raise TypeError("event_dataframe must be a pandas DataFrame or None.")
            self.table = event_dataframe
        else:
            columns = {
                "atom_index": pd.Series(dtype="int64"),
                "saddle_positions": pd.Series(dtype="object"),
                "final_positions": pd.Series(dtype="object"),
                "neighbors": pd.Series(dtype="object"),
                "energy_barrier": pd.Series(dtype="float64"),
                "k": pd.Series(dtype="float64"),
                "num_reference_event": pd.Series(dtype="int64"),
                "refined": pd.Series(dtype="str"),
            }
            self.table = pd.DataFrame(columns)

    def prune_for_recycling(
        self,
        executed_idx: int,
        system: System,
        positions_pre: np.ndarray,
    ) -> None:
        """Replace `self.table` with the rows that survive the recycler's filter.

        If no recycler is attached, clear the table (matches the prior
        end-of-step `del active_table` behavior).
        """
        if self.recycler is None:
            self.table = self.table.iloc[0:0].reset_index(drop=True)
        else:
            self.table = self.recycler.select_recyclable(
                self,
                executed_idx,
                system,
                positions_pre,
            )
            # Mark carried-over rows in the events log: a recycled row shows as
            # 'RT'/'RF' (recycled; originally refined/unrefined) instead of
            # 'T'/'F'. Idempotent for rows recycled over multiple steps.
            if len(self.table) > 0:
                self.table["refined"] = [
                    "R" + r if isinstance(r, str) and not r.startswith("R") else r
                    for r in self.table["refined"]
                ]

    def existing_pairs(self) -> set[tuple[int, int]]:
        """Return `(atom_index, num_reference_event)` tuples already in the table.

        Used by `Refinement.execute` to skip pairs that survived the last step
        and don't need to be refined again.
        """
        if len(self.table) == 0:
            return set()
        return set(
            zip(
                self.table["atom_index"].astype(int).tolist(),
                self.table["num_reference_event"].astype(int).tolist(),
            )
        )

    def add_events(
        self, events: EventRefinementOutput | list[EventRefinementOutput]
    ) -> None:
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
        self, event_refinement_output: EventRefinementOutput
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
                "atom_index": event_refinement_output.central_atom_index,
                "saddle_positions": event_refinement_output.saddle_positions,
                "final_positions": event_refinement_output.min2_positions,
                "neighbors": event_refinement_output.neighbors,
                "energy_barrier": event_refinement_output.dE_forward,
                "k": compute_rate_Eyring(
                    event_refinement_output.dE_forward, self.config
                ),
                "num_reference_event": event_refinement_output.num_reference_event,
                "refined": event_refinement_output.refined,
            }
        )
        return dfactive

    def remove(self, ind: int | list[int]) -> None:
        """Remove event at row = ind

        Parameters
        ----------
        ind : int
            index of the row to be removed
        """
        self.table = self.table.drop(ind)
        self.table = self.table.reset_index(drop=True)

    def remove_duplicates(self, cell, neighbors_list: NeighborsList = None) -> None:
        """Loop over all active events in the DataFrame, check if there are duplicates by computing delr."""

        duplicates = []
        # 1. Check duplicates on central atoms : to be sure
        # Sub dataframes with events grouped by central_atom and dE
        tol_energy = 0.1  # eV
        grouped = []

        for idx, row in self.table.iterrows():
            central_atom = row["atom_index"]
            dE = row["energy_barrier"]

            subset = self.table[
                (self.table["atom_index"] == central_atom)
                & (abs(self.table["energy_barrier"] - dE) < tol_energy)
            ]
            grouped.append((idx, subset))

        # For each group, check duplicated by computing delr

        for idx, subset in grouped:
            pos_ref = np.array(self.table.loc[idx, "saddle_positions"])
            nb_ref = self.table.loc[idx, "neighbors"]
            for jdx in subset.index:
                if jdx <= idx:
                    continue  # dont compute twice
                pos_comp = np.array(self.table.loc[jdx, "saddle_positions"])
                nb_comp = self.table.loc[jdx, "neighbors"]
                # Align on atom id via the stored per-row `neighbors` ordering
                # before comparing: recycled rows keep their event-time ordering
                # while fresh rows follow the current NeighborsList, so a raw
                # element-wise compute_delr would compare non-corresponding
                # atoms -- or crash outright when the two stored environments
                # have different sizes. None/length-mismatched neighbours are
                # not-comparable -> keep both rows.
                aligned = align_positions_by_neighbors(
                    nb_ref, pos_ref, nb_comp, pos_comp
                )
                if aligned is None:
                    continue
                aligned_ref, aligned_comp, sets_equal = aligned
                # Same central atom: differing neighbour *sets* means a different
                # local environment (membership drifted after the system moved),
                # so the two rows are not duplicates.
                if not sets_equal:
                    continue
                delr = compute_delr(aligned_ref, aligned_comp, cell)
                if delr < self.config.psr.matching_score_thr:
                    # print('Removing event with delr',delr)
                    duplicates.append(jdx)

        # 2. Check duplicates due to symmetric events applied on different central atoms.
        # Group by same generic event if generic event has symmetries meaning that the same generic event has been applied to same central atom
        # The alignment uses each row's STORED `neighbors` ordering (the
        # `neighbors` column) instead of a fresh get_neighbors('rcut', ...)
        # re-derivation: `saddle_positions` is ordered by the neighbours captured
        # at event time, and re-deriving the order from the current NeighborsList
        # scatters coordinates onto the wrong atoms for recycled rows.
        # `neighbors_list` is retained only as the gate that enables this
        # symmetric pass.
        if (
            neighbors_list is not None
        ):  # need neighbors list to remove symmetric duplicates
            counts = self.table.groupby(["atom_index", "num_reference_event"]).size()
            symmetric_num_ref = counts[counts > 1].index.get_level_values(1).unique()

            # Loop on all num_ref symmetric event
            for num_ref in symmetric_num_ref:
                subset = self.table[self.table["num_reference_event"] == num_ref]
                indices = subset.index.to_list()

                for i, idx in enumerate(indices):  # Loop over indice of subset
                    central_atom1 = subset.loc[idx, "atom_index"]
                    env1 = subset.loc[
                        idx, "neighbors"
                    ]  # stored ids for saddle_positions of row idx

                    for jdx in indices[i + 1 :]:  # to not compare two times
                        central_atom2 = subset.loc[jdx, "atom_index"]
                        if (
                            central_atom1 != central_atom2
                        ):  # if yes already done in part 1.
                            env2 = subset.loc[
                                jdx, "neighbors"
                            ]  # stored ids for saddle_positions of row jdx
                            if env1 is None or env2 is None:
                                continue  # ordering unknown -> not comparable

                            if central_atom1 not in set(
                                np.asarray(env2, dtype=int).tolist()
                            ):  # TODO : To check, but should not be a duplicate
                                continue

                            # extract saddle positions
                            sad_pos1 = subset.loc[idx, "saddle_positions"]
                            sad_pos2 = subset.loc[jdx, "saddle_positions"]

                            # Align on the SHARED atoms via the stored neighbour
                            # ids. Unlike part 1 (same central atom, full-set
                            # match required), a symmetric event on a different
                            # central atom legitimately spans a different rcut
                            # shell, so here we deliberately compare over the
                            # intersection and ignore `sets_equal`.
                            aligned = align_positions_by_neighbors(
                                env1, sad_pos1, env2, sad_pos2
                            )
                            if aligned is None:
                                continue  # no shared atoms / length mismatch -> keep both
                            sad_pos1, sad_pos2, _sets_equal = aligned

                            # now we can compare
                            delr = compute_delr(sad_pos1, sad_pos2, cell)
                            if delr < self.config.psr.matching_score_thr:
                                duplicates.append(jdx)

        self.remove(duplicates)

    def save(self, outfile: str = "active_table.pickle") -> None:
        """Save the reference event table to a pickle file.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'active_table.pickle'.

        """
        self.table.to_pickle(outfile)
