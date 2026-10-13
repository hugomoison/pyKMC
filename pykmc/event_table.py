"""Module implementing Classes to manage reference events and active events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging
import pandas as pd
from .rate_constant import compute_rate_Eyring
from .parameters import Parameters
import numpy as np
from .environments.graph_nauty import combine_ids
from .atomic_environment import compute_atomic_environment_id
from .system import System, Configuration
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
    ShapeID,
    SaddleID,
)
from .point_set_registration import simple_ira, check_match
from .utils.geometry import compute_delr_max, minimum_image_distance, unwrap_around
from .shape_table import ShapeTable


_LOGGER = logging.getLogger("log")


if TYPE_CHECKING:
    from .event_recycling import Recycling


class ReferenceEventTable:
    """Store reference events and manage them.

    Parameters
    ----------
    params : Parameters
        The atomic simulations configuration.

    """

    def __init__(self, params: Parameters) -> None:
        self.params = params
        self._initialize_table()
        self.shapes = ShapeTable(params)

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
                min1=ev.min1,
                saddle=ev.saddle,
                min2=ev.min2,
                move_atom_idx=ev.move_atom_index,
                dE_forward=ev.dE_forward,
                dE_backward=ev.dE_backward,
            )
            results_is_valid_events.append(res)
            if res.is_ok():
                self.add(res.ok_value())
            self._record_knowledge_from_result(res)
        # df_valid_events = self.get_valid_events(results_is_valid_events)

        # Check if events in results are not the same :

        # for df in df_valid_events:
        #    self.add(df)

        return results_is_valid_events

    def _record_knowledge_from_result(
        self, res: Result[pd.DataFrame, ErrorInfo]
    ) -> None:
        """Update persistent per-`ShapeID` knowledge from one `is_valid_new_event` outcome.

        Only outcomes that reach the duplicate-vs-new decision -- `Ok`
        (a genuinely new forward+backward pair) or `Err(EVENT_NOT_NEW)` (a
        rediscovered forward match) -- carry any knowledge; energy/asymmetry/
        not-distinct rejections are not a sighting of any shape's event
        population and are ignored here, mirroring
        `pykmc.adaptive_search.record_draw`'s exclusion.

        Called unconditionally, regardless of which atom's search produced
        `res` or whether its shape has an open per-step session -- this is
        what lets a disconnected/opportunistic discovery still update the
        record for whichever shape it actually belongs to.
        """
        if res.is_ok():
            added = res.ok_value()
            for _, row in added.iterrows():
                self.shapes.record_shape_knowledge(
                    row["id_initial"], int(row["sid_initial"]), int(row["idx_ref"]), float(row["k"])
                )
            return

        err = res.err_value()
        if err.type is not ErrorType.EVENT_NOT_NEW:
            return
        matched_idx_ref = err.variables["matched_idx_ref"]
        match = self.table.loc[self.table["idx_ref"] == matched_idx_ref]
        if len(match):
            self.shapes.record_shape_knowledge(
                match.iloc[0]["id_initial"],
                int(match.iloc[0]["sid_initial"]),
                int(matched_idx_ref),
                float(match.iloc[0]["k"]),
            )

    def resolve_forward_and_backward_rows(
        self, res: Result[pd.DataFrame, ErrorInfo]
    ) -> tuple[pd.Series, pd.Series] | None:
        """Return (forward_row, backward_row) for one `is_valid_new_event` outcome.

        A fresh `Ok` already carries both rows directly. An
        `Err(EVENT_NOT_NEW)` only carries the matched row's `idx_ref`, so its
        other side is fetched via `idx_backward` -- trusted, not
        re-verified: `is_valid_new_event` guarantees every accepted pair is
        real and mutual.

        Returns `None` for any outcome that isn't evidence about an event's
        shape (energy filters, `EVENT_MINIMA_NOT_DISTINCT`, ...), mirroring
        `pykmc.adaptive_search.record_draw`'s exclusion.
        """
        if res.is_ok():
            added = res.ok_value()
            return added.iloc[0], added.iloc[1]

        err = res.err_value()
        if err.type is not ErrorType.EVENT_NOT_NEW:
            return None
        fwd_row = self.table.loc[
            self.table["idx_ref"] == err.variables["matched_idx_ref"]
        ].iloc[0]
        bwd_row = self.table.loc[
            self.table["idx_ref"] == int(fwd_row["idx_backward"])
        ].iloc[0]
        return fwd_row, bwd_row

    def is_valid_new_event(
        self,
        min1: Configuration,
        saddle: Configuration,
        min2: Configuration,
        move_atom_idx: int,
        dE_forward: float,
        dE_backward: float,
    ) -> Result[pd.DataFrame, ErrorInfo]:
        """Check if the event has the required conditions to be added to the table DataFrame based on the configuration's parameters.

        Parameters
        ----------
        min1 : Configuration
            event's types/positions/cell of the first minimum.
        saddle : Configuration
            event's types/positions/cell of the saddle point.
        min2 : Configuration
            event's types/positions/cell of the second minimum.
        move_atom_idx : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.

        Returns
        -------
        Result[pd.DataFrame, ErrorInfo]
            The results of the operation.

        """
        # Energy bounds
        emin = self.params.eventsearch.emin_event
        emax = self.params.eventsearch.emax_event
        backward_emin = self.params.eventsearch.backward_emin_event
        energy_asymmetry = self.params.eventsearch.energy_asymmetry

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
                min1=min1,
                saddle=saddle,
                min2=min2,
                index_move=move_atom_idx,
                dE_forward=dE_forward,
                dE_backward=dE_backward,
            )

            if dfevent_forward["id_initial"] == dfevent_forward["id_final"]:
                # Same coarse topology on both sides -- ambiguous from the
                # coarse id alone: this could be a genuine self-reversing
                # event (e.g. a vacancy hop, where the local graph hash is
                # identical before/after but the two configurations are
                # real, distinct points), or a degenerate ARTn search that
                # never actually moved anywhere. Fine IRA on the actual
                # positions disambiguates, replacing the old raw whole-array
                # `compute_delr_max(min1_positions, min2_positions)` check.
                full = self.params.atomicenvironment.coloring_mode == "full"
                initial_configuration = dfevent_forward["initial_configuration"]
                final_configuration = dfevent_forward["final_configuration"]
                distinct_check = simple_ira(
                    initial_configuration,
                    final_configuration,
                    self.params.ira.kmax_factor,
                    full=full,
                )
                distinct_check = check_match(
                    distinct_check, self.params.psr.matching_score_thr
                )
                if distinct_check.is_ok():
                    return Err(
                        ErrorInfo(
                            type=ErrorType.EVENT_MINIMA_NOT_DISTINCT,
                            message=(
                                "min1 and min2 are the same shape within "
                                "matching_score_thr {}"
                            ).format(self.params.psr.matching_score_thr),
                            variables={
                                "matching_score": distinct_check.ok_value().matching_score
                            },
                        )
                    )

            forward_is_new, forward_matched_idx_ref = self.is_new_event(
                dfevent=dfevent_forward
            )
            if forward_is_new:  # check if event not already in the catalog
                # Always mint a fresh, dedicated backward partner together
                # with the forward row, keeping idx_backward a strict 1:1
                # pairing.
                dfevent = pd.concat(
                    [
                        dfevent_forward.to_frame().T,
                        dfevent_backward.to_frame().T,
                    ],
                    ignore_index=True,
                )
                return Ok(dfevent)  # return foward and backward event

            else:
                return Err(
                    ErrorInfo(
                        type=ErrorType.EVENT_NOT_NEW,
                        message="Found event already in reference table",
                        details="Same topology",
                        variables={"matched_idx_ref": forward_matched_idx_ref},
                    )
                )

    def _rows_with_shape(
        self, df: pd.DataFrame, id_column: str, sid_column: str, shape: ShapeID | SaddleID
    ) -> pd.DataFrame:
        """Rows of `df` whose `(id_column, sid_column)` pair equals `shape`."""
        return df[(df[id_column] == shape.id) & (df[sid_column] == shape.sid)]

    def is_new_event(self, dfevent: pd.Series) -> tuple[bool, int | None]:
        """Check if the constructed event Series is already in the table.

        Parameters
        ----------
        dfevent : pd.Series
            the event's Serie.

        Returns
        -------
        tuple[bool, int | None]
            (is_new, matched_idx_ref). matched_idx_ref is None when the event is
            new, otherwise the idx_ref of the pre-existing row it matched.

        """
        # Coarse topology pre-filter, narrowed to the specific initial
        # ShapeID; the sid_saddle check below decides true pathway-level
        # duplication within that ShapeID:
        subset = self._rows_with_shape(
            self.table,
            "id_initial",
            "sid_initial",
            ShapeID(dfevent["id_initial"], int(dfevent["sid_initial"])),
        )
        if len(subset) == 0:
            return True, None

        # if same id, check if same dE
        tol = 0.25
        dE = dfevent["dE_forward"]
        subset = subset[(subset["dE_forward"] - dE).abs() <= tol]
        if len(subset) == 0:
            return True, None

        # dfevent's saddle shape was already resolved into a sid_saddle by
        # _build_event_series() (via resolve_sid("saddle", ...)) before this method
        # was ever called -- reuse that instead of re-deriving it with a
        # second IRA pass over every row in subset.
        matches = self._rows_with_shape(
            subset,
            "id_saddle",
            "sid_saddle",
            SaddleID(dfevent["id_saddle"], int(dfevent["sid_saddle"])),
        )
        if len(matches) == 0:
            return True, None
        return False, int(matches.iloc[0]["idx_ref"])

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
            # idx_backward is -1 ("unknown yet") only for a true self-backward
            # event; if the backward reaction was already known elsewhere in
            # the table, is_valid_new_event already resolved idx_backward to
            # that pre-existing row's idx_ref, and must not be overwritten here.
            if dfevent["idx_backward"].iloc[0] == -1:
                dfevent["idx_backward"] = ref
        else:
            dfevent.loc[0, "idx_ref"] = ref
            dfevent.loc[0, "idx_backward"] = ref + 1
            dfevent.loc[1, "idx_ref"] = ref + 1
            dfevent.loc[1, "idx_backward"] = ref

        self.table = pd.concat([self.table, dfevent], ignore_index=True)

    def live_events(self, atom_shapes: dict[int, ShapeID]):
        """Yield (atom_idx, event) for every live atom whose actual `ShapeID` matches a catalogued event.

        An atom sharing an event's `id_initial` does not necessarily share
        its actual `ShapeID`; an event with no currently-matching atom
        simply produces no pairs.

        This is the primitive behind "which known reactions currently apply,
        and to which atoms": used by `KMC.build_refinement_candidates()`
        (via `KMC.resolve_new_shapes()`'s own already-resolved map) to
        build this step's refinement worklist, and by
        `BasinGenericEventExplorer.explore()` (via its own map, resolved
        against the candidate basin state) to build basin connectivity.

        Parameters
        ----------
        atom_shapes : dict[int, ShapeID]
            Every relevant atom's already-resolved `ShapeID`; an atom with
            no known shape is simply absent (or mapped to `None`), never
            reclassified here.

        Yields
        ------
        tuple[int, pd.Series]
            One live atom index and the one catalogued event its actual
            `ShapeID` matches.

        """
        atoms_by_shape: dict[ShapeID, list[int]] = {}
        for at_idx, shape in atom_shapes.items():
            if shape is None:
                continue
            atoms_by_shape[shape] = atoms_by_shape.get(shape, []) + [at_idx]

        live_ids = {shape.id for shape in atoms_by_shape}
        subset = self.table[self.table["id_initial"].isin(live_ids)]

        for _idx, row in subset.iterrows():
            shape = ShapeID(row["id_initial"], int(row["sid_initial"]))
            for at_idx in atoms_by_shape.get(shape, []):
                yield at_idx, row

    def _build_event_series(
        self,
        min1: Configuration,
        saddle: Configuration,
        min2: Configuration,
        index_move: int,
        dE_forward: float,
        dE_backward: float,
    ) -> tuple[pd.Series, pd.Series]:
        """Build foward and backward events Series.

        Parameters
        ----------
        min1 : Configuration
            event's types/positions/cell of the first minimum.
        saddle : Configuration
            event's types/positions/cell of the saddle point.
        min2 : Configuration
            event's types/positions/cell of the second minimum.
        index_move : int
            index of the atom that move the most during the event.
        dE_forward : float
            Energy barrier of the foward event.
        dE_backward : float
            Energy barrier of the backward event.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            tuple containing :
            - a pd.Series of the foward reaction.
            - a pd.Series of the backward reaction.

        """
        min1_positions = min1.positions
        saddle_positions = saddle.positions
        min2_positions = min2.positions
        cell = min1.cell
        full = self.params.atomicenvironment.coloring_mode == "full"

        # TODO need to see how to deal with different style for atomic environment ID
        # Compute all needed topology ID, one throwaway NeighborsList per
        # configuration (initial/saddle/final), all anchored at index_move:
        id_min1, min1neighbors_list = compute_atomic_environment_id(
            min1, index_move, self.params
        )
        id_saddle, _saddleneighbors_list = compute_atomic_environment_id(
            saddle, index_move, self.params
        )
        id_min2, min2neighbors_list = compute_atomic_environment_id(
            min2, index_move, self.params
        )
        # query_ball_point can hand back Python lists; coerce to arrays so the
        # element-wise comparisons (np.where) and type indexing below behave.
        neighbor_list_forward = np.asarray(
            min1neighbors_list.neighbors_list["rcut"][index_move]
        )
        neighbor_list_backward = np.asarray(
            min2neighbors_list.neighbors_list["rcut"][index_move]
        )

        # Store each cluster in the periodic image closest to its own central atom,
        # since the live side of the PSR is unwrapped the same way. The other two
        # configurations are then placed in the image closest to their own initial
        # position, so that saddle - initial and final - initial are the true
        # minimum-image displacements even for an atom that wraps mid-event.
        # Each call builds the stage/direction's Configuration snapshot directly,
        # reused both to resolve its sid below and to populate the stored event row.
        config_min1_forward = unwrap_around(
            min1[neighbor_list_forward], min1_positions[index_move]
        )
        config_saddle_forward = unwrap_around(
            saddle[neighbor_list_forward], config_min1_forward
        )
        config_min2_forward = unwrap_around(
            min2[neighbor_list_forward], config_min1_forward
        )

        config_min2_backward = unwrap_around(
            min2[neighbor_list_backward], min2_positions[index_move]
        )
        config_saddle_backward = unwrap_around(
            saddle[neighbor_list_backward], config_min2_backward
        )
        config_min1_backward = unwrap_around(
            min1[neighbor_list_backward], config_min2_backward
        )

        # Index of the central/moving atom within each direction's clusters
        # (uniform across initial/saddle/final since all three share one
        # neighbor list per direction) -- known before any shape resolution,
        # so it can seed the IRA search against each representative's own
        # move_atom_idx instead of falling back to a geometric center.
        move_atom_idx_forward = np.where(neighbor_list_forward == index_move)[0][0]
        move_atom_idx_backward = np.where(neighbor_list_backward == index_move)[0][0]

        # Decorate id_initial/id_saddle with a shape id each: both are only
        # coarse topology hashes and can independently collide, so resolve
        # which distinct geometry (among those already grouped under that
        # hash) this event actually belongs to, per id-type. id_final's sid
        # is not independently resolved: the forward event's final state is
        # exactly the backward event's initial state (same physical
        # configuration), so its sid is the one already resolved above.
        sid_initial_forward = self.shapes.resolve_sid(
            "initial", id_min1, config_min1_forward, move_atom_idx=move_atom_idx_forward
        )
        sid_initial_backward = self.shapes.resolve_sid(
            "initial", id_min2, config_min2_backward, move_atom_idx=move_atom_idx_backward
        )
        # config_saddle_forward/config_saddle_backward are two different
        # neighbor-list views of the exact same physical saddle point (both
        # sliced from the one `saddle` Configuration) -- resolved once and
        # shared between both directions, rather than independently per
        # direction, so the same saddle can never be split into two distinct
        # SaddleIDs by an imperfect geometric match between its two views.
        sid_saddle = self.shapes.resolve_sid(
            "saddle", id_saddle, config_saddle_forward, move_atom_idx=move_atom_idx_forward
        )
        sid_final_forward = sid_initial_backward
        sid_final_backward = sid_initial_forward

        # Symmetries : colour-aware detection only in full coloring mode.
        sym_matrix, sym_perm = unique_symmetries(
            config_min1_forward, config_min2_forward, self.params.ira.sym_thr, full=full
        )

        # dr :
        dra_forward = minimum_image_distance(
            min1_positions[neighbor_list_forward][move_atom_idx_forward],
            saddle_positions[neighbor_list_forward][move_atom_idx_forward],
            cell,
        )
        dra_backward = minimum_image_distance(
            min2_positions[neighbor_list_backward][move_atom_idx_backward],
            saddle_positions[neighbor_list_backward][move_atom_idx_backward],
            cell,
        )

        dfevent_forward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "initial_configuration": config_min1_forward,
                "saddle_configuration": config_saddle_forward,
                "final_configuration": config_min2_forward,
                "dE_forward": dE_forward,
                "dE_backward": dE_backward,
                "k": compute_rate_Eyring(dE_forward, self.params),
                "event_id": combine_ids(id_min1, id_saddle, id_min2),
                "id_initial": id_min1,
                "sid_initial": sid_initial_forward,
                "id_saddle": id_saddle,
                "sid_saddle": sid_saddle,
                "id_final": id_min2,
                "sid_final": sid_final_forward,
                "move_atom_idx": move_atom_idx_forward,
                "sym_matrix": sym_matrix,
                "sym_perm": sym_perm,
                "idx_backward": -1,  # unknown yet,
                "dra": dra_forward,
            }
        )

        sym_matrix, sym_perm = unique_symmetries(
            config_min2_backward, config_min1_backward, self.params.ira.sym_thr, full=full
        )
        dfevent_backward = pd.Series(
            {
                "idx_ref": -1,  # unknown yet
                "initial_configuration": config_min2_backward,
                "saddle_configuration": config_saddle_backward,
                "final_configuration": config_min1_backward,
                "dE_forward": dE_backward,
                "dE_backward": dE_forward,
                "k": compute_rate_Eyring(dE_backward, self.params),
                "event_id": combine_ids(id_min2, id_saddle, id_min1),
                "id_initial": id_min2,
                "sid_initial": sid_initial_backward,
                "id_saddle": id_saddle,
                "sid_saddle": sid_saddle,
                "id_final": id_min1,
                "sid_final": sid_final_backward,
                "move_atom_idx": move_atom_idx_backward,
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
        if self.params.control.reference_table is not None:
            self.table = pd.read_pickle(self.params.control.reference_table)
        else:
            self.table = pd.DataFrame(
                {
                    "idx_ref": pd.Series(dtype="int64"),
                    "initial_configuration": pd.Series(dtype="object"),
                    "saddle_configuration": pd.Series(dtype="object"),
                    "final_configuration": pd.Series(dtype="object"),
                    "dE_forward": pd.Series(dtype="float64"),
                    "dE_backward": pd.Series(dtype="float64"),
                    "k": pd.Series(dtype="float64"),
                    "event_id": pd.Series(dtype="str"),
                    "id_initial": pd.Series(dtype="str"),
                    "sid_initial": pd.Series(dtype="int64"),
                    "id_saddle": pd.Series(dtype="str"),
                    "sid_saddle": pd.Series(dtype="int64"),
                    "id_final": pd.Series(dtype="str"),
                    "sid_final": pd.Series(dtype="int64"),
                    "move_atom_idx": pd.Series(dtype="int64"),
                    "sym_matrix": pd.Series(dtype="object"),
                    "sym_perm": pd.Series(dtype="object"),
                    "idx_backward": pd.Series(dtype="int64"),
                    "dra": pd.Series(dtype="float64"),
                }
            )

    def remove(self, idx_refs: list[int], protect: set[int] | None = None) -> None:
        """Remove events with idx_ref in `idx_refs`, and their backward partners.

        Parameters
        ----------
        idx_refs : list[int]
            idx_ref values of the events to be removed.
        protect : set[int] | None
            idx_ref values whose whole forward/backward pair must never be removed
        """

        idx_refs = set(idx_refs)  # make a set if there are doublons

        backward_refs = set(
            self.table.loc[self.table["idx_ref"].isin(idx_refs), "idx_backward"].astype(
                int
            )
        )  # find set idx backwards

        all_refs = idx_refs | backward_refs  # all ref to remove

        if protect:
            protect_backward = set(
                self.table.loc[
                    self.table["idx_ref"].isin(protect), "idx_backward"
                ].astype(int)
            )
            all_refs -= set(protect) | protect_backward  # a pair is never split

        removed = self.table[self.table["idx_ref"].isin(all_refs)]
        self.table = self.table[~self.table["idx_ref"].isin(all_refs)].reset_index(
            drop=True
        )  # keep event not (~) in all refs

        # A removed row's idx_ref may still linger in its shape's
        # rediscovery_counts/known_rates -- self.shapes.forget prunes it,
        # without deleting the shape's own catalog entry (a shape with zero
        # table rows is a normal state, not staleness).
        for _, row in removed.iterrows():
            shape = ShapeID(row["id_initial"], int(row["sid_initial"]))
            self.shapes.forget(shape, int(row["idx_ref"]))

    def save(self, outfile: str = "reference_table.pickle") -> None:
        """Save the reference event table to a pickle file.

        Also saves the persistent minima/saddle shape catalogs via
        `self.shapes.save()`, independent of `outfile` -- `reference_table.pickle`
        stays a plain DataFrame pickle so existing external tooling reading
        it is unaffected.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'reference_table.pickle'.

        """
        self.table.to_pickle(outfile)
        self.shapes.save()


class ActiveEventTable:
    """Store active events and manage them.

    Parameters
    ----------
    params : Parameters
        The atomic simulations configuration.
    event_dataframe : pd.DataFrame, optional
        An table with active event use to initialize the table. by default 'None'.

    """

    def __init__(
        self,
        params: Parameters,
        event_dataframe: pd.DataFrame = None,
        recycler: "Recycling | None" = None,
    ):
        self.params = params
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
                "saddle_configuration": pd.Series(dtype="object"),
                "final_configuration": pd.Series(dtype="object"),
                "dE_forward": pd.Series(dtype="float64"),
                "k": pd.Series(dtype="float64"),
                "num_reference_event": pd.Series(dtype="int64"),
                "refined": pd.Series(dtype="str"),
                "neighbors": pd.Series(dtype="object"),
            }
            self.table = pd.DataFrame(columns)

    def prune_for_recycling(
        self,
        executed_idx: int,
        system: System,
        configuration_pre: Configuration,
        reference_table: ReferenceEventTable,
    ) -> None:
        """Replace `self.table` with the rows that survive the recycler's filter.

        If no recycler is attached, clear the table (matches the prior
        end-of-step `del active_table` behavior).
        """
        if self.recycler is None:
            self.table = self.table.iloc[0:0].reset_index(drop=True)
        else:
            self.table = self.recycler.select_recyclable(
                self, executed_idx, system, configuration_pre
            )
            # A recycled row's reference may have been purged from
            # reference_table this step (a sibling instance failed
            # reconstruction); drop it so it can't be selected later with a
            # dead num_reference_event.
            valid_refs = set(reference_table.table["idx_ref"])
            self.table = self.table[
                self.table["num_reference_event"].isin(valid_refs)
            ].reset_index(drop=True)

    def drop_stale_rows(self, neighbors_list: NeighborsList) -> int:
        """Remove rows whose stored atom list no longer matches the current one.

        A row's `saddle_configuration` and `final_configuration` are indexed against
        the neighbour list that existed when it was built. Once the system moves, that
        atom list can gain, lose or reorder members, and applying the stored arrays
        against the new list would move the wrong atoms. Only recycled rows can
        reach here with a stale list.

        Parameters
        ----------
        neighbors_list : NeighborsList
            The neighbour lists of the system in its current state.

        Returns
        -------
        int
            Number of rows removed.

        """
        stale = [
            idx
            for idx, row in self.table.iterrows()
            if list(row["neighbors"])
            != list(neighbors_list.get_neighbors("rcut", int(row["atom_index"])))
        ]
        if stale:
            self.remove(stale)
        return len(stale)

    def existing_pairs(self) -> set[tuple[int, int]]:
        """Return `(atom_index, num_reference_event)` tuples already in the table."""
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
                "saddle_configuration": event_refinement_output.saddle,
                "final_configuration": event_refinement_output.min2,
                "dE_forward": event_refinement_output.dE_forward,
                "k": compute_rate_Eyring(
                    event_refinement_output.dE_forward, self.params
                ),
                "num_reference_event": event_refinement_output.num_reference_event,
                "refined": event_refinement_output.refined,
                "neighbors": event_refinement_output.neighbors,
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

    def remove_duplicates(self, neighbors_list: NeighborsList = None) -> None:
        """Loop over all active events in the DataFrame, check if there are duplicates by computing delr."""

        duplicates: list[int] = []
        duplicates_central: list[int] = []
        duplicates_symmetric: list[int] = []
        # 1. Check duplicates on central atoms : to be sure
        # atom_index equality is a necessary condition for any pair to be
        # considered, so group by it once (vectorized) instead of re-scanning
        # the whole table per row; the dE tolerance window still can't be a
        # groupby key, so it's checked within each (small) group below.
        tol_energy = 0.1  # eV
        dE_arr = self.table["dE_forward"].to_numpy()
        config_arr = self.table["saddle_configuration"].to_numpy()

        for _atom_index, idx_list in self.table.groupby("atom_index").groups.items():
            idx_list = list(idx_list)
            for i, idx in enumerate(idx_list):
                for jdx in idx_list[i + 1 :]:  # dont compute twice
                    if abs(dE_arr[idx] - dE_arr[jdx]) >= tol_energy:
                        continue
                    delr = compute_delr_max(config_arr[idx], config_arr[jdx])
                    if delr < self.params.psr.matching_score_thr:
                        # print('Removing event with delr',delr)
                        duplicates.append(jdx)
                        duplicates_central.append(jdx)

        # 2. Check duplicates due to symmetric events applied on different central atoms.
        # Group by same generic event if generic event has symmetries meaning that the same generic event has been applied to same central atom
        if (
            neighbors_list is not None
        ):  # need neighbors list to remove symmetric duplicates
            counts = self.table.groupby(["atom_index", "num_reference_event"]).size()
            symmetric_num_ref = counts[counts > 1].index.get_level_values(1).unique()
            atom_index_arr = self.table["atom_index"].to_numpy()

            # Loop on all num_ref symmetric event
            for num_ref in symmetric_num_ref:
                subset = self.table[self.table["num_reference_event"] == num_ref]
                indices = subset.index.to_list()

                for i, idx in enumerate(indices):  # Loop over indice of subset
                    central_atom1 = atom_index_arr[idx]
                    env1 = neighbors_list.get_neighbors(
                        "rcut", central_atom1
                    )  # list of atom in env1

                    for jdx in indices[i + 1 :]:  # to not compare two times
                        central_atom2 = atom_index_arr[jdx]
                        if (
                            central_atom1 != central_atom2
                        ):  # if yes already done in part 1.
                            env2 = neighbors_list.get_neighbors("rcut", central_atom2)
                            # intersection of atoms in atomic environments
                            common = set(env1) & set(env2)

                            if not common:  # it's not a duplicate since they don't share atoms in their atomic environments
                                continue

                            if (
                                central_atom1 not in env2
                            ):  # TODO : To check, but should not be a duplicate
                                continue

                            # know we want to compare positions of share atoms, need to map.
                            map1 = {
                                a: k for k, a in enumerate(env1)
                            }  # so we know that the first position is atom xxx, ect, eg {345:0, 439:1, ....}
                            map2 = {a: k for k, a in enumerate(env2)}  # same for env2

                            # map atom when they are in common
                            index1 = [map1[a] for a in common]
                            index2 = [map2[a] for a in common]

                            # now we can compare the shared-atom sub-configurations
                            delr = compute_delr_max(
                                config_arr[idx][index1], config_arr[jdx][index2]
                            )
                            if delr < self.params.psr.matching_score_thr:
                                duplicates.append(jdx)
                                duplicates_symmetric.append(jdx)

        unique_duplicates = sorted(set(duplicates))
        if unique_duplicates:
            self.remove(unique_duplicates)
            _LOGGER.info(
                "\t :=> Removed %d duplicate active events (central=%d, symmetric=%d).",
                len(unique_duplicates),
                len(set(duplicates_central)),
                len(set(duplicates_symmetric)),
            )
            _LOGGER.info(
                "\t :=> Duplicate active event indices removed: %s",
                unique_duplicates,
            )
        else:
            _LOGGER.info("\t :=> No duplicate active events detected.")

    def save(self, outfile: str = "active_table.pickle") -> None:
        """Save the reference event table to a pickle file.

        Parameters
        ----------
        outfile : str, optional
            path to the output file, by default 'active_table.pickle'.

        """
        self.table.to_pickle(outfile)
