"""Module for executing off-lattice kinetic Monte Carlo (KMC) simulations.

This module defines the `KMC` class.
"""

from pykmc import (
    NeighborsList,
    AtomicEnvironment,
    ActiveEventTable,
    Parameters,
    Reconstruction,
    Configuration,
)
import random

from pykmc.enginemanager.lmpi.pool import Manager
from pykmc.event_table import ReferenceEventTable
from .result import (
    EventSearchOutput,
    KMCLoopInfo,
    ErrorInfo,
    ErrorType,
    Result,
    AtomicEnvironmentInfo,
    ReferenceEventSearchInfo,
    ReferenceValidEventsInfo,
    RefinementsInfo,
    RefinementCandidate,
    EventRefinementOutput,
    ReconstructionOutput,
    AdaptiveSearchInfo,
    Err,
    Ok,
    ShapeID,
)
from .log import fmt_hash
import numpy as np
from ase.io import write
from ase import Atoms
from .algorithms import rejection_free
import sys
import pandas as pd
import pickle
from .initializer import Initializer
from .info_simulation import (
    info_atomic_environments,
    info_reference_event_searches,
    info_is_valid_reference_events,
    info_refinements,
    info_active_events,
    info_basin_events,
    info_adaptive_search,
)
from .eventsearch import EventSearch
from .refinement import Refinement
from .log import Colors
from .otfml import OTFMLController, OTFMLStreamCheckpoint
import time
from .utils import push_towards, compute_delr_max
from .basins.detection import DetectorThreshold
from .basins import BasinsGenericEvents
from .event_recycling import DistanceRecycling, Recycling
from .bias import Bias
from .adaptive_search import (
    ShapeSearchStats,
    AdaptiveSearchSession,
    AdaptiveSearchResult,
)


# NOTE can maybe reimplment tries if empty catalog
# TODO: Add reconstruction info


class KMC:
    """Manage and execute the Kinetic Monte Carlo (KMC) simulation.

    This class acts as the central controller coordinating all phases of the simulation:
    initialization, event search, event refinement, event selection, system updates,
    minimization, logging, and termination.

    Attributes
    ----------
    params : Parameters
        The parameters of the simulation.
    loggers : LogKMC
        Handle logging of simulation progress.
    system : System
        The atomic system.
    engine : Engine
        The E/F engine used.
    neighbors_list : NeighborsList
        Store neighbors of atoms in the system.
    atomic_environment : AtomicEnvironment
        Store atomic environment of atoms in the system
    reference_table : ReferenceEventTable
        Store generic events that can be apply to the system.
    visited_environment : set[str]
        Track atomic environments already explored. Those for which event searches as been previously done.
    total_energy : float
        The total energy of the system.

    """

    def __init__(self, params: Parameters) -> None:
        self.params = params
        self.loggers = None
        self.system = None
        self.manager: Manager = None
        self.engine = None
        self.neighbors_list: NeighborsList = None
        self.atomic_environment: AtomicEnvironment = None
        self.reference_table: ReferenceEventTable = None
        self.visited_environments: set[str | bytes] = None
        self.total_energy = None
        self.potential_energy = None
        self.active_table: ActiveEventTable | None = None
        self._pre_exec_configuration: Configuration | None = None
        self.bias: Bias | None = None
        self.otfml = OTFMLController(self)
        self.recycler: Recycling | None = None
        if self.params.control.recycle:
            if self.params.eventrecycling.style == "displacement":
                self.recycler = DistanceRecycling(
                    movement_thr=self.params.eventrecycling.movement_thr,
                    distance_thr=self.params.eventrecycling.distance_thr,
                )

    def run(self) -> None:
        """Run the simulation."""
        # Initialize the simulation, KMC attributes and minimize the system
        # self._initialize()
        self.manager.initialize_sessions(self.params, self.system)
        self.minimize_system(configuration=self.system.configuration)
        self.neighbors_list = NeighborsList(
            self.system,
            self.params.atomicenvironment.rnei,
            self.params.atomicenvironment.rcut,
            self.params.atomicenvironment.rnei_pairs,
        )
        self.atomic_environment = AtomicEnvironment(
            self.params.atomicenvironment.style,
            self.neighbors_list.neighbors_list["rnei"],
            self.neighbors_list.neighbors_list["rcut"],
            self.params.atomicenvironment.neighbors_add,
            coordination_threshold=self.params.atomicenvironment.coordination_threshold,
            types=self.system.types,
            coloring_mode=self.params.atomicenvironment.coloring_mode,
        )
        self.inactive_ae = (
            AtomicEnvironment(
                style="region",
                region=self.params.inactive_atoms,
                configuration=self.system.configuration,
            )
            if self.params.inactive_atoms is not None
            else None
        )
        self.frozen_ae = (
            AtomicEnvironment(
                style="region",
                region=self.params.frozen_atoms,
                configuration=self.system.configuration,
            )
            if self.params.frozen_atoms is not None
            else None
        )
        # Set new positions to all sessions/engine :
        self.manager.use_local()
        self.manager.set_all_positions(self.system.positions)

        if self.params.control.restart_file is None:
            # Write initial step to file
            self._append_snapshot_to_trajectory()
            last_step = 0
            total_time = 0.0
            self.loggers.table_line_info_kmc(
                "output",
                0,
                self.total_energy,
                None,
                None,
                None,
                total_time,
                None,
                None,
                None,
                None,
            )

        else:  # read restart file
            self.loggers.info("log", ":=> Reading restart file")
            restart_info = np.load(self.params.control.restart_file)
            last_step = restart_info["last_step"]
            total_time = restart_info["last_time"]
            self.loggers.info(
                "log",
                ":=> last step = {}, last_end_time = {}ps".format(
                    last_step, total_time
                ),
            )

        # LOOP KMC PARAMETERS
        nkmc_steps = self.params.control.n_steps
        last_step += 1
        nsearch = self.params.eventsearch.nsearch

        # Build the persistent active event table once, with the recycler
        # plugin (built in __init__) attached.
        self.active_table = ActiveEventTable(self.params, recycler=self.recycler)

        # KMC LOOP
        for step in range(last_step, nkmc_steps + last_step):
            start_real = time.time()
            start_cpu = time.process_time()

            self.loggers.info(
                "log",
                "{}{}Step : {}{}".format(
                    Colors.BOLD.value, Colors.YELLOW.value, step, Colors.RESET.value
                ),
            )

            # == Find Current atomic environments that has not been visited ==
            new_shapes, atom_shapes = self.resolve_new_shapes()

            if self.params.control.recycle and len(self.active_table.table) > 0:
                self.loggers.info(
                    "log",
                    "\t :=> Recycling {} events from the previous step".format(
                        len(self.active_table.table)
                    ),
                )

            # == FIND NEW GENERIC EVENTS ==
            if self.params.eventsearch.adaptive_search:
                ##=>Adaptively search each new topology in small batches until
                ##   its rate-weighted undiscovered-mass estimate converges.
                (
                    search_results,
                    all_search_results_for_info,
                    results_is_valid_events,
                    shape_search_stats,
                ) = self.adaptive_event_search(new_shapes)
            else:
                ##=>List of atoms(central) on which we gonna perfom an event search
                central_atom_research_list = self.central_atoms_research(
                    new_shapes, nsearch
                )

                ##=>Perform event search on each atom in central_atom_research_list
                event_search = self.execute_event_searches(central_atom_research_list)

                # == ADD NEW GENERIC EVENTS TO REFERENCE EVENT TABLE ==
                ##=>Check if the event is valid, ie if not already present and has a valid energy barrier if yes add it to the reference table
                search_results = event_search.get_successes_results()
                inactive_set = (
                    set(self.inactive_ae.get_atoms_with_id("in"))
                    if self.inactive_ae is not None
                    else set()
                )
                if inactive_set:
                    search_results = [
                        r
                        for r in search_results
                        if r.move_atom_index not in inactive_set
                    ]
                results_is_valid_events = self.add_reference_events(search_results)
                shape_search_stats = None
                all_search_results_for_info = event_search.results
                # No Good-Turing estimate here, so no continuous credit
                # mechanism -- each shape an actually-dispatched atom
                # resolves to is simply marked completed, regardless of what
                # was found (resolve_live_shape mints a shape backed by the
                # atom's own geometry if this exact shape never matched
                # anything, so a search that finds nothing is still marked
                # complete instead of silently forgotten). A different shape
                # sharing the same coarse id that never got a random draw
                # this round stays unmarked, so it's picked up again by
                # resolve_new_shapes() next step.
                for atom_idx in central_atom_research_list:
                    shape = self.reference_table.shapes.resolve_live_shape(
                        self.system, self.neighbors_list, self.atomic_environment, atom_idx, mint=True
                    )
                    self.reference_table.shapes.mark_shape_completed(shape)

            ##=>Close simulation if no events in the reference table
            if len(self.reference_table.table) == 0:
                self.loggers.error(
                    "log",
                    "No events have been found, empty reference events table. \n \tTry to increase nsearch or saddle point search algorithm's parameters. \n \tClosing the simulation.",
                )
                self._close()

            # == Update variables ==
            l_ids = list(set(self.atomic_environment.atomic_environment_list))
            self.visited_environments.update(
                set(l_ids).difference(self.visited_environments)
            )
            # == Refinement ==
            ##=>Refines all selected candidates (skipping (atom, ref_event) pairs already carried over)
            refinement = self.execute_refinements(
                existing_pairs=self.active_table.existing_pairs(),
                atom_shapes=atom_shapes,
            )

            # == ADD ACTIVE EVENT TO ACTIVE EVENT TABLE ==
            # The persistent self.active_table is extended in place; recycled
            # rows from the previous step are already present.
            self.add_active_events(refinement.get_successes_results())
            active_table = self.active_table

            active_table.remove_duplicates(self.neighbors_list)  # To be sure
            self.loggers.info(
                "log",
                "\t :=> {} active events after removing duplicates.".format(
                    len(active_table.table)
                ),
            )

            # == Update System ==
            self.manager.use_global()
            (
                result_reconstruction,
                delta_t,
                ktot,
                idx_selected_event,
                err_reference,
                err_ae,
            ) = self.reconstruction(active_table)
            num_ref_selected = active_table.table.loc[idx_selected_event].at[
                "num_reference_event"
            ]
            idx_selected_display, events_info = info_active_events(
                self.system.types,
                self.reference_table,
                active_table,
                idx_selected_event,
            )
            if len(err_reference) != 0:
                selected_topo = self.reference_table.table[
                    self.reference_table.table["idx_ref"] == num_ref_selected
                ]["id_initial"].values[0]
                self.loggers.info(
                    "log",
                    "\t :=> Removing reference event from which reconstruction failed.",
                )
                self.reference_table.remove(
                    list(set(err_reference)), protect={num_ref_selected}
                )
                self.loggers.info(
                    "log",
                    "\t :=> Removing topology from known environments from which reconstruction failed.",
                )
                self.visited_environments = self.visited_environments.difference(
                    set(err_ae) - {selected_topo}
                )
            # INFO :
            self.loggers.events_file_step_first_line("events", step)
            self.loggers.events_applicable_info_line("events", idx_selected_display)
            self.loggers.events_write("events", events_info)

            # TODO: Temporary, need to unified kmc main loop and basin operations + ugly
            detector = DetectorThreshold()
            # Pre-execution snapshot for event recycling (needed before update_positions below)
            if self.params.control.recycle:
                self._pre_exec_configuration = self.system.configuration.copy()
            # IF selected event shows we are in a basin
            if self.params.control.basin and detector.detect(
                active_table.table.iloc[idx_selected_event],
                self.reference_table.table,
                self.params.basin.energy_thr,
                is_refined=True,
            ):
                self.loggers.info("log", "\t :=> System is in a Basin.")
                self.loggers.info("log", "\t :=> Exploring the Basin.")
                # get basin info/explore
                basin = BasinsGenericEvents(
                    self.params,
                    self.reference_table,
                    self.visited_environments,
                    self.manager,
                )
                self.system.update_positions(
                    result_reconstruction.ok_value().min1_configuration
                )
                result_basin = basin.execute(self.system)
                if result_basin.is_ok():  # Basin did no fail
                    # move system to a state connected to the exit_state
                    self.system.update_positions(
                        result_basin.ok_value().initial_system_configuration
                    )
                    self.neighbors_list = basin.states[
                        result_basin.ok_value().from_state
                    ].neighbors_list
                    # construct new active table with only event : new_actual_state - > exit_state
                    tmp_active_table = ActiveEventTable(self.params)
                    basin_local_types = np.asarray(self.system.types)[
                        result_basin.ok_value().neighbors
                    ]
                    tmp_event = EventRefinementOutput(
                        central_atom_index=result_basin.ok_value().central_atom,
                        saddle=result_basin.ok_value().saddle_configuration.with_types(
                            basin_local_types
                        ),
                        E_saddle=-1,
                        min2=result_basin.ok_value().final_configuration.with_types(
                            basin_local_types
                        ),
                        dE_forward=result_basin.ok_value().dE_forward,
                        num_reference_event=result_basin.ok_value().num_reference_event,
                        neighbors=result_basin.ok_value().neighbors,
                    )
                    tmp_active_table.add_events(tmp_event)
                    # reconstruct event
                    self.manager.use_global()
                    result_basin_reconstruction = self._reconstruction_active_event(
                        0, tmp_active_table
                    )
                    if result_basin_reconstruction.is_ok():
                        self.system.update_positions(
                            result_basin_reconstruction.ok_value().min2_configuration
                        )
                        self.total_energy = (
                            result_basin_reconstruction.ok_value().min2_etot
                        )
                        delta_t = result_basin.ok_value().t_exit
                        ktot = result_basin.ok_value().k_tot
                        idx_selected_event = 0
                        active_table.table = tmp_active_table.table

                        # INFO
                        idx_exit_event, basin_info = info_basin_events(
                            self.system.types,
                            self.reference_table,
                            basin.connectivity_table,
                            result_basin.ok_value().exit_state,
                        )
                        self.loggers.events_basin_info_line("events", idx_exit_event)
                        self.loggers.events_write("events", basin_info)

                    else:
                        self.loggers.info(
                            "log",
                            "\t :=> Reconstruction Exit State Basin fails with error {}, back to original event".format(
                                result_basin_reconstruction.err_value()
                            ),
                        )
                        self.system.update_positions(basin.states[0].system.positions)
                        self.system.update_positions(
                            result_reconstruction.ok_value().min2_configuration
                        )
                else:
                    self.loggers.info(
                        "log",
                        "\t :=> Basin fails with error : {}, back to original event".format(
                            result_basin.err_value()
                        ),
                    )
                    self.system.update_positions(
                        result_reconstruction.ok_value().min2_configuration
                    )
                if basin.connectivity_table is not None:
                    basin.connectivity_table.save(
                        "basin_connectivity_" + str(step) + ".pickle"
                    )
                # Basin super-event spans many atoms; recycling is deferred (the
                # prune below runs with the recycler detached).
                prune_detach_recycler = True
            else:
                self.system.update_positions(
                    result_reconstruction.ok_value().min2_configuration
                )
                self.total_energy = result_reconstruction.ok_value().min2_etot
                prune_detach_recycler = False
            total_time += delta_t * 10**-12  # time is in seconds

            ###=> Synchronise all lammps instances with new positions
            self.manager.use_local()
            self.manager.set_all_positions(positions=self.system.positions)
            ##=>Minimize

            # == Log informations ==
            atomic_environment_info = self.get_info_atomic_environments(
                new_shapes
            )
            reference_event_searches_info = self.get_info_reference_event_searches(
                all_search_results_for_info
            )
            is_valid_events_info = self.get_info_is_valid_reference_events(
                results_is_valid_events
            )
            refinements_info = self.get_info_refinements(refinement.results)
            adaptive_search_info = self.get_info_adaptive_search(shape_search_stats)
            kmc_loop_info = KMCLoopInfo(
                step=step,
                atomic_environment_info=atomic_environment_info,
                reference_event_searches_info=reference_event_searches_info,
                valid_event_info=is_valid_events_info,
                refinements_info=refinements_info,
                adaptive_search_info=adaptive_search_info,
            )
            self.loggers.info("info", kmc_loop_info.output_msg())
            if adaptive_search_info is not None and adaptive_search_info.n_capped > 0:
                self.loggers.info(
                    "log",
                    "\t :=> WARNING: {} topolog{} hit the adaptive search ceiling "
                    "without converging; may still have undiscovered events.".format(
                        adaptive_search_info.n_capped,
                        "y" if adaptive_search_info.n_capped == 1 else "ies",
                    ),
                )

            elapsed_real = time.time() - start_real
            elapsed_cpu = time.process_time() - start_cpu

            event_id_selected = fmt_hash(
                self.reference_table.table[
                    self.reference_table.table["idx_ref"] == num_ref_selected
                ]["event_id"].values[0]
            )
            self.loggers.table_line_info_kmc(
                "output",
                step,
                self.total_energy,
                active_table.table.loc[idx_selected_event].at["dE_forward"],
                delta_t * 10**-12,
                active_table.table.loc[idx_selected_event].at["k"],
                total_time,
                ktot,
                num_ref_selected,
                event_id_selected,
                elapsed_cpu,
                elapsed_real,
            )

            # == Event recycling: prune the active table for the next step ==
            # Must run AFTER the step log above, which reads the executed event's
            # row; with no recycler (recycle = False, the default) the prune clears
            # the whole table and the lookup would raise KeyError.
            if prune_detach_recycler:
                saved_recycler = self.active_table.recycler
                self.active_table.recycler = None
                self.active_table.prune_for_recycling(
                    idx_selected_event,
                    self.system,
                    self._pre_exec_configuration,
                    self.reference_table,
                )
                self.active_table.recycler = saved_recycler
            else:
                self.active_table.prune_for_recycling(
                    idx_selected_event,
                    self.system,
                    self._pre_exec_configuration,
                    self.reference_table,
                )
                if self.params.control.recycle:
                    self.loggers.info(
                        "log",
                        "\t :=> {} events flagged for recycling".format(
                            len(self.active_table.table)
                        ),
                    )

            # == Update variables ==
            self.neighbors_list = NeighborsList(
                self.system,
                self.params.atomicenvironment.rnei,
                self.params.atomicenvironment.rcut,
                self.params.atomicenvironment.rnei_pairs,
            )
            n_stale = self.active_table.drop_stale_rows(self.neighbors_list)
            if n_stale:
                self.loggers.info(
                    "log",
                    "\t :=> Dropped {} recycled events whose atomic environment changed.".format(
                        n_stale
                    ),
                )
            self.atomic_environment = AtomicEnvironment(
                self.params.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.params.atomicenvironment.neighbors_add,
                coordination_threshold=self.params.atomicenvironment.coordination_threshold,
                types=self.system.types,
                coloring_mode=self.params.atomicenvironment.coloring_mode,
            )
            self.inactive_ae = (
                AtomicEnvironment(
                    style="region",
                    region=self.params.inactive_atoms,
                    configuration=self.system.configuration,
                )
                if self.params.inactive_atoms is not None
                else None
            )
            self.frozen_ae = (
                AtomicEnvironment(
                    style="region",
                    region=self.params.frozen_atoms,
                    configuration=self.system.configuration,
                )
                if self.params.frozen_atoms is not None
                else None
            )

            # == Save Reference Table and List visited environment :
            self._save()
            self._append_snapshot_to_trajectory()
            del active_table
            # == Check if only cristalline environments ==
            if set(list(self.atomic_environment.atomic_environment_list)) == {
                "crystal"
            }:
                self.loggers.info("log", ":=> Only atoms with cristalline environment")
                self._close()
        self._save_restart_file(step, total_time)
        self._close()

    def resolve_new_shapes(self) -> tuple[dict[ShapeID, list[int]], dict[int, ShapeID]]:
        """Resolve every atom's live `ShapeID` and group the ones still needing search.

        An atom classified `"crystal"` never needs search and is never
        resolved: under the `cna/graph`-family styles, a bulk/perfectly-
        coordinated atom never gets a real topology hash computed for it in
        the first place (see `AtomicEnvironment.compute_cnagraph`), so there
        is no shape for it to classify into or search for.

        Every other atom is resolved with `mint=True`: if its live geometry
        matches nothing already catalogued under its `id_initial`, this is
        exactly what "discovering a new shape" means, so it becomes a fresh
        `ShapeKnowledge` entry (`status="unsearched"`) right here rather than
        being left as an unresolved gap -- there is no separate "does this
        classify at all" question anymore, only "has this exact `ShapeID`'s
        search reached `"completed"` yet" (`ShapeTable.shape_knowledge`).
        The eligible worklist is grouped by `ShapeID`, not the coarse
        `id_initial`: dispatch now knows exactly which shape each atom
        belongs to before searching it, so sampling targets one shape at a
        time -- grouping by the coarse id would let one shape's abundant
        atom count crowd out a rarer, still-unsearched shape sharing the
        same `id_initial` from ever being drawn.

        Returns
        -------
        dict[ShapeID, list[int]]
            shape -> atom indices resolved to it and still needing search
            (excludes shapes already `"completed"`).
        dict[int, ShapeID]
            Every non-crystal atom's own resolved `ShapeID` this step --
            reused directly by `live_events()` later this same step instead
            of reclassifying.

        """
        eligible: dict[ShapeID, list[int]] = {}
        atom_shapes: dict[int, ShapeID] = {}
        for atom_idx, id_initial in enumerate(
            self.atomic_environment.atomic_environment_list
        ):
            if id_initial == "crystal":
                continue
            shape = self.reference_table.shapes.resolve_live_shape(
                self.system, self.neighbors_list, self.atomic_environment, atom_idx, mint=True
            )
            atom_shapes[atom_idx] = shape
            if self.reference_table.shapes.get_shape_knowledge(shape).status == "completed":
                continue
            eligible[shape] = eligible.get(shape, []) + [atom_idx]

        self.loggers.info(
            "log",
            "\t :=> {} shapes need a fresh search".format(len(eligible)),
        )
        return eligible, atom_shapes

    def central_atoms_research(
        self, new_shapes: dict[ShapeID, list[int]], nsearch: int
    ) -> list[int]:
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table.

        For each new shape it adds nseach atoms having that shape to the list.

        Parameters
        ----------
        new_shapes : dict[ShapeID, list[int]]
            shape -> eligible atom indices, from `resolve_new_shapes()`.
        nsearch : int
            Number of searches per shape.

        Returns
        -------
        list[int]
            List of central atoms

        Raises
        ------
        IndexError
            If no atoms are found for a given shape, random.choice will raise an IndexError.

        """
        central_atom_research_list = []
        inactive_set = (
            set(self.inactive_ae.get_atoms_with_id("in"))
            if self.inactive_ae is not None
            else set()
        )
        for atoms in new_shapes.values():
            tmp1 = [i for i in atoms if i not in inactive_set] if inactive_set else atoms
            if not tmp1:
                continue  # no eligible atoms for this environment
            # Randomly choose nsearch atoms that have that environment
            tmp2 = [random.choice(tmp1) for _i in range(nsearch)]
            central_atom_research_list += tmp2
        return central_atom_research_list

    def execute_event_searches(
        self, central_atom_research_list: list[int]
    ) -> EventSearch:
        """Execute an event search for each atom index in central_atom_research_list.

        Parameters
        ----------
        central_atom_research_list : list[int]
            The list of atom index on which we want to perform and event search.

        Returns
        -------
        EventSearch
            The EventSearch class containing results of the event searches.

        """
        event_search = EventSearch(
            self.params,
            self.system,
            self.manager,
            self.loggers,
        )
        event_search.execute(central_atom_research_list)
        self.otfml.retry_extrapolating("search", event_search)
        return event_search

    def add_reference_events(
        self, events: list[EventSearchOutput]
    ) -> list[pd.DataFrame]:
        """Add events to the reference table.

        Parameters
        ----------
        events : list[EventSearchOutput]
            List containing EventSearchOutput dataclass of successful events.

        Returns
        -------
        list[pd.DataFrame]
            List of event dataframe that has been added to the reference event table.

        """
        results_is_valid_events = self.reference_table.add_events(events)
        self.loggers.info(
            "log",
            "\t :=> Adding {} events to the reference table".format(
                len([e for e in results_is_valid_events if e.is_ok()])
            ),
        )
        return results_is_valid_events

    def adaptive_event_search(
        self, new_shapes: dict[ShapeID, list[int]]
    ) -> AdaptiveSearchResult:
        """Count-weighted adaptive event search for a batch of new shapes.

        A fixed-size in-flight window, sized to the worker pool: whenever a
        slot is free, `AdaptiveSearchSession` picks the next-highest-priority
        shape still needing search (per the chosen adaptive_stopping_rule's
        own undiscovered-mass estimate, with an escalating cap -- see
        `pykmc.adaptive_search`) and this method dispatches it right away,
        without waiting on any other in-flight search. Each shape's stats are
        updated the moment its own result lands, so a slow-converging shape
        never blocks or is blocked by the others.

        The dispatch loop below makes one priority-ordered decision per pass
        -- submit if there's room and work, else wait for something
        outstanding, else run a due checkpoint, else stop -- so filling the
        pool from empty, draining it before a checkpoint, and terminating
        are all just consequences of which branch wins, not separate loops.

        OTF-ML's extrapolation retry (needed because retraining repositions
        every session directly, bypassing the job queue, so it may only run
        with nothing in flight) is owned entirely by `otfml_checkpoint`
        (`pykmc.otfml.OTFMLStreamCheckpoint`), including when it's inactive for
        this phase -- this method only ever talks to its narrow interface
        (`should_defer`/`track`/`admits_more`/`due`/`run`), never OTF-ML
        vocabulary directly.

        Parameters
        ----------
        new_shapes : dict[ShapeID, list[int]]
            shape -> eligible atom indices, from `resolve_new_shapes()`.

        Returns
        -------
        AdaptiveSearchResult
            NamedTuple of (search_results, raw_results, valid_results, stats)
            -- see `pykmc.adaptive_search.AdaptiveSearchResult`.

        """
        params = self.params
        inactive_set = (
            set(self.inactive_ae.get_atoms_with_id("in"))
            if self.inactive_ae is not None
            else set()
        )

        session = AdaptiveSearchSession(
            list(new_shapes.keys()), self.reference_table.shapes.shape_knowledge
        )

        atoms_by_shape: dict[ShapeID, list[int]] = {}
        for shape, atoms in new_shapes.items():
            if inactive_set:
                atoms = [i for i in atoms if i not in inactive_set]
            if not atoms:
                session.mark_no_atoms(shape)
                continue
            atoms_by_shape[shape] = atoms

        event_search = EventSearch(self.params, self.system, self.manager, self.loggers)
        pool_size = len(self.manager.sessions)
        otfml_checkpoint = OTFMLStreamCheckpoint(self.otfml, "search", pool_size)

        all_search_results: list[EventSearchOutput] = []
        all_valid_results: list[Result] = []
        in_flight = {}

        def credit_result(shape, output):
            # Persistent knowledge (ShapeTable.shape_knowledge) is already
            # updated unconditionally inside add_reference_events ->
            # add_events, for whichever shape the outcome actually belongs
            # to. session.credit() below only credits this ephemeral
            # session when the outcome's forward or backward row is
            # actually this task's own live shape -- an opportunistic find
            # whose row belongs to some other shape is real knowledge, just
            # not about *this* shape's distribution.
            output = event_search._center_event_positions(output)
            if inactive_set and output.move_atom_index in inactive_set:
                return
            all_search_results.append(output)
            valid_result = self.add_reference_events([output])[0]
            all_valid_results.append(valid_result)

            # The shape is resolved fresh here (rather than reused from
            # dispatch time) because add_reference_events, just above, may
            # have just catalogued this exact atom's sid for the first time.
            resolved_shape = self.reference_table.shapes.resolve_live_shape(
                self.system, self.neighbors_list, self.atomic_environment,
                output.central_atom_index, mint=False,
            )
            if resolved_shape is None:
                return
            resolved = self.reference_table.resolve_forward_and_backward_rows(valid_result)
            if resolved is None:
                return
            session.credit(resolved_shape, *resolved)

        def process_result(task, shape):
            result = event_search.results[task.task_id]
            if result.is_ok():
                credit_result(shape, result.ok_value())
            if session.advance_one(shape, params):
                self.loggers.info(
                    "log",
                    "\t :=> Adaptive search: shape {}#{} still unconverged past half its "
                    "budget, escalating toward the cap ({} searches).".format(
                        fmt_hash(shape.id),
                        shape.sid,
                        params.eventsearch.adaptive_max_searches,
                    ),
                )

        while True:
            pending_shape = None
            if otfml_checkpoint.admits_more() and len(in_flight) < pool_size:
                pending_shape = session.pick_next(list(session.open_shapes), params)

            if pending_shape is not None:
                atom = random.choice(atoms_by_shape[pending_shape])
                in_flight[event_search.submit(atom)] = pending_shape
                session.record_dispatch(pending_shape)
            elif in_flight:
                future, task, result = event_search.wait_next(list(in_flight))
                shape = in_flight.pop(future)
                if not otfml_checkpoint.should_defer(result):
                    process_result(task, shape)
                otfml_checkpoint.track(task, shape)
            elif otfml_checkpoint.due():
                for task, shape in otfml_checkpoint.run(event_search):
                    process_result(task, shape)
            else:
                break

        session.finalize(self.reference_table.shapes.mark_shape_completed)
        search_result = AdaptiveSearchResult(
            all_search_results, list(event_search.results), all_valid_results, session.stats
        )
        return search_result

    def build_refinement_candidates(
        self,
        atom_shapes: dict[int, ShapeID],
        existing_pairs: set[tuple[int, int]] | None = None,
    ) -> list[RefinementCandidate]:
        """Select this step's refinement candidates and their verify-vs-trust decision.

        Narrows the reference table to events whose `ShapeID`
        (`id_initial`/`sid_initial`) actually matches at least one live atom
        (via `ReferenceEventTable.live_events()`), then resolves each
        candidate's verify-vs-trust decision from the step's rate budget
        (`e_thr`). A candidate flagged "trust" is not excluded -- it still
        needs PSR/IRA-aligned positions to become a valid active-table row;
        only whether `Refinement` dispatches a real ARTn call for it changes.

        Parameters
        ----------
        atom_shapes : dict[int, ShapeID]
            Every non-crystal atom's already-resolved `ShapeID` this step,
            from `resolve_new_shapes()` -- passed straight through to
            `live_events()` so it can look shapes up instead of
            reclassifying.
        existing_pairs : set[tuple[int, int]] | None, optional
            `(atom_index, num_reference_event)` pairs already present in the
            persistent active table (carried over from the previous step).
            These are skipped.

        Returns
        -------
        list[RefinementCandidate]
            The full, symmetry-expanded worklist for `Refinement.execute()`.

        """
        existing_pairs = existing_pairs or set()

        raw_entries = []
        matched_rows = []
        seen_ref_idx: set[int] = set()
        supposed_ktot = 0.0
        for at_idx, dfevent in self.reference_table.live_events(atom_shapes):
            ref_idx = int(dfevent["idx_ref"])
            if (at_idx, ref_idx) in existing_pairs:
                continue
            if ref_idx not in seen_ref_idx:
                seen_ref_idx.add(ref_idx)
                matched_rows.append(dfevent)
            for symmetry_index, _sym in enumerate(dfevent.at["sym_matrix"]):
                raw_entries += [(at_idx, dfevent, symmetry_index)]
                supposed_ktot += dfevent.at["k"]

        if not raw_entries:
            return []

        e_thr = self._refinement_energy_threshold(
            pd.DataFrame(matched_rows), supposed_ktot
        )

        return [
            RefinementCandidate(
                central_atom_index=at_idx,
                dfevent=dfevent,
                symmetry_index=symmetry_index,
                verify=dfevent.at["dE_forward"] <= e_thr,
            )
            for at_idx, dfevent, symmetry_index in raw_entries
        ]

    def _refinement_energy_threshold(
        self, matched_rows: pd.DataFrame, supposed_ktot: float
    ) -> float:
        """Adaptive barrier above which a refinement candidate is trusted rather than ARTn-verified.

        Parameters
        ----------
        matched_rows : pd.DataFrame
            Reference-table rows that actually produced at least one
            refinement candidate this step -- narrower than the coarse
            `id_initial` pre-filter, so the threshold search below isn't
            skewed by rows with no live atom currently matching their shape.
        supposed_ktot : float
            Total catalogued rate of every candidate this step, before any
            real ARTn verification.

        Returns
        -------
        float
            The `dE_forward` cutoff: candidates above it are trusted as-is.

        """
        tol = self.params.control.refine_thr
        k_thr = supposed_ktot * tol

        # get energy corresponding to the first k value just under k_thr
        mask = matched_rows["k"] < k_thr
        if mask.any():
            e_value = matched_rows.loc[mask].sort_values("k").iloc[-1]["dE_forward"]
        else:  # refine no event
            e_value = 0.0
        e_value += 0.1  # to be sure want using condition
        return e_value

    def execute_refinements(
        self,
        atom_shapes: dict[int, ShapeID],
        existing_pairs: set[tuple[int, int]] | None = None,
    ) -> Refinement:
        """Refine every candidate selected by `build_refinement_candidates()`.

        Parameters
        ----------
        atom_shapes : dict[int, ShapeID]
            Every non-crystal atom's already-resolved `ShapeID` this step,
            passed straight through to `build_refinement_candidates()`.
        existing_pairs : set[tuple[int, int]] | None, optional
            `(atom_index, num_reference_event)` pairs already present in the
            persistent active table (carried over from the previous step).
            These are skipped during refinement.

        Returns
        -------
        Refinement
            The refinement class with results.

        """
        candidates = self.build_refinement_candidates(
            atom_shapes, existing_pairs=existing_pairs
        )
        refinement = Refinement(
            self.params,
            self.loggers,
            self.system,
            self.neighbors_list,
            self.manager,
        )
        refinement.execute(candidates)
        self.otfml.retry_extrapolating("refine", refinement)
        return refinement

    def add_active_events(
        self, events: list[EventRefinementOutput]
    ) -> ActiveEventTable:
        """Create a new ActiveEventTable, add active events and return it.

        Parameters
        ----------
        events : list[RefinementsInfo]
            List of events to be added.

        Returns
        -------
        ActiveEventTable
            The active event table object.

        """
        # Extend the persistent active table (initialised once in `run()`).
        # Any rows surviving from the previous step are already present and
        # are not re-added because Refinement skipped them via existing_pairs.
        self.active_table.add_events(events)
        return self.active_table

    def _select_event(
        self,
        active_table: ActiveEventTable,
    ) -> tuple[int, float, float]:
        """Select an event in the active table based on the refection free algorithm.

        Uses ``self.bias`` when set and enabled; otherwise performs a standard
        unbiased rejection-free selection.  ``delta_t`` and ``ktot`` are always
        derived from the rates of the pool at the moment of acceptance.

        Parameters
        ----------
        active_table : ActiveEventTable
            The ActiveEventTable object with active events.

        Returns
        -------
        tuple[int, float, float]
            A typle containing :
            - int: Index of the selected event in the ActiveEventTable table.
            - float: time increment associated with the event.
            - float: total rate constant of the active events.

        """
        l_k = np.array(
            [active_table.table.loc[i].at["k"] for i in range(len(active_table.table))]
        )
        if self.bias is None:
            idx_selected_event, delta_t, ktot = rejection_free(l_k)
        else:
            idx_selected_event, delta_t, ktot = self.bias.select(
                rejection_free,
                l_k,
                active_table,
                self.system,
                self.reference_table,
                self.atomic_environment,
                self.neighbors_list,
            )
        return idx_selected_event, delta_t, ktot

    def reconstruction(self, active_table):
        # TODO make a Result

        err_reference = []
        err_ae = []
        while len(active_table.table) > 0:
            ##=>Select event
            idx_selected_event, delta_t, ktot = self._select_event(active_table)
            selected_event = active_table.table.loc[idx_selected_event]
            self.loggers.info(
                "log",
                (
                    "\n\t :=> Selected event context: "
                    f"idx={idx_selected_event}, "
                    f"atom_index={selected_event.at['atom_index']}, "
                    f"reference_event={selected_event.at['num_reference_event']}, "
                    f"k={selected_event.at['k']:.6e}, "
                    f"Ea={selected_event.at['dE_forward']:.6f} eV"
                ),
            )
            ##=>Reconstruct event
            self.loggers.info("log", "\t :=> Event Reconstruction")
            result_reconstruction = self._reconstruction_active_event(
                idx_selected_event, active_table
            )
            if result_reconstruction.is_ok():
                num_ref_event = active_table.table.loc[idx_selected_event].at[
                    "num_reference_event"
                ]
                event_id = self.reference_table.table[
                    self.reference_table.table["idx_ref"] == num_ref_event
                ]["event_id"].values[0]
                self.loggers.info(
                    "log",
                    f"\t :=> Reconstruction succeeded (reference event {num_ref_event}, event_id={fmt_hash(event_id)}, Ea={selected_event.at['dE_forward']:.6f} eV).",
                )
                break
            else:
                num_ref_event = active_table.table.loc[idx_selected_event].at[
                    "num_reference_event"
                ]
                err = result_reconstruction.err_value()
                err_type = getattr(err, "type", "UNKNOWN")
                self.loggers.info(
                    "log",
                    (
                        f"\t :=> Reconstruction fails (reference event {num_ref_event}) "
                        f"[type={err_type}] : {err.message}"
                    ),
                )
                ae_topo = self.reference_table.table[
                    self.reference_table.table["idx_ref"] == num_ref_event
                ]["id_initial"].values[0]
                err_reference += [num_ref_event]
                err_ae += [ae_topo]

                self.loggers.info("log", "\t :=> Removing active event.")
                active_table.remove(idx_selected_event)
        else:
            self.loggers.error("log", "All event reconstuctions failed.")
            self._close()

        return (
            result_reconstruction,
            delta_t,
            ktot,
            idx_selected_event,
            err_reference,
            err_ae,
        )

    def _reconstruction_active_event(
        self, idx_selected_event: int, active_table: AtomicEnvironment
    ):
        central_atom = active_table.table.loc[idx_selected_event].at["atom_index"]
        neighbors = self.neighbors_list.get_neighbors("rcut", central_atom)
        saddle_configuration = active_table.table.loc[idx_selected_event].at[
            "saddle_configuration"
        ].copy()
        local_types = np.asarray(self.system.types)[neighbors]
        supposed_final = active_table.table.loc[idx_selected_event].at[
            "final_configuration"
        ].with_types(local_types)
        supposed_initial = self.system.configuration[neighbors]

        # Move the system to the saddle point
        self.system.update_positions(saddle_configuration, atom_idx=neighbors)

        # try to reconstruct
        result = Reconstruction(self.params, self.manager).reconstruct(
            supposed_initial,
            supposed_final,
            self.system.configuration,
            self.params.psr.matching_score_thr,
            neighbors,
        )
        # result with min1, saddle, min2 pos

        # Back to original positions, in case reconstruction fails
        self.system.update_positions(supposed_initial, atom_idx=neighbors)
        return result

    def _apply_event(
        self, idx_selected_event: int, active_table: ActiveEventTable
    ) -> None:
        """Apply an active event to the system.

        Parameters
        ----------
        idx_selected_event : int
            index of the selected event in the active_table's table
        active_table : ActiveEventTable
            The ActiveEventTable okbject with active events.

        """
        final_configuration = active_table.table.loc[idx_selected_event].at["final_configuration"]
        self.system.update_positions(final_configuration)

    def minimize_system(self, configuration: Configuration) -> None:
        """Minimize the system and update its positions."""
        if self.otfml.is_enabled_for_phase("minimize"):
            self.otfml.retry_extrapolating_minimization(
                lambda: self._minimize_system_once(configuration=configuration)
            )
            return
        self._minimize_system_once(configuration=configuration)

    def _minimize_system_once(self, configuration: Configuration) -> None:
        """Perform a single minimization without OTF retry handling."""
        if self.params.control.restart_file is None:
            self.loggers.info("log", ":=> Minimizing the system")
        else:
            self.loggers.info("log", ":=> Computing energies")
        if self.otfml.is_enabled_for_phase("minimize"):
            self.manager.global_reset_otf_flags()
        new_configuration, total_energy = self.manager.global_minimize_with_results(
            self.params, configuration=configuration
        )
        if self.params.control.restart_file is None:
            self.system.update_positions(new_configuration)
        self.total_energy = total_energy
        self.potential_energy = self.manager.global_get_potential_energy()
        if self.otfml.is_enabled_for_phase("minimize"):
            return self.manager.global_get_otf_flags()
        return None

    def get_info_atomic_environments(
        self, new_shapes: dict[ShapeID, list[int]]
    ) -> AtomicEnvironmentInfo:
        """Get atomic environments informations for outputs.

        See :func:`pykmc.info_simulation.info_atomic_environments`.

        Parameters
        ----------
        new_shapes : dict[ShapeID, list[int]]
            shape -> eligible atom indices needing search.

        Returns
        -------
        AtomicEnvironmentInfo
            The Dataclass with atomic environments informations.

        """
        return info_atomic_environments(self, new_shapes)

    def get_info_reference_event_searches(
        self,
        results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]],
    ) -> ReferenceEventSearchInfo:
        """Get reference event searches informations for outputs.

        See :func:`pykmc.info_simulation.info_reference_event_searches`.

        Parameters
        ----------
        results_reference_event_searches : list[Result[EventSearchOutput, ErrorInfo]]
            The list of Result from event searches.

        Returns
        -------
        ReferenceEventSearchInfo
            The Dataclass with reference event searches informations.

        """
        return info_reference_event_searches(results_reference_event_searches)

    def get_info_is_valid_reference_events(
        self, results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]]
    ) -> ReferenceValidEventsInfo:
        """Get informations on whether or not an event is valid.

        See :func:`pykmc.info_simulation.info_is_valid_reference_events`.

        Parameters
        ----------
        results_is_valid_events : list[Result[pd.DataFrame, ErrorInfo]]
            List of Results from ReferenceEventTable.is_valid_event().

        Returns
        -------
        ReferenceValidEventsInfo
            The Dataclass with information on whether an event is valid or not.

        """
        return info_is_valid_reference_events(results_is_valid_events)

    def get_info_refinements(
        self, results_refinements: list[Result[EventSearchOutput, ErrorType]]
    ) -> RefinementsInfo:
        """Get informations on refined events.

        See :func:`pykmc.info_simulation.info_refinements`.

        Parameters
        ----------
        results_refinements : list[Result[EventSearchOutput, ErrorType]]
           List of Results from the refinements.

        Returns
        -------
        RefinementsInfo
           The dataclass with refinements informations.

        """
        return info_refinements(results_refinements)

    def get_info_adaptive_search(
        self, shape_search_stats: dict[ShapeID, ShapeSearchStats] | None
    ) -> AdaptiveSearchInfo | None:
        """Get adaptive event-search convergence informations for outputs.

        See :func:`pykmc.info_simulation.info_adaptive_search`.

        Parameters
        ----------
        shape_search_stats : dict[ShapeID, ShapeSearchStats] | None
            Per-shape stats from `adaptive_event_search`, or None when
            `params.eventsearch.adaptive_search` is False.

        Returns
        -------
        AdaptiveSearchInfo | None
            The dataclass with adaptive search convergence informations, or
            None outside adaptive mode or when there were no new shapes to
            report on this step (avoids logging an empty block every step).

        """
        if not shape_search_stats:
            return None
        return info_adaptive_search(shape_search_stats)

    def _initialize(self) -> None:
        """Initialize the KMC attributes.

        See :func:`pykmc.Initializer.initialize()`.

        """
        Initializer(self).initialize()

    def _append_snapshot_to_trajectory(self) -> None:
        """Append the configurations positions to the trajectory file."""
        atoms = Atoms(
            self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        write(self.params.control.trajectory_output, atoms, append=True)

    def _save(self) -> None:
        """Save the reference event table and the list of visited environments."""
        self.reference_table.save("reference_table.pickle")
        self.loggers.reference_table_write("reference_table", self.reference_table)
        with open(self.params.control.visited_environments_output, "wb") as file:
            pickle.dump(self.visited_environments, file)

    def _save_restart_file(self, last_step, last_time):
        """
        Save end simulation informations
        """
        np.savez(
            "restart_" + str(last_step) + ".npz",
            last_step=last_step,
            last_time=last_time,
        )

    def _close(self) -> None:
        """Close the simulation."""
        self.loggers.info("log", ":=> End of simulation")
        self.manager.close_all()
        sys.exit()
