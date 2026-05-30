"""Module for executing off-lattice kinetic Monte Carlo (KMC) simulations.

This module defines the `KMC` class.
"""

from pykmc import NeighborsList, AtomicEnvironment, ActiveEventTable, Config, Reconstruction
import random
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
    EventRefinementOutput,
    ReconstructionOutput,
    Err,
    Ok
)
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
    info_basin_events
)
from .eventsearch import EventSearch
from .refinement import Refinement
from .log import Colors
import time
from .utils import push_towards, compute_delr
import copy
from .basins.detection import DetectorThreshold
from .basins import BasinsGenericEvents
from .bias import Bias


# NOTE can maybe reimplment tries if empty catalog
#TODO: Add reconstruction info

class KMC:
    """Manage and execute the Kinetic Monte Carlo (KMC) simulation.

    This class acts as the central controller coordinating all phases of the simulation:
    initialization, event search, event refinement, event selection, system updates,
    minimization, logging, and termination.

    Attributes
    ----------
    config : Config
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

    def __init__(self, config: Config) -> None:
        self.config = config
        self.loggers = None
        self.system = None
        self.manager = None
        self.engine = None
        self.neighbors_list = None
        self.atomic_environment = None
        self.reference_table = None
        self.visited_environments = None
        self.total_energy = None
        self.potential_energy = None
        self.bias: Bias | None = None

    def run(self) -> None:
        """Run the simulation."""
        # Initialize the simulation, KMC attributes and minimize the system
        #self._initialize()
        self.manager.initialize_sessions(self.config, self.system)
        self.minimize_system()
        self.neighbors_list = NeighborsList(
                self.system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
        self.atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
            )
        self.inactive_ae = (
            AtomicEnvironment(
                style="region",
                region=self.config.inactive_atoms,
                positions=self.system.positions,
                atom_types=self.system.types,
            ) if self.config.inactive_atoms is not None else None
        )
        self.frozen_ae = (
            AtomicEnvironment(
                style="region",
                region=self.config.frozen_atoms,
                positions=self.system.positions,
                atom_types=self.system.types,
            ) if self.config.frozen_atoms is not None else None
        )
        #Set new positions to all sessions/engine :
        self.manager.use_local()
        self.manager.set_all_positions(self.system.positions)

        if self.config.control.restart_file is None:
        # Write initial step to file
            self._append_snapshot_to_trajectory()
            last_step = 0
            total_time = 0.0

        else : #read restart file
            self.loggers.info("log", ":=> Reading restart file")
            restart_info = np.load(self.config.control.restart_file)
            last_step = restart_info["last_step"]
            total_time = restart_info["last_time"]
            self.loggers.info("log", ":=> last step = {}, last_end_time = {}ps".format(last_step, total_time))

        # LOOP KMC PARAMETERS
        nkmc_steps = self.config.control.n_steps
        last_step +=1
        nsearch = self.config.eventsearch.nsearch



        # KMC LOOP
        for step in range(last_step, nkmc_steps+last_step):
            start_real = time.time()
            start_cpu = time.process_time()

            self.loggers.info(
                "log",
                "{}{}Step : {}{}".format(
                    Colors.BOLD.value, Colors.YELLOW.value, step, Colors.RESET.value
                ),
            )

            # == Find Current atomic environments that has not been visited ==
            new_environments = self.get_new_environments()

            # == FIND NEW GENERIC EVENTS ==
            ##=>List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(
                new_environments, nsearch
            )

            ##=>Perform event search on each atom in central_atom_research_list
            event_search = self.execute_event_searches(central_atom_research_list)

            # == ADD NEW GENERIC EVENTS TO REFERENCE EVENT TABLE ==
            ##=>Check if the event is valid, ie if not already present and has a valid energy barrier if yes add it to the reference table
            search_results = event_search.get_successes_results()
            if self.inactive_ae is not None:
                inactive_set = set(self.inactive_ae.get_atoms_with_id("in"))
                search_results = [r for r in search_results if r.move_atom_index not in inactive_set]
            results_is_valid_events = self.add_reference_events(search_results)

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
            ##=>Subset of reference_event_table with generic event that can be apply to the current step (ie event_id in atomic environment)
            subset_reference_event_table = self.reference_table.has_id_subset_table(
                self.atomic_environment.atomic_environment_list
            )
            ##=>Refines all event in subset
            refinement = self.execute_refinements(subset_reference_event_table)

            # == ADD ACTIVE EVENT TO ACTIVE EVENT TABLE ==
            active_table = self.add_active_events(refinement.get_successes_results())
            active_table.remove_duplicates(self.system.cell, self.neighbors_list)  #To be sure
            self.loggers.info("log", "\t :=> {} active events after removing duplicates.".format(len(active_table.table)))


            # == Update System ==
            self.manager.use_global()
            result_reconstruction, delta_t, ktot, idx_selected_event, err_reference, err_ae = self.reconstruction(active_table)
            events_info = info_active_events(self.system.types, self.reference_table, active_table)
            if len(err_reference) != 0 :
                self.loggers.info("log", "\t :=> Removing reference event from which reconstruction failed.")
                self.reference_table.remove(list(set(err_reference)))
                self.loggers.info("log", "\t :=> Removing topology from known environments from which reconstruction failed.")
                self.visited_environments = self.visited_environments.difference(set(err_ae))
            events_info = events_info.output_msg()




            #INFO :
            self.loggers.events_file_step_first_line("events", step)
            self.loggers.events_applicable_info_line("events", idx_selected_event)
            self.loggers.info("events", events_info)

                #TODO: Temporary, need to unified kmc main loop and basin operations + ugly
            detector = DetectorThreshold()
                #IF selected event shows we are in a basin
            if self.config.control.basin and detector.detect(active_table.table.iloc[idx_selected_event], self.reference_table.table, self.config.basin.energy_thr, True) :
                self.loggers.info("log","\t :=> System is in a Basin." )
                self.loggers.info("log","\t :=> Exploring the Basin." )
                #get basin info/explore
                basin = BasinsGenericEvents(self.config, self.reference_table, self.visited_environments, self.manager)
                self.system.update_positions(result_reconstruction.ok_value().min1_positions)
                result_basin = basin.execute(self.system)
                if result_basin.is_ok() : #Basin did no fail
                #move system to a state connected to the exit_state
                    self.system.update_positions(result_basin.ok_value().initial_system_positions)
                    self.neighbors_list = basin.states[result_basin.ok_value().from_state].neighbors_list
                #construct new active table with only event : new_actual_state - > exit_state
                    tmp_active_table = ActiveEventTable(self.config)
                    tmp_event = EventRefinementOutput(central_atom_index=result_basin.ok_value().central_atom,
                                                      saddle_positions=result_basin.ok_value().saddle_positions,
                                                      E_saddle=-1,
                                                      min2_positions=result_basin.ok_value().final_positions,
                                                      dE_forward=result_basin.ok_value().energy_barrier,
                                                      num_reference_event=result_basin.ok_value().num_reference_event)
                    neighbors = result_basin.ok_value().neighbors
                    tmp_active_table.add_events(tmp_event)
                #reconstruct event
                    self.manager.use_global()
                    result_basin_reconstruction = self._reconstruction_active_event(0, tmp_active_table)
                    if result_basin_reconstruction.is_ok() :
                        self.system.update_positions(result_basin_reconstruction.ok_value().min2_positions)
                        self.total_energy = result_basin_reconstruction.ok_value().min2_etot
                        delta_t = result_basin.ok_value().t_exit
                        ktot = result_basin.ok_value().k_tot
                        idx_selected_event = 0
                        active_table.table = tmp_active_table.table

                        #INFO
                        idx_exit_event, basin_info = info_basin_events(self.system.types, self.reference_table, basin.connectivity_table, result_basin.ok_value().exit_state)
                        basin_info = basin_info.output_msg()
                        self.loggers.events_basin_info_line("events",idx_exit_event )
                        self.loggers.info("events", basin_info)


                    else :
                       self.loggers.info("log", "\t :=> Reconstruction Exit State Basin fails with error {}, back to original event".format(result_basin_reconstruction.err_value()))
                       self.system.update_positions(basin.states[0].system.positions)
                       self.system.update_positions(result_reconstruction.ok_value().min2_positions)
                else :
                    self.loggers.info("log", "\t :=> Basin fails with error : {}, back to original event".format(result_basin.err_value()))
                    self.system.update_positions(result_reconstruction.ok_value().min2_positions)
                if basin.connectivity_table is not None :
                    basin.connectivity_table.save('basin_connectivity_'+str(step)+'.pickle')
                #update delta_t, ktot (use basin infos)
            else :
                self.system.update_positions(result_reconstruction.ok_value().min2_positions)
                self.total_energy = result_reconstruction.ok_value().min2_etot
            total_time += delta_t * 10**-12  # time is in seconds

            ###=> Synchronise all lammps instances with new positions
            self.manager.use_local()
            self.manager.set_all_positions(positions=self.system.positions)
            ##=>Minimize

            # == Log informations ==
            atomic_environment_info = self.get_info_atomic_environments(
                new_environments
            )
            reference_event_searches_info = self.get_info_reference_event_searches(
                event_search.results
            )
            is_valid_events_info = self.get_info_is_valid_reference_events(
                results_is_valid_events
            )
            refinements_info = self.get_info_refinements(refinement.results)
            kmc_loop_info = KMCLoopInfo(
                step=step,
                atomic_environment_info=atomic_environment_info,
                reference_event_searches_info=reference_event_searches_info,
                valid_event_info=is_valid_events_info,
                refinements_info=refinements_info,
            )
            self.loggers.info("info", kmc_loop_info.output_msg())


            elapsed_real = time.time() - start_real
            elapsed_cpu = time.process_time() - start_cpu

            self.loggers.table_line_info_kmc(
                "output",
                step,
                delta_t * 10**-12,
                total_time,
                active_table.table.loc[idx_selected_event].at["num_reference_event"],
                active_table.table.loc[idx_selected_event].at["energy_barrier"],
                active_table.table.loc[idx_selected_event].at["k"],
                ktot,
                self.total_energy,
                elapsed_cpu,
                elapsed_real
            )

            # == Update variables ==
            self.neighbors_list = NeighborsList(
                self.system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
            self.atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
            )
            self.inactive_ae = (
                AtomicEnvironment(
                    style="region",
                    region=self.config.inactive_atoms,
                    positions=self.system.positions,
                    atom_types=self.system.types,
                ) if self.config.inactive_atoms is not None else None
            )
            self.frozen_ae = (
                AtomicEnvironment(
                    style="region",
                    region=self.config.frozen_atoms,
                    positions=self.system.positions,
                    atom_types=self.system.types,
                ) if self.config.frozen_atoms is not None else None
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

    def get_new_environments(self) -> list[str]:
        """Get atomic environments of the current system that has not been already explored.

        Returns
        -------
        list[str]
            The atomic environments of the current system that are encounter for the first time.

        """
        new_environments = self.atomic_environment.get_new_environments(
            self.visited_environments
        )
        self.loggers.info(
            "log",
            "\t :=> {} new atomic environments found".format(len(new_environments)),
        )
        return new_environments

    def central_atoms_research(
        self, new_environments: list[str], nsearch: int
    ) -> list[int]:
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table.

        For each new environment it adds nseach atoms having that environment to the list.

        Parameters
        ----------
        new_environments : list[str]
            List of atomic environment ID.
        nsearch : int
            Number of searches per atomic environment.

        Returns
        -------
        list[int]
            List of central atoms

        Raises
        ------
        IndexError
            If no atoms are found for a given environment, random.choice will raise an IndexError.

        """
        central_atom_research_list = []
        inactive_set = (
            set(self.inactive_ae.get_atoms_with_id("in"))
            if self.inactive_ae is not None else set()
        )
        # for each atomic environment hash in new_environment
        for env in new_environments:
            # find all index having that hash
            tmp1 = [
                i
                for i, e in enumerate(self.atomic_environment.atomic_environment_list)
                if e == env
            ]
            if inactive_set:
                tmp1 = [i for i in tmp1 if i not in inactive_set]
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
        event_search = EventSearch(self.config, self.system, self.manager, self.loggers)
        event_search.execute(central_atom_research_list)
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

    def execute_refinements(self, df_reference_events: pd.DataFrame) -> Refinement:
        """Refine all events in df_reference_events for all atoms on which they can be apply.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            Subset of the reference table with events that can be apply to the current system.

        Returns
        -------
        Refinement
            The refinement class with results.

        """
        refinement = Refinement(
            self.config,
            self.loggers,
            self.system,
            self.neighbors_list,
            self.atomic_environment,
            self.manager,
        )
        #refinement.execute(df_reference_events, self.potential_energy)
        refinement.execute(df_reference_events, self.total_energy)
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
        active_table = ActiveEventTable(self.config)
        active_table.add_events(events)
        return active_table

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
                rejection_free, l_k, active_table,
                self.system, self.reference_table, self.atomic_environment
            )
        return idx_selected_event, delta_t, ktot


    def reconstruction(self, active_table) :
            #TODO make a Result

            err_reference = []
            err_ae = []
            while len(active_table.table) > 0 :
                ##=>Select event
                idx_selected_event, delta_t, ktot = self._select_event(active_table)
                ##=>Reconstruct event
                self.loggers.info("log", "\t :=> Event Reconstruction")
                result_reconstruction = self._reconstruction_active_event(idx_selected_event, active_table)
                if result_reconstruction.is_ok() :
                    break
                else :
                    num_ref_event = active_table.table.loc[idx_selected_event].at['num_reference_event']
                    self.loggers.info("log", "\t :=> Reconstruction fails (reference event {}) :  {}".format(num_ref_event, result_reconstruction.err_value().message))
                    ae_topo = self.reference_table.table[self.reference_table.table['idx_ref'] == num_ref_event]['event_id'].values[0]
                    err_reference.append(num_ref_event)
                    err_ae.append(ae_topo)

                    self.loggers.info("log", "\t :=> Removing active event.")
                    active_table.remove(idx_selected_event)
            else :
                self.loggers.error("log", "All event reconstuctions failed.")
                self._close()
            return result_reconstruction, delta_t, ktot, idx_selected_event, err_reference, err_ae

    def _reconstruction_active_event(self, idx_selected_event: int, active_table: AtomicEnvironment) :
        central_atom = active_table.table.loc[idx_selected_event].at["atom_index"]
        neighbors = self.neighbors_list.get_neighbors("rcut", central_atom)
        saddle_positions = copy.deepcopy(active_table.table.loc[idx_selected_event].at["saddle_positions"])
        supposed_final_positions = copy.deepcopy(active_table.table.loc[idx_selected_event].at["final_positions"])
        supposed_initial_positions = copy.deepcopy(self.system.positions[neighbors])




        #Move the system to the saddle point
        self.system.update_positions(new_positions= saddle_positions, atom_idx = neighbors)

        #try to reconstruct
        result = Reconstruction(self.config, self.manager, types=self.system.types).reconstruct(supposed_initial_positions, supposed_final_positions, self.system.positions, self.system.cell, self.config.psr.matching_score_thr, neighbors)
        #result with min1, saddle, min2 pos

        #Back to original positions, in case reconstruction fails
        self.system.update_positions(new_positions = supposed_initial_positions, atom_idx = neighbors)
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
        new_positions = active_table.table.loc[idx_selected_event].at["final_positions"]
        self.system.update_positions(new_positions)

    def minimize_system(self, positions = None) -> None:
        """Minimize the system and update its positions."""
        if self.config.control.restart_file is None:
            self.loggers.info("log", ":=> Minimizing the system")
        else :
            self.loggers.info("log", ":=> Computing energies")
        new_positions, total_energy = self.manager.global_minimize_with_results(self.config, positions=positions, types=self.system.types)
        #TEST
        #future = self.manager.minimize_with_results(self.config, positions=positions)
        #new_positions, total_energy = future.result()
        #np.savetxt('before_min.dat', self.system.positions)
        #np.savetxt('after_min.dat', new_positions)
        if self.config.control.restart_file is None :
            self.system.update_positions(new_positions)
        self.total_energy = total_energy
        self.potential_energy = self.manager.global_get_potential_energy()

    def get_info_atomic_environments(
        self, new_environments: list[str]
    ) -> AtomicEnvironmentInfo:
        """Get atomic environments informations for outputs.

        See :func:`pykmc.info_simulation.info_atomic_environments`.

        Parameters
        ----------
        new_environments : list[str]
            List of new environments detected.

        Returns
        -------
        AtomicEnvironmentInfo
            The Dataclass with atomic environments informations.

        """
        return info_atomic_environments(self, new_environments)

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
        write(self.config.control.trajectory_output, atoms, append=True)

    def _save(self) -> None:
        """Save the reference event table and the list of visited environments."""
        self.reference_table.save("reference_table.pickle")
        with open(self.config.control.visited_environments_output, "wb") as file:
            pickle.dump(self.visited_environments, file)

    def _save_restart_file(self, last_step, last_time) :
        """
        Save end simulation informations
        """
        np.savez("restart_"+str(last_step)+".npz",
                 last_step = last_step,
                 last_time = last_time)


    def _close(self) -> None:
        """Close the simulation."""
        self.loggers.info("log", ":=> End of simulation")
        self.manager.close_all()
        sys.exit()

