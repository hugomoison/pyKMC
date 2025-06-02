"""Module for executing off-lattice kinetic Monte Carlo (KMC) simulations.

This module defines the `KMC` class.
"""

from pykmc import NeighborsList, AtomicEnvironment, ActiveEventTable, Config
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
)
from .eventsearch import EventSearch
from .refinement import Refinement
from .log import Colors
from basin import Basin


# TODO fix reconstruction = False
# NOTE can maybe reimplment tries if empty catalog
# TODO add select histo refinement


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
    visited_environment : set[str|bytes]
        Track atomic environments already explored. Those for which event searches as been previously done.
    total_energy : float
        The total energy of the system.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.loggers = None
        self.system = None
        self.engine = None
        self.neighbors_list = None
        self.atomic_environment = None
        self.reference_table = None
        self.visited_environments = None
        self.total_energy = None

    def run(self) -> None:
        """Run the simulation."""
        # Initialize the simulation, KMC attributes and minimize the system
        self._initialize()
        # Write initial step to file
        self._append_snapshot_to_trajectory()

        # LOOP KMC PARAMETERS
        nkmc_steps = self.config.control.n_steps
        time = 0.0  # in seconds
        nsearch = self.config.eventsearch.nsearch

        # KMC LOOP
        for step in range(nkmc_steps):
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
            results_is_valid_events = self.add_reference_events(
                event_search.get_successes_results()
            )

            ##=>Close simulation if no events in the reference table
            if len(self.reference_table.table) == 0:
                self.loggers.error(
                    "log",
                    "No events have been found, empty reference events table. \n \tTry to increase nsearch or saddle point search algorithm's parameters. \n \tClosing the simulation.",
                )
                self._close()

            ################################
            #CONTRUCT CURRENT ACTIVE EVENTS#
            ################################
            active_table = self.refinement() 
            ###############################
            #SELECT EVENT IN ACTIVE TABLE # 
            ###############################
            idx_selected_event, delta_t = self._select_event(active_table)
            
            
            # == Refinement ==
            ##=>Subset of reference_event_table with generic event that can be apply to the current step (ie event_id in atomic environment)
            subset_reference_event_table = self.reference_table.has_id_subset_table(
                self.atomic_environment.atomic_environment_list
            )

            ##=>Refines all event in subset
            refinement = self.execute_refinements(subset_reference_event_table)

            # == ADD ACTIVE EVENT TO ACTIVE EVENT TABLE ==
            active_table = self.add_active_events(refinement.get_successes_results())

            # == Update System ==
            ##=>Select event
            idx_selected_event, delta_t, ktot = self._select_event(active_table)
            time += delta_t * 10**-12  # time is in seconds

            ##=>Move system
            self._apply_event(idx_selected_event, active_table)

            ##=>Minimize
            self.minimize_system()

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

            self.loggers.table_line_info_kmc(
                "output",
                step + 1,
                delta_t * 10**-12,
                time,
                active_table.table.loc[idx_selected_event].at["num_reference_event"],
                active_table.table.loc[idx_selected_event].at["energy_barrier"],
                active_table.table.loc[idx_selected_event].at["k"],
                ktot,
                self.total_energy,
            )

            # == Update variables ==
            l_ids = list(set(self.atomic_environment.atomic_environment_list))
            self.visited_environments.update(
                set(l_ids).difference(self.visited_environments)
            )
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

            # == Save Reference Table and List visited environment :
            self._save()
            self._append_snapshot_to_trajectory()

            # == Check if only cristalline environments ==
            if set(list(self.atomic_environment.atomic_environment_list)) == {
                "crystal"
            }:
                self.loggers.info("log", ":=> Only atoms with cristalline environment")
                self._close()

    def get_new_environments(self) -> list[str | bytes]:
        """Get atomic environments of the current system that has not been already explored.

        Returns
        -------
        list[str|bytes]
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
        self, new_environments: list[str | bytes], nsearch: int
    ) -> list[int]:
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table.

        For each new environment it adds nseach atoms having that environment to the list.

        Parameters
        ----------
        new_environments : list[str|bytes]
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
        # for each atomic environment hash in new_environment
        for env in new_environments:
            # find all index having that hash
            tmp1 = [
                i
                for i, e in enumerate(self.atomic_environment.atomic_environment_list)
                if e == env
            ]
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
        event_search = EventSearch(self.system, self.engine, self.loggers)
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
            self.engine,
        )
        refinement.execute(df_reference_events)
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

    def _select_event(self, active_table: ActiveEventTable) -> tuple[int, float, float]:
        """Select an event in the active table based on the refection free algorithm.

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
        # list of rate constant
        l_k = np.array(
            [active_table.table.loc[i].at["k"] for i in range(len(active_table.table))]
        )
        idx_selected_event, delta_t, ktot = rejection_free(l_k)
        return idx_selected_event, delta_t, ktot

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

    def minimize_system(self) -> None:
        """Minimize the system and update its positions."""
        self.loggers.info("log", ":=> Minimizing the system")
        new_positions, total_energy = self.engine.minimize(self.system)
        self.system.update_positions(new_positions)
        self.total_energy = total_energy

    def get_info_atomic_environments(
        self, new_environments: list[str | bytes]
    ) -> AtomicEnvironmentInfo:
        """Get atomic environments informations for outputs.

        See :func:`pykmc.info_simulation.info_atomic_environments`.

        Parameters
        ----------
        new_environments : list[str | bytes]
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

    def _close(self) -> None:
        """Close the simulation."""
        self.loggers.info("log", ":=> End of simulation")
        sys.exit()
