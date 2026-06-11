"""Module implementing the EventSearch class that deals with the event search procedure."""

from .result import EventSearchOutput
from .system import System
from .enginemanager.lmpi.pool import Manager
from .log import LogKMC
from .utils.geometry import translate
import numpy as np


class EventSearch:
    """Perform event searches and manage results.

    Parameters
    ----------
    system : System
        The atomic system.
    engine : Engine
        The engine used to perform the event search.
    loggers : LogKMC
        The KMC simulation loggers.

    """

    def __init__(
        self, config, system: System, manager: Manager, loggers: LogKMC
    ) -> None:
        self.config = config
        self.system = system
        self.manager = manager
        self.loggers = loggers
        self.results = None

    def execute(self, central_atom_research_list: list[int]) -> None:
        """Execute an event search for each central atom in the central_atom_research_list list.

        It stores the results of the event searches in self.results

        Parameters
        ----------
        central_atom_research_list : list[int]
            list of central atom around which we will perform the event search.

        """
        self.results = []
        self.loggers.info(
            "log",
            "\t :=> Searching {} reference events".format(
                len(central_atom_research_list)
            ),
        )
        if self.config.control.active_volume == True:
            if self.config.activevolume.ract <= self.config.atomicenvironment.rcut:
                raise ValueError(
                    "Active Volume radius is smaller than cutoff radius. Please increase ract or decrease rcut"
                )
            futures = self.manager.partn_search(
                config=self.config,
                central_atom=central_atom_research_list,
                positions=self.system.positions.copy(),
                cell=self.system.cell.copy(),
                types=self.system.types.copy(),
            )
        else:
            futures = self.manager.partn_search(
                config=self.config,
                central_atom=central_atom_research_list,
                positions=self.system.positions.copy(),
                types=self.system.types.copy(),
            )
        for f in futures:
            self.results.append(f.result())

            self.loggers.progress_bar(
                "progress", len(self.results), len(central_atom_research_list)
            )
        # self.results = [f.result() for f in futures]

        # for i, at_idx in enumerate(central_atom_research_list):
        #    event_search_output = self.engine.search_event(self.system, at_idx)
        #    self.results.append(event_search_output)
        #    self.loggers.progress_bar(
        #        "progress", i + 1, len(central_atom_research_list)
        #    )

    def _center_event_positions(
        self, event_search_output: EventSearchOutput
    ) -> EventSearchOutput:
        """Translate positions of the events so that the atom that move the most during the event is at the center of the simulation box.

        It is used to that when we store positions in the reference table around the atom that move the most we don't have periodic bound problems.

        Parameters
        ----------
        event_search_output : EventSearchOutput
            The dataclass countaining the event search outputs.

        Returns
        -------
        EventSearchOutput
            The dataclass countaining the event search outputs with translated positions.

        """
        # Translate atoms so that the atom that moves the most is at the center of the cell at start event, prevent pbc problem with psr
        cell = self.system.cell
        ax, ay, az = cell[0][0], cell[1][1], cell[2][2]
        # displacement
        move_atom_idx = event_search_output.move_atom_index
        dx, dy, dz = (
            ax / 2 - event_search_output.min1_positions[move_atom_idx][0],
            ay / 2 - event_search_output.min1_positions[move_atom_idx][1],
            az / 2 - event_search_output.min1_positions[move_atom_idx][2],
        )
        displacement = np.array([dx, dy, dz])
        event_search_output.min1_positions = translate(
            event_search_output.min1_positions, displacement, cell
        )
        event_search_output.saddle_positions = translate(
            event_search_output.saddle_positions, displacement, cell
        )
        event_search_output.min2_positions = translate(
            event_search_output.min2_positions, displacement, cell
        )
        event_search_output.cell = cell
        return event_search_output

    def get_successes_results(self) -> list[EventSearchOutput]:
        """Return a list of only successful event searches.

        Returns
        -------
        list[EventSearchOutput]
            List of successful event searches.

        """
        return [
            self._center_event_positions(e.ok_value())
            for e in self.results
            if e.is_ok()
        ]
