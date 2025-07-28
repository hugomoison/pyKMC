"""Engine interface module.

This module defines the `Engine` class, which serves as a unified interface for
atomistic simulation engines (e.g., LAMMPS). It abstracts away backend-specific
details and exposes a common API for tasks like energy minimization and event search.
"""

from .lammpsengine import LammpsEngine
from .config import Config
from .system import System
from .result import EventSearchOutput, EventRefinementOutput, Result, ErrorInfo
import numpy as np
import os


class Engine:
    """Unified interface to different E/F Engines.

    It delegates operations to the appropriate engine-specific implementation.

    Parameters
    ----------
    config : Config
        Parameters of the simulation.

    Raises
    ------
    ValueError
        If the specified engine type in the config is not recognized.

    See Also
    --------
    LammpsEngine : Backend class for the LAMMPS engine implementation.

    """

    def __init__(self, config: Config) -> None:
        self.engine_type = config.control.engine
        match self.engine_type:
            case "lammps":
                self.engine = LammpsEngine(config)
            case _:
                raise ValueError("Engine type unknown")

    def minimize(self, system: System) -> tuple[np.ndarray, float]:
        """Minimize the given System.

        Parameters
        ----------
        system : System
            The atomic System.

        Returns
        -------
        tuple[np.ndarray, float] :
            A tuple containing :
            - New atomic positions after the minimization
            - Total energy of the minimized system.

        """
        minimized_positions, total_energy = self.engine.minimize(system)
        return minimized_positions, total_energy

    def search_event(
        self, system: System, central_atom_idx: int
    ) -> Result[EventSearchOutput, ErrorInfo]:
        """Perform an event search around a given central atom.

        Depending on the selected event search style (e.g., 'partn'), the method calls the corresponding backend function.

        Parameters
        ----------
        system : System
            The atomic system.
        central_atom_idx : int
            Index of the central atom around which the event search is perfromed.

        Returns
        -------
        Result[EventSearchOutput,ErrorInfo]
            The result of the event search.

        Raises
        ------
        Exception
            If the event search style specified in the config is unknown.

        """
        match self.engine.config.eventsearch.style:
            case "partn":
                original_stdout_fd = os.dup(1)
                devnull = os.open(os.devnull, os.O_WRONLY)
                # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
                os.dup2(devnull, 1)
                result = self.engine.pARTn(system, central_atom_idx)
                # Restore original stdout (fd 1)
                os.dup2(original_stdout_fd, 1)
                os.close(original_stdout_fd)
                os.close(devnull)
            case _:
                raise Exception("Event Search style unknown")
        return result

    def refine_event(
        self, system: System, central_atom_idx: int
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Perform an event refinement around a given central atom.

        Depending on the selected event search style (e.g., 'partn'), the method calls the corresponding backend function.

        Parameters
        ----------
        system : System
            The atomic system.
        central_atom_idx : int
            Index of the central atom around which the event search is perfromed.

        Returns
        -------
        Result[EventSearchOutput,ErrorInfo]
            The result of the event search.

        Raises
        ------
        Exception
            If the event search style specified in the config is unknown.

        """
        match self.engine.config.eventsearch.style:
            case "partn":
                original_stdout_fd = os.dup(1)
                devnull = os.open(os.devnull, os.O_WRONLY)
                # Redirect stdout (fd 1) to /dev/null, only way to deal with pARTn error write
                os.dup2(devnull, 1)
                result = self.engine.pARTn_refine_event(system, central_atom_idx)
                # Restore original stdout (fd 1)
                os.dup2(original_stdout_fd, 1)
                os.close(original_stdout_fd)
                os.close(devnull)
            case _:
                raise Exception("Event Search style unknown")
        return result

    def compute_potential_energy(self, system: System) -> float:
        """Compute the potential energy of the system.

        Parameters
        ----------
        system : System
            The atomic system.

        Returns
        -------
        float
            The potential energy.

        """
        potential_energy = self.engine.compute_potential_energy(system)
        return potential_energy

    def compute_distances(self, system: System) -> None:
        """Compute distances, here for future implementation.

        Parameters
        ----------
        system : System
            The atomic system.

        """
        self.engine.compute_distances(system)

    def neighbors(self, system: System) -> None:
        """Compute neighbors, here for future implementation.

        Parameters
        ----------
        system : System
            The atomic system.

        """
        self.engine.neighbors(system)

    def _close(self) -> None:
        """Close the simulation.
        """
        self.engine._close()