"""KMC Simulation Initialization Module.

This module contains the `Initializer` class, which takes a reference to a `KMC` object
and sets up its attributes necessary for running the simulation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kmc import KMC
from .log import LogKMC, LOGGING_CONFIG
from .system import System
from .neighbors_list import NeighborsList
from .atomic_environment import AtomicEnvironment
from .event_table import ReferenceEventTable
from .bias import DirectionBias, PointBias, TopoBias
import pickle


class Initializer:
    """Initializer for the KMC class.

    Parameters
    ----------
    kmc : KMC
        KMC object initialized based on its configuration.

    """

    def __init__(self, kmc: "KMC") -> None:
        self.kmc = kmc

    def initialize(self) -> None:
        """Initialize the entire KMC object before starting the simulation."""
        self.initialize_loggers()
        self.initialize_system()
        self.initialize_neighbors_list()
        self.initialize_atomic_environments()
        self.initialize_reference_table()
        self._initialize_visited_environments()
        self.initialize_bias()

        self.kmc.loggers.new_line("log")
        self.kmc.loggers.info("log", "===========================")
        self.kmc.loggers.info("log", "= Starting KMC simulation =")
        self.kmc.loggers.info("log", "===========================")

        if self.kmc.config.control.restart_file is None : 
            self.kmc.loggers.table_line_info_kmc(
            "output", 0, 0.0, 0.0, None, None, None, None, self.kmc.total_energy
        )
        else : 
            self.kmc.loggers.info("log", ":=> Restarting")

    def initialize_loggers(self) -> None:
        """Initialize the loggers and create their files."""
        self.kmc.loggers = LogKMC(LOGGING_CONFIG, verbosity=self.kmc.config.control.verbosity)
        self.kmc.loggers.title("log")
        self.kmc.loggers.write_parameters("log", self.kmc.config)
        if self.kmc.config.control.restart_file is None : 
            self.kmc.loggers.output_file_header("output")
            self.kmc.loggers.events_file_header("events")

    def initialize_system(self) -> None:
        """Read and initialize the system from the intial configuration file."""
        self.kmc.loggers.info(
            "log",
            ":=> Reading initial configuration file : {}".format(
                self.kmc.config.control.initial_config
            ),
        )
        self.kmc.system = System.create_from_file(
            self.kmc.config.control.initial_config
        )

    def initialize_neighbors_list(self) -> None:
        """Construct a new Neighbors List."""
        self.kmc.loggers.info("log", ":=> Constructing Neighbors Lists")
        self.kmc.neighbors_list = NeighborsList(
            self.kmc.system,
            self.kmc.config.atomicenvironment.rnei,
            self.kmc.config.atomicenvironment.rcut,
        )

    def initialize_atomic_environments(self) -> None:
        """Construct a new Atomic Environment."""
        self.kmc.loggers.info("log", ":=> Computing Atomic Environments")
        self.kmc.atomic_environment = AtomicEnvironment(
            self.kmc.config.atomicenvironment.style,
            self.kmc.neighbors_list.neighbors_list["rnei"],
            self.kmc.neighbors_list.neighbors_list["rcut"],
            self.kmc.config.atomicenvironment.neighbors_add,
        )

    def initialize_reference_table(self) -> None:
        """Initialize the Reference Event Table."""
        if self.kmc.config.control.reference_table is not None:
            self.kmc.loggers.info(
                "log",
                ":=> Reading Reference table file {}".format(
                    self.kmc.config.control.reference_table
                ),
            )
        else:
            self.kmc.loggers.info("log", ":=> Generate a empty reference table")
        self.kmc.reference_table = ReferenceEventTable(self.kmc.config)

    def initialize_bias(self) -> None:
        """Instantiate the bias object from the config, or set it to None."""
        bc = self.kmc.config.bias
        if bc is None or not self.kmc.config.control.bias:
            self.kmc.bias = None
            return
        match bc.style:
            case "direction":
                self.kmc.bias = DirectionBias(
                    bc.direction, bc.atom_indices, bc.threshold,
                    mode=bc.mode, bias_weight=bc.bias_weight, pass_unlisted=bc.pass_unlisted,
                )
            case "point":
                self.kmc.bias = PointBias(
                    bc.target_point, bc.atom_indices, bc.threshold,
                    mode=bc.mode, bias_weight=bc.bias_weight, pass_unlisted=bc.pass_unlisted,
                )
            case "topo":
                self.kmc.bias = TopoBias(
                    bc.topo_source, bc.topo_target,
                    mode=bc.mode, bias_weight=bc.bias_weight, pass_unlisted=bc.pass_unlisted,
                )

    def _initialize_visited_environments(self) -> None:
        """Initialize visited environment from file if specified, else initialize as {'crystal'}."""
        if self.kmc.config.control.visited_environments is not None:
            self.kmc.loggers.info(
                "log",
                ":=> Initiating visited environment from file {}".format(
                    self.kmc.config.control.visited_environments
                ),
            )
            try:
                with open(self.kmc.config.control.visited_environments, "rb") as file:
                    loaded_set_environments = pickle.load(file)
                self.kmc.visited_environments = loaded_set_environments
            except Exception as e:
                raise Exception("Can't read visited environment file.") from e
        else:
            self.kmc.visited_environments = set(["crystal"])
        if (
            self.kmc.config.control.visited_environments
            and not self.kmc.config.control.reference_table
        ):
            self.kmc.loggers.warning(
                "log",
                "Visited environments are read from file while no reference table was provided",
            )
