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

        if self.kmc.params.control.restart_file is not None:
            self.kmc.loggers.info("log", ":=> Restarting")

    def initialize_loggers(self) -> None:
        """Initialize the loggers and create their files."""
        self.kmc.loggers = LogKMC(
            LOGGING_CONFIG, verbosity=self.kmc.params.control.verbosity
        )
        self.kmc.loggers.title("log")
        self.kmc.loggers.write_parameters("log", self.kmc.params)
        if self.kmc.params.control.restart_file is None:
            self.kmc.loggers.output_file_header("output")
            self.kmc.loggers.events_file_header("events")
            self.kmc.loggers.reference_table_file_header("reference_table")

    def initialize_system(self) -> None:
        """Read and initialize the system from the intial configuration file."""
        self.kmc.loggers.info(
            "log",
            ":=> Reading initial configuration file : {}".format(
                self.kmc.params.control.initial_config
            ),
        )
        self.kmc.system = System.create_from_file(
            self.kmc.params.control.initial_config
        )

    def initialize_neighbors_list(self) -> None:
        """Construct a new Neighbors List."""
        self.kmc.loggers.info("log", ":=> Constructing Neighbors Lists")
        self.kmc.neighbors_list = NeighborsList(
            self.kmc.system,
            self.kmc.params.atomicenvironment.rnei,
            self.kmc.params.atomicenvironment.rcut,
            self.kmc.params.atomicenvironment.rnei_pairs,
        )

    def initialize_atomic_environments(self) -> None:
        """Construct a new Atomic Environment."""
        self.kmc.loggers.info("log", ":=> Computing Atomic Environments")
        self.kmc.atomic_environment = AtomicEnvironment(
            self.kmc.params.atomicenvironment.style,
            self.kmc.neighbors_list.neighbors_list["rnei"],
            self.kmc.neighbors_list.neighbors_list["rcut"],
            self.kmc.params.atomicenvironment.neighbors_add,
            coordination_threshold=self.kmc.params.atomicenvironment.coordination_threshold,
            types=self.kmc.system.types,
            coloring_mode=self.kmc.params.atomicenvironment.coloring_mode,
        )

    def initialize_reference_table(self) -> None:
        """Initialize the Reference Event Table."""
        if self.kmc.params.control.reference_table is not None:
            self.kmc.loggers.info(
                "log",
                ":=> Reading Reference table file {}".format(
                    self.kmc.params.control.reference_table
                ),
            )
        else:
            self.kmc.loggers.info("log", ":=> Generate a empty reference table")
        self.kmc.reference_table = ReferenceEventTable(self.kmc.params)

    def initialize_bias(self) -> None:
        """Instantiate the bias object from the params, or set it to None."""
        bc = self.kmc.params.bias
        if bc is None or not self.kmc.params.control.bias:
            self.kmc.bias = None
            return
        match bc.style:
            case "direction":
                self.kmc.bias = DirectionBias(
                    bc.direction,
                    bc.atom_indices,
                    bc.threshold,
                    mode=bc.mode,
                    bias_weight=bc.bias_weight,
                    pass_unlisted=bc.pass_unlisted,
                    require_central=bc.require_central,
                    step_interval=bc.step_interval,
                    thr_boost=bc.thr_boost,
                )
            case "point":
                self.kmc.bias = PointBias(
                    bc.target_point,
                    bc.atom_indices,
                    bc.threshold,
                    mode=bc.mode,
                    bias_weight=bc.bias_weight,
                    pass_unlisted=bc.pass_unlisted,
                    require_central=bc.require_central,
                    thr_boost=bc.thr_boost,
                )
            case "topo":
                self.kmc.bias = TopoBias(
                    bc.atom_source_idx,
                    self.kmc.atomic_environment,
                    atom_target_idx=bc.atom_target_idx,
                    direction=bc.direction,
                    threshold=bc.threshold,
                    mode=bc.mode,
                    bias_weight=bc.bias_weight,
                    pass_unlisted=bc.pass_unlisted,
                    thr_boost=bc.thr_boost,
                )

    def _initialize_visited_environments(self) -> None:
        """Initialize visited environment from file if specified, else initialize as {'crystal'}."""
        if self.kmc.params.control.visited_environments is not None:
            self.kmc.loggers.info(
                "log",
                ":=> Initiating visited environment from file {}".format(
                    self.kmc.params.control.visited_environments
                ),
            )
            try:
                with open(self.kmc.params.control.visited_environments, "rb") as file:
                    loaded_set_environments = pickle.load(file)
                self.kmc.visited_environments = loaded_set_environments
            except Exception as e:
                raise Exception("Can't read visited environment file.") from e
        else:
            self.kmc.visited_environments = set(["crystal"])
        if (
            self.kmc.params.control.visited_environments
            and not self.kmc.params.control.reference_table
        ):
            self.kmc.loggers.warning(
                "log",
                "Visited environments are read from file while no reference table was provided",
            )
