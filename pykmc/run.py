"""Entry point for running a Kinetic Monte Carlo (KMC) simulation.

This script parses command-line arguments to load the simulation configuration
from an input file, initializes the KMC simulation, and runs it.
"""

import argparse
from .kmc import KMC
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from .config import Config


def main() -> None:
    """Parse input arguments and launch the KMC simulation.

    The function reads a configuration file specified by the user,
    creates a `KMC` instance, and runs the simulation.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input", type=str, required=True, help="inputs file")
    args = parser.parse_args()

    # Config
    config = Config.from_ini_file(args.input)
    # KMC
    factory = ManagerFactory(
        n_sessions=config.control.n_sessions,
        use_rank_0=config.control.engine_use_rank_0,
        has_global=True,
    )
    manager = factory.launch()
    if manager is not None:  # On rank 0
        kmc = KMC(config)
        kmc._initialize()
        kmc.manager = manager
        #        kmc = KMC(config)
        try:
            kmc.run()
        except Exception:
            manager.close_all()
            raise


if __name__ == "__main__":
    main()
