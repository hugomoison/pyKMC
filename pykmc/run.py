"""Entry point for running a Kinetic Monte Carlo (KMC) simulation.

This script parses command-line arguments to load the simulation configuration
from an input file, initializes the KMC simulation, and runs it.
"""

import argparse
from mpi4py import MPI
from .kmc import KMC
from pykmc.factory import EngineManagerFactory
from .config import Config
from .rate_constant import RateConstant


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
    comm = MPI.COMM_WORLD
    group_size = (
        (comm.Get_size() - 1)
        if config.control.group_size == -1
        else config.control.group_size
    )
    # KMC
    # Every rank builds the rate constant: the factory needs its engine extensions.
    rate_constant = RateConstant.create(
        config.rateconstant.style, config=config.rateconstant
    )
    factory = EngineManagerFactory(
        engine_style=config.control.engine,
        n_workers=config.control.n_sessions,
        comm=comm,
        engine_config=config.lammps,
        group_size=group_size,
        engine_extensions=list(rate_constant.engine_extensions),
    )
    manager = factory.launch()
    if manager is not None:  # On rank 0
        kmc = KMC(config, rate_constant=rate_constant)
        kmc.manager = manager
        try:
            kmc._initialize()
            kmc.run()
        except SystemExit:
            pass
        except BaseException:
            comm.Abort(1)
            raise


if __name__ == "__main__":
    main()
