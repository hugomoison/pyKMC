#!/usr/bin/env python3
"""Short integration check; does not run a KMC simulation or write a LAMMPS log."""

import sys

import ase
import numpy
import mpi4py
from mpi4py import MPI
import pykmc
import ira_mod
import pypARTn
from lammps import lammps


def main():
    if not pykmc.__file__:
        raise RuntimeError("pykmc resolved to a namespace, not the installed package")
    print("Python:", sys.executable)
    print("Python version:", sys.version.split()[0])
    print("pyKMC:", pykmc.__file__)
    print("ASE:", ase.__version__, "NumPy:", numpy.__version__)
    print("mpi4py:", mpi4py.__version__)
    print("MPI:", MPI.Get_library_version().strip())
    print("IRA:", ira_mod.IRA().get_version())
    lmp = None
    artn = None
    try:
        lmp = lammps(cmdargs=["-log", "none"])
        if not lmp.has_style("pair", "eam/alloy"):
            raise RuntimeError("LAMMPS lacks pair_style eam/alloy; review the build packages")
        artn = pypARTn.artn(engine="lmp")
        lmp.command(f'plugin load "{artn.lib._name}"')
        print("LAMMPS library:", lmp.lib._name)
        print("pARTn library:", artn.lib._name)
    finally:
        if lmp is not None:
            lmp.close()
        if artn is not None:
            artn.destroy()
    print("PASS: imports, IRA, LAMMPS eam/alloy and pARTn plugin loading")


if __name__ == "__main__":
    main()
