"""Lammps Engine interface module."""


__all__ = ["LammpsEngine"]
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from lammps import lammps
from ..config import Config
from .partn import pARTn_search, pARTn_refine_event
from ..system import System
from ..result import Result, EventSearchOutput, EventRefinementOutput, ErrorInfo

class LammpsEngine:
    """Backend class to interface with the LAMMPS atomistic simulation engine.

    This class encapsulates all LAMMPS-specific logic required for performing
    simulation tasks. It is intended to be used internally by the `Engine`.

    Liam Smyth (July 24th, 2025)
    Initializes 2 lammps instances.
    First is used as 'Big Lammps', runs minimization, energy calculations
    Second is 'Small Lammps', runs partn processes

    Parameters
    ----------
    config : Config
        Parameters of the simulations.

    """

    def __init__(self, config: Config) -> None:
        self.config = config

        # MPI :
        from mpi4py import MPI

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.lmp_mpi=lammps(comm=self.comm, cmdargs=["-screen", "none"])
        self.lmp=lammps(cmdargs=["-screen", "none"])

    def _initialize_default(self, system: System, lmp_instance: lammps) -> None:
        """Lammps initialization. Based on the system it initalize the simulation box, positions, pbc, masses.

        Parameters
        ----------
        system : System
            The atomic system.
        lmp_instance : lammps
            The lammps instance.

        """
        # parameters
        natoms = len(system.types)
        cell = system.cell
        types = system.types
        x = system.positions.flatten()  # Lammps format

        xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]

        ind = np.linspace(0, natoms - 1, natoms).astype(int)
        ind += 1  # Lammps id start at 1
        # map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
        map_type = {
            atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
            for i, atom_type in enumerate(sorted(set(types)))
        }
        types = [map_type[element]["ref"] for element in types]  # map to integer

        # lammps command
        lmp_instance.command("units metal")
        lmp_instance.command("atom_style atomic")
        lmp_instance.command("dimension 3")
        lmp_instance.command("boundary p p p")
        lmp_instance.command("atom_modify sort 0 0.0")
        lmp_instance.command(
            "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
        )
        lmp_instance.command("create_box {} box".format(len(map_type)))
        lmp_instance.create_atoms(natoms, ind, types, x)
        # Set masses
        for key in map_type.keys():
            lmp_instance.command(
                "mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"])
            )
        # Label atoms name to type :
        lmp_instance.command(
            "labelmap atom "
            + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
        )

    def _initialize_potential(self, lmp_instance: lammps) -> None:
        """Initialize the potential based on the config.

        Parameters
        ----------
        lmp_instance : lammps
            The lammps instance.

        """
        pair_style = self.config.lammps.pair_style
        pair_coeff = self.config.lammps.pair_coeff
        lmp_instance.command("pair_style {}".format(pair_style))
        lmp_instance.command("pair_coeff {}".format(pair_coeff))


    def minimize(self, system: System) -> tuple[np.ndarray, float]:
        """Minimize the system based on the config min_stayle and minimize command.

        Parameters
        ----------
        system : System
            The atomic system

        Returns
        -------
        tuple[np.ndarray, float] :
            A tuple containing :
            - New atomic positions after the minimization
            - Total energy of the minimized system.

        """

        # Lammps default parameters
        self._initialize_default(system, self.lmp_mpi)
        # Initialize potential
        self._initialize_potential(self.lmp_mpi)
        # Minimization
        self.lmp_mpi.command("min_style {}".format(self.config.lammps.min_style))   #Convert to using instance instead of initiating
        self.lmp_mpi.command("minimize {}".format(self.config.lammps.minimize))
        positions = self.lmp_mpi.gather_atoms("x", 1, 3)
        total_energy = self.lmp_mpi.get_thermo("etotal")

        self._reset(system, self.lmp_mpi)

        if self.rank == 0:
             # convert ctype positions into a numpy array
             positions = np.ctypeslib.as_array(positions)
             positions = np.reshape(positions, (-1, 3))
             return positions, total_energy
        else:
            return None

    def pARTn(
        self, system: System, central_atom: int
    ) -> Result[EventSearchOutput, ErrorInfo]:
        """Perform an event search around the central atom using pARTn.

        Parameters
        ----------
        system : System
            The atomic system.
        central_atom : int
            The central atom index.

        Returns
        -------
        Result[EventSearchOutput, ErrorInfo]
            The result of the event search.

        """
        # Parameters
        # Lammps default parameters :
        self._initialize_default(system, self.lmp)
        # Initialize potential
        self._initialize_potential(self.lmp)
        # pARTn search :
        result = pARTn_search(self.lmp, self.config, central_atom)

        self._reset(system, self.lmp)

        if result.is_ok():
            result.ok_value().cell = system.cell
        return result

    def pARTn_refine_event(
        self, system: System, central_atom: int
    ) -> Result[EventRefinementOutput, ErrorInfo]:
        """Perform an event refinement around the central atom using pARTn.

        Parameters
        ----------
        system : System
            The atomic system.
        central_atom : int
            The central atom index.

        Returns
        -------
        Result[EventRefinementOutput, ErrorInfo]
            The result of the event refinement.

        """
        self._initialize_default(system, self.lmp)
        self._initialize_potential(self.lmp)
        result = pARTn_refine_event(self.lmp, self.config, central_atom)
        self._reset(system, self.lmp)

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
            The potential energy of the system.

        """
        self._initialize_default(system, self.lmp_mpi)
        self._initialize_potential(self.lmp_mpi)

        self.lmp_mpi.command("compute c1 all pe")
        self.lmp_mpi.command("run 0")
        potential_energy = self.lmp_mpi.extract_compute("c1", 0, 0)

        self._reset(system, self.lmp_mpi)
        return potential_energy

    def _reset(self, system: System, lmp_instance: lammps) -> None:
        """Resets LAMMPS instance to original configuration

                Parameters
                ----------
                system : System
                    The atomic system.

                lmp_instance : lammps
                    The lammps instance.

                """
        lmp_instance.command("clear")

    def compute_distances(self, system: System) -> None:
        """Define a future operation to be implemented.

        Parameters
        ----------
        system : System
            The atomic system.

        """
        pass

    def neighbors(self, system: System) -> None:
        """Define a future operation to be implemented.

        Parameters
        ----------
        system : System
            The atomic system.

        """
        pass
    def _close(self) -> None:
        """ Closes all instances of lammps
        """
        self.lmp.close()
        self.lmp_mpi.finalize()