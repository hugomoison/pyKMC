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

    Parameters
    ----------
    config : Config
        Parameters of the simulations.

    """

    def __init__(self, config: Config) -> None:
        self.config = config

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
        from mpi4py import MPI

        # MPI :
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # nprocs = comm.Get_size()
        lmp = lammps(comm=comm, cmdargs=["-screen", "none"])

        # Lammps default parameters
        self._initialize_default(system, lmp)
        # Initialize potential
        self._initialize_potential(lmp)
        # Minimization
        # lmp.command('min_style {}'.format(min_style))
        # lmp.command('minimize {} {} {} {}'.format(etol, ftol, maxiter, maxeval))
        lmp.command("min_style {}".format(self.config.lammps.min_style))
        lmp.command("minimize {}".format(self.config.lammps.minimize))
        # gather all positions
        # id = lmp.numpy.extract_atom("id")
        positions = lmp.gather_atoms("x", 1, 3)
        total_energy = lmp.get_thermo("etotal")
        if rank == 0:
            # convert ctype positions into a numpy array
            positions = np.ctypeslib.as_array(positions)
            positions = np.reshape(positions, (-1, 3))
            lmp.close()
            return positions, total_energy
        else:
            lmp.close()
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

        lmp = lammps(cmdargs=["-screen", "none"])
        # Lammps default parameters :
        self._initialize_default(system, lmp)
        # Initialize potential
        self._initialize_potential(lmp)
        # pARTn search :
        result = pARTn_search(lmp, self.config, central_atom)
        if result.is_ok():
            result.ok_value().cell = system.cell
        lmp.close()
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
        lmp = lammps(cmdargs=["-screen", "none"])
        self._initialize_default(system, lmp)
        self._initialize_potential(lmp)
        result = pARTn_refine_event(lmp, self.config, central_atom)
        lmp.close()
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
        lmp = lammps(cmdargs=["-screen", "none"])
        self._initialize_default(system, lmp)
        self._initialize_potential(lmp)

        lmp.command("compute c1 all pe")
        lmp.command("run 0")
        potential_energy = lmp.extract_compute("c1", 0, 0)
        lmp.close()
        return potential_energy

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
