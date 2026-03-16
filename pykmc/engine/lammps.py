from lammps import lammps
import numpy as np
import ctypes
from typing import Protocol 
from .base import Engine
from ase import Cell
from ase.data import atomic_masses, atomic_numbers
try : 
    from mpi4py import MPI
except ImportError : 
    pass
try:
    from pykmc.manager.registry import registrable
except ImportError:
    def registrable(name: str | None = None):
        def decorator(func):
            return func
        return decorator

class LammpsConfigProtocol(Protocol) : 
    """
    Protocol defining the configuration interface for LammpsEngine.

    Attributes
    ----------
    pair_style : str
        LAMMPS pair_style command string (e.g. "eam/alloy").
    pair_coeff : str
        LAMMPS pair_coeff command string (e.g. "* * potential.eam Ni").
    min_style : str
        Minimization algorithm (e.g. "cg", "fire").
    minimize : str
        Minimization convergence parameters (e.g. "1e-6 1e-8 1000 10000").
    verbosity : int
        Log verbosity. 0 disables log file, any other value enables it.
    """
    pair_style: str 
    pair_coeff: str 
    min_style: str 
    minimize: str
    verbosity: int

class LammpsEngine(Engine) : 
    """ 
    LAMMPS engine wrapper.

    This class is designed to be used in a master-worker MPI pattern.
    All LAMMPS commands are executed collectively across all ranks,
    but data extraction methods (get_positions, get_total_energy, etc.)
    only return values on rank 0 — other ranks return None.

    Parameters
    ----------
    config : LammpsConfigProtocol
        Configuration object containing simulation parameters:
        pair_style, pair_coeff, min_style, minimize, verbosity.
    comm : MPI.COMM, optional
        MPI communicator for parallel execution. If None, runs in serial.
    engine_id : int, optional
        Unique identifier for this engine instance, used for log file naming.
        Default is 0.

    Examples
    --------
    Serial usage:
        engine = LammpsEngine(config=my_config)
        engine.start()
        engine.initialize_parameters()
        engine.initialize_system(types, positions, cell, pbc)
        engine.initialize_potential()

    MPI usage:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        engine = LammpsEngine(config=my_config, comm=comm, engine_id=0)
        engine.start()
        ... 
        pe = engine.get_total_energy()
        # ... only rank 0 receives data from extraction methods
    """

    def __init__(self, config: LammpsConfigProtocol, comm: "MPI.COMM"|None = None, engine_id: int = 0) -> None : 
        self.config = config
        self.comm = comm
        self.engine_id = engine_id

    #Convenience
    @property
    def _is_rank0(self) -> bool:
        return self.comm is None or self.comm.Get_rank() == 0

    @registrable("start")
    def start(self) -> None : 
        engine_log = "none" if self.config.verbosity == 0 else f"lammps.log.{self.engine_id}"
        self.lmp = lammps(comm=self.comm, cmdargs = ['screen', 'none', '-log', engine_log])

    @registrable("close")
    def close(self) -> None : 
        self.lmp.close()

    @registrable("initialize_parameters")
    def initialize_parameters(self) -> None : 
        self.lmp.command("units metal")
        self.lmp.command("atom_style atomic")
        self.lmp.command("dimension 3")
        self.lmp.command("atom_modify map array") #! necessary for scatter atoms
        self.lmp.command("atom_modify sort 0 0.0") #! necessary for partn

    @registrable("initialize_system")
    def initialize_system(self, types: list[str]|np.ndarray[str], positions: np.ndarray, cell: Cell , pbc: list[bool]|np.ndarray[bool]) -> None : 

        #system parameters 
        natoms = len(types)
        x = positions.flatten() #Lammps format 

            #cell 
        xhi = cell[0, 0]
        yhi = cell[1, 1]
        zhi = cell[2, 2]
            # non diagonal terms
        xy = cell[1, 0]
        xz = cell[2, 0]
        yz = cell[2, 1]

        # boundary 
        boundary = " ".join("p" if p else "f" for p in pbc)
        self.lmp.command(f"boundary {boundary}")

        

        ind = np.arange(1, natoms+1) #Lammps ids start at 1
        # map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
        map_type = {
            atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
            for i, atom_type in enumerate(sorted(set(types)))
        }
        int_types = [map_type[element]["ref"] for element in types]  # map to integer

        # lammps create system
        #ortho 
        if np.allclose([xy, xz, yz], 0):
            self.lmp.command(f"region box block 0.0 {xhi} 0.0 {yhi} 0.0 {zhi}")
        #triclinic
        else:
            self.lmp.command(
                f"region box prism 0.0 {xhi} 0.0 {yhi} 0.0 {zhi} {xy} {xz} {yz}"
            )
        self.lmp.command("create_box {} box".format(len(map_type)))
        self.lmp.create_atoms(natoms, ind, int_types, x)
        # Set masses
        for key in map_type.keys():
            self.lmp.command(
                "mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"])
            )
        # Label atoms name to type :
        self.lmp.command(
            "labelmap atom "
            + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
        )

    @registrable("initialize_potential")
    def initialize_potential(self, rcut: float = None) -> None :
        pair_style = self.config.pair_style 
        pair_coeff = self.config.pair_coeff
        
        if rcut is None : 
            self.lmp.command("pair_style {}".format(pair_style))
            self.lmp.command("pair_coeff {}".format(pair_coeff))
        else : 
            self.lmp.command("pair_style hybrid/overlay {} zero {} full".format(pair_style, rcut))
            self.lmp.command("pair_coeff {}".format(pair_coeff))
            self.lmp.command("pair_coeff * * zero")

    @registrable("get_positions")
    def get_positions(self) -> np.ndarray | None : 
        result = self.lmp.gather_atoms("x", 1, 3)
        if self._is_rank0 :
            # convert ctype positions into a numpy array
            result = np.ctypeslib.as_array(result)
            result = np.reshape(result, (-1, 3))
            return result
        else : 
            return None

    @registrable("set_positions") 
    def set_positions(self, positions: np.ndarray) -> None : 
        positions = positions.flatten().astype(np.float64)
        positions = np.ascontiguousarray(positions)
        c_array = (ctypes.c_double * len(positions))(*positions)
        self.lmp.scatter_atoms("x", 1, 3, c_array)

    @registrable("get_total_energy")
    def get_total_energy(self, positions: np.ndarray = None, recompute:bool = True) -> float|None : 
        if positions is not None :
            self.set_positions(positions=positions)
        #Get total energy
        if recompute : 
            self.lmp.command("run 0 post no")
        result = self.lmp.get_thermo("etotal")
        if self._is_rank0 :
            return result
        else : 
            return None

    @registrable("get_potential_energy")
    def get_potential_energy(self, positions: np.ndarray = None, recompute: bool = True) -> float|None :
        if positions is not None :
            self.set_positions(positions=positions)

        # Check if compute exists (rank 0 only)
        define_compute = False
        if self._is_rank0:
            if self.lmp.extract_compute("c_pe", 0, 0) is None:
                define_compute = True

        # Broadcast decision to all ranks
        if self.comm is not None:
            define_compute = self.comm.bcast(define_compute, root=0)

        if define_compute:
            self.lmp.command("compute c_pe all pe")

        # Always run to get up-to-date value
        if recompute :
            self.lmp.command("run 0 post no")
        result = self.lmp.extract_compute("c_pe", 0, 0)

        if self._is_rank0:
            return result
        return None

        
    @registrable("minimize") 
    def minimize(self, positions: np.ndarray = None) -> None : 
        if positions is not None : 
            self.set_positions(positions=positions)
        self.lmp.command("min_style {}".format(self.config.min_style))
        self.lmp.command("minimize {}".format(self.config.minimize))

    @registrable("minimize_with_results")
    def minimize_with_results(self, positions=None) -> tuple[np.ndarray, float] | None:
        self.minimize(positions=positions) 
        new_positions = self.get_positions()
        total_energy = self.get_total_energy(recompute=False)
        if self._is_rank0 : 
            return new_positions, total_energy
        else : 
            return None
