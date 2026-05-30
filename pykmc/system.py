"""Defines the System class for representing atomic systems.

It provides functionalities for updating positions and handling periodic
boundary conditions.
"""

from __future__ import annotations
from ase.io import read
from ase.cell import Cell
import numpy as np
import ase.geometry
import logging

class System:
    """Represents an atomic system with its properties.

    This class provides a way to store and manage the fundamental
    characteristics of an atomic configuration, including atom types,
    spatial positions, simulation box dimensions, periodic boundary conditions,
    and original atom indices.
    It follows ASE conventions: positions always (N,3), pbc always (3,), 
    cell always ase.cell.Cell. 
    Dimentionality (1D, 2D, 3D) is expressed via pbc, not by reducing 
    array dimensions.

    Attributes
    ----------
    types : np.ndarray of str, shape (N,), optional
        Atomic types (e.g., 'H', 'O', 'C') where N is the number of atoms.
        Defaults to None.
    positions : np.ndarray of float, shape (N, 3), optional
        Atomic Cartesian coordinates. Each row represents an atom's
        (x, y, z) position. Defaults to None.
    cell : ase.cell.Cell
        Simulation box cell. Defaults to None.
    pbc : np.ndarray of bool, shape (3,), optional
        Flags for periodic boundary conditions (x, y, z). Defaults to None.
    index : np.ndarray of int, shape (N,), optional
        Original indices of the atoms. Defaults to None.
    logger : logging.Logger 
            Logger to log informations. Default None


    """

    def __init__(self, types: np.ndarray | None = None, positions: np.ndarray | None = None, cell: np.ndarray | None = None
                 , pbc: np.ndarray | None = None, index: np.ndarray | None = None, logger: logging.Logger | None = None) -> None:

        self.types = types
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self.index = index
        self._logger = logger

    #------------#
    # Properties #
    #------------#
    #To force ASE convention

    @property 
    def cell(self) -> Cell | None : 
        return self._cell

    @cell.setter 
    def cell(self, value: Cell | np.ndarray | None) -> None : 
        self._cell = Cell.new(value) if value is not None else None

    @property 
    def pbc(self) -> np.ndarray | None : 
        return self._pbc 
    
    @pbc.setter 
    def pbc(self, value: np.ndarray | bool | None) -> None : 
        if value is None : 
            self._pbc = None 
        else : 
            self._pbc = np.broadcast_to(np.asarray(value, dtype=bool), (3,)).copy()
    
    @property 
    def n_atoms(self) -> int | None : 
        """ 
        Convenience method
        """
        return None if self.positions is None else len(self.positions)

    #--------------#
    # Class method #
    #--------------#

    @classmethod
    def create_from_file(cls, file_path: str, logger: logging.Logger | None = None) -> System:
        """Create a System object from a structure file.

        This method reads an atomic configuration file (e.g., .xyz, .vasp, .xsf)
        using ASE, and initializes a new System instance with the corresponding
        atomic positions, types, cell, and periodic boundary conditions.

        Parameters
        ----------
        file_path : str
            Path to the input structure file.
        logger : logging.Logger 
            Logger to log informations. Default None

        Returns
        -------
        System
            A new instance of System populated from the file data.

        Raises
        ------
        ValueError
            If the file cannot be read or parsed into an ASE Atoms object.

        """
        # Create ase.Atoms from file
        try:
            atoms = read(file_path, parallel=False, index=-1)
        except Exception as e:
            raise ValueError(f"Can't create System from file {file_path}: {e}") from e

        # Create new System instance
        new_system = cls(
            types = atoms.get_chemical_symbols(), 
            positions = atoms.get_positions(), 
            cell = atoms.get_cell(),
            pbc = atoms.get_pbc(),
            index = np.arange(len(atoms)), 
            logger=logger
        )

        #Log warning if wrong initialization indicating file with uncomplete informations.
        if new_system._logger is not None : 
            if new_system.cell is None or not np.any(new_system.cell) : 
                new_system._logger.warning("System loaded without a valid cell - simulation will not work properly.")
            if new_system.pbc is None or not np.any(new_system.pbc) : 
                new_system._logger.warning("All pbc are False - no periodic boundary conditions set.")

        #Wrap positions
        new_system.update_positions(new_system.positions)

        return new_system

    #---------#
    # Methods #
    #---------#

    def update_positions(
        self, new_positions: np.ndarray, atom_idx: np.ndarray | None = None, clamp_negative: bool = True) -> None:
        """Update the atomic positions of the system.

        This method allows updating either all atomic positions or a subset
        of them specified by their indices. After updating, positions are
        wrapped back into the simulation cell if PBC are enabled
        , and any small negative coordinates are clamped to zero.

        Parameters
        ----------
        new_positions : np.ndarray of float, shape (N,3) or (M, 3)
            - If `atom_idx` is None, this array should have shape `(N, 3)`,
              where N is the total number of atoms in the system.
            - If `atom_idx` is provided, this array should have shape `(M, 3)`,
              where M is the number of atoms being updated (i.e., `len(atom_idx)`).
        atom_idx : np.ndarray of int, shape (M,3) optional
            A 1D NumPy array of integers specifying the indices of the atoms
            whose positions are to be updated. If `None` (default), all atoms'
            positions are updated.
        clam_negative: bool 
            Clamp negative values, after wrapping, to 0. Default True.

        Notes
        -----
        - Positions are wrapped using `self.wrap_positions` based on `self.cell` and `self.pbc`.
        - Small negative position values are set to zero to prevent issues with
          spatial search algorithms (e.g., KD-trees) due to floating-point inaccuracies.

        """

        if atom_idx is None :
            self.positions = new_positions
        
        else : 
            self.positions[atom_idx] = new_positions

        #Always wrap positions : 
        if self.cell is not None and self.pbc is not None : 
            self.positions = ase.geometry.wrap_positions(positions=self.positions, cell=self.cell, pbc=self.pbc)
        
        if clamp_negative : 
        #Clamp small negative values to zero to avoid issues with KD-Tree 
        #This handles floating-point inacurracies that might result in value like -1e-10 (e.g. when using Lammps)
            self.positions[self.positions < 0] = 0

    #--------#
    # Dunder #
    #--------#

    def __len__(self) -> int: 
        return self.n_atoms if self.n_atoms is not None else 0

    def __repr__(self) -> str : 
        return ( 
            f"System(n_atoms={self.n_atoms}, pbc={self.pbc}, cell={np.array(self.cell) if self.cell is not None else None})"
        )


