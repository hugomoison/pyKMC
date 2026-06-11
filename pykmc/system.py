"""Defines the System class for representing atomic systems.

It provides functionalities for updating positions and handling periodic
boundary conditions.
"""

from __future__ import annotations
from ase.io import read
import numpy as np
import ase.geometry


class System:
    """Represents an atomic system with its properties.

    This class provides a way to store and manage the fundamental
    characteristics of an atomic configuration, including atom types,
    spatial positions, simulation box dimensions, periodic boundary conditions,
    and original atom indices.

    Attributes
    ----------
    types : np.ndarray of str, shape (N), optional
        Atomic types (e.g., 'H', 'O', 'C') where N is the number of atoms.
        Defaults to None.
    positions : np.ndarray of float, shape (N, 3), optional
        Atomic Cartesian coordinates. Each row represents an atom's
        (x, y, z) position. Defaults to None.
    cell : np.ndarray of float, shape (3, 3), optional
        Simulation box cell. Defaults to None.
    pbc : np.ndarray of bool, shape (3), optional
        Flags for periodic boundary conditions (x, y, z). Defaults to None.
    index : np.ndarray of int, shape (N,), optional
        Original indices of the atoms. Defaults to None.

    """

    def __init__(
        self,
        types: np.ndarray | None = None,
        positions: np.ndarray | float = None,
        cell: np.ndarray | None = None,
        pbc: np.ndarray | None = None,
        index: np.ndarray | None = None,
    ) -> None:
        self.types = types
        self.positions = positions
        self.cell = cell
        self.pbc = pbc
        self.index = index

    @classmethod
    def create_from_file(cls, file_path: str) -> System:
        """Create a System object from a structure file.

        This method reads an atomic configuration file (e.g., .xyz, .vasp, .xsf)
        using ASE, and initializes a new System instance with the corresponding
        atomic positions, types, cell, and periodic boundary conditions.

        Parameters
        ----------
        file_path : str
            Path to the input structure file.

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
        new_system = cls()
        # update attributes
        new_system.types = atoms.get_chemical_symbols()
        new_system.positions = atoms.get_positions()
        new_system.cell = atoms.get_cell()
        new_system.pbc = atoms.get_pbc()
        new_system.index = np.arange(len(new_system.types))

        #Wrap positions
        new_system.update_positions(new_system.positions)

        return new_system

    def update_positions(
        self, new_positions: np.ndarray, atom_idx: np.ndarray | None = None
    ) -> None:
        """Update the atomic positions of the system.

        This method allows updating either all atomic positions or a subset
        of them specified by their indices. After updating, positions are
        wrapped back into the simulation cell if PBC are enabled, and any
        small negative coordinates are clamped to zero.

        Parameters
        ----------
        new_positions : np.ndarray of float, shape (N,3)
            - If `atom_idx` is None, this array should have shape `(N, 3)`,
              where N is the total number of atoms in the system.
            - If `atom_idx` is provided, this array should have shape `(M, 3)`,
              where M is the number of atoms being updated (i.e., `len(atom_idx)`).
        atom_idx : np.ndarray of int, shape (M,3) optional
            A 1D NumPy array of integers specifying the indices of the atoms
            whose positions are to be updated. If `None` (default), all atoms'
            positions are updated.

        Notes
        -----
        - Positions are wrapped using `self.wrap_positions` based on `self.cell` and `self.pbc`.
        - Small negative position values are set to zero to prevent issues with
          spatial search algorithms (e.g., KD-trees) due to floating-point inaccuracies.

        """
        if atom_idx is None:
            self.positions = new_positions
            self.positions = self.wrap_positions(
                self.positions, cell=self.cell, pbc=self.pbc
            )
            # Clamp small negative positions to zero to avoid issues with KD-trees.
            # This handles floating-point inaccuracies that might result in values like -1e-10.
            # Only clamp in periodic dimensions; non-periodic z can have valid negative coords.
            self._clamp_negative_positions()

        else:
            self.positions[atom_idx] = new_positions
            self.positions = self.wrap_positions(
                self.positions, cell=self.cell, pbc=self.pbc
            )
            self._clamp_negative_positions()

    def _clamp_negative_positions(self) -> None:
        """Clamp small negative positions to zero in periodic dimensions only."""
        if self.pbc is not None and hasattr(self.pbc, '__iter__') and not np.all(self.pbc):
            for dim in range(3):
                if self.pbc[dim]:
                    self.positions[:, dim] = np.where(
                        self.positions[:, dim] < 0, 0, self.positions[:, dim]
                    )
        else:
            self.positions[self.positions < 0] = 0

    def remove_atom(self, atom_idx: int) -> None:
        """Remove an atom from the system.

        Deletes the atom at the given index from positions, types, and index arrays.

        Parameters
        ----------
        atom_idx : int
            Index of the atom to remove.

        Raises
        ------
        IndexError
            If atom_idx is out of range.

        """
        if atom_idx < 0 or atom_idx >= len(self.positions):
            raise IndexError(
                f"atom_idx {atom_idx} out of range for system with "
                f"{len(self.positions)} atoms"
            )
        self.positions = np.delete(self.positions, atom_idx, axis=0)
        if isinstance(self.types, np.ndarray):
            self.types = np.delete(self.types, atom_idx, axis=0)
        elif isinstance(self.types, list):
            del self.types[atom_idx]
        if self.index is not None:
            self.index = np.delete(self.index, atom_idx, axis=0)

    def wrap_positions(
        self, positions: np.ndarray, cell: np.ndarray, pbc: bool | np.ndarray = True
    ) -> np.ndarray:
        """Wrap atomic positions back into the primary unit cell.

        This method is a convenience wrapper for `ase.geometry.wrap_positions`.

        Parameters
        ----------
        positions : np.ndarray of float, shape (N, 3)
            Atomic coordinates to be wrapped.
        cell : np.ndarray of float, shape (3, 3)
            Simulation box
        pbc : bool or np.ndarray of bool, shape (3), optional
            Whether periodic boundary conditions are applied along each direction.
            Defaults to True (all directions).

        Returns
        -------
        np.ndarray of float, shape (N, 3)
            A new array with the wrapped positions.

        See Also
        --------
        ase.geometry.wrap_positions : Refer to ASE documentation for full details.

        """
        return ase.geometry.wrap_positions(positions=positions, cell=cell, pbc=pbc)
