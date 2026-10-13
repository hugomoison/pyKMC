"""Defines the System class for representing atomic systems.

It provides functionalities for updating positions and handling periodic
boundary conditions.
"""

from __future__ import annotations
from dataclasses import dataclass
from ase.io import read
import numpy as np
import ase.geometry


@dataclass
class Configuration:
    """A bare types/positions/cell snapshot of an atomic configuration.

    Unlike `System`, carries no `pbc`/`index` and no mutators -- just the
    three values that must always travel together wherever a piece of code
    needs a specific geometry (an engine call, a candidate ARTn result),
    rather than the live, owned system state `System` represents.

    Attributes
    ----------
    types : np.ndarray of str, shape (N)
        Atomic types (e.g., 'H', 'O', 'C') where N is the number of atoms.
    positions : np.ndarray of float, shape (N, 3)
        Atomic Cartesian coordinates.
    cell : np.ndarray of float, shape (3, 3)
        Simulation box cell.

    """

    types: np.ndarray | list
    positions: np.ndarray
    cell: np.ndarray

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (("types", self.types), ("positions", self.positions), ("cell", self.cell))
            if value is None
        ]
        if missing:
            raise ValueError(f"Configuration requires {', '.join(missing)} to be set (got None)")

    def copy(self) -> Configuration:
        """Return a new `Configuration` with each set field `.copy()`'d."""
        return Configuration(
            types=self.types.copy(),
            positions=self.positions.copy(),
            cell=self.cell.copy(),
        )

    def __add__(self, other: Configuration | np.ndarray) -> Configuration:
        """Add positions (an `other.positions`, or a raw displacement array); `types`/`cell` come from `self`."""
        other_positions = other.positions if isinstance(other, Configuration) else other
        return Configuration(positions=self.positions + other_positions, types=self.types, cell=self.cell)

    def __sub__(self, other: Configuration | np.ndarray) -> Configuration:
        """Subtract positions (an `other.positions`, or a raw array); `types`/`cell` come from `self`."""
        other_positions = other.positions if isinstance(other, Configuration) else other
        return Configuration(positions=self.positions - other_positions, types=self.types, cell=self.cell)

    def __mul__(self, scalar: float) -> Configuration:
        """Scale positions by `scalar`; `types`/`cell` come from `self`."""
        return Configuration(positions=self.positions * scalar, types=self.types, cell=self.cell)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Configuration:
        """Divide positions by `scalar`; `types`/`cell` come from `self`."""
        return Configuration(positions=self.positions / scalar, types=self.types, cell=self.cell)

    def __getitem__(self, indices) -> Configuration:
        """Return the sub-`Configuration` at `indices`; `cell` comes from `self`."""
        return Configuration(
            positions=self.positions[indices],
            types=np.asarray(self.types)[indices],
            cell=self.cell,
        )

    def __setitem__(self, indices, value: Configuration | np.ndarray) -> None:
        """Write positions (an `other.positions`, or a raw array) into `indices` in place; the inverse of `__getitem__`.

        Also writes `types` at `indices` when `value` is a `Configuration`.
        """
        if isinstance(value, Configuration):
            types = np.asarray(self.types)
            types[indices] = value.types
            self.types = types
            self.positions[indices] = value.positions
        else:
            self.positions[indices] = value

    def with_types(self, types) -> Configuration:
        """Return a copy of this `Configuration` with `types` replacing its own; `positions`/`cell` come from `self`."""
        return Configuration(positions=self.positions, types=types, cell=self.cell)

    def __len__(self) -> int:
        """Number of atoms."""
        return len(self.positions)


class System:
    """Represents an atomic system with its properties.

    This class provides a way to store and manage the fundamental
    characteristics of an atomic configuration, including atom types,
    spatial positions, simulation box dimensions, periodic boundary conditions,
    and original atom indices.

    Attributes
    ----------
    types : np.ndarray of str, shape (N),
        Atomic types (e.g., 'H', 'O', 'C') where N is the number of atoms.
        Defaults to None.
    positions : np.ndarray of float, shape (N, 3)
        Atomic Cartesian coordinates. Each row represents an atom's
        (x, y, z) position. Defaults to None.
    cell : np.ndarray of float, shape (3, 3)
        Simulation box cell. Defaults to None.
    pbc : np.ndarray of bool, shape (3)
        Flags for periodic boundary conditions (x, y, z). Defaults to None.
    index : np.ndarray of int, shape (N,)
        Original indices of the atoms. Defaults to None.

    """

    def __init__(
        self,
        types: np.ndarray,
        positions: np.ndarray | float,
        cell: np.ndarray,
        pbc: np.ndarray,
        index: np.ndarray | None = None,
    ) -> None:
        self._configuration = Configuration(types=types, positions=positions, cell=cell)
        self.pbc = pbc
        self.index = index

    @property
    def configuration(self) -> Configuration:
        """The `types`/`positions`/`cell` bundle backing this system."""
        return self._configuration

    @classmethod
    def from_configuration(
        cls,
        configuration: Configuration,
        pbc: np.ndarray | None = None,
        index: np.ndarray | None = None,
    ) -> System:
        """Build a `System` from a `Configuration` snapshot (e.g. another `System`'s, possibly `.copy()`'d).

        `index` defaults to `arange(len(configuration.types))` when not given.
        """
        return cls(
            types=configuration.types,
            positions=configuration.positions,
            cell=configuration.cell,
            pbc=pbc,
            index=index if index is not None else np.arange(len(configuration.types)),
        )

    @property
    def types(self):
        return self._configuration.types

    @types.setter
    def types(self, value) -> None:
        self._configuration.types = value

    @property
    def positions(self):
        return self._configuration.positions

    @positions.setter
    def positions(self, value) -> None:
        self._configuration.positions = value

    @property
    def cell(self):
        return self._configuration.cell

    @cell.setter
    def cell(self, value) -> None:
        self._configuration.cell = value

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
        new_system = cls(
            types=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
            index=np.arange(len(atoms)),
        )

        # Wrap positions
        new_system.update_positions(new_system.positions)

        return new_system

    def update_positions(
        self, new_positions: np.ndarray | Configuration, atom_idx: np.ndarray | None = None
    ) -> None:
        """Update the atomic positions of the system.

        This method allows updating either all atomic positions or a subset
        of them specified by their indices. After updating, positions are
        wrapped back into the simulation cell if PBC are enabled, and any
        small negative coordinates are clamped to zero.

        Parameters
        ----------
        new_positions : np.ndarray of float, shape (N,3), or Configuration
            A `Configuration` is equivalent to passing its `.positions`.
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
        if isinstance(new_positions, Configuration):
            new_positions = new_positions.positions

        if atom_idx is None:
            self.positions = new_positions
        else:
            self.positions[atom_idx] = new_positions

        self.positions = self.wrap_positions(self.positions, cell=self.cell, pbc=self.pbc)
        # Clamp small negative positions to zero to avoid issues with KD-trees.
        # This handles floating-point inaccuracies that might result in values like -1e-10.
        self.positions[self.positions < 0] = 0

    def __add__(self, other: Configuration | np.ndarray) -> System:
        """Update this System's positions in place to `self.configuration + other`; returns `self`."""
        self.update_positions(self.configuration + other)
        return self

    def __sub__(self, other: Configuration | np.ndarray) -> System:
        """Update this System's positions in place to `self.configuration - other`; returns `self`."""
        self.update_positions(self.configuration - other)
        return self

    def __mul__(self, scalar: float) -> System:
        """Update this System's positions in place to `self.configuration * scalar`; returns `self`."""
        self.update_positions(self.configuration * scalar)
        return self

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> System:
        """Update this System's positions in place to `self.configuration / scalar`; returns `self`."""
        self.update_positions(self.configuration / scalar)
        return self

    def __len__(self) -> int:
        """Number of atoms."""
        return len(self.positions)

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
