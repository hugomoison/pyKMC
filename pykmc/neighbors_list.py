"""Manage atomic neighbor lists for an `System` using radial cutoffs."""

from scipy.spatial import cKDTree
from .system import System
from .config import Config
import numpy as np


class NeighborsList:
    """Store and manage neighbor lists for atoms in a system.

    Builds neighbor lists and environment lists based on two cutoff radii (`rnei` and `rcut`)

    Attributes
    ----------
    system : System
        The atomic system.
    rnei : float
        First neighbor radial cutoff distance.
    rcut : float
        Environment radial cutoff distance.
    neighbors_list : dict[list[int]]
        Pre-calculated neighbor lists: `{'rnei': [...], 'rcut': [...]}`.

    """

    def __init__(self, system: System, rnei: float, rcut: float = None) -> None:
        self.system = system
        self.rnei = rnei
        self.rcut = rcut
        if rcut is not None:
            self.neighbors_list = {"rnei": [], "rcut": []}
        else:
            self.neighbors_list = {"rnei": []}
        self._build_neighbors_list()

    def _build_neighbors_list(self) -> None:
        """Build and populates the `neighbors_list`."""
        # Construct the kdTree
        positions = self.system.positions
        box = [self.system.cell[0][0], self.system.cell[1][1], self.system.cell[2][2]]
        tree = cKDTree(positions, boxsize=box)

        # Find first neighbors and atoms in environments
        for i in range(len(positions)):
            neighbors = tree.query_ball_point(positions[i], self.rnei)
            neighbors.remove(i)  # don't have self as neighbor
            self.neighbors_list["rnei"].append(neighbors)
            if self.rcut is not None:
                neighbors = tree.query_ball_point(positions[i], self.rcut)
                self.neighbors_list["rcut"].append(neighbors)

    def get_neighbors(self, cutoff_type: float, idx: int) -> list[int]:
        """Retrieve the neighbor list for a specific atom and cutoff.

        Parameters
        ----------
        cutoff_type : str
            The cutoff type ('rnei' or 'rcut').
        idx : int
            The index of the atom.

        Returns
        -------
        list of int
            Indices of neighboring atoms.

        """
        return self.neighbors_list[cutoff_type][idx]

    def update_neighbors(self, list_atoms: np.ndarray) -> None:
        """Update placeholder for future implementation.

        Parameters
        ----------
        list_atoms : np.ndarray
            list of atoms

        """
        pass
