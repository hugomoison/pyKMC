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
        if rcut is not None : 
            self.neighbors_list = {"rnei": [], "rcut": []}
        else : 
            self.neighbors_list = {"rnei" : []}
        self._build_neighbors_list()

    def _build_neighbors_list(self) -> None:
        """Build and populates the `neighbors_list`."""
        positions = self.system.positions
        cell_diag = np.array([self.system.cell[0][0], self.system.cell[1][1], self.system.cell[2][2]])
        pbc = self.system.pbc if self.system.pbc is not None else np.array([True, True, True])

        if np.all(pbc):
            # Fully periodic: use boxsize (existing fast path)
            tree = cKDTree(positions, boxsize=cell_diag.tolist())
            for i in range(len(positions)):
                neighbors = tree.query_ball_point(positions[i], self.rnei)
                neighbors.remove(i)  # don't have self as neighbor
                self.neighbors_list["rnei"].append(neighbors)
                if self.rcut is not None:
                    neighbors = tree.query_ball_point(positions[i], self.rcut)
                    self.neighbors_list["rcut"].append(neighbors)
        else:
            # Mixed PBC: create ghost images in periodic directions
            shifts = [[-1, 0, 1] if pbc[d] else [0] for d in range(3)]
            all_positions = []
            index_map = []
            for sx in shifts[0]:
                for sy in shifts[1]:
                    for sz in shifts[2]:
                        shift_vec = np.array([sx * cell_diag[0], sy * cell_diag[1], sz * cell_diag[2]])
                        all_positions.append(positions + shift_vec)
                        index_map.extend(range(len(positions)))
            all_positions = np.vstack(all_positions)
            index_map = np.array(index_map)
            tree = cKDTree(all_positions)

            n_real = len(positions)
            for i in range(n_real):
                raw = tree.query_ball_point(positions[i], self.rnei)
                mapped = sorted(set(index_map[j] for j in raw) - {i})
                self.neighbors_list["rnei"].append(mapped)
                if self.rcut is not None:
                    raw = tree.query_ball_point(positions[i], self.rcut)
                    # Keep the central atom, matching the fully periodic branch:
                    # downstream event building locates the moving atom inside
                    # its own rcut environment.
                    mapped = sorted(set(index_map[j] for j in raw))
                    self.neighbors_list["rcut"].append(mapped)

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
