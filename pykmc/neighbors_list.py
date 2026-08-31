"""Manage atomic neighbor lists for an `System` using radial cutoffs."""

from scipy.spatial import cKDTree
from .system import System
from .parameters import Parameters
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
    rnei_pairs : dict[tuple[str, str], float] | None
        Optional per-species-pair override of `rnei`; pairs not listed fall back to `rnei`.
    atom_indices : list[int] | None
        Optional restriction: only these atoms get real neighbor lists computed
        (the rest are left as `[]`). Still indexed by global atom id, so
        `get_neighbors` behaves identically either way -- just cheaper to build
        when only a few atoms' local environments are actually needed.
    neighbors_list : dict[list[int]]
        Pre-calculated neighbor lists: `{'rnei': [...], 'rcut': [...]}`.

    """

    def __init__(
        self,
        system: System,
        rnei: float,
        rcut: float = None,
        rnei_pairs: dict[tuple[str, str], float] | None = None,
        atom_indices: list[int] | None = None,
    ) -> None:
        self.system = system
        self.rnei = rnei
        self.rcut = rcut
        self.rnei_pairs = rnei_pairs
        self.atom_indices = atom_indices
        natoms = len(system)
        if rcut is not None:
            self.neighbors_list = {"rnei": [[] for _ in range(natoms)], "rcut": [[] for _ in range(natoms)]}
        else:
            self.neighbors_list = {"rnei": [[] for _ in range(natoms)]}
        self._build_neighbors_list()

    def _pair_rnei(self, i: int, j: int) -> float:
        """Look up the rnei cutoff for the species pair of atoms `i` and `j`."""
        types = self.system.types
        return self.rnei_pairs.get(tuple(sorted((types[i], types[j]))), self.rnei)

    def _build_neighbors_list(self) -> None:
        """Build and populates the `neighbors_list`."""
        # Construct the kdTree
        positions = self.system.positions
        box = [self.system.cell[0][0], self.system.cell[1][1], self.system.cell[2][2]]
        tree = cKDTree(positions, boxsize=box)
        box_arr = np.array(box)
        query_rnei = (
            max(self.rnei, *self.rnei_pairs.values()) if self.rnei_pairs else self.rnei
        )

        # Find first neighbors and atoms in environments.
        # query_ball_point leaves single-point results unsorted, and its order
        # depends on the whole position array, so an unrelated atom moving can
        # reorder an untouched atom's list. Positions stored against a list are
        # applied back positionally, so the order has to be reproducible.
        indices = self.atom_indices if self.atom_indices is not None else range(len(positions))
        for i in indices:
            neighbors = sorted(tree.query_ball_point(positions[i], query_rnei))
            neighbors.remove(i)  # don't have self as neighbor
            if self.rnei_pairs:
                delta = positions[neighbors] - positions[i]
                delta -= box_arr * np.round(delta / box_arr)
                distances = np.linalg.norm(delta, axis=1)
                neighbors = [
                    j
                    for j, d in zip(neighbors, distances)
                    if d <= self._pair_rnei(i, j)
                ]
            self.neighbors_list["rnei"][i] = neighbors
            if self.rcut is not None:
                self.neighbors_list["rcut"][i] = sorted(
                    tree.query_ball_point(positions[i], self.rcut)
                )

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
