"""Manage atomic neighbor lists for an `System` using radial cutoffs."""

from __future__ import annotations
import numpy as np
from .backend import NeighborsBackend

class NeighborsList:
    """Store and manage neighbor lists for atoms in a system.

    Builds neighbor lists and environment lists based on two cutoff radii (`rnei` and `rcut`)

    Attributes
    ----------
    backend : NeighborsBackend 
        Backend used to compute neighbors lists.
    neighbors_list : dict[list[int]]
        Pre-calculated neighbor lists: `{'rnei': [...], 'rcut': [...]}`.

    """

    def __init__(self, backend: NeighborsBackend) -> None:
        self.backend = backend
        self.neighbors_list = self.backend.build()


    @property
    def r_neighbors_cutoff(self) -> float:
        return self.backend.r_neighbors_cutoff

    @property
    def r_env_cutoff(self) -> float:
        return self.backend.r_env_cutoff

    

    def get_neighbors(self, cutoff_type: str, idx: int) -> list[int]:
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
    def __repr__(self) -> str:
        return (
        f"NeighborsList(n_atoms={len(self.neighbors_list['neighbors'])}, "
        f"r_neighbors_cutoff={self.r_neighbors_cutoff}, "
        f"r_env_cutoff={self.r_env_cutoff}, "
        f"backend={type(self.backend).__name__},"
        f"backend_properties={self.backend})"
    )