"""Defines the `AtomicEnvironment` class for characterizing and computing local atomic environments."""

from __future__ import annotations

import numpy as np
from .environments import cna, coordination, graph, identify_diamond, region
from .config import RegionConfig


class AtomicEnvironment:
    """Computes and stores atomic environment ID based on a specified style.

    Attributes
    ----------
    style : str
        The atomic environment style (e.g., 'cna', 'graph', 'cna/graph').
    neighbors_list : list[list[int]]
       first neighbors lists
    environment_list : list[list[int]] or None
        Optional. lists of atoms in environments (used for 'graph' or 'cna/graph' styles).
    neighbors_add : int or None
        Optional. When `style` is 'cna/graph', specifies the N-th shell of neighbors whose graph IDs should also be computed.
    atomic_environment_list : list
        Computed atomic environment ID for each atom. **Populated during initialization**
        based on the chosen `style`.

    Raises
    ------
    Exception
        If the specified 'AtomicEnvironment' style in `config` is unknown.

    """

    def __init__(
        self,
        style: str,
        neighbors_list: list[list[int]] | None = None,
        environment_list: list[list[int]] | None = None,
        neighbors_add: int = 0,
        types: list[str] | None = None,
        coordination_threshold: int | None = None,
        region: RegionConfig | None = None,
        positions: np.ndarray | None = None,
        atom_types: list[str] | None = None,
    ) -> None:
        self.style = style
        self.neighbors_list = neighbors_list
        self.environment_list = environment_list
        self.neighbors_add = neighbors_add
        self.types = types
        self.coordination_threshold = coordination_threshold

        # Compute the atomic environment ID and store it in self.atomic_environment_list
        match self.style:
            case "cna":
                self.atomic_environment_list = self.compute_cna()
            case "coordination":
                self.atomic_environment_list = self.compute_coordination()
            case "graph":
                self.atomic_environment_list = self.compute_graph(
                    neighbors_list, environment_list
                )
            case "cna/graph":
                self.atomic_environment_list = self.compute_cnagraph(
                    neighbors_list, environment_list
                )
            case "coordination/graph":
                self.atomic_environment_list = self.compute_coordinationgraph(
                    neighbors_list, environment_list
                )
            case "diamond/graph" :
                self.atomic_environment_list = self.compute_diamondgraph(neighbors_list, environment_list)
            case "region":
                self.atomic_environment_list = self.compute_region(
                    region, positions, atom_types
                )
            case _:
                raise Exception("Atomic environment style unknown")



    def get_atoms_with_id(self, id: str) -> list[int] : 
        """Return list of atom indices whose environment matches the given ID.

        Parameters
        ----------
        id : str 
            The match ID.
        Returns
        -------
        list[int]
            List of atom indices
        """
        return [i for i, e in enumerate(self.atomic_environment_list) if e == id]
    
    def get_new_environments(self, visited_environments: set[str]) -> list[str] : 
        """ 
        Return list of atomic environment ID that are in the current self.environment_list but not in visited_environments
        """
        #return list([]) #Set if you want to only test refinements
        return list(set(self.atomic_environment_list).difference(visited_environments))

    def compute_region(
        self,
        r: RegionConfig | None,
        positions: np.ndarray | None,
        atom_types: list[str] | None,
    ) -> list[str]:
        """See :py:func:`.environments.region` for details."""
        return region(r, positions, atom_types)

    def compute_cna(self) -> list[str]:
        """See :py:func:`.environments.cna` for details on CNA computation."""
        return cna(self.neighbors_list)

    def compute_coordination(self) -> list[str]:
        """See :py:func:`.environments.coordination` for details on coordination classification."""
        return coordination(self.neighbors_list, self.coordination_threshold)

    def compute_graph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[str]:
        """See :py:func:`.environment.graph` for detail on Graph Topology computation."""
        return graph(neighbors_list, environment_list, types=self.types)

    def compute_cnagraph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[str]:
        """Compute CNA and then Graph Topology for all atoms that have a non cristalline environment.

        Parameters
        ----------
        neighbors_list : list[list[int]]
            first neighbors lists
        environment_list : list[list[int]]
            Optional. lists of atoms in environments (used for 'graph' or 'cna/graph' styles).

        Returns
        -------
        list[str]
            atomic environment ID for each atom

        """
        # Compute CNA ID
        list_hash = cna(neighbors_list)
        non_crystal_idx = (
            np.where(np.array(list_hash) == "noncrystal")[0].astype(int).tolist()
        )

        # If radd_cna != None add neighbors of non crystal from cna
        if self.neighbors_add > 0:
            tmp = []
            for _i in range(self.neighbors_add):  # Do it recursively
                for idx in non_crystal_idx:
                    tmp += neighbors_list[idx]
            non_crystal_idx += tmp
            non_crystal_idx = list(set(non_crystal_idx))
        # Compute graph topo for all non cristalline atoms
        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx, types=self.types)
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]

        return list_hash

    def compute_coordinationgraph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[str]:
        """Compute coordination-based classification, then Graph Topology for non-crystal atoms.

        Parameters
        ----------
        neighbors_list : list[list[int]]
            first neighbors lists
        environment_list : list[list[int]]
            lists of atoms in environments (used for graph computation).

        Returns
        -------
        list[str]
            atomic environment ID for each atom

        """
        # Compute coordination-based classification
        list_hash = coordination(neighbors_list, self.coordination_threshold)
        non_crystal_idx = (
            np.where(np.array(list_hash) == "noncrystal")[0].astype(int).tolist()
        )

        # If neighbors_add > 0, expand non-crystal set to include neighbors
        if self.neighbors_add > 0:
            tmp = []
            for _i in range(self.neighbors_add):
                for idx in non_crystal_idx:
                    tmp += neighbors_list[idx]
            non_crystal_idx += tmp
            non_crystal_idx = list(set(non_crystal_idx))

        # Compute graph topo for all non-crystal atoms
        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx, types=self.types)
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]

        return list_hash
    
    def compute_diamondgraph(self, neighbors_list, environment_list) : 
        #Compute identify diamant ID 
        list_hash = identify_diamond(neighbors_list)
        non_crystal_idx = (
            np.where(np.array(list_hash) == "noncrystal")[0].astype(int).tolist()
        )

        # If radd_cna != None add neighbors of non crystal from cna
        if self.neighbors_add > 0:
            tmp = []
            for _i in range(self.neighbors_add):  # Do it recursively
                for idx in non_crystal_idx:
                    tmp += neighbors_list[idx]
            non_crystal_idx += tmp
            non_crystal_idx = list(set(non_crystal_idx))
        # Compute graph topo for all non cristalline atoms
        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx)
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]
        
        return list_hash