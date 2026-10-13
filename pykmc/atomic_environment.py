"""Defines the `AtomicEnvironment` class for characterizing and computing local atomic environments."""

from __future__ import annotations

import numpy as np
from .environments import cna, coordination, graph, identify_diamond, region
from .parameters import Parameters, RegionParameters
from .system import System, Configuration
from .neighbors_list import NeighborsList


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
        If the specified 'AtomicEnvironment' style in `params` is unknown.

    """

    def __init__(
        self,
        style: str,
        neighbors_list: list[list[int]] | None = None,
        environment_list: list[list[int]] | None = None,
        neighbors_add: int = 0,
        types: list[str] | None = None,
        coloring_mode: str = "full",
        region: RegionParameters | None = None,
        configuration: Configuration | None = None,
        coordination_threshold: int | None = None,
    ) -> None:
        self.style = style
        self.neighbors_list = neighbors_list
        self.environment_list = environment_list
        self.neighbors_add = neighbors_add
        self.coordination_threshold = coordination_threshold
        self.types = types
        self.coloring_mode = coloring_mode

        # Compute the atomic environment ID and store it in self.atomic_environment_list
        match self.style:
            case "cna":
                self.atomic_environment_list = self.compute_cna()
            case "graph":
                self.atomic_environment_list = self.compute_graph(
                    neighbors_list, environment_list
                )
            case "cna/graph":
                self.atomic_environment_list = self.compute_cnagraph(
                    neighbors_list, environment_list
                )
            case "coordination":
                self.atomic_environment_list = self.compute_coordination()
            case "coordination/graph":
                self.atomic_environment_list = self.compute_coordinationgraph(
                    neighbors_list, environment_list
                )
            case "diamond/graph":
                self.atomic_environment_list = self.compute_diamondgraph(
                    neighbors_list, environment_list
                )
            case "region":
                self.atomic_environment_list = self.compute_region(region, configuration)
            case _:
                raise Exception("Atomic environment style unknown")

    def get_atoms_with_id(self, id: str) -> list[int]:
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

    def compute_region(
        self,
        r: RegionParameters | None,
        configuration: Configuration | None,
    ) -> list[str]:
        """See :py:func:`.environments.region` for details."""
        return region(r, configuration)

    def compute_cna(self) -> list[str]:
        """See :py:func:`.environments.cna` for details on CNA computation."""
        return cna(self.neighbors_list)

    def compute_graph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[str]:
        """See :py:func:`.environment.graph` for detail on Graph Topology computation."""
        return graph(
            neighbors_list,
            environment_list,
            types=self.types if self.coloring_mode == "full" else None,
        )

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
        list_graphs_hash = graph(
            neighbors_list,
            environment_list,
            non_crystal_idx,
            types=self.types if self.coloring_mode == "full" else None,
        )
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]

        return list_hash

    def compute_coordination(self) -> list[str]:
        """See :py:func:`.environments.coordination` for the coordination-number classifier."""
        # The params validator guarantees a threshold for coordination styles.
        assert self.coordination_threshold is not None, (
            "coordination_threshold must be set"
        )
        return coordination(self.neighbors_list, self.coordination_threshold)

    def compute_coordinationgraph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[str]:
        """Classify by coordination, then compute Graph Topology IDs for the non-crystal atoms.

        Parameters
        ----------
        neighbors_list : list[list[int]]
            first neighbors lists
        environment_list : list[list[int]]
            lists of atoms in environments (used for the graph computation)

        Returns
        -------
        list[str]
            atomic environment ID for each atom

        """
        # Coordination-number classification (validator guarantees a threshold for these styles)
        assert self.coordination_threshold is not None, (
            "coordination_threshold must be set"
        )
        list_hash = coordination(neighbors_list, self.coordination_threshold)
        non_crystal_idx = (
            np.where(np.array(list_hash) == "noncrystal")[0].astype(int).tolist()
        )

        # Optionally extend to the N-th neighbour shell of each non-crystal atom
        if self.neighbors_add > 0:
            tmp = []
            for _i in range(self.neighbors_add):  # Do it recursively
                for idx in non_crystal_idx:
                    tmp += neighbors_list[idx]
            non_crystal_idx += tmp
            non_crystal_idx = list(set(non_crystal_idx))

        # Compute graph topology only for the non-crystalline atoms (uncolored graph())
        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx)
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]

        return list_hash

    def compute_diamondgraph(self, neighbors_list, environment_list):
        # Compute identify diamant ID
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
        list_graphs_hash = graph(
            neighbors_list,
            environment_list,
            non_crystal_idx,
            types=self.types if self.coloring_mode == "full" else None,
        )
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]

        return list_hash


def compute_atomic_environment_id(
    configuration: Configuration,
    atom_idx: int,
    params: Parameters,
) -> tuple[str, NeighborsList]:
    """Compute the coarse graph-hash id of `atom_idx`'s rcut neighborhood in `configuration`.

    Builds a throwaway `System` from `configuration` -- which need not be the
    live system (e.g. an ARTn-returned minimum) -- and runs the same
    graph-hash recipe used everywhere else in the catalogue, so a candidate
    geometry's local shape can be hashed the same way a live atom's or a
    catalogued row's is. Always uses the graph/nauty algorithm, regardless of
    `params.atomicenvironment.style` -- catalogued shapes must stay
    comparable to each other independent of how the live system is being
    classified.

    Only `atom_idx`'s own rcut ball needs real neighbor lists (the graph
    hash only ever looks at that ball and its members' rnei lists), so this
    builds the `NeighborsList` in two small passes -- one atom, then just
    that atom's ball -- instead of the whole system.

    Parameters
    ----------
    configuration : Configuration
        Full-system types/positions/cell to build the neighborhood from.
    atom_idx : int
        Index of the atom whose neighborhood to hash.
    params : Parameters
        Needs `.atomicenvironment.rnei`/`.rcut`/`.rnei_pairs`/`.coloring_mode`.

    Returns
    -------
    tuple[str, NeighborsList]
        The graph-hash id, and the `NeighborsList` used to compute it (most
        callers also need `neighbors_list.get_neighbors("rcut", atom_idx)` to
        slice out the corresponding local cluster).

    """
    system = System.from_configuration(configuration)
    rnei = params.atomicenvironment.rnei
    rcut = params.atomicenvironment.rcut
    rnei_pairs = params.atomicenvironment.rnei_pairs
    probe = NeighborsList(system, rnei, rcut, rnei_pairs, atom_indices=[atom_idx])
    ball = probe.get_neighbors("rcut", atom_idx)
    neighbors_list = NeighborsList(system, rnei, rcut, rnei_pairs, atom_indices=ball)
    graph_types = (
        configuration.types if params.atomicenvironment.coloring_mode == "full" else None
    )
    id_ = graph(
        neighbors_list.neighbors_list["rnei"],
        neighbors_list.neighbors_list["rcut"],
        atom_idx=[atom_idx],
        types=graph_types,
    )[0]
    return id_, neighbors_list
