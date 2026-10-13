"""Tools to compute graphs and graph's ID using NAUTY."""

import hashlib

import pynauty
import numpy as np


def encode_cert(value: bytes | str) -> str:
    """Return a stable string ID for a graph certificate or existing ID."""
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    return value


def combine_ids(id1: str, id2: str, id3: str) -> str:
    """Return a single ID derived from three graph IDs."""
    return hashlib.sha256((id1 + id2 + id3).encode()).hexdigest()


def graph(
    neighbors_list: list[list[int]],
    environment_list: list[list[int]],
    atom_idx: list[int] = None,
    types: list[str] = None,
) -> list[str]:
    """Compute atoms's graph ID.

    Parameters
    ----------
    neighbors_list : list[list[int]]
        First neighbors lists.
    environment_list : list[list[int]]
        Lists of atoms in environments.
    atom_idx : list[int], optional
        List of atom index for which we compute their graph ID, if set to None compute for all atoms, by default None.

    Returns
    -------
    list[str]
        List of graph ID

    """
    # from mpi4py import MPI

    ## MPI
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # nprocs = comm.Get_size()

    ## Split index atoms in approximatively even number sublist
    if atom_idx is None:  # graph for all atoms in system
        local_index = np.arange(len(neighbors_list))
    else:
        local_index = atom_idx
    #    split = np.array_split(range(len(neighbors_list)), nprocs)
    # else:
    #    split = np.array_split(atom_idx, nprocs)  # when using cna/graph
    # local_index = split[rank]
    list_g = make_graph(local_index, neighbors_list, environment_list, types)

    list_hash = []

    for g in list_g:
        list_hash.append(encode_cert(pynauty.certificate(g)))
    # list_hash = comm.gather(list_hash, root=0)
    # if rank == 0:
    #    list_hash = [gcertificate for e in list_hash for gcertificate in e]
    return list_hash
    # else:
    #    return None


def make_graph(
    atoms_idx: int,
    neighbors_list: list[list[int]],
    environment_list: list[list[int]],
    types: list[str] = None,
) -> list[pynauty.Graph]:
    """Create graphs.

    Parameters
    ----------
    atoms_idx : int
        List of atom index for which we compute their graph.
    neighbors_list : list[list[int]]
        First neighbors lists.
    environment_list : list[list[int]]
       Lists of atoms in environments.

    Returns
    -------
    list[pynauty.Graph]
        List of atoms's graphs.

    """
    list_g = []
    for idx in atoms_idx:
        # To reorder indexes from 0 to number of vertexes-1 (for pynauty)
        local_to_global = {
            local_idx: global_idx
            for local_idx, global_idx in enumerate(environment_list[idx])
        }
        global_to_local = {v: k for k, v in local_to_global.items()}
        # Dictionary to map graph indexes to system indexes
        adjacency_dict = {local_idx: [] for local_idx in local_to_global.keys()}

        for i, at in enumerate(environment_list[idx]):
            for neighbor in neighbors_list[at]:
                if neighbor in environment_list[idx]:
                    adjacency_dict[i].append(global_to_local[neighbor])
        # Build vertex coloring from element types if provided
        vertex_coloring = []
        if types is not None:
            local_types = [types[global_idx] for global_idx in environment_list[idx]]
            unique_types = sorted(set(local_types))
            for element in unique_types:
                vertex_coloring.append(
                    {i for i, t in enumerate(local_types) if t == element}
                )

        g = pynauty.Graph(
            number_of_vertices=len(environment_list[idx]),
            adjacency_dict=adjacency_dict,
            directed=False,
            vertex_coloring=vertex_coloring,
        )

        list_g.append(g)

    return list_g
