"""Tools to compute graphs and graph's ID using NAUTY."""

import pynauty
import numpy as np


def graph(
    neighbors_list: list[list[int]],
    environment_list: list[list[int]],
    atom_idx: list[int] = None,
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
    list_g = make_graph(local_index, neighbors_list, environment_list)

    list_hash = []

    for g in list_g:
        list_hash.append(pynauty.certificate(g).hex())
    # list_hash = comm.gather(list_hash, root=0)
    # if rank == 0:
    #    list_hash = [gcertificate for e in list_hash for gcertificate in e]
    return list_hash
    # else:
    #    return None


def make_graph(
    atoms_idx: int, neighbors_list: list[list[int]], environment_list: list[list[int]]
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
        graph = pynauty.Graph(
            number_of_vertices=len(environment_list[idx]),
            adjacency_dict=adjacency_dict,
            directed=False,
        )

        list_g.append(graph)

    return list_g
