import pandas as pd
from ase import Atoms
from ase.io import read
from ase.visualize import view
import numpy as np
from ase.neighborlist import NeighborList

catalog_file_path = "./NiSIAandVacancy/event_searches/catalog.pickle"
catalog = pd.read_pickle(catalog_file_path)


def event_result(idx):
    print("{} events in the catalog".format(len(catalog)))
    dE = catalog.loc[idx].at["energy_barrier"]
    print("Energy barrier = {}eV ".format(dE))

    # atom that moves the most :
    dist = (
        catalog.loc[idx].at["initial_positions"]
        - catalog.loc[idx].at["saddle_positions"]
    ) ** 2
    dist = dist.sum(axis=-1)
    dist = np.sqrt(dist)
    index_move = np.argmax(dist)

    traj_event = []
    pos = ["initial_positions", "saddle_positions", "final_positions"]
    z = len(catalog.loc[idx].at["initial_positions"]) * [20]
    z[index_move] = 30
    for pp in pos:
        traj_event.append(Atoms(numbers=z, positions=catalog.loc[idx].at[pp]))
    # atom that moves the most :
    dist = (
        catalog.loc[idx].at["initial_positions"]
        - catalog.loc[idx].at["saddle_positions"]
    ) ** 2
    dist = dist.sum(axis=-1)
    dist = np.sqrt(dist)
    index_move = np.argmax(dist)
    return view(traj_event, viewer="ngl")


from ipywidgets import interact

interact(event_result, idx=[i for i in range(len(catalog))])
