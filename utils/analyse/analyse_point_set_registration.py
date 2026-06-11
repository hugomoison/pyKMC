import pandas as pd
from ase.io import read, write
from ase import Atoms
import numpy as np

# Read input system
init_config_file = "./initial_config.xyz"
system = read(init_config_file)
# Read catalog
catalog = pd.read_pickle("catalog.pickle")

# psr
psr = pd.read_pickle("psr_event_0.pickle")

# Read event :
idx_cat = psr["n event"][0]
coords2 = catalog.loc[idx_cat].at["initial_positions"]
nat2 = len(coords2)
typ2 = nat2 * ["Ni"]

# Read PSR :
rmat = psr.loc[0].at["R"]
tr = psr.loc[0].at["T"]
perm = psr.loc[0].at["P"]
dh = psr.loc[0].at["dh"]

print("Distance DH = ", dh)

# get rcutevent env system :
rcutevent = 7.0
atom_index = psr.loc[0].at["central atom index"]
print(atom_index)
ind = np.linspace(
    0, system.get_global_number_of_atoms() - 1, system.get_global_number_of_atoms()
).astype(int)
dist = system.get_distances(atom_index, ind, mic=True)
neighbor_list = np.where(dist < rcutevent)[0]

coords1 = system.get_positions()[neighbor_list]
nat1 = len(coords1)
typ1 = nat1 * ["Cu"]


pos = np.concatenate((coords1, coords2), axis=0)
at_name = typ1 + typ2

traj = [Atoms(at_name, positions=pos)]

# change coords2 :
for i in range(nat2):
    coords2[i] = np.matmul(rmat, coords2[i]) + tr
coords2[:] = coords2[perm]

pos = np.concatenate((coords1, coords2), axis=0)
at_name = typ1 + typ2

traj.append(Atoms(at_name, positions=pos))


write("event_0_psr_reconstruction.xyz", traj)
