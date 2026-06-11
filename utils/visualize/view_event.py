from ase.io import read
from ase.visualize import view

traj = read("/root/pyKMC/examples/Ni_fcc_499at_monovacancy/test.xyz", index=":")
view(traj)
