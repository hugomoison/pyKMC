import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell
import random
from ase.visualize import view

# Generate fcc Ni
output = "./Ni_sia_octa.xyz"
# Ni fcc
alat = 3.52
atoms = bulk("Ni", crystalstructure="fcc", a=alat, cubic=True)
atoms = atoms.repeat((10, 10, 10))
print("Number of atoms = ", atoms.get_global_number_of_atoms())
print("Cell is : ", atoms.get_cell())

cell = atoms.get_cell()
# Identify interstitial site

# Octahedral and tetrahedral site for fcc
octahedral_sites = (
    np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    * alat
)
tetrahedral_sites = (
    np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
        ]
    )
    * alat
)

# Repeat interstitial sites on all the box
all_octahedral_sites = []
all_tetrahedral_sites = []
for i in range(0, 8):
    for j in range(0, 8):
        for k in range(0, 8):
            base_shift = np.array([i, j, k]) * alat
            for site in octahedral_sites:
                all_octahedral_sites.append(site + base_shift)
            for site in tetrahedral_sites:
                all_tetrahedral_sites.append(site + base_shift)


# Choose a site :
chosen_site = all_octahedral_sites[1240]
print("SI site at : ", chosen_site)
# Add atom in site
atoms.extend(Atoms("Ni", positions=[chosen_site]))

# save to file
atoms.write(output)
