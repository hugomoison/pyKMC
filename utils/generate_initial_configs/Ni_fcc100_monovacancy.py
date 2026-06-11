from ase.build import fcc100
from ase.io import write
from ase.io.lammpsdata import write_lammps_data

# Parameters :
output = "/Volumes/D1/pyKMC/examples/Ni_fcc100_monovacancy/initial_config"


# creat fcc100 surface
atoms = fcc100("Ni", size=(32, 32, 1), vacuum=1)
# remove center atom
atoms.pop(512 - 16)
# set all z positions to 0
pos = atoms.get_positions()
for p in pos:
    p[2] = 0
atoms.set_positions(pos)
# write to file
write(output + ".xyz", atoms)
write_lammps_data(output + ".lmp", atoms)
