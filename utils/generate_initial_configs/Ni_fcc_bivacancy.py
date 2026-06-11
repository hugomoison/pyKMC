from ase.build import bulk
from ase.io import write
from ase.io.lammpsdata import write_lammps_data

# Parameters :
output = "/root/pyKMC/examples/Ni_fcc_2047at_monovacancy/initial_config"
# Ni fcc
atoms = bulk("Ni", crystalstructure="fcc", a=3.52, cubic=True)
atoms = atoms.repeat((8, 8, 8))
print("Number of atoms = ", atoms.get_global_number_of_atoms())
print("Cell is : ", atoms.get_cell())


# remove center atom
atoms.pop(1166)
# atoms.pop(1706)
# write to file
write(output + ".xyz", atoms)
write_lammps_data(output + ".lmp", atoms)
