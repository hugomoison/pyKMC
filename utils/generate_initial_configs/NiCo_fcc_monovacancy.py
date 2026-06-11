from ase.build import bulk
from ase.io import write
from ase.io.lammpsdata import write_lammps_data
import random

# Parameters :
output = "/root/pyKMC/examples/NiFe_fcc_2047at_monovacancy/initial_config"
# Ni fcc
atoms = bulk("Ni", crystalstructure="fcc", a=3.52, cubic=True)
atoms = atoms.repeat((8, 8, 8))
print("Number of atoms = ", atoms.get_global_number_of_atoms())
print("Cell is : ", atoms.get_cell())


# Replace Ni atoms by Co :
tau = 20  # replace tau% of Ni at
number = int(atoms.get_global_number_of_atoms() * tau / 100)
print("Replacing randomly {} Ni atoms with Co".format(number))

atomlist = atoms.get_chemical_symbols()

for i in range(number):
    # find 'Ni' index in atomlist
    indices_ni = [i for i, elem in enumerate(atomlist) if elem == "Ni"]
    # randomly choose an indice
    index_choisi = random.choice(indices_ni)
    atomlist[index_choisi] = "Fe"

atoms.set_chemical_symbols(atomlist)


# remove center atom
atoms.pop(1166)
# atoms.pop(1706)
# write to file
write(output + ".xyz", atoms)
