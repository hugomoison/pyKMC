def modify_lammps_data_2D(lammps_data_file) : 
    """
    Change the line zlo, zhi in a lammps_data_file so zlo/zhi straddle 0.0, ie zlo = -0.5 and zhi = 0.5 
    Usefull for 2D simulation since with ASE we need to add vacuum and so we can't really control the cell parameter
    """
    #TODO should raise error if zlo and zhi not find in file
    zlo = -0.5
    zhi = 0.5

    #Open file
    with open(lammps_data_file, 'r') as file : 
        lines = file.readlines() 

   # Modify the zlo/zhi line : 
    for i, line in enumerate(lines):
        if "zlo" in line and "zhi" in line:
            lines[i] = f"{zlo:.4f} {zhi:.4f} zlo zhi\n"

    #Write new lammps file 
    with open(lammps_data_file, 'w') as file:
        file.writelines(lines) 


def initialize_default_lammps(atoms, lmp_instance) : 
    """ 
    Initialize the lammps simulation with default settings, metal units, boundary conditions, atomic_style atomic and create box and atoms for 
    the current system.
    """
    from ase.data import atomic_numbers, atomic_masses
    import numpy as np
    #System parameters
    natoms = atoms.get_global_number_of_atoms()
    cell = atoms.get_cell()
    xhi, yhi, zhi = cell[0][0], cell[1,1], cell[2,2]
    type = atoms.get_chemical_symbols()
    ind = np.linspace(0, atoms.get_global_number_of_atoms()-1, atoms.get_global_number_of_atoms()).astype(int)
    ind += 1 #Lammps id start at 1

    #map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
    map_type = {atom_type: {'ref' :i+1, 'mass' : atomic_masses[atomic_numbers[atom_type]]} for i, atom_type in enumerate(sorted(set(type)))}
    type = [map_type[element]['ref'] for element in type] #map to integer
    x = atoms.get_positions().flatten()

    #lammps command
    lmp_instance.command('units metal')
    lmp_instance.command('atom_style atomic')
    lmp_instance.command('dimension 3') 
    lmp_instance.command('boundary p p p')
    lmp_instance.command('atom_modify sort 0 1')
    lmp_instance.command('region box block 0.0 {} 0.0 {} 0.0 {}'.format(xhi, yhi, zhi))
    lmp_instance.command('create_box {} box'.format(len(map_type)))
    lmp_instance.create_atoms(natoms, ind,type, x)
    #Set masses
    for key in map_type.keys() : 
        lmp_instance.command('mass {} {}'.format(map_type[key]['ref'], map_type[key]['mass']))
    #Label atoms name to type : 
    #print('labelmap atom '+ " ".join(f"{e['ref']} {key}" for key, e in map_type.items()))
#    for key in map_type.keys() : 
    lmp_instance.command('labelmap atom '+ " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items()))
    