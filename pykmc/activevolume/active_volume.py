from ase import Atoms, Atom
from ase.build import bulk
from lammps import lammps
from ase.data import atomic_numbers, atomic_masses
import pypARTn
import numpy as np
import ctypes
from ase.geometry import find_mic
from ..system import System
from ..config import Config
from ..neighbors_list import NeighborsList
from ..environments import cna_order_parameter

from scipy.spatial import cKDTree


def define_AV(engine, config, central_atom_idx: int, positions, cell):
    # Defining parameters
    # Radius of whole active volume in Ang
    r_a = config.activevolume.ract  # Ensure AV is larger than topology analysis
    # Defines the radius of atoms that can move.
    r_m = config.activevolume.rmov

    # NEED TO ADD WARNING IF R_A<R_M

    # neighbors=engine.lmp.get_neighlist(central_atom_idx+1) # Lammps id start at 1
    # print(neighbors)

    center = positions[central_atom_idx]

    inner_movable_idx = []
    buffer_idx = []
    total_active_idx = []
    non_active_idx = []

    for i, pos in enumerate(positions):
        diff = pos - center
        diff_mic, distance = find_mic(diff, cell, pbc=True)
        if np.abs(distance) <= r_m:
            inner_movable_idx.append(i)
            total_active_idx.append(i)  # Inner is also part of total active
        elif np.abs(distance) > r_m and np.abs(distance) <= r_a:  # Can change to make it between r_m and r_a
            buffer_idx.append(i)
            total_active_idx.append(i)
        else:
            non_active_idx.append(i)

    buffer_idx = np.array(sorted(buffer_idx))
    av_idx = np.array(sorted(total_active_idx))

    av_positions = positions[av_idx]

    #print(len(av_idx)," atoms in AV,", len(av_idx)-len(buffer_idx), "movable atoms")

    return av_positions, av_idx, buffer_idx

def make_AV(engine, av_indices, buffer_indices):

    # Define the buffer group based on the new LAMMPS IDs
    # We need to find which index in 'av_indices' corresponds to 'buffer_indices'
    engine_buffer_ids = []
    buffer_set = set(buffer_indices)
    for i, original_id in enumerate(av_indices):
        if original_id in buffer_set:
            engine_buffer_ids.append(i + 1)  # LAMMPS IDs are 1-based

    if engine_buffer_ids:
        engine.command(f"group buffer id {' '.join(map(str, engine_buffer_ids))}")
        engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
    else:
        engine.command(f"group buffer empty")
        engine.command("fix f_buffer buffer setforce 0.0 0.0 0.0")
        print('No buffer atoms defined')

    engine.command('run 0 post no')


def build_type_map(type) -> dict:
    """Map each distinct element symbol in `type` to a 1-based LAMMPS atom type ID
    (alphabetical order) and its atomic mass, eg {'Fe': {'ref': 1, 'mass': 55.85}, 'Ni': {'ref': 2, 'mass': 58.69}}.
    """
    return {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(sorted(set(type)))
    }


def reset(engine, config, cell, map_type) -> None:
    """
    Clear lammps instance, preps it for the new sim.
    `map_type` (see `build_type_map`) determines how many atom types the box
    needs and their per-type masses, so multi-element systems get one LAMMPS
    atom type per element instead of a single hardcoded type.
    """

    engine.command('clear')
    initialize_parameters(engine)
    # Create cell
    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
    engine.command(
        "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
    )
    engine.command('create_box {} box'.format(len(map_type)))
    initialize_potential(engine, config)
    # Set masses
    for entry in map_type.values():
        engine.command("mass {} {}".format(entry["ref"], entry["mass"]))
    # Label atoms name to type :
    engine.command(
        "labelmap atom "
        + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
    )

def clear(engine):
    '''
    Clears lammps instance
    '''
    engine.command('clear')

def redefine_atoms(engine, positions, type=None) -> None:
    '''
        Check to see if current lammps system has enough atoms
        If not, deletes all atoms then redefines them
        '''
    if type is None:
        type = [1]*len(positions)
    new_positions = positions.flatten().astype(np.float64)
    ids = np.arange(1, len(positions) + 1, dtype=np.int32)
    engine.lmp.create_atoms(len(positions), ids, type, x=new_positions)
    engine.command("comm_style tiled")
    engine.command("balance 1.1 rcb")
    engine.command("neigh_modify every 1 delay 0 check yes")
    engine.command('fix 1 all setforce 0.0 0.0 0.0')
    engine.command('run 0')
    engine.command('unfix 1')

def partn_search_AV(engine, config, central_atom_idx: int, positions, cell, type) -> [np.array, int]:
    map_type = build_type_map(type)
    reset(engine, config, cell, map_type)
    av_positions, av_idx, buffer_idx = define_AV(engine, config, central_atom_idx, positions, cell)

    #Need to map type to positions
    atom_map = np.array(av_idx, dtype=int)
    type = np.array([map_type[element]["ref"] for element in type]) # map to integer

    av_type=type[atom_map]

    redefine_atoms(engine, av_positions, av_type)
    make_AV(engine, av_idx, buffer_idx)
    return atom_map, np.array(np.where(atom_map == central_atom_idx)[0]+1)

def partn_refine_AV(engine, config, central_atom_idx:int, positions, cell, type, saddle_idx, saddle_positions, neighbors) -> [float, np.array, int]:
    '''
        Receive the system with the central atom index, define an active volume around this atom, then update the positions
        with those for the saddle.

        This was added in order to get the activation energy for an event, as the traditional method does not work for
        Active Volumes.
    '''

    # Liam Smyth change: self_define_AV_new now returns (atom_map, central_lammps_id) and already
    # does the type-mapping/redefine_atoms/make_AV internally, so this used to unpack 3 values
    # (av_positions, av_idx, buffer_idx) and redo that work — fixed to match the current
    # 2-value contract and just rebuild av_positions from atom_map.
    atom_map, central_lammps_id = self_define_AV_new(engine, config, central_atom_idx, positions, cell, type, neighbors)
    av_positions = positions[atom_map]

    if config.activevolume.AV_debug == True:
        E_before = get_potential_energy(engine)
        engine.command("min_style {}".format(config.lammps.min_style))
        engine.command("minimize 1.0e-6 1.0e-8 10 10")
        E_init=get_potential_energy(engine)
        print("Before minimization: ", E_before, "After minimization: ", E_init)
        print("% Difference:", abs((E_before - E_init)/E_init*100),"%")
    else:
        E_init=get_potential_energy(engine)

    core_idx=[]
    core_ids=[]
    for i, atom_idx in enumerate(saddle_idx):
        index = int(np.where(atom_map == atom_idx)[0])  # index in atom map where this value is true
        av_positions[index] = saddle_positions[i]
        core_idx.append(index)  # Atom id
        core_ids.append(index + 1)
    set_positions(engine, av_positions)

    engine.command('fix 1 all setforce 0.0 0.0 0.0')
    engine.command('run 0')
    engine.command('unfix 1')

    # Want to minimize initially to speed up refinement process
    engine.command(f"group core id {' '.join(map(str, core_ids))}")
    engine.command('fix f_core core setforce 0.0 0.0 0.0')
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize 1.0e-6 1.0e-8 10 10")
    engine.command('unfix f_core')

    return E_init, atom_map, (np.where(atom_map == central_atom_idx)[0]+1)

def position_results_AV(config, artn, atom_map, positions) -> [np.array, np.array, np.array, int]:


    min1positions = artn.extract("tau_min1")
    min2positions = artn.extract("tau_min2")
    saddlepositions = artn.extract("tau_sad")

    # find atom that moves the most
    dist = (min1positions - saddlepositions) ** 2
    dist = dist.sum(axis=-1)
    dist = np.sqrt(dist)
    dist[dist > config.atomicenvironment.rcut] = (
        0
    # if atom moves more that rcutevent, consider that it crosses the cell (happens with lammps), so distance = 0 to not consider it as the one that moves the most
    )
    index_move = np.argmax(dist)

    index_move_mapped = atom_map[index_move]

    min1positions_mapped = positions.copy()
    min2positions_mapped = positions.copy()
    saddlepositions_mapped = positions.copy()

    for i, atom_idx in enumerate(atom_map):
        saddlepositions_mapped[atom_idx][0] = saddlepositions[i][0]
        saddlepositions_mapped[atom_idx][1] = saddlepositions[i][1]
        saddlepositions_mapped[atom_idx][2] = saddlepositions[i][2]

        min1positions_mapped[atom_idx][0] = min1positions[i][0]
        min1positions_mapped[atom_idx][1] = min1positions[i][1]
        min1positions_mapped[atom_idx][2] = min1positions[i][2]

        min2positions_mapped[atom_idx][0] = min2positions[i][0]
        min2positions_mapped[atom_idx][1] = min2positions[i][1]
        min2positions_mapped[atom_idx][2] = min2positions[i][2]

    return  min1positions_mapped, min2positions_mapped, saddlepositions_mapped, index_move_mapped


def initialize_parameters(engine):
    engine.command("units metal")
    engine.command("atom_style atomic")
    engine.command("dimension 3")
    engine.command("boundary p p p")
    engine.command("atom_modify map array")  # ! necessary for scatter atoms
    engine.command("atom_modify sort 0 0.0")  # ! necessary for partn


def initialize_system(engine, system):
    # system parameters
    natoms = len(system.types)
    cell = system.cell
    types = system.types
    x = system.positions.flatten()  # Lammps format

    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]

    ind = np.linspace(0, natoms - 1, natoms).astype(int)
    ind += 1  # Lammps id start at 1
    map_type = build_type_map(types)
    types = [map_type[element]["ref"] for element in types]  # map to integer

    # lammps create system
    engine.command(
        "region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi)
    )
    engine.command("create_box {} box".format(len(map_type)))
    engine.lmp.create_atoms(natoms, ind, types, x)
    # Set masses
    for key in map_type.keys():
        engine.command(
            "mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"])
        )
    # Label atoms name to type :
    engine.command(
        "labelmap atom "
        + " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items())
    )


def initialize_potential(engine, config):
    pair_style = config.lammps.pair_style
    pair_coeff = config.lammps.pair_coeff
    engine.command("pair_style {}".format(pair_style))
    engine.command("pair_coeff {}".format(pair_coeff))


def minimize(engine, config, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize {}".format(config.lammps.minimize))


def get_total_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    # Get total energy
    engine.command("run 0")
    result = engine.lmp.get_thermo("etotal")
    if engine.rank == 0:
        return result


def get_potential_energy(engine, positions=None):
    if positions is not None:
        set_positions(engine=engine, positions=positions)
    # get potential energy
    engine.command("compute c1 all pe")
    engine.command("run 0")
    result = engine.lmp.extract_compute("c1", 0, 0)
    engine.command("uncompute c1")
    return result


def get_positions(engine):
    result = engine.lmp.gather_atoms("x", 1, 3)
    if engine.rank == 0:
        # convert ctype positions into a numpy array
        result = np.ctypeslib.as_array(result)
        result = np.reshape(result, (-1, 3))
        return result


def set_positions(engine, positions):
    positions = positions.flatten().astype(np.float64)
    positions = np.ascontiguousarray(positions)
    c_array = (ctypes.c_double * len(positions))(*positions)
    engine.lmp.scatter_atoms("x", 1, 3, c_array)


def self_define_AV(engine, config, central_atom_idx: int, positions, cell):
    '''
    Self defines AV based on input parameters.
    If pe is selected, the potential energy of each atom is used
    If stress is selected, the stress of each atom is used
    '''

    neighbors=engine.lmp.get_neighlist(central_atom_idx+1) # Lammps id start at 1
    print(neighbors)
    
    
    # Defining parameters
    # Radius of whole active volume in Ang
    r_a = config.activevolume.ract  # Ensure AV is larger than topology analysis
    # Defines the radius of atoms that can move.
    r_m = config.activevolume.rmov

    # NEED TO ADD WARNING IF R_A<R_M

    center = positions[central_atom_idx]

    inner_movable_idx = []
    buffer_idx = []
    total_active_idx = []
    non_active_idx = []

    for i, pos in enumerate(positions):
        diff = pos - center
        diff_mic, distance = find_mic(diff, cell, pbc=True)
        if np.abs(distance) <= r_m:
            inner_movable_idx.append(i)
            total_active_idx.append(i)  # Inner is also part of total active
        elif np.abs(distance) > r_m and np.abs(distance) <= r_a:  # Can change to make it between r_m and r_a
            buffer_idx.append(i)
            total_active_idx.append(i)
        else:
            non_active_idx.append(i)

    buffer_idx = np.array(sorted(buffer_idx))
    av_idx = np.array(sorted(total_active_idx))

    av_positions = positions[av_idx]

    print(len(av_idx)," atoms in AV,", len(av_idx)-len(buffer_idx), "movable atoms")

    return av_positions, av_idx, buffer_idx

# def make_neighbors_list(config, positions, cell):
#     """Makes and populates the `neighbors_list`."""
#     neighbors_list = {"rnei": [], "rcut": []}
#     # Construct the kdTree
#     box = [cell[0][0], cell[1][1], cell[2][2]]
#     tree = cKDTree(positions, boxsize=box)

#     # Find first neighbors and atoms in environments
#     for i in range(len(positions)):

#         neighbors = tree.query_ball_point(positions[i], config.atomicenvironment.rnei)
#         neighbors.remove(i)  # don't have self as neighbor
#         neighbors_list["rnei"].append(neighbors)
#         if config.atomicenvironment.rcut is not None : 
#             neighbors = tree.query_ball_point(positions[i], config.atomicenvironment.rcut)
#             neighbors_list["rcut"].append(neighbors)

#     # Ragged (variable-length) neighbor lists: use object arrays so they can be
#     # indexed with an array of atom indices, e.g. neighbors_list["rnei"][core].
#     neighbors_list["rnei"] = np.array(neighbors_list["rnei"], dtype=object)
#     if config.atomicenvironment.rcut is not None:
#         neighbors_list["rcut"] = np.array(neighbors_list["rcut"], dtype=object)

#     return neighbors_list


def self_define_AV_new(engine, config, central_atom_idx: int, positions, cell, type, neighbors) -> [np.array, int]:
    np.set_printoptions(threshold=np.inf)
    #==================================================================================================
    # Build the element -> LAMMPS type id map once, and the per-atom type array
    # (same length/order as `positions`) so every reset()/redefine_atoms() call
    # below keeps the correct species for multi-element systems.
    map_type = build_type_map(type)
    type_mapped = np.array([map_type[element]["ref"] for element in type])

    reset(engine, config, cell, map_type)
    redefine_atoms(engine, positions, type_mapped)
    # Start with the atoms within the cutoff radius around the central atom.
    core = np.array(neighbors.get_neighbors('rcut',central_atom_idx), dtype=int)
    # Expand the initial core by including the first-neighbor shell of those atoms.
    neighbor_lists = [np.asarray(neighbors.get_neighbors('rnei',idx), dtype=int) for idx in core]
    if neighbor_lists:
        first_shell = np.unique(np.concatenate(neighbor_lists))
    else:
        first_shell = np.array([], dtype=int)
    core = np.unique(np.concatenate((core, first_shell, np.array([central_atom_idx], dtype=int))))

    #==================FOR PE/ATOM==========================
    # engine.command("compute c1 all pe/atom")
    # engine.command("run 0")
    # result = engine.lmp.numpy.extract_compute("c1", 1, 1)
    # engine.command("uncompute c1")

    # reset(engine, config, cell)
    # redefine_atoms(engine, positions[core])
    # engine.command(f"write_dump all atom core.atoms.{central_atom_idx}.1")

    #========================FORCE METHOD============================
    # Compute per-atom force magnitude as a local response metric.
    # This is often more informative than raw per-atom potential energy for
    # identifying atoms that belong to the defect core versus the surrounding bulk.
    # engine.command("run 0")
    # forces = engine.lmp.gather_atoms("f", 1, 3)
    # if forces is None:
    #     raise RuntimeError("Unable to gather per-atom forces from LAMMPS")
    # forces = np.ctypeslib.as_array(forces)
    # forces = np.reshape(forces, (-1, 3))
    # force_magnitude = np.linalg.norm(forces, axis=1)
    # result = force_magnitude
    #===============================================================

    #========================STRESS METHOD (Liam Smyth change)============================
    # Compute per-atom stress tensor as an alternative local response metric to force
    # magnitude. LAMMPS' stress/atom returns 6 components per atom (xx, yy, zz, xy, xz, yz),
    # in pressure*volume units. Reduced here to the von Mises equivalent stress, a single
    # # scalar magnitude per atom analogous to force_magnitude above.
    # engine.command("compute c_stress all stress/atom NULL")
    # engine.command("run 0")
    # stress = engine.lmp.numpy.extract_compute("c_stress", 1, 2)
    # engine.command("uncompute c_stress")
    # if stress is None:
    #     raise RuntimeError("Unable to gather per-atom stress from LAMMPS")
    # sxx, syy, szz, sxy, sxz, syz = (stress[:, i] for i in range(6))
    # von_mises_stress = np.sqrt(
    #     0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    #     + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2)
    # )
    # result = von_mises_stress
    # # print('Results', result)
    #===============================================================
    result = get_results(engine, config, neighbors)

    mask = np.ones(len(result), dtype=bool)
    mask[core]=False
    # print(result[neighbors])
    converge=False
    #Include way to remove duplicates
    # Liam Smyth change: get_neighbors() only accepts a scalar atom index, but this used to be
    # called with the whole `core` array at once, which raised "only integer scalar arrays can
    # be converted to a scalar index" — fixed to loop per-atom like the pattern above.
    core_neighbor_lists = [np.asarray(neighbors.get_neighbors('rnei', idx), dtype=int) for idx in core]
    buffer = np.unique(np.concatenate(core_neighbor_lists, dtype=object)).astype(int)
    buffer = buffer[np.isin(buffer, core, invert=True)]
    non_converged=buffer
    error=0.0001
    '''
    Old method, makes spherical AV
    '''
    # while converge==False:
    #     core_avg=np.mean(result[core])
    #     buffer_avg=np.mean(result[buffer])
    #     all_avg=np.mean(result)
    #     per_diff=abs((all_avg-buffer_avg)/all_avg)
    #     if per_diff<=error: 
    #         converge=True #AV is large enough, we're good
    #     else:
    #         print('ADDING TO BUFFER')
    #         #Also remove duplicates from buffer and atoms that are within the core
    #         core=np.unique(np.concatenate((core, buffer) ,dtype=object)).astype(int)
    #         buffer_new=np.unique(np.concatenate(neighbors['rnei'][core],dtype=object)).astype(int)
    #         buffer_new=buffer_new[np.isin(buffer_new, core, invert=True)]
    #         buffer=np.array(buffer_new)
    '''
    New method, allows AV to grow
    '''
    all_avg = np.mean(result)
    converged = False
    key = 1
    threshold = 5e-4
    error=0.1
    # Liam Smyth change: how many bulk standard deviations above the bulk mean an atom's
    # metric (von Mises stress, or force_magnitude if you switch result back) must exceed
    # to be considered a real outlier rather than background noise. Tune per system.
    n_sigma = 5
    num=1
    while not converged:
        mask = np.ones(len(result), dtype=bool)
        mask[buffer] = False
        mask[core] = False
        non_selected = np.average(result[mask == True])

        # # Promote buffer atoms that are fully surrounded by the core to the core.
        core_neighbor_set = set(core)
        surrounded = []
        for atom in buffer:
            atom_neighbors = set(np.asarray(neighbors.get_neighbors('rnei',atom), dtype=int))
            if atom_neighbors.issubset(core_neighbor_set):
                surrounded.append(atom)

        # Non-converged atoms are those in the buffer with a large force magnitude
        # or whose neighbors are all already in the core.

        #non_converged=buffer[result[buffer]>threshold] #For forces

        # Liam Smyth change: relative difference from the bulk mean blows up because the
        # bulk mean sits near zero at equilibrium, so tiny absolute noise looks like a huge
        # percent difference — unstable regardless of the `error` threshold chosen. Replaced
        # with a mean + N*std outlier test against the bulk (non-AV) atoms, which is scale
        # invariant with respect to the background noise floor.
        non_converged=check_criteria(config, result, buffer, non_selected, mask)


        #Stress
        # bulk_std = np.std(result[mask == True])
        # non_converged = buffer[result[buffer] > non_selected + n_sigma * bulk_std]
    

        if len(non_converged) != 0:
            key = 1
            num+=1
            print('Adding Atoms')
            core = np.unique(np.concatenate((core, non_converged), dtype=object)).astype(int)
            buffer = buffer[np.isin(buffer, non_converged, invert=True)]
            # Liam Smyth change: this was a bare generator expression missing its enclosing
            # brackets/list comprehension, so np.unique() received a generator object instead of
            # an array and raised "int() argument must be ... not 'generator'" — fixed to build
            # the per-atom neighbor lists first, then concatenate.
            non_converged_neighbor_lists = [np.asarray(neighbors.get_neighbors('rnei', atom), dtype=int) for atom in non_converged]
            if non_converged_neighbor_lists:
                new_buffer = np.unique(np.concatenate(non_converged_neighbor_lists)).astype(int)
            else:
                new_buffer = np.array([], dtype=int)
            new_buffer = new_buffer[np.isin(new_buffer, core, invert=True)]
            buffer = np.unique(np.concatenate((buffer, new_buffer), dtype=object)).astype(int)
            reset(engine, config, cell, map_type)
            redefine_atoms(engine, positions[core], type_mapped[core])
            engine.command(f"write_dump all atom core.atoms.{num}.{central_atom_idx}")
        else:
            converged = True

        # After growth, prune core atoms that do not have enough core neighbors.
        core_neighbor_counts = []
        for atom in core:
            atom_neighbors = set(np.asarray(neighbors.get_neighbors('rnei',atom), dtype=int))
            core_neighbors = atom_neighbors & core_neighbor_set
            core_neighbor_counts.append(len(core_neighbors))

        # if len(core) > 0:
        #     core_neighbor_fraction = np.array(core_neighbor_counts) / np.maximum(1, np.array([len(np.asarray(neighbors['rnei'][atom], dtype=int)) for atom in core]))
        #     weak_core_atoms = core[core_neighbor_fraction < 0.5]
        #     if len(weak_core_atoms) > 0:
        #         core = core[np.isin(core, weak_core_atoms, invert=True)]
        #         buffer = np.unique(np.concatenate((buffer, weak_core_atoms), dtype=object)).astype(int)
        #         buffer = buffer[np.isin(buffer, core, invert=True)]
        #         converged = False


    print('AV DEFINED')
    #Now add atoms within cutoff radius of the buffer atoms
    # Liam Smyth change: same "whole array passed as scalar index" bug as above — get_neighbors()
    # was called with the full `buffer` array instead of looping per-atom, fixed the same way.
    #============================LARGE FROZEN BUFFER REGION=============================
    # buffer_neighbor_lists = [np.asarray(neighbors.get_neighbors('rcut', atom), dtype=int) for atom in buffer]
    # if buffer_neighbor_lists:
    #     buffer_extra = np.unique(np.concatenate(buffer_neighbor_lists)).astype(int)
    # else:
    #     buffer_extra = np.array([], dtype=int)
    # buffer=np.unique(np.concatenate((buffer, buffer_extra),dtype=object)).astype(int)
    # buffer = buffer[np.isin(buffer, core, invert=True)]
    # print('Length of buffer', len(buffer))

    #===================================================================================

    #===========================SOFT BUFFER WITH FROZEN LAYER============================
    buffer_neighbor_lists = [np.asarray(neighbors.get_neighbors('rcut', atom), dtype=int) for atom in buffer]
    if buffer_neighbor_lists:
        buffer_extra = np.unique(np.concatenate(buffer_neighbor_lists)).astype(int)
    else:
        buffer_extra = np.array([], dtype=int)
    core=np.unique(np.concatenate((core, buffer),dtype=object)).astype(int)
    reset(engine, config, cell, map_type)
    redefine_atoms(engine, positions[core], type_mapped[core])
    engine.command(f"write_dump all atom core.atoms.{central_atom_idx}.{num+1}")
    buffer = buffer_extra[np.isin(buffer_extra, core, invert=True)]
    print('Length of buffer', len(buffer))
    
    #==================================Buffer fully defined, now prune=========================
    # core_neighbor_set = set(core)
    # core_neighbor_counts = []
    # for atom in core:
    #     atom_neighbors = set(np.asarray(neighbors.get_neighbors('rnei',atom), dtype=int))
    #     core_neighbors = atom_neighbors & core_neighbor_set
    #     core_neighbor_counts.append(len(core_neighbors))
    # core_neighbor_fraction = np.array(core_neighbor_counts) / np.maximum(1, np.array([len(np.asarray(neighbors.get_neighbors('rnei',atom), dtype=int)) for atom in core]))
    # weak_core_atoms = core[core_neighbor_fraction < 0.25]
    # if len(weak_core_atoms) > 0:
    #     core = core[np.isin(core, weak_core_atoms, invert=True)]
    #     buffer = np.unique(np.concatenate((buffer, weak_core_atoms), dtype=object)).astype(int)
    #     buffer = buffer[np.isin(buffer, core, invert=True)]
    #===========================================================================================

    all_idx=np.concatenate((core, buffer) ,dtype=object).astype(int)

    print('Total atoms in AV', len(all_idx))

    av_positions=positions[all_idx]
    #===============================================================================================
    reset(engine, config, cell, map_type)

    # av_positions, av_idx, buffer_idx = define_AV(engine, config, central_atom_idx, positions, cell)
    #Need to map type to positions
    atom_map = np.array(all_idx, dtype=int)
    av_type=type_mapped[atom_map]
    key=0
    if key==1:
        redefine_atoms(engine, positions[core], type_mapped[core])
        engine.command(f"write_dump all atom core.atoms.{central_atom_idx}")
        print('Core defined')
        reset(engine, config, cell, map_type)
        redefine_atoms(engine, positions[buffer], type_mapped[buffer])
        engine.command(f"write_dump all atom buffer.atoms.{central_atom_idx}")
        print('Buffer defined')
        reset(engine, config, cell, map_type)

    redefine_atoms(engine, positions[all_idx], av_type)


    make_AV(engine, all_idx, buffer)

    if config.activevolume.AV_debug == True:
            E_before = get_potential_energy(engine)
            engine.command("min_style {}".format(config.lammps.min_style))
            engine.command("minimize 1.0e-6 1.0e-8 10 10")
            E_init=get_potential_energy(engine)
            # print("Before minimization: ", E_before, "After minimization: ", E_init)
            # print("% Difference:", abs((E_before - E_init)/E_init*100),"%")
    else:
        E_init=get_potential_energy(engine)

    return atom_map, np.array(np.where(atom_map == central_atom_idx)[0]+1)


def get_results(engine, config, neighbors=None):
    key={'pe','stress','cna'}

    if config.activevolume.method == 'pe':
        engine.command("compute c1 all pe/atom")
        engine.command("run 0")
        result = engine.lmp.numpy.extract_compute("c1", 1, 1)
        engine.command("uncompute c1")

    elif config.activevolume.method == 'cna':
        # Continuous CNA crystallinity order parameter per atom, in [0, 1]:
        # 0 = matches a reference crystal structure (FCC/HCP/BCC/ICO), 1 = no
        # match at all. See `cna_order_parameter` for the signature-overlap logic.
        result = np.array(cna_order_parameter(neighbors.neighbors_list['rnei']))

    elif config.activevolume.method == 'stress':
        # Compute per-atom stress tensor as an alternative local response metric to force
        # magnitude. LAMMPS' stress/atom returns 6 components per atom (xx, yy, zz, xy, xz, yz),
        # in pressure*volume units. Reduced here to the von Mises equivalent stress, a single
        # # scalar magnitude per atom analogous to force_magnitude above.
        engine.command("compute c_stress all stress/atom NULL")
        engine.command("run 0")
        stress = engine.lmp.numpy.extract_compute("c_stress", 1, 2)
        engine.command("uncompute c_stress")
        if stress is None:
            raise RuntimeError("Unable to gather per-atom stress from LAMMPS")
        sxx, syy, szz, sxy, sxz, syz = (stress[:, i] for i in range(6))
        von_mises_stress = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2)
        )
        result = von_mises_stress
        # print('Results', result)
    
    return result

def check_criteria(config, result, buffer, non_selected, mask):
    if config.activevolume.method == 'pe':
        criteria=config.activevolume.threshold
        difference=abs((result[buffer]-non_selected)/non_selected) #absolute difference
        non_converged=buffer[difference>criteria]

    elif config.activevolume.method == 'stress':
        #Stress
        n_sigma=config.activevolume.threshold
        bulk_std = np.std(result[mask == True])
        non_converged = buffer[result[buffer] > non_selected + n_sigma * bulk_std]

    elif config.activevolume.method == 'cna':
        # result is 0 (crystalline) or 1 (non-crystalline) per atom. Buffer atoms
        # still classified as non-crystalline (above threshold) keep the AV growing;
        # the AV stops growing once every buffer atom has settled back to crystalline.
        criteria=config.activevolume.threshold
        non_converged = buffer[result[buffer] > criteria]

    return non_converged