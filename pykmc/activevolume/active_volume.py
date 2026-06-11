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


def define_AV(config, central_atom_idx: int, positions, cell):
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
        elif (
            np.abs(distance) > r_m and np.abs(distance) <= r_a
        ):  # Can change to make it between r_m and r_a
            buffer_idx.append(i)
            total_active_idx.append(i)
        else:
            non_active_idx.append(i)

    buffer_idx = np.array(sorted(buffer_idx))
    av_idx = np.array(sorted(total_active_idx))

    av_positions = positions[av_idx]

    # print(len(av_idx)," atoms in AV,", len(av_idx)-len(buffer_idx), "movable atoms")

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
        print("No buffer atoms defined")

    engine.command("run 0 post no")


def reset(engine, config, cell) -> None:
    """
    Clear lammps instance, preps it for the new sim:
    """

    engine.command("clear")
    initialize_parameters(engine)
    # Create cell
    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box 1 box")  # NEEDS TO BE UPDATED FOR ALLOYS
    initialize_potential(engine, config)


def clear(engine):
    """
    Clears lammps instance
    """
    engine.command("clear")


def redefine_atoms(engine, positions, type=None) -> None:
    """
    Check to see if current lammps system has enough atoms
    If not, deletes all atoms then redefines them
    """
    if type is None:
        type = [1] * len(positions)
    new_positions = positions.flatten().astype(np.float64)
    ids = np.arange(1, len(positions) + 1, dtype=np.int32)
    engine.lmp.create_atoms(len(positions), ids, type, x=new_positions)
    engine.command("comm_style tiled")
    engine.command("balance 1.1 rcb")
    engine.command("neigh_modify every 1 delay 0 check yes")
    engine.command("fix 1 all setforce 0.0 0.0 0.0")
    engine.command("run 0")
    engine.command("unfix 1")


def partn_search_AV(
    engine, config, central_atom_idx: int, positions, cell, type
) -> [np.array, int]:
    reset(engine, config, cell)
    av_positions, av_idx, buffer_idx = define_AV(
        config, central_atom_idx, positions, cell
    )

    # Need to map type to positions
    atom_map = np.array(av_idx, dtype=int)
    map_type = {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(sorted(set(type)))
    }
    type = np.array([map_type[element]["ref"] for element in type])  # map to integer

    av_type = type[atom_map]

    redefine_atoms(engine, av_positions, av_type)
    make_AV(engine, av_idx, buffer_idx)
    return atom_map, np.array(np.where(atom_map == central_atom_idx)[0] + 1)


def partn_refine_AV(
    engine,
    config,
    central_atom_idx: int,
    positions,
    cell,
    type,
    saddle_idx,
    saddle_positions,
) -> [float, np.array, int]:
    """
    Receive the system with the central atom index, define an active volume around this atom, then update the positions
    with those for the saddle.

    This was added in order to get the activation energy for an event, as the traditional method does not work for
    Active Volumes.
    """

    reset(engine, config, cell)
    av_positions, av_idx, buffer_idx = define_AV(
        config, central_atom_idx, positions, cell
    )

    # Need to map types to positions
    atom_map = np.array(av_idx, dtype=int)
    map_type = {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(sorted(set(type)))
    }
    type = np.array([map_type[element]["ref"] for element in type])  # map to integer

    av_type = type[atom_map]

    redefine_atoms(engine, av_positions, av_type)
    make_AV(engine, av_idx, buffer_idx)

    if config.activevolume.AV_debug == True:
        E_before = get_potential_energy(engine)
        engine.command("min_style {}".format(config.lammps.min_style))
        engine.command("minimize 1.0e-6 1.0e-8 10 10")
        E_init = get_potential_energy(engine)
        print("Before minimization: ", E_before, "After minimization: ", E_init)
        print("% Difference:", abs((E_before - E_init) / E_init * 100), "%")
    else:
        E_init = get_potential_energy(engine)

    core_idx = []
    core_ids = []
    for i, atom_idx in enumerate(saddle_idx):
        index = int(
            np.where(atom_map == atom_idx)[0]
        )  # index in atom map where this value is true
        av_positions[index] = saddle_positions[i]
        core_idx.append(index)  # Atom id
        core_ids.append(index + 1)
    set_positions(engine, av_positions)

    engine.command("fix 1 all setforce 0.0 0.0 0.0")
    engine.command("run 0")
    engine.command("unfix 1")

    # Want to minimize initially to speed up refinement process
    engine.command(f"group core id {' '.join(map(str, core_ids))}")
    engine.command("fix f_core core setforce 0.0 0.0 0.0")
    engine.command("min_style {}".format(config.lammps.min_style))
    engine.command("minimize 1.0e-6 1.0e-8 10 10")
    engine.command("unfix f_core")

    return E_init, atom_map, (np.where(atom_map == central_atom_idx)[0] + 1)


def position_results_AV(
    config, artn, atom_map, positions
) -> [np.array, np.array, np.array, int]:

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

    return (
        min1positions_mapped,
        min2positions_mapped,
        saddlepositions_mapped,
        index_move_mapped,
    )


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
    # map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
    map_type = {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(sorted(set(types)))
    }
    types = [map_type[element]["ref"] for element in types]  # map to integer

    # lammps create system
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    engine.lmp.create_atoms(natoms, ind, types, x)
    # Set masses
    for key in map_type.keys():
        engine.command("mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"]))
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
