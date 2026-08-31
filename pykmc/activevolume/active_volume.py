from ase.data import atomic_numbers, atomic_masses
import numpy as np
from ase.geometry import find_mic
from ..system import Configuration
from ..utils.geometry import compute_distances
from ..enginemanager.lmpi.engine_primitives import (
    initialize_parameters,
    initialize_potential,
    set_positions,
    get_potential_energy,
)


def define_AV(params, central_atom_idx: int, configuration):
    positions = configuration.positions
    cell = configuration.cell
    # Defining parameters
    # Radius of whole active volume in Ang
    r_a = params.activevolume.ract  # Ensure AV is larger than topology analysis
    # Defines the radius of atoms that can move.
    r_m = params.activevolume.rmov

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


def reset(engine, params, cell, map_type) -> None:
    """
    Clear lammps instance, preps it for the new sim:
    """

    engine.command("clear")
    initialize_parameters(engine)
    # Create cell
    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    for key in map_type:
        engine.command("mass {} {}".format(map_type[key]["ref"], map_type[key]["mass"]))
    initialize_potential(engine, params)


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


def _setup_AV(
    engine, params, central_atom_idx: int, configuration
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reset the LAMMPS instance and (re)build the active volume around `central_atom_idx`.

    Returns `(atom_map, av_positions, buffer_idx)`: `atom_map` holds each AV
    atom's index into `configuration` (in the same order as the LAMMPS atoms
    `redefine_atoms` just created), `av_positions` their positions, and
    `buffer_idx` the subset of `atom_map` held fixed as the buffer shell.
    """
    types = configuration.types
    map_type = {
        atom_type: {"ref": i + 1, "mass": atomic_masses[atomic_numbers[atom_type]]}
        for i, atom_type in enumerate(sorted(set(types)))
    }
    reset(engine, params, configuration.cell, map_type)
    av_positions, av_idx, buffer_idx = define_AV(
        params, central_atom_idx, configuration
    )

    atom_map = np.array(av_idx, dtype=int)
    type_refs = np.array([map_type[element]["ref"] for element in types])  # map to integer
    av_type = type_refs[atom_map]

    redefine_atoms(engine, av_positions, av_type)
    make_AV(engine, av_idx, buffer_idx)
    return atom_map, av_positions, buffer_idx


def partn_search_AV(
    engine, params, central_atom_idx: int, configuration
) -> tuple[np.ndarray, np.ndarray]:
    atom_map, _, _ = _setup_AV(engine, params, central_atom_idx, configuration)
    return atom_map, np.where(atom_map == central_atom_idx)[0] + 1


def partn_refine_AV(
    engine,
    params,
    central_atom_idx: int,
    configuration,
    saddle_idx,
    saddle_positions,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Receive the system with the central atom index, define an active volume around this atom, then update the positions
    with those for the saddle.

    This was added in order to get the activation energy for an event, as the traditional method does not work for
    Active Volumes.
    """

    atom_map, av_positions, _ = _setup_AV(engine, params, central_atom_idx, configuration)

    if params.activevolume.AV_debug == True:
        E_before = get_potential_energy(engine)
        engine.command("min_style {}".format(params.lammps.min_style))
        engine.command("minimize 1.0e-6 1.0e-8 10 10")
        E_init = get_potential_energy(engine)
        print("Before minimization: ", E_before, "After minimization: ", E_init)
        print("% Difference:", abs((E_before - E_init) / E_init * 100), "%")
    else:
        E_init = get_potential_energy(engine)

    core_idx = np.searchsorted(atom_map, saddle_idx)  # atom_map is sorted ascending
    av_positions[core_idx] = saddle_positions
    core_ids = core_idx + 1
    set_positions(engine, av_positions)

    engine.command("fix 1 all setforce 0.0 0.0 0.0")
    engine.command("run 0")
    engine.command("unfix 1")

    # Want to minimize initially to speed up refinement process
    engine.command(f"group core id {' '.join(map(str, core_ids))}")
    engine.command("fix f_core core setforce 0.0 0.0 0.0")
    engine.command("min_style {}".format(params.lammps.min_style))
    engine.command("minimize {}".format(params.lammps.frz_min))
    engine.command("unfix f_core")

    return E_init, atom_map, (np.where(atom_map == central_atom_idx)[0] + 1)


def position_results_AV(
    params, artn, atom_map, configuration
) -> tuple[Configuration, Configuration, Configuration, int]:
    types = configuration.types
    cell = configuration.cell

    min1positions = artn.extract("tau_min1")
    min2positions = artn.extract("tau_min2")
    saddlepositions = artn.extract("tau_sad")

    # find atom that moves the most (PBC-aware, so an atom crossing the
    # periodic boundary between min1 and saddle isn't mistaken for a small mover)
    dist = compute_distances(
        Configuration(positions=min1positions, types=types, cell=cell),
        Configuration(positions=saddlepositions, types=types, cell=cell),
    )
    index_move_mapped = atom_map[np.argmax(dist)]

    min1_configuration = configuration.copy()
    min1_configuration[atom_map] = min1positions
    min2_configuration = configuration.copy()
    min2_configuration[atom_map] = min2positions
    saddle_configuration = configuration.copy()
    saddle_configuration[atom_map] = saddlepositions

    return min1_configuration, min2_configuration, saddle_configuration, index_move_mapped
