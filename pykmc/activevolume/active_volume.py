from ase import Atoms, Atom
from ase.build import bulk
from lammps import lammps
import pypARTn
import numpy as np
import ctypes
from typing import Protocol
from ase.geometry import find_mic
from ..system import System
from ..config import Config
from ..region import Region, SphereParams


def define_AV(central_atom_idx: int, positions, cell, rmov: float, rbuff: float):
    """Select the atoms of the active volume around `central_atom_idx`.

    The active volume is the sphere of radius ``rmov + rbuff`` around the
    central atom (minimum image). Inside it, atoms at a distance ``<= rmov``
    can move, the others (``rmov < d <= rmov + rbuff``) form the fixed buffer.

    Parameters
    ----------
    central_atom_idx : int
        Index of the atom at the center of the active volume.
    positions : np.ndarray, shape (N, 3)
        Positions of the full system.
    cell : np.ndarray, shape (3, 3)
        Simulation cell.
    rmov : float
        Radius (Angstrom) of the movable atoms.
    rbuff : float
        Thickness (Angstrom) of the fixed buffer around the movable atoms.

    Returns
    -------
    av_positions : np.ndarray, shape (M, 3)
        Positions of the active volume atoms.
    av_idx : np.ndarray, shape (M,)
        Sorted indices of the active volume atoms.
    buffer_idx : np.ndarray
        Sorted indices of the buffer atoms, a subset of `av_idx`.

    """

    # Define AV sphere
    region_av = Region.create(
        "sphere",
        config=SphereParams(center_atom=central_atom_idx, radius=rmov + rbuff),
    )
    av_idx = np.asarray(region_av.select(positions, cell=cell), dtype=int)

    # Distances to the central atom, to find buffer region
    _, distances = find_mic(
        positions[av_idx] - positions[central_atom_idx], cell, pbc=True
    )
    buffer_idx = av_idx[distances > rmov]

    return positions[av_idx], av_idx, buffer_idx


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


class _CommandEngine(Protocol):
    """Minimal engine surface the active-volume box rebuild drives."""

    def command(self, cmd: str) -> None:
        """Run one LAMMPS command string."""
        ...

    def initialize_potential(self) -> None:
        """Send the engine's pair_style and pair_coeff commands."""
        ...


def reset(
    engine: _CommandEngine,
    cell: np.ndarray,
    map_type: dict[str, dict],
) -> None:
    """Clear the LAMMPS instance and rebuild an empty box for the active volume.

    ``map_type`` is the ``{symbol: {"ref", "mass"}}`` map
    ``LammpsEngine.map_types`` built from the full-system types. The box gets
    one LAMMPS atom type per species and one ``mass`` per type, both before
    ``pair_coeff``: a multi-element
    ``pair_coeff`` (``eam/alloy``, ``eam/fs``) needs the full type count to
    exist, and pair styles that do not set masses themselves (``lj/cut``,
    ``mlip``, ``sw``) need them before the first ``run 0``. The potential is
    then loaded by the engine itself, as for the full system.
    """
    engine.command("clear")
    initialize_parameters(engine)
    # Create cell
    xhi, yhi, zhi = cell[0][0], cell[1, 1], cell[2, 2]
    engine.command("region box block 0.0 {} 0.0 {} 0.0 {}".format(xhi, yhi, zhi))
    engine.command("create_box {} box".format(len(map_type)))
    for entry in map_type.values():
        engine.command("mass {} {}".format(entry["ref"], entry["mass"]))
    engine.initialize_potential()


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
    engine, central_atom_idx: int, positions, cell, type, rmov: float, rbuff: float
) -> [np.array, int]:
    """Rebuild the engine box with only the active volume around `central_atom_idx`.

    ``rmov`` is the radius of the movable atoms and ``rbuff`` the thickness of
    the fixed buffer (see :func:`define_AV`).
    """
    int_types, map_type = engine.map_types(type)
    reset(engine, cell, map_type=map_type)
    av_positions, av_idx, buffer_idx = define_AV(
        central_atom_idx, positions, cell, rmov, rbuff
    )

    atom_map = np.array(av_idx, dtype=int)
    av_type = int_types[atom_map]

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
    rmov: float,
    rbuff: float,
    debug: bool = False,
) -> [float, np.array, int]:
    """
    Receive the system with the central atom index, define an active volume around this atom, then update the positions
    with those for the saddle.

    This was added in order to get the activation energy for an event, as the traditional method does not work for
    Active Volumes.

    ``rmov`` and ``rbuff`` are the active volume radii (see :func:`define_AV`).
    With ``debug``, the energy of the active volume is printed before and after
    a short minimization, to check that the active volume is large enough.
    """

    atom_map, central_lammps_id = partn_search_AV(
        engine, central_atom_idx, positions, cell, type, rmov, rbuff
    )
    av_positions = positions[atom_map]

    if debug:
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
            np.where(atom_map == atom_idx)[0].item()
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
    engine.command("minimize {}".format(config.lammps.frz_min))
    engine.command("unfix f_core")

    return E_init, atom_map, central_lammps_id


def position_results_AV(
    config, artn, atom_map, positions, cell
) -> [np.array, np.array, np.array, int]:

    min1positions = artn.extract("tau_min1")
    min2positions = artn.extract("tau_min2")
    saddlepositions = artn.extract("tau_sad")

    # find atom that moves the most; minimum image, as pARTn returns wrapped
    # coordinates and the atom that hops may cross the cell boundary
    _, dist = find_mic(min1positions - saddlepositions, cell, pbc=True)
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
    types, map_type = engine.map_types(types)

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
