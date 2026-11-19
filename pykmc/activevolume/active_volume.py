import numpy as np
from ase.cell import Cell
from ase.geometry import find_mic
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


def reset(engine, positions, types, cell, species) -> None:
    """Clear the LAMMPS instance and rebuild it with the active volume atoms only.

    The box is rebuilt by the engine itself, as for the full system.
    ``species`` is the full-system species list: the active volume can miss a
    species, and the LAMMPS types must keep the full-system numbering (a
    multi-element ``pair_coeff`` such as ``eam/alloy`` needs every type).
    """
    engine.command("clear")
    engine.initialize_parameters()
    engine.initialize_system(
        types=types,
        positions=positions,
        cell=Cell.new(cell),
        pbc=[True, True, True],
        species=species,
    )
    engine.initialize_potential()
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
    av_positions, av_idx, buffer_idx = define_AV(
        central_atom_idx, positions, cell, rmov, rbuff
    )

    atom_map = np.array(av_idx, dtype=int)
    av_types = np.asarray(type)[atom_map]

    reset(engine, av_positions, av_types, cell, species=type)
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
        E_before = engine.get_potential_energy()
        engine.command("min_style {}".format(config.lammps.min_style))
        engine.command("minimize 1.0e-6 1.0e-8 10 10")
        E_init = engine.get_potential_energy()
        print("Before minimization: ", E_before, "After minimization: ", E_init)
        print("% Difference:", abs((E_before - E_init) / E_init * 100), "%")
    else:
        E_init = engine.get_potential_energy()

    core_idx = []
    core_ids = []
    for i, atom_idx in enumerate(saddle_idx):
        index = int(
            np.where(atom_map == atom_idx)[0].item()
        )  # index in atom map where this value is true
        av_positions[index] = saddle_positions[i]
        core_idx.append(index)  # Atom id
        core_ids.append(index + 1)
    engine.set_positions(av_positions)

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
    atom_map, positions, cell, min1positions, min2positions, saddlepositions
) -> [np.array, np.array, np.array, int]:
    """Map the active volume results of an event search back to the full system.

    ``min1positions``, ``min2positions`` and ``saddlepositions`` are the
    active volume positions returned by pARTn, already converted to the
    frame of the system (``LammpsEngine._positions_from_lammps``). Atom ``i``
    of the active volume is atom ``atom_map[i]`` of the full system; the
    atoms outside the active volume keep their ``positions``.

    Returns
    -------
    min1, min2, saddle : np.ndarray, shape (N, 3)
        Full-system positions at the two minima and at the saddle.
    index_move : int
        Full-system index of the atom that moves the most between the first
        minimum and the saddle.
    """
    # find atom that moves the most; minimum image, as pARTn returns wrapped
    # coordinates and the atom that hops may cross the cell boundary
    _, dist = find_mic(min1positions - saddlepositions, cell, pbc=True)
    index_move_mapped = atom_map[np.argmax(dist)]

    min1positions_mapped = positions.copy()
    min2positions_mapped = positions.copy()
    saddlepositions_mapped = positions.copy()
    min1positions_mapped[atom_map] = min1positions
    min2positions_mapped[atom_map] = min2positions
    saddlepositions_mapped[atom_map] = saddlepositions

    return (
        min1positions_mapped,
        min2positions_mapped,
        saddlepositions_mapped,
        index_move_mapped,
    )
