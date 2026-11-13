"""Tests for the Hessian region builder."""

from pathlib import Path

import ase.build
import ase.io
import numpy as np
from ase.geometry import find_mic

from pykmc.rate_constant.strategies.htst.region import (
    build_region,
    frozen_overlap,
    wrapped_on_open_axis,
)
from pykmc.utils.geometry import translate

DATA_DIR = Path(__file__).parents[2] / "data"
PBC = (True, True, True)


def _per_atom_distances(positions, center_index, cell):
    """Distances to one atom, measured one atom at a time like `define_AV`."""
    center = positions[center_index]
    return np.array([find_mic(pos - center, cell, pbc=True)[1] for pos in positions])


def _fcc_vacancy_hop():
    """Return min1, saddle, min2 and the cell of a vacancy hop in fcc Ni."""
    atoms = ase.build.bulk("Ni", "fcc", a=3.52, cubic=True) * (4, 4, 4)
    cell = np.asarray(atoms.cell)
    positions = atoms.get_positions()
    vacancy = positions[0]
    min1 = positions[1:]
    mover = int(np.argmin(np.linalg.norm(min1 - vacancy, axis=1)))
    min2 = min1.copy()
    min2[mover] = vacancy
    saddle = min1.copy()
    saddle[mover] = 0.5 * (min1[mover] + vacancy)
    return min1, saddle, min2, cell, mover


def test_av_region_matches_define_av_rule() -> None:
    atoms = ase.io.read(DATA_DIR / "initial_config_Ni_fcc_4000at_monovacancy+sia.xyz")
    positions = atoms.get_positions()

    for center in (0, 3999, 417, 1806, 2950):
        distances = _per_atom_distances(positions, center, atoms.cell)
        for rmov, ract in ((4.0, 6.0), (9.0, 16.0)):
            # the active volume is periodic on every axis, whatever the system is
            for pbc in (PBC, (True, True, False)):
                region = build_region(
                    "av", positions, atoms.cell, pbc, center, rmov, ract
                )

                assert np.array_equal(region.present, np.flatnonzero(distances <= ract))
                assert np.array_equal(
                    region.present[region.displaced],
                    np.flatnonzero(distances <= rmov),
                )
                assert region.pbc == PBC
                assert region.n_zero_modes == 0


def test_av_region_includes_atom_exactly_at_rmov() -> None:
    atoms = ase.io.read(DATA_DIR / "initial_config_Ni_fcc_4000at_monovacancy+sia.xyz")
    positions = atoms.get_positions()
    center = 417
    distances = _per_atom_distances(positions, center, atoms.cell)

    # Each atom in turn sits exactly on a boundary. The atoms of one neighbour
    # shell are at the same distance within rounding, so which of them are inside
    # depends on how the distance is computed.
    for boundary_atom in np.flatnonzero((distances > 0.0) & (distances < 12.0)):
        radius = distances[boundary_atom]
        inside = np.flatnonzero(distances <= radius)

        at_rmov = build_region(
            "av", positions, atoms.cell, PBC, center, radius, radius + 7.0
        )
        at_ract = build_region(
            "av", positions, atoms.cell, PBC, center, 0.5 * radius, radius
        )

        assert boundary_atom in at_rmov.present[at_rmov.displaced]
        assert np.array_equal(at_rmov.present[at_rmov.displaced], inside)
        assert np.array_equal(at_ract.present, inside)


def test_reduced_region_is_centred_on_the_saddle_mover() -> None:
    min1, saddle, min2, cell, mover = _fcc_vacancy_hop()

    region = build_region("reduced", saddle, cell, PBC, mover, 4.0, 9.0)

    distances = find_mic(saddle - saddle[mover], cell, pbc=True)[1]
    assert np.array_equal(
        region.present[region.displaced], np.flatnonzero(distances <= 4.0)
    )
    # A sphere around the position of the mover at a minimum holds other atoms.
    for minimum in (min1, min2):
        distances = find_mic(saddle - minimum[mover], cell, pbc=True)[1]
        assert not np.array_equal(
            region.present[region.displaced], np.flatnonzero(distances <= 4.0)
        )


def test_reduced_present_contains_displaced() -> None:
    _, saddle, _, cell, mover = _fcc_vacancy_hop()

    region = build_region("reduced", saddle, cell, PBC, mover, 4.0, 6.0)

    distances = find_mic(saddle - saddle[mover], cell, pbc=True)[1]
    assert np.array_equal(region.present, np.flatnonzero(distances <= 6.0))
    assert np.array_equal(
        region.present[region.displaced], np.flatnonzero(distances <= 4.0)
    )
    assert 0 < region.displaced.size < region.present.size < len(saddle)
    assert region.n_zero_modes == 0


def test_reduced_region_follows_the_system_pbc() -> None:
    _, saddle, _, cell, _ = _fcc_vacancy_hop()
    # an atom on the low z face: its neighbours across that face are periodic images
    center = int(np.argmin(saddle[:, 2]))
    open_z = (True, True, False)

    periodic = build_region("reduced", saddle, cell, PBC, center, 4.0, 6.0)
    slab = build_region("reduced", saddle, cell, open_z, center, 4.0, 6.0)

    distances = find_mic(saddle - saddle[center], cell, pbc=open_z)[1]
    assert np.array_equal(slab.present, np.flatnonzero(distances <= 6.0))
    assert np.array_equal(
        slab.present[slab.displaced], np.flatnonzero(distances <= 4.0)
    )
    assert slab.pbc == open_z
    assert slab.present.size < periodic.present.size
    assert np.all(np.isin(slab.present, periodic.present))


def test_full_region_has_three_zero_modes() -> None:
    _, saddle, _, cell, _ = _fcc_vacancy_hop()

    region = build_region("full", saddle, cell, (True, True, False))

    assert np.array_equal(region.present, np.arange(len(saddle)))
    assert np.array_equal(region.displaced, np.arange(len(saddle)))
    assert region.pbc == (True, True, False)
    assert region.n_zero_modes == 3


def test_wrapped_on_open_axis_detects_recentring_wrap() -> None:
    slab = ase.build.fcc100("Ni", size=(4, 4, 8), a=3.52, vacuum=4.0)
    cell = np.asarray(slab.cell)
    pbc = (True, True, False)
    start = slab.get_positions()
    region = build_region("full", start, cell, pbc)
    shift = np.array([0.0, 0.0, 0.5 * cell[2, 2]])

    wrapped = translate(start.copy(), shift, cell)
    unwrapped = start + shift

    assert wrapped_on_open_axis(region, (wrapped,), start, cell, pbc)
    assert not wrapped_on_open_axis(region, (unwrapped,), start, cell, pbc)


def test_frozen_overlap_ignores_shell_atoms() -> None:
    _, saddle, _, cell, mover = _fcc_vacancy_hop()
    region = build_region("reduced", saddle, cell, PBC, mover, 4.0, 6.0)
    displaced_ids = region.present[region.displaced]
    shell_ids = np.setdiff1d(region.present, displaced_ids)

    assert frozen_overlap(region, shell_ids[:3]) == 0
    assert frozen_overlap(region, np.array([shell_ids[0], displaced_ids[0]])) == 1
    assert frozen_overlap(region, np.array([], dtype=int)) == 0
