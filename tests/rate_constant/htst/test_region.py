"""Tests for the Hessian region builder."""

from pathlib import Path
from types import SimpleNamespace

import ase.build
import ase.io
import numpy as np
from ase.geometry import find_mic

from pykmc.config import ActiveVolume, RateConstantConfig
from pykmc.rate_constant.strategies.htst.region import (
    build_region,
    frozen_overlap,
    moving_atom,
    region_record,
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

    forward_center = moving_atom(min1, saddle, cell, PBC)
    backward_center = moving_atom(min2, saddle, cell, PBC)
    forward = build_region("reduced", saddle, cell, PBC, forward_center, 4.0, 9.0)
    backward = build_region("reduced", saddle, cell, PBC, backward_center, 4.0, 9.0)

    assert forward_center == backward_center == mover
    assert np.array_equal(forward.present, backward.present)
    assert np.array_equal(forward.displaced, backward.displaced)
    distances = find_mic(saddle - saddle[mover], cell, pbc=True)[1]
    assert np.array_equal(
        forward.present[forward.displaced], np.flatnonzero(distances <= 4.0)
    )
    # A sphere around the position of the mover at a minimum holds other atoms.
    for minimum in (min1, min2):
        distances = find_mic(saddle - minimum[mover], cell, pbc=True)[1]
        assert not np.array_equal(
            forward.present[forward.displaced], np.flatnonzero(distances <= 4.0)
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


def test_moving_atom_uses_minimum_image() -> None:
    min1, saddle, _, cell, mover = _fcc_vacancy_hop()
    # Atom 5 crosses a periodic face: a raw displacement of almost one cell length.
    crossing = 5
    assert crossing != mover
    min1[crossing, 0] = 0.01
    saddle[crossing, 0] = cell[0, 0] - 0.01

    assert moving_atom(min1, saddle, cell, PBC) == mover
    assert moving_atom(min1, saddle, cell, (False, True, True)) == crossing


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


def test_region_record_format() -> None:
    def config(active_volume, free_radius):
        return SimpleNamespace(
            control=SimpleNamespace(active_volume=active_volume),
            rateconstant=RateConstantConfig(style="htst", free_radius=free_radius),
            activevolume=ActiveVolume(ract=16.0, rmov=9.0),
        )

    assert region_record(config(True, 8.0), None) == "av rmov=9 ract=16"
    assert region_record(config(True, "full"), None) == "av rmov=9 ract=16"
    assert region_record(config(False, 8.0), 13.0) == "reduced free_radius=8 shell=13"
    assert region_record(config(False, "full"), None) == "full"
