"""Tests for PBC-aware helpers in pykmc.utils.geometry."""

import ase.build
import numpy as np
import pytest
from ase.geometry import find_mic

from pykmc.utils.geometry import (
    av_membership,
    minimum_image_distance,
    per_atom_displacement,
)

CELL = np.diag([10.0, 10.0, 10.0])


def test_minimum_image_distance_no_wrap() -> None:
    """Pair well inside the box: plain Euclidean distance."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([4.0, 5.0, 1.0])
    assert minimum_image_distance(a, b, CELL) == pytest.approx(5.0)


def test_minimum_image_distance_wraps_across_boundary() -> None:
    """Pair straddling the boundary: wrapped distance beats the naive one."""
    a = np.array([0.5, 5.0, 5.0])
    b = np.array([9.5, 5.0, 5.0])
    naive = float(np.linalg.norm(b - a))
    wrapped = minimum_image_distance(a, b, CELL)
    assert wrapped == pytest.approx(1.0)
    assert wrapped < naive


def test_minimum_image_distance_matches_per_atom_displacement() -> None:
    """Single-pair helper agrees with the vectorized one on a (1, 3) pair."""
    a = np.array([0.5, 9.7, 2.0])
    b = np.array([9.5, 0.3, 2.4])
    expected = per_atom_displacement(a[None, :].copy(), b[None, :].copy(), CELL)[0]
    assert minimum_image_distance(a, b, CELL) == pytest.approx(float(expected))


@pytest.mark.parametrize("cubic", [True, False])
def test_av_membership_matches_one_atom_at_a_time(cubic: bool) -> None:
    """Same split as measuring each distance alone, ties at a radius included."""
    atoms = ase.build.bulk("Ni", "fcc", a=3.52, cubic=cubic) * (4, 4, 4)
    positions, cell = atoms.get_positions(), atoms.cell
    centre = 5
    distances = np.array(
        [find_mic(pos - positions[centre], cell, pbc=True)[1] for pos in positions]
    )
    # radii exactly at a shell of neighbours, and between shells
    shells = np.unique(np.round(distances, 8))
    for r_inner, r_outer in ((distances[17], distances[40]), (shells[2], shells[4])):
        inner, buffer = av_membership(positions, centre, r_inner, r_outer, cell)
        assert np.array_equal(inner, np.flatnonzero(distances <= r_inner))
        assert np.array_equal(
            buffer, np.flatnonzero((distances > r_inner) & (distances <= r_outer))
        )


def test_av_membership_on_a_tilted_cell_measures_each_atom_alone() -> None:
    """On this cell `find_mic` on all atoms at once finds a shorter image of atom 1."""
    cell = np.array([[18.23, 0.0, 0.0], [-16.35, 9.6, 0.0], [4.7, -4.76, 15.73]])
    positions = np.array([[0.0, 0.0, 0.0], [-13.38, 1.06, 10.41], [10.86, -6.57, 9.66]])
    distances = np.array([find_mic(pos, cell, pbc=True)[1] for pos in positions])
    batched = find_mic(positions, cell, pbc=True)[1]
    assert batched[1] <= 7.38 < distances[1]

    inner, buffer = av_membership(positions, 0, 1.0, 7.38, cell)
    assert np.array_equal(inner, [0])
    assert np.array_equal(
        buffer, np.flatnonzero((distances > 1.0) & (distances <= 7.38))
    )
