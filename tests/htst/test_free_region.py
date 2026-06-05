"""Tests for free_region module."""

import numpy as np

from pykmc.htst.free_region import select_free_indices


def test_selects_within_radius_with_pbc() -> None:
    """Test selection of free atoms within radius with periodic boundary conditions."""
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = np.array([True, True, True])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # center (idx 0)
            [1.0, 0.0, 0.0],  # inside
            [9.5, 0.0, 0.0],  # 0.5 away across the periodic boundary -> inside
            [5.0, 0.0, 0.0],  # outside
        ]
    )
    free = select_free_indices(
        positions, center_index=0, radius=2.0, cell=cell, pbc=pbc
    )
    assert set(free.tolist()) == {0, 1, 2}
    assert list(free) == sorted(free)  # sorted, deterministic
