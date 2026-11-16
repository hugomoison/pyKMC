"""Tests for the Region facade and the geometric region strategies."""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.atomic_environment import AtomicEnvironment
from pykmc.config import RegionConfig
from pykmc.region import Region, RegionStrategy

POSITIONS = np.array(
    [
        [5.0, 5.0, 5.0],  # 0: center
        [6.5, 5.0, 5.0],  # 1: d = 1.5
        [7.0, 5.0, 5.0],  # 2: d = 2.0 (on the sphere)
        [8.0, 5.0, 5.0],  # 3: d = 3.0
        [5.0, 5.0, 9.0],  # 4: d = 4.0, z above
        [0.5, 5.0, 5.0],  # 5: d = 4.5, 1.5 through the x boundary with L = 6
    ]
)
TYPES = ["Fe", "Fe", "O", "Fe", "C", "Fe"]


def sphere(radius: float = 2.0, center=(5.0, 5.0, 5.0)):
    return SimpleNamespace(center=center, radius=radius)


def test_registered_shapes() -> None:
    assert {"sphere", "shell", "box", "plane"} <= set(RegionStrategy._registry)


def test_unknown_shape_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        Region.create("not_a_shape", config=sphere())


def test_sphere_includes_boundary() -> None:
    region = Region.create("sphere", config=sphere(2.0))
    assert region.select(POSITIONS) == [0, 1, 2]


def test_sphere_invert() -> None:
    region = Region.create("sphere", config=sphere(2.0), invert=True)
    assert region.select(POSITIONS) == [3, 4, 5]


def test_sphere_uses_minimum_image_with_cell() -> None:
    cell = np.diag([6.0, 20.0, 20.0])
    region = Region.create("sphere", config=sphere(2.0))
    # Without the cell, atom 5 is at 4.5; across the x boundary it is at 1.5.
    assert 5 not in region.select(POSITIONS)
    assert 5 in region.select(POSITIONS, cell=cell)


def test_shell() -> None:
    config = SimpleNamespace(center=(5.0, 5.0, 5.0), radius=3.0, inner_radius=2.0)
    assert Region.create("shell", config=config).select(POSITIONS) == [2, 3]


def test_box() -> None:
    config = SimpleNamespace(lo=(4.0, 4.0, 4.0), hi=(7.0, 6.0, 6.0))
    assert Region.create("box", config=config).select(POSITIONS) == [0, 1, 2]


def test_box_wraps_positions_with_cell() -> None:
    config = SimpleNamespace(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    positions = np.array([[10.5, 0.5, 0.5], [5.0, 0.5, 0.5]])
    region = Region.create("box", config=config)
    assert region.select(positions) == []
    assert region.select(positions, cell=np.diag([10.0, 10.0, 10.0])) == [0]


@pytest.mark.parametrize(
    ("side", "expected"), [("above", [4]), ("below", [0, 1, 2, 3, 5])]
)
def test_plane(side, expected) -> None:
    config = SimpleNamespace(normal="z", threshold=6.0, side=side)
    assert Region.create("plane", config=config).select(POSITIONS) == expected


def test_plane_rejects_inside_side() -> None:
    config = SimpleNamespace(normal="z", threshold=6.0, side="inside")
    with pytest.raises(ValueError, match="above"):
        Region.create("plane", config=config)


def test_union_of_types_indices_and_shape() -> None:
    region = Region.create(
        "sphere", config=sphere(1.0), types=["C"], indices=[3, 100, -1]
    )
    # 0 from the sphere, 3 from indices (100 and -1 ignored), 4 from types.
    assert region.select(POSITIONS, TYPES) == [0, 3, 4]


def test_invert_does_not_apply_to_types_and_indices() -> None:
    region = Region.create(
        "sphere", config=sphere(2.0), types=["O"], indices=[1], invert=True
    )
    assert region.select(POSITIONS, TYPES) == [1, 2, 3, 4, 5]


def test_types_require_atom_types() -> None:
    region = Region(types=["Fe"])
    with pytest.raises(ValueError, match="atom_types"):
        region.mask(POSITIONS)


def test_no_criteria_selects_nothing() -> None:
    assert Region().select(POSITIONS, TYPES) == []


@pytest.mark.parametrize(
    ("ini", "expected"),
    [
        (
            {"region_type": "sphere", "region_center": "5 5 5", "region_radius": 2.0},
            [0, 1, 2],
        ),
        (
            {
                "region_type": "sphere",
                "region_center": "5 5 5",
                "region_radius": 2.0,
                "side": "outside",
            },
            [3, 4, 5],
        ),
        (
            {
                "region_type": "shell",
                "region_center": "5 5 5",
                "region_radius": 3.0,
                "region_inner_radius": 2.0,
            },
            [2, 3],
        ),
        (
            {
                "region_type": "plane",
                "region_normal": "z",
                "region_threshold": 6.0,
                "side": "above",
            },
            [4],
        ),
        ({"types": "O C"}, [2, 4]),
        ({"indices": "1 5"}, [1, 5]),
    ],
)
def test_from_config(ini, expected) -> None:
    config = RegionConfig(**ini)
    assert Region.from_config(config).select(POSITIONS, TYPES) == expected


def test_atomic_environment_region_style_delegates_to_region() -> None:
    config = RegionConfig(region_type="sphere", center=[5, 5, 5], radius=2.0)
    ae = AtomicEnvironment(
        style="region", region=config, positions=POSITIONS, atom_types=TYPES
    )
    assert ae.atomic_environment_list == ["in", "in", "in", "out", "out", "out"]
    assert ae.get_atoms_with_id("in") == [0, 1, 2]


def test_sphere_centered_on_atom_follows_it() -> None:
    region = Region.create("sphere", config=SimpleNamespace(center_atom=0, radius=2.0))
    assert region.select(POSITIONS) == [0, 1, 2]
    moved = POSITIONS.copy()
    moved[0] = [8.0, 5.0, 5.0]
    assert region.select(moved) == [0, 1, 2, 3]


def test_shell_centered_on_atom_with_cell() -> None:
    config = SimpleNamespace(center_atom=5, radius=2.0, inner_radius=1.0)
    cell = np.diag([6.0, 20.0, 20.0])
    # Atom 5 at x = 0.5: atoms 0 (x = 5.0) and 3 (x = 8.0) are 1.5 away through
    # the x boundary, atom 2 (x = 7.0) is 0.5 away, inside the hole.
    assert Region.create("shell", config=config).select(POSITIONS, cell=cell) == [0, 3]


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(radius=2.0),
        SimpleNamespace(center=(0.0, 0.0, 0.0), center_atom=0, radius=2.0),
    ],
)
def test_sphere_requires_exactly_one_center(config) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Region.create("sphere", config=config)


def test_center_atom_out_of_range() -> None:
    region = Region.create("sphere", config=SimpleNamespace(center_atom=6, radius=1.0))
    with pytest.raises(IndexError, match="center_atom=6"):
        region.mask(POSITIONS)


def test_config_center_atom() -> None:
    config = RegionConfig(region_type="sphere", region_center_atom=4, region_radius=1.0)
    assert config.center_atom == 4
    assert Region.from_config(config).select(POSITIONS) == [4]


@pytest.mark.parametrize(
    "ini",
    [
        {"region_type": "sphere", "region_radius": 1.0},
        {
            "region_type": "shell",
            "region_center": "0 0 0",
            "region_center_atom": 1,
            "region_radius": 2.0,
            "region_inner_radius": 1.0,
        },
    ],
)
def test_config_requires_exactly_one_center(ini) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RegionConfig(**ini)


def test_config_rejects_negative_center_atom() -> None:
    with pytest.raises(ValueError):
        RegionConfig(region_type="sphere", region_center_atom=-1, region_radius=1.0)
