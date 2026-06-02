"""System element typing for LAMMPS-dump configs (no embedded species).

A LAMMPS dump (``ITEM: TIMESTEP`` …) carries only integer atom *types*, no
chemical species. ASE maps integer type ``i`` to atomic number ``i`` (type 1 →
"H"), which silently mislabels e.g. a Ni system as Hydrogen and corrupts any
mass-dependent physics (notably HTST ν₀). When the pair_coeff element list is
known, ``System.create_from_file`` should map LAMMPS types → those elements.
"""

from __future__ import annotations

from pykmc.system import System

_DUMP = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z
1 1 1.0 1.0 1.0
2 1 2.0 1.0 1.0
3 1 1.0 2.0 1.0
"""


def _write_dump(tmp_path: object) -> str:
    """Write a 3-atom (type 1) LAMMPS dump named .xyz, as in the JB_Test case."""
    p = tmp_path / "initial_config.xyz"
    p.write_text(_DUMP)
    return str(p)


def test_dump_maps_to_pair_coeff_element(tmp_path: object) -> None:
    """With pair_coeff elements, dump type 1 maps to the right element (Ni), not H."""
    system = System.create_from_file(_write_dump(tmp_path), elements=["Ni"])
    assert set(system.types) == {"Ni"}
    assert len(system.types) == 3


def test_dump_without_elements_keeps_ase_default(tmp_path: object) -> None:
    """Backward compatible: no elements supplied → ASE default (type 1 → H)."""
    system = System.create_from_file(_write_dump(tmp_path))
    assert len(system.types) == 3
    assert set(system.types) == {"H"}


def test_extxyz_species_not_remapped(tmp_path: object) -> None:
    """A config that already carries the right species is left untouched."""
    p = tmp_path / "ni.xyz"
    p.write_text(
        '3\n'
        'Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Ni 1.0 1.0 1.0\nNi 2.0 1.0 1.0\nNi 1.0 2.0 1.0\n"
    )
    system = System.create_from_file(str(p), elements=["Ni"])
    assert set(system.types) == {"Ni"}
