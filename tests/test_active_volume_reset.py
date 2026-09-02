"""Tests for multi-species support in the active-volume ``reset()``.

``reset()`` used to hard-code ``create_box 1 box``: every active-volume (AV)
event search rebuilt the LAMMPS instance as a one-type box, so a multi-element
``pair_coeff`` (``eam/alloy ... Cr Ni``) was rejected inside ``reset()`` and,
under a pair style that does not check the type count, the second species'
atoms were created into a one-type box and LAMMPS segfaulted at the next
``run 0``. ``reset()`` also emitted no ``mass``, so pair styles that do not set
masses themselves (``lj/cut``, ``mlip``, ``sw``) failed at the first ``run 0``
even for pure Ni.

The fake-engine tests pin the emitted command stream. The real-LAMMPS tests
(serial, no pARTn search) check that the AV type map reproduces the main
engine's integer types, and drive the actual ``partn_search_AV`` entry point
on Ni and NiCr cells under ``lj/cut`` and under the shipped NiFeCr
``eam/alloy`` potential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.cell import Cell
from ase.data import atomic_masses, atomic_numbers

pytest.importorskip("lammps")
pytest.importorskip("pypARTn")

from pykmc.activevolume import active_volume as av  # noqa: E402

_CELL = np.diag([20.0, 20.0, 50.0])
_EAM_FILE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "NiCr_fcc_447at_slab_monovacancy"
    / "Bonny_2013_NiFeCr.eam"
)


class _FakeEngine:
    """Record every LAMMPS command string instead of executing it."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, cmd: str) -> None:
        """Store ``cmd`` verbatim."""
        self.commands.append(cmd)


@dataclass
class _EamCfg:
    """LAMMPS config shim for the fake engine (strings are never parsed)."""

    pair_style: str = "eam/alloy"
    pair_coeff: str = "* * Bonny_2013_NiFeCr.eam Cr Ni"


@dataclass
class _ResetCfg:
    """Config shim carrying only what ``reset()`` reads."""

    lammps: _EamCfg = field(default_factory=_EamCfg)


def _index_of(commands: list[str], prefix: str) -> int:
    """Return the index of the first command that starts with ``prefix``."""
    return next(i for i, c in enumerate(commands) if c.startswith(prefix))


def test_map_types_matches_engine_rule() -> None:
    """``map_types`` follows the engine's alphabetical ``sorted(set(types))`` rule."""
    int_types, map_type = av.map_types(["Ni", "Cr", "Ni", "Fe"])
    assert list(map_type) == ["Cr", "Fe", "Ni"]
    assert [map_type[k]["ref"] for k in map_type] == [1, 2, 3]
    assert map_type["Ni"]["mass"] == atomic_masses[atomic_numbers["Ni"]]
    assert int_types.tolist() == [3, 1, 3, 2]


def test_reset_sets_one_mass_per_species() -> None:
    """``reset()`` emits one ``mass`` per species with the main engine's values."""
    engine = _FakeEngine()
    _, map_type = av.map_types(["Ni", "Cr", "Ni"])
    av.reset(engine, _ResetCfg(), _CELL, map_type)
    masses = [c for c in engine.commands if c.startswith("mass ")]
    assert masses == [
        "mass 1 {}".format(atomic_masses[atomic_numbers["Cr"]]),
        "mass 2 {}".format(atomic_masses[atomic_numbers["Ni"]]),
    ]


def test_reset_orders_box_and_masses_before_potential() -> None:
    """``create_box N`` and ``mass`` precede ``pair_coeff`` (eam/alloy needs the box)."""
    engine = _FakeEngine()
    _, map_type = av.map_types(["Ni", "Cr", "Ni"])
    av.reset(engine, _ResetCfg(), _CELL, map_type)
    i_box = engine.commands.index("create_box 2 box")
    i_mass = _index_of(engine.commands, "mass ")
    i_coeff = _index_of(engine.commands, "pair_coeff")
    assert i_box < i_mass < i_coeff


# ---------------------------------------------------------------------------
# Real LAMMPS (serial)
# ---------------------------------------------------------------------------


@dataclass
class _EngineCfg:
    """Engine config; the ``lj/cut`` default sets no masses, so ``reset()`` has to."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"
    min_style: str = "cg"
    minimize: str = "1e-6 1e-8 1000 10000"
    frz_min: str = "1e-4 1e-6 100 1000"
    verbosity: int = 0


@dataclass
class _AVCfg:
    """Active-volume radii sized for a 4x4x4 FCC cell (14.1 Angstrom)."""

    ract: float = 6.0
    rmov: float = 3.0
    AV_debug: bool = False


@dataclass
class _SearchCfg:
    """Config shim for ``partn_search_AV``."""

    lammps: _EngineCfg = field(default_factory=_EngineCfg)
    activevolume: _AVCfg = field(default_factory=_AVCfg)


def _engine_cfg(potential: str) -> _EngineCfg:
    """Return the engine config for ``potential`` (``lj/cut`` or ``eam/alloy``)."""
    if potential == "eam/alloy":
        if not _EAM_FILE.is_file():
            pytest.skip(f"{_EAM_FILE} not present (examples/ not shipped)")
        return _EngineCfg(pair_style="eam/alloy", pair_coeff=f"* * {_EAM_FILE} Cr Ni")
    return _EngineCfg()


def _require_serial() -> None:
    """Skip under ``mpirun``: these tests drive one serial LAMMPS instance."""
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial test: run without mpirun")


def _fcc_ni_cell(cr_stride: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build a 4x4x4 FCC Ni cell; every ``cr_stride``-th atom is Cr (0 = none)."""
    atoms = bulk("Ni", crystalstructure="fcc", a=3.524, cubic=True)
    atoms = atoms.repeat([4, 4, 4])
    types = atoms.get_chemical_symbols()
    if cr_stride:
        for i in range(0, len(types), cr_stride):
            types[i] = "Cr"
    return types, atoms.get_positions(), np.array(atoms.get_cell())


def test_map_types_matches_lammps_engine_types() -> None:
    """``map_types`` reproduces the integer types the main engine assigns."""
    _require_serial()
    from pykmc.engine.lammps import LammpsEngine

    types, positions, cell = _fcc_ni_cell(5)
    engine = LammpsEngine(config=_EngineCfg(), comm=None)
    engine.start()
    try:
        engine.initialize_parameters()
        engine.initialize_system(
            types=types, positions=positions, cell=Cell(cell), pbc=[True] * 3
        )
        engine.initialize_potential()
        int_types, map_type = av.map_types(types)
        assert engine.lmp.extract_global("ntypes") == len(map_type)
        got = np.ctypeslib.as_array(engine.lmp.gather_atoms("type", 0, 1)).copy()
        np.testing.assert_array_equal(got, int_types)
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("cr_stride", "n_species", "potential"),
    [(0, 1, "lj/cut"), (5, 2, "lj/cut"), (5, 2, "eam/alloy")],
    ids=["Ni-lj", "NiCr-lj", "NiCr-eam"],
)
def test_partn_search_AV_crop_holds_every_species(
    cr_stride: int, n_species: int, potential: str
) -> None:
    """The AV crop is built with the main engine's integer types and masses.

    ``partn_search_AV`` is the real entry point up to (not including) the
    pARTn search, so no plugin load or MPI launch is needed.
    """
    _require_serial()
    from pykmc.engine.lammps import LammpsEngine

    types, positions, cell = _fcc_ni_cell(cr_stride)
    config = _SearchCfg(lammps=_engine_cfg(potential))
    engine = LammpsEngine(config=config.lammps, comm=None)
    engine.start()
    try:
        atom_map, central_id = av.partn_search_AV(
            engine, config, 0, positions, cell, types
        )
        lmp = engine.lmp
        int_types, _ = av.map_types(types)
        expected = int_types[atom_map]
        assert set(expected.tolist()) == set(range(1, n_species + 1))

        assert lmp.extract_global("ntypes") == n_species
        assert lmp.get_natoms() == len(atom_map)
        got = np.ctypeslib.as_array(lmp.gather_atoms("type", 0, 1)).copy()
        np.testing.assert_array_equal(got, expected)
        assert atom_map[central_id[0] - 1] == 0
    finally:
        engine.close()
