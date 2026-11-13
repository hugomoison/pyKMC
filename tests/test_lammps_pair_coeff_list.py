"""Tests for ``[Lammps] pair_coeff`` given as a list.

``pair_style`` is one LAMMPS command, written as in a LAMMPS input (a hybrid
style included). ``pair_coeff`` can be several commands: a value in square
brackets is a comma-separated list, and a value without brackets is one
command, kept as written. Either way the config holds a ``list[str]``.
``LammpsEngine.initialize_potential`` sends one ``pair_style`` command, then
one ``pair_coeff`` command per element, and the active-volume ``reset()``
calls that same method.

The bracket parser is shared with ``nperp_limitation`` and
``r_nperp_limitation``, so their parsing is checked here too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from pydantic import ValidationError

from pykmc.config import Config, LammpsConfig, PartnConfig

_INPUT = Path(__file__).resolve().parent / "data" / "input.in"


def _ini_with(tmp_path: Path, pair_style: str, pair_coeff: str) -> Path:
    """Copy ``tests/data/input.in`` with its ``pair_style`` and ``pair_coeff`` lines replaced."""
    out = []
    for line in _INPUT.read_text().splitlines():
        if line.lstrip().startswith("pair_style"):
            out.append(pair_style)
        elif line.lstrip().startswith("pair_coeff"):
            out.append(pair_coeff)
        else:
            out.append(line)
    path = tmp_path / "input.in"
    path.write_text("\n".join(out) + "\n")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_existing_input_gives_one_command() -> None:
    """An existing input is unchanged: one pair_style string, one pair_coeff command."""
    config = Config.from_ini_file(str(_INPUT))
    assert config.lammps.pair_style == "eam/alloy"
    assert config.lammps.pair_coeff == [
        "* * ./tests/data/Ni_v6_2.0_LKBeland2016.eam Ni"
    ]


def test_command_with_comma_is_kept_whole(tmp_path: Path) -> None:
    """Without brackets, a comma (here in a file name) does not split the command."""
    path = _ini_with(
        tmp_path,
        "pair_style = eam/alloy",
        "pair_coeff = * * ./Cu,revision.eam.alloy Cu",
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.pair_coeff == ["* * ./Cu,revision.eam.alloy Cu"]


def test_bracket_list(tmp_path: Path) -> None:
    """Square brackets give one command per element; pair_style stays as written."""
    path = _ini_with(
        tmp_path,
        "pair_style = hybrid/overlay eam/alloy zbl 1.0 2.0",
        "pair_coeff = [* * eam/alloy my_file.eam Ni , 1 1 zbl 28.0 28.0]",
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.pair_style == "hybrid/overlay eam/alloy zbl 1.0 2.0"
    assert config.lammps.pair_coeff == [
        "* * eam/alloy my_file.eam Ni",
        "1 1 zbl 28.0 28.0",
    ]


def test_bracket_list_on_continuation_lines(tmp_path: Path) -> None:
    """A list can span indented continuation lines."""
    path = _ini_with(
        tmp_path,
        "pair_style = hybrid/overlay eam/alloy zbl 1.0 2.0",
        "pair_coeff = [* * eam/alloy my_file.eam Ni,\n    1 1 zbl 28.0 28.0]",
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.pair_coeff == [
        "* * eam/alloy my_file.eam Ni",
        "1 1 zbl 28.0 28.0",
    ]


def test_python_list() -> None:
    """A list passed from Python is used as given."""
    cfg = LammpsConfig(
        pair_style="hybrid lj/cut 6.0 zbl 3.0 4.0",
        pair_coeff=["1 1 lj/cut 0.52 2.274", "1 2 zbl 28.0 28.0"],
    )
    assert cfg.pair_coeff == ["1 1 lj/cut 0.52 2.274", "1 2 zbl 28.0 28.0"]


@pytest.mark.parametrize("value", ["[]", "[ , ]", [], "none", "None"])
def test_no_pair_coeff_is_rejected(value) -> None:
    """At least one pair_coeff command is required."""
    with pytest.raises(ValidationError):
        LammpsConfig(pair_style="eam/alloy", pair_coeff=value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("[4, 8, 12, 16, -1]", [4, 8, 12, 16, -1]),
        ("4, 8, -1", [4, 8, -1]),  # without brackets, as before
        ("[4, 8", [4, 8]),  # as before
        ("7", [7]),
        ("[]", []),
        ("none", None),
        ([5, 6], [5, 6]),
    ],
)
def test_nperp_limitation_lists(value, expected) -> None:
    """The integer lists keep their previous parsing through the shared parser."""
    cfg = PartnConfig(nperp_limitation=value, r_nperp_limitation=value)
    assert cfg.nperp_limitation == expected
    assert cfg.r_nperp_limitation == expected


def test_nperp_limitation_rejects_non_integers() -> None:
    with pytest.raises(ValidationError):
        PartnConfig(nperp_limitation="[4, x]")


# ---------------------------------------------------------------------------
# Command stream (no LAMMPS run)
# ---------------------------------------------------------------------------


class _Recorder:
    """Stand-in for the ``lammps`` object: record every command string."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, cmd: str) -> None:
        self.commands.append(cmd)

    def close(self) -> None:
        pass


_HYBRID = LammpsConfig(
    pair_style="hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
    pair_coeff="[* * lj/cut 0.52 2.274, 1 1 zbl 28.0 28.0]",
    verbosity=0,
)
_HYBRID_COMMANDS = [
    "pair_style hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
    "pair_coeff * * lj/cut 0.52 2.274",
    "pair_coeff 1 1 zbl 28.0 28.0",
]
_SINGLE = LammpsConfig(
    pair_style="lj/cut 6.0", pair_coeff="* * 0.52 2.274", verbosity=0
)
_SINGLE_COMMANDS = ["pair_style lj/cut 6.0", "pair_coeff * * 0.52 2.274"]


def _recording_engine(cfg: LammpsConfig):
    """A ``LammpsEngine`` whose LAMMPS instance only records commands."""
    pytest.importorskip("lammps")
    from pykmc.engine.lammps import LammpsEngine

    engine = LammpsEngine(config=cfg)
    engine.lmp = _Recorder()
    return engine


@pytest.mark.parametrize(
    "cfg, expected", [(_HYBRID, _HYBRID_COMMANDS), (_SINGLE, _SINGLE_COMMANDS)]
)
def test_engine_initialize_potential(cfg, expected) -> None:
    """One pair_style command, then one pair_coeff command per element."""
    engine = _recording_engine(cfg)
    engine.initialize_potential()
    assert engine.lmp.commands == expected


@pytest.mark.parametrize(
    "cfg, expected", [(_HYBRID, _HYBRID_COMMANDS), (_SINGLE, _SINGLE_COMMANDS)]
)
def test_active_volume_reset_loads_potential_through_engine(cfg, expected) -> None:
    """The active-volume ``reset()`` ends with the engine's own potential commands."""
    pytest.importorskip("pypARTn")
    from pykmc.activevolume import active_volume as av

    engine = _recording_engine(cfg)
    _, map_type = engine.map_types(["Ni"])
    av.reset(engine, np.diag([20.0, 20.0, 20.0]), map_type)
    assert engine.lmp.commands[-len(expected) :] == expected
    assert sum(c.startswith("pair_style") for c in engine.lmp.commands) == 1


# ---------------------------------------------------------------------------
# Real LAMMPS (serial)
# ---------------------------------------------------------------------------

_LJ = LammpsConfig(pair_style="lj/cut 6.0", pair_coeff="* * 0.52 2.274", verbosity=0)
_ZBL = LammpsConfig(pair_style="zbl 3.0 4.0", pair_coeff="1 1 28.0 28.0", verbosity=0)


def _require_serial() -> None:
    pytest.importorskip("lammps")
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial test must run without mpirun")


def _ni_cell(n: int):
    return bulk("Ni", "fcc", a=3.524, cubic=True).repeat([n, n, n])


def _energy(cfg: LammpsConfig) -> float:
    """Potential energy of a 32-atom fcc Ni cell loaded through ``initialize_potential``."""
    from pykmc.engine.lammps import LammpsEngine

    atoms = _ni_cell(2)
    engine = LammpsEngine(config=cfg)
    engine.start()
    try:
        engine.initialize_parameters()
        engine.initialize_system(
            types=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )
        engine.initialize_potential()
        return engine.get_potential_energy()
    finally:
        engine.close()


def test_hybrid_overlay_energy() -> None:
    """The hybrid/overlay energy is the lj/cut energy plus the zbl energy."""
    _require_serial()
    e_hybrid, e_lj, e_zbl = _energy(_HYBRID), _energy(_LJ), _energy(_ZBL)
    assert abs(e_zbl) > 1.0  # the zbl term is not negligible in this cell
    assert e_hybrid == pytest.approx(e_lj + e_zbl, rel=1e-12, abs=1e-10)


@dataclass
class _AVCfg:
    ract: float = 6.0
    rmov: float = 3.0
    AV_debug: bool = False


@dataclass
class _SearchCfg:
    """Config shim for ``partn_search_AV``."""

    lammps: LammpsConfig
    activevolume: _AVCfg = field(default_factory=_AVCfg)


def _active_volume_energy(cfg: LammpsConfig) -> float:
    """Energy of the active-volume crop that ``partn_search_AV`` builds in a 256-atom Ni cell."""
    pytest.importorskip("pypARTn")
    from pykmc.activevolume import active_volume as av
    from pykmc.engine.lammps import LammpsEngine

    atoms = _ni_cell(4)
    engine = LammpsEngine(config=cfg)
    engine.start()
    try:
        av.partn_search_AV(
            engine,
            _SearchCfg(lammps=cfg),
            0,
            atoms.get_positions(),
            np.array(atoms.get_cell()),
            atoms.get_chemical_symbols(),
        )
        engine.lmp.command("run 0")
        return engine.lmp.get_thermo("pe")
    finally:
        engine.close()


def test_active_volume_hybrid_overlay_energy() -> None:
    """In the active volume too, the hybrid energy is the sum of the two sub-styles."""
    _require_serial()
    e_hybrid = _active_volume_energy(_HYBRID)
    e_lj, e_zbl = _active_volume_energy(_LJ), _active_volume_energy(_ZBL)
    assert abs(e_zbl) > 1.0
    assert e_hybrid == pytest.approx(e_lj + e_zbl, rel=1e-12, abs=1e-10)
