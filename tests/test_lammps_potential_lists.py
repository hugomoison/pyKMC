"""Tests for list values of ``[Lammps] pair_style`` and ``pair_coeff``.

A single value keeps the previous behaviour. A value in square brackets is a
comma-separated list:

- a ``pair_style`` list becomes one ``pair_style hybrid/overlay`` command;
- a ``pair_coeff`` list gives one ``pair_coeff`` command per element, in order.

Both places that load the potential (``LammpsEngine.initialize_potential`` and
the active-volume ``initialize_potential``, called by ``reset()``) use the same
helper, ``lammps_potential_commands``.

The tests cover:

- parsing through ``Config.from_ini_file`` and ``LammpsConfig``;
- the commands built by ``lammps_potential_commands``;
- the command stream of both places that load the potential;
- a real serial LAMMPS run where the energy of a ``hybrid/overlay`` potential
  given as lists equals the sum of its two sub-styles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from pydantic import ValidationError

from pykmc.config import Config, LammpsConfig, lammps_potential_commands

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


def test_single_values_are_unchanged() -> None:
    """An existing input keeps plain strings and gives the same two commands as before."""
    config = Config.from_ini_file(str(_INPUT))
    assert config.lammps.pair_style == "eam/alloy"
    assert config.lammps.pair_coeff == "* * ./tests/data/Ni_v6_2.0_LKBeland2016.eam Ni"
    assert lammps_potential_commands(config.lammps) == [
        "pair_style eam/alloy",
        "pair_coeff * * ./tests/data/Ni_v6_2.0_LKBeland2016.eam Ni",
    ]


def test_single_value_with_comma_stays_one_string() -> None:
    """Only a value in square brackets is split; commas elsewhere are kept."""
    cfg = LammpsConfig(pair_style="table linear 1000", pair_coeff="1 1 a,b.table X")
    assert cfg.pair_coeff == "1 1 a,b.table X"


def test_bracket_lists(tmp_path: Path) -> None:
    """Square brackets give lists; elements are stripped."""
    path = _ini_with(
        tmp_path,
        "pair_style = [eam/alloy , zbl 1.0 2.0]",
        "pair_coeff = [* * eam/alloy my_file.eam Ni, 1 1 zbl 28.0 28.0]",
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.pair_style == ["eam/alloy", "zbl 1.0 2.0"]
    assert config.lammps.pair_coeff == [
        "* * eam/alloy my_file.eam Ni",
        "1 1 zbl 28.0 28.0",
    ]


def test_bracket_list_on_continuation_lines(tmp_path: Path) -> None:
    """A list can span indented continuation lines."""
    path = _ini_with(
        tmp_path,
        "pair_style = [eam/alloy, zbl 1.0 2.0]",
        "pair_coeff = [* * eam/alloy my_file.eam Ni,\n    1 1 zbl 28.0 28.0]",
    )
    config = Config.from_ini_file(str(path))
    assert config.lammps.pair_coeff == [
        "* * eam/alloy my_file.eam Ni",
        "1 1 zbl 28.0 28.0",
    ]


def test_python_list() -> None:
    """A list passed from Python is accepted; items are stripped, empty ones dropped."""
    cfg = LammpsConfig(
        pair_style=[" lj/cut 6.0", "zbl 3.0 4.0 "],
        pair_coeff=["* * lj/cut 0.52 2.274", " 1 1 zbl 28.0 28.0 ", ""],
    )
    assert cfg.pair_style == ["lj/cut 6.0", "zbl 3.0 4.0"]
    assert cfg.pair_coeff == ["* * lj/cut 0.52 2.274", "1 1 zbl 28.0 28.0"]


@pytest.mark.parametrize("value", ["[]", "[ , ]", []])
def test_empty_list_is_rejected(value) -> None:
    """An empty list would send no pair_style or pair_coeff, so it is an error."""
    with pytest.raises(ValidationError):
        LammpsConfig(pair_style="eam/alloy", pair_coeff=value)


# ---------------------------------------------------------------------------
# Commands built by the helper
# ---------------------------------------------------------------------------


def test_lists_give_hybrid_overlay_and_one_pair_coeff_per_element() -> None:
    """The lists need not have the same length (one zbl line per pair of types)."""
    cfg = LammpsConfig(
        pair_style=["eam/alloy", "zbl 1.0 2.0"],
        pair_coeff=[
            "* * eam/alloy CuNi.eam.alloy Cu Ni",
            "1 1 zbl 29.0 29.0",
            "1 2 zbl 29.0 28.0",
            "2 2 zbl 28.0 28.0",
        ],
    )
    assert lammps_potential_commands(cfg) == [
        "pair_style hybrid/overlay eam/alloy zbl 1.0 2.0",
        "pair_coeff * * eam/alloy CuNi.eam.alloy Cu Ni",
        "pair_coeff 1 1 zbl 29.0 29.0",
        "pair_coeff 1 2 zbl 29.0 28.0",
        "pair_coeff 2 2 zbl 28.0 28.0",
    ]


def test_single_pair_style_with_pair_coeff_list() -> None:
    """A hybrid style other than hybrid/overlay can still be written by hand."""
    cfg = LammpsConfig(
        pair_style="hybrid lj/cut 6.0 zbl 3.0 4.0",
        pair_coeff=["1 1 lj/cut 0.52 2.274", "1 2 zbl 28.0 28.0"],
    )
    assert lammps_potential_commands(cfg) == [
        "pair_style hybrid lj/cut 6.0 zbl 3.0 4.0",
        "pair_coeff 1 1 lj/cut 0.52 2.274",
        "pair_coeff 1 2 zbl 28.0 28.0",
    ]


# ---------------------------------------------------------------------------
# Command stream (no LAMMPS run)
# ---------------------------------------------------------------------------


class _Recorder:
    """Record every LAMMPS command string instead of executing it."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, cmd: str) -> None:
        self.commands.append(cmd)

    def close(self) -> None:
        pass


@dataclass
class _ShimCfg:
    """Plain config object with string values (the shape other tests use)."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"


_HYBRID = LammpsConfig(
    pair_style=["lj/cut 6.0", "zbl 3.0 4.0"],
    pair_coeff=["* * lj/cut 0.52 2.274", "1 1 zbl 28.0 28.0"],
    verbosity=0,
)
_HYBRID_COMMANDS = [
    "pair_style hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
    "pair_coeff * * lj/cut 0.52 2.274",
    "pair_coeff 1 1 zbl 28.0 28.0",
]


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (_HYBRID, _HYBRID_COMMANDS),
        (_ShimCfg(), ["pair_style lj/cut 6.0", "pair_coeff * * 0.52 2.274"]),
    ],
)
def test_engine_initialize_potential(cfg, expected) -> None:
    """``LammpsEngine.initialize_potential`` sends exactly the helper's commands."""
    pytest.importorskip("lammps")
    from pykmc.engine.lammps import LammpsEngine

    engine = LammpsEngine(config=cfg)
    engine.lmp = _Recorder()
    engine.initialize_potential()
    assert engine.lmp.commands == expected


@dataclass
class _ResetCfg:
    """Config shim carrying only what the active-volume ``reset()`` reads."""

    lammps: object = field(default_factory=lambda: _HYBRID)


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (_HYBRID, _HYBRID_COMMANDS),
        (_ShimCfg(), ["pair_style lj/cut 6.0", "pair_coeff * * 0.52 2.274"]),
    ],
)
def test_active_volume_reset_ends_with_potential(cfg, expected) -> None:
    """Active-volume ``reset()`` ends with the same commands as the engine."""
    pytest.importorskip("lammps")
    pytest.importorskip("pypARTn")
    from pykmc.activevolume import active_volume as av
    from pykmc.engine.lammps import LammpsEngine

    engine = _Recorder()
    _, map_type = LammpsEngine.map_types(["Ni"])
    av.reset(engine, _ResetCfg(lammps=cfg), np.diag([20.0, 20.0, 20.0]), map_type)
    assert engine.commands[-len(expected) :] == expected


# ---------------------------------------------------------------------------
# Real LAMMPS (serial)
# ---------------------------------------------------------------------------


def _energy(cfg: LammpsConfig) -> float:
    """Potential energy of a 32-atom fcc Ni cell loaded through ``initialize_potential``."""
    from pykmc.engine.lammps import LammpsEngine

    atoms = bulk("Ni", "fcc", a=3.524, cubic=True).repeat([2, 2, 2])
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


def test_hybrid_overlay_energy_from_lists() -> None:
    """The hybrid energy given as lists is the lj/cut energy plus the zbl energy."""
    pytest.importorskip("lammps")
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial test must run without mpirun")

    lj = LammpsConfig(pair_style="lj/cut 6.0", pair_coeff="* * 0.52 2.274", verbosity=0)
    zbl = LammpsConfig(
        pair_style="zbl 3.0 4.0", pair_coeff="1 1 28.0 28.0", verbosity=0
    )

    e_hybrid, e_lj, e_zbl = _energy(_HYBRID), _energy(lj), _energy(zbl)
    assert abs(e_zbl) > 1.0  # the zbl term is not negligible in this cell
    assert e_hybrid == pytest.approx(e_lj + e_zbl, rel=1e-12, abs=1e-10)
