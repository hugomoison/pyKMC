"""Tests for ``[Lammps] setup_commands``.

``setup_commands`` holds extra LAMMPS commands that pyKMC sends right after
``pair_style`` and ``pair_coeff``. The motivating case is a ``hybrid/overlay``
potential, which needs one ``pair_coeff`` per sub-style (for example an MTP
plus a tabulated short-range repulsion), while pyKMC sends ``pair_coeff`` as a
single LAMMPS command.

The INI value holds one full command per line and is never split on commas,
so commands such as ``variable v equal max(1,2)`` stay intact.

The tests cover:

- parsing through ``Config.from_ini_file`` and ``LammpsConfig``;
- the command stream of both places that load the potential
  (``LammpsEngine.initialize_potential`` and the active-volume ``reset()``);
- a real serial LAMMPS run where the energy of a ``hybrid/overlay`` potential
  loaded through ``initialize_potential`` equals the sum of its two sub-styles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from pykmc.config import Config, LammpsConfig

_INPUT = Path(__file__).resolve().parent / "data" / "input.in"


def _ini_with(tmp_path: Path, block: str) -> Path:
    """Copy ``tests/data/input.in`` and insert ``block`` after its ``pair_coeff`` line."""
    lines = _INPUT.read_text().splitlines()
    i = next(
        k for k, line in enumerate(lines) if line.lstrip().startswith("pair_coeff")
    )
    lines[i + 1 : i + 1] = block.splitlines()
    path = tmp_path / "input.in"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_setup_commands_absent_is_none() -> None:
    """An input without the key keeps the previous behaviour."""
    config = Config.from_ini_file(str(_INPUT))
    assert config.lammps.setup_commands is None


@pytest.mark.parametrize("block", ["setup_commands =", "setup_commands = None"])
def test_setup_commands_empty_or_none(tmp_path: Path, block: str) -> None:
    """An empty value or ``None`` means no extra command."""
    config = Config.from_ini_file(str(_ini_with(tmp_path, block)))
    assert config.lammps.setup_commands is None


def test_setup_commands_single_line_keeps_commas_and_brackets(tmp_path: Path) -> None:
    """A single-line value is one command; commas and brackets are not separators."""
    path = _ini_with(tmp_path, "setup_commands = variable v equal max(1,2)+c_x[1]")
    config = Config.from_ini_file(str(path))
    assert config.lammps.setup_commands == ["variable v equal max(1,2)+c_x[1]"]


def test_setup_commands_single_command_on_continuation_line(tmp_path: Path) -> None:
    """A single command written on an indented continuation line stays whole."""
    path = _ini_with(tmp_path, "setup_commands =\n    variable v equal max(1,2)")
    config = Config.from_ini_file(str(path))
    assert config.lammps.setup_commands == ["variable v equal max(1,2)"]


def test_setup_commands_one_command_per_line(tmp_path: Path) -> None:
    """Each continuation line is one command, in order; blank lines are dropped."""
    block = (
        "setup_commands =\n"
        "    pair_coeff 1 1 table repulsion.table NLH\n"
        "\n"
        "    variable w equal c_x[1]"
    )
    config = Config.from_ini_file(str(_ini_with(tmp_path, block)))
    assert config.lammps.setup_commands == [
        "pair_coeff 1 1 table repulsion.table NLH",
        "variable w equal c_x[1]",
    ]


def test_setup_commands_python_list() -> None:
    """A list passed from Python is accepted; items are stripped, empty ones dropped."""
    cfg = LammpsConfig(
        pair_style="lj/cut 6.0",
        pair_coeff="* * 0.52 2.274",
        setup_commands=["  pair_coeff 1 1 zbl 28.0 28.0 ", ""],
    )
    assert cfg.setup_commands == ["pair_coeff 1 1 zbl 28.0 28.0"]


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
    """Config object without ``setup_commands`` (older shape, e.g. other tests)."""

    pair_style: str = "lj/cut 6.0"
    pair_coeff: str = "* * 0.52 2.274"


_HYBRID = LammpsConfig(
    pair_style="hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
    pair_coeff="* * lj/cut 0.52 2.274",
    setup_commands=["pair_coeff 1 1 zbl 28.0 28.0", "variable v equal max(1,2)"],
    verbosity=0,
)


def test_engine_sends_setup_commands_after_pair_coeff() -> None:
    """``LammpsEngine.initialize_potential``: pair_style, pair_coeff, then setup_commands."""
    pytest.importorskip("lammps")
    from pykmc.engine.lammps import LammpsEngine

    engine = LammpsEngine(config=_HYBRID)
    engine.lmp = _Recorder()
    engine.initialize_potential()
    assert engine.lmp.commands == [
        "pair_style hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
        "pair_coeff * * lj/cut 0.52 2.274",
        "pair_coeff 1 1 zbl 28.0 28.0",
        "variable v equal max(1,2)",
    ]


def test_engine_without_setup_commands_is_unchanged() -> None:
    """A config object without the attribute sends exactly pair_style and pair_coeff."""
    pytest.importorskip("lammps")
    from pykmc.engine.lammps import LammpsEngine

    engine = LammpsEngine(config=_ShimCfg())
    engine.lmp = _Recorder()
    engine.initialize_potential()
    assert engine.lmp.commands == ["pair_style lj/cut 6.0", "pair_coeff * * 0.52 2.274"]


@dataclass
class _ResetCfg:
    """Config shim carrying only what the active-volume ``reset()`` reads."""

    lammps: object = field(default_factory=lambda: _HYBRID)


def test_active_volume_reset_sends_setup_commands_last() -> None:
    """Active-volume ``reset()`` ends with pair_style, pair_coeff, then setup_commands."""
    pytest.importorskip("lammps")
    pytest.importorskip("pypARTn")
    from pykmc.activevolume import active_volume as av
    from pykmc.engine.lammps import LammpsEngine

    engine = _Recorder()
    _, map_type = LammpsEngine.map_types(["Ni"])
    av.reset(engine, _ResetCfg(), np.diag([20.0, 20.0, 20.0]), map_type)
    assert engine.commands[-4:] == [
        "pair_style hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
        "pair_coeff * * lj/cut 0.52 2.274",
        "pair_coeff 1 1 zbl 28.0 28.0",
        "variable v equal max(1,2)",
    ]


def test_active_volume_reset_without_setup_commands_is_unchanged() -> None:
    """Without the attribute, ``reset()`` still ends with pair_coeff."""
    pytest.importorskip("lammps")
    pytest.importorskip("pypARTn")
    from pykmc.activevolume import active_volume as av
    from pykmc.engine.lammps import LammpsEngine

    engine = _Recorder()
    _, map_type = LammpsEngine.map_types(["Ni"])
    av.reset(engine, _ResetCfg(lammps=_ShimCfg()), np.diag([20.0] * 3), map_type)
    assert engine.commands[-2:] == [
        "pair_style lj/cut 6.0",
        "pair_coeff * * 0.52 2.274",
    ]


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


def test_hybrid_overlay_energy_includes_setup_commands_substyle() -> None:
    """The hybrid energy is the lj/cut energy plus the zbl energy set in setup_commands."""
    pytest.importorskip("lammps")
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial test must run without mpirun")

    hybrid = LammpsConfig(
        pair_style="hybrid/overlay lj/cut 6.0 zbl 3.0 4.0",
        pair_coeff="* * lj/cut 0.52 2.274",
        setup_commands=["pair_coeff 1 1 zbl 28.0 28.0"],
        verbosity=0,
    )
    lj = LammpsConfig(pair_style="lj/cut 6.0", pair_coeff="* * 0.52 2.274", verbosity=0)
    zbl = LammpsConfig(
        pair_style="zbl 3.0 4.0", pair_coeff="1 1 28.0 28.0", verbosity=0
    )

    e_hybrid, e_lj, e_zbl = _energy(hybrid), _energy(lj), _energy(zbl)
    assert abs(e_zbl) > 1.0  # the zbl term is not negligible in this cell
    assert e_hybrid == pytest.approx(e_lj + e_zbl, rel=1e-12, abs=1e-10)
