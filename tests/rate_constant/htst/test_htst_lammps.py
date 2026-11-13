"""Real-LAMMPS test of the HTST prefactor of a vacancy hop in fcc Ni.

The fixture `tests/data/htst_ni_fcc_vacancy_hop.npz` holds one event of the
2047-atom cell of `examples/Ni_fcc_2047at_monovacancy`, relaxed at fixed box
(forces below 5e-12 eV/A) with the Ni v6 EAM potential. The event was found by
pARTn: a symmetric first-neighbour vacancy hop (moving atom 1423, energy barrier
1.188043 eV). The positions of the minimum, the saddle and the final minimum are
translated rigidly into [0, L). It is loaded with `allow_pickle=False`.
"""

import glob
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

lammps = pytest.importorskip("lammps")

from ase.cell import Cell  # noqa: E402
from mpi4py import MPI  # noqa: E402

from pykmc import constants  # noqa: E402
from pykmc.config import Config  # noqa: E402
from pykmc.engine import Engine  # noqa: E402
from pykmc.rate_constant import RateConstant, RateRequest  # noqa: E402
from pykmc.rate_constant.strategies.htst.lammps_extension import (  # noqa: E402
    HTSTLammpsExtension,
)
from pykmc.result import Ok  # noqa: E402

DATA_DIR = Path(__file__).parents[2] / "data"


def _has_phonon() -> bool:
    lmp = lammps.lammps(comm=MPI.COMM_SELF, cmdargs=["-log", "none", "-screen", "none"])
    try:
        return "PHONON" in lmp.installed_packages
    finally:
        lmp.close()


pytestmark = [
    pytest.mark.skipif(
        MPI.COMM_WORLD.Get_size() > 1, reason="serial test; run without mpirun"
    ),
    pytest.mark.skipif(not _has_phonon(), reason="LAMMPS has no PHONON package"),
]

INPUT = """[Control]
initial_config = ./initial_config.xyz
n_steps = 1
engine = lammps

[Lammps]
pair_style = eam/alloy
pair_coeff = * * {potential} Ni
verbosity = 0

[AtomicEnvironment]
style = cna/graph
rnei = 3.0
rcut = 6.5

[EventSearch]
style = partn
nsearch = 1

[pARTn]

[RateConstant]
style = htst
k0 = 10
T = 300
free_radius = 6.0
hessian_dir = {hessian_dir}

[PSR]
style = ira

[IRA]
"""


class _Future:
    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


class _Loggers:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, name: str, message: str) -> None:
        self.lines.append(("info", message))

    def warning(self, name: str, message: str) -> None:
        self.lines.append(("warning", message))

    def error(self, name: str, message: str) -> None:
        self.lines.append(("error", message))


class _OneEngineManager:
    """Run every operation on one engine, like a pool of one serial session."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def broadcast(self, op: str, **kwargs: Any) -> None:
        getattr(self._engine, op)(**kwargs)

    def __getattr__(self, op: str) -> Any:
        return lambda **kwargs: _Future(getattr(self._engine, op)(**kwargs))


def test_reference_event_gets_its_nu0(tmp_path, monkeypatch) -> None:
    # The scratch engine would write its log into the current directory.
    monkeypatch.chdir(tmp_path)
    event = np.load(DATA_DIR / "htst_ni_fcc_vacancy_hop.npz", allow_pickle=False)
    min1, saddle = event["min1_positions"], event["saddle_positions"]
    cell, dE = event["cell"], float(event["dE_forward"])
    hessian_dir = tmp_path / "hess"
    input_file = tmp_path / "input.in"
    input_file.write_text(
        INPUT.format(
            potential=DATA_DIR / "Ni_v6_2.0_LKBeland2016.eam", hessian_dir=hessian_dir
        )
    )
    config = Config.from_ini_file(str(input_file))
    rate_constant = RateConstant.create("htst", config=config.rateconstant)
    assert rate_constant.engine_extensions == (HTSTLammpsExtension,)
    engine = Engine.create("lammps", config=config.lammps, comm=MPI.COMM_SELF)
    for extension in rate_constant.engine_extensions:
        extension(engine)

    # The shell of fixed atoms is two force cutoffs.
    info = engine.htst_capability_check(config=config, species=["Ni"])
    assert info["force_cutoff"] == 6.5

    system = SimpleNamespace(
        cell=Cell(cell),
        pbc=(True, True, True),
        types=["Ni"] * len(min1),
        positions=min1,
    )
    loggers = _Loggers()
    rate_constant.setup(config, _OneEngineManager(engine), system, loggers)  # type: ignore[arg-type]

    # moving atom 1423: 83 atoms within free_radius = 6 A of its saddle position
    (result,) = rate_constant.compute(
        [
            RateRequest(
                dE=dE,
                initial_positions=min1,
                saddle_positions=saddle,
                central_atom=1423,
            )
        ]
    )

    assert isinstance(result, Ok), loggers.lines
    output = result.ok_value()
    assert abs(output.prefactor - 22.0389) <= 1e-4
    assert output.k == output.prefactor * math.exp(-dE / (constants.KB * 300.0))
    assert not [message for level, message in loggers.lines if level != "info"]

    # No scratch file is left, and the scratch engine wrote no log.
    assert os.listdir(hessian_dir) == []
    assert glob.glob("lammps.log.*-htst") == []
