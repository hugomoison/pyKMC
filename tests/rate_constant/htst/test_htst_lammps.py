"""Real-LAMMPS test of the HTST prefactor of a vacancy hop in fcc Ni.

The fixture `tests/data/htst_ni_fcc_vacancy_hop.npz` holds one event of the
2047-atom cell of `examples/Ni_fcc_2047at_monovacancy`, relaxed at fixed box
(forces below 5e-12 eV/A) with the Ni v6 EAM potential. The event was found by
pARTn: a symmetric first-neighbour vacancy hop (moving atom 1423, energy barrier
1.188043 eV). The positions of the minimum, the saddle and the final minimum are
translated rigidly into [0, L). It is loaded with `allow_pickle=False`.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pytest

lammps = pytest.importorskip("lammps")

from mpi4py import MPI  # noqa: E402

from pykmc.config import Config  # noqa: E402
from pykmc.engine import Engine  # noqa: E402
from pykmc.event_table import ActiveEventTable  # noqa: E402
from pykmc.rate_constant.strategies.htst.region import build_region, moving_atom  # noqa: E402
from pykmc.rate_constant.strategies.htst.lammps_extension import (  # noqa: E402
    HTSTLammpsExtension,
)
from pykmc.rate_constant import PrefactorOutcome, RateConstant  # noqa: E402
from pykmc.result import EventRefinementOutput, Ok  # noqa: E402

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


def test_finite_nu0_reaches_the_active_table(tmp_path, monkeypatch) -> None:
    # The scratch engine would write its log into the current directory.
    monkeypatch.chdir(tmp_path)
    event = np.load(DATA_DIR / "htst_ni_fcc_vacancy_hop.npz", allow_pickle=False)
    min1, saddle = event["min1_positions"], event["saddle_positions"]
    cell, dE = event["cell"], float(event["dE_forward"])
    pbc = (True, True, True)
    hessian_dir = tmp_path / "hess"
    input_file = tmp_path / "input.in"
    input_file.write_text(
        INPUT.format(
            potential=DATA_DIR / "Ni_v6_2.0_LKBeland2016.eam", hessian_dir=hessian_dir
        )
    )
    config = Config.from_ini_file(str(input_file))
    engine = Engine.create("lammps", config=config.lammps, comm=MPI.COMM_SELF)
    HTSTLammpsExtension(engine)

    # The shell of fixed atoms is two force cutoffs.
    info = engine.htst_capability_check(config=config, species=["Ni"])
    assert info["force_cutoff"] == 6.5

    mover = moving_atom(min1, saddle, cell, pbc)
    region = build_region("reduced", saddle, cell, pbc, mover, 6.0, 6.0 + 13.0)
    assert mover == 1423
    assert len(region.displaced) == 83

    outcome = engine.compute_htst_prefactors(
        config=config,
        saddle=saddle[region.present],
        minima=[min1[region.present]],
        types=["Ni"] * len(region.present),
        species=["Ni"],
        cell=cell,
        pbc=region.pbc,
        displaced=region.displaced,
        n_zero_modes=region.n_zero_modes,
    )
    assert isinstance(outcome[0], Ok)
    nu0 = outcome[0].ok_value()
    assert abs(nu0 - 22.0389) <= 1e-4

    # A refined event waits for its nu0 with its full-size saddle.
    rate_constant = RateConstant.create("htst", config=config.rateconstant)
    table = ActiveEventTable(config, rate_constant=rate_constant)
    table.add_events(
        EventRefinementOutput(
            central_atom_index=mover,
            saddle_positions=saddle[region.present],
            E_saddle=dE,
            min2_positions=None,
            dE_forward=dE,
            num_reference_event=0,
            refined="T",
            prefactor_status="pending",
            full_saddle=saddle,
        )
    )
    table.set_prefactor(0, PrefactorOutcome(nu0, "ok"))

    constant = RateConstant.create("constant", config=config.rateconstant)
    assert np.isfinite(table.table.at[0, "prefactor"])
    assert table.table.at[0, "prefactor_status"] == "ok"
    assert table.table.at[0, "full_saddle"] is None
    assert table.table.at[0, "k"] == rate_constant.compute_rate(dE, prefactor=nu0)
    assert table.table.at[0, "k"] != constant.compute_rate(dE)

    # No scratch file is left, and the scratch engine wrote no log.
    assert os.listdir(hessian_dir) == []
    assert glob.glob("lammps.log.*-htst") == []
