"""End-to-end ``style = htst`` run (opt-in, heavy).

Validates the live runtime path: a fresh KMC run computes ν₀ per reference event
(via ``KMC.attach_prefactors`` → the manager fan-out → the engine op) and stores
it in the reference table. This was confirmed manually on a 383-atom Ni(100)
slab (the live run reached ``attach_prefactors`` and produced finite ν₀ for all
searched events); it is encoded here as an opt-in test because it needs MPI, a
prepared run directory, and the pre-existing ``partn_refine(type=...)`` bug in
``refinement.py`` fixed so the loop completes and writes the reference table.

Opt in with ``RUN_HTST_E2E=1`` and ``HTST_E2E_DIR`` pointing at a run directory
that contains ``initial_config.xyz``, the potential, and ``input_htst.in``
(``[RateConstant] style = htst``). Set ``MPIRUN`` / ``HTST_E2E_NPROC`` as needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("lammps")

import pandas as pd  # noqa: E402

_OPT_IN = os.environ.get("RUN_HTST_E2E") == "1"


@pytest.mark.skipif(not _OPT_IN, reason="set RUN_HTST_E2E=1 to run the heavy e2e")
def test_htst_run_writes_nu0_column() -> None:
    """A short style=htst run completes and writes a ν₀-bearing reference table."""
    import subprocess

    run_dir = Path(os.environ.get("HTST_E2E_DIR", ""))
    assert run_dir.is_dir(), "set HTST_E2E_DIR to a prepared run directory"
    mpirun = os.environ.get("MPIRUN", "mpirun")
    nproc = os.environ.get("HTST_E2E_NPROC", "4")

    subprocess.run(
        [mpirun, "-n", nproc, "python", "-m", "pykmc", "-in", "input_htst.in"],
        cwd=run_dir,
        check=True,
    )

    table = pd.read_pickle(run_dir / "reference_table.pickle")
    assert "nu0" in table.columns
    assert len(table) > 0
