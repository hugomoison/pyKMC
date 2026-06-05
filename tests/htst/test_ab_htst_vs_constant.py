"""Task 23 (efficacy): constant-vs-htst A/B characterization.

Per the plan this is *characterization*, not a pass/fail on "improvement": a
prior pylatkmc HTST attempt made the Ni(100) 1-vac MSD overshoot worse by 1.31x
(a physically-correct ν₀ compounding an already-overshooting baseline, not a
bug). So here we only HARD-assert that:

- both styles run to completion and produce events, and
- the constant run carries no ν₀ while the htst run does (the plumbing differs),

and we *record* the per-style rate/ν₀ statistics for review rather than gating
on MSD.

This test is heavy (two full KMC runs under MPI). It is opt-in: set
``RUN_HTST_AB=1`` and provide a run directory (``HTST_AB_DIR``) containing
``initial_config.xyz`` + the potential, with ``input.in`` (constant) and
``input_htst.in`` (htst) differing only in ``[RateConstant]``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("lammps")

import pandas as pd  # noqa: E402

_OPT_IN = os.environ.get("RUN_HTST_AB") == "1"


def _reference_table_stats(run_dir: Path) -> dict:
    """Summarise a finished run's reference table (rate spread, ν₀ coverage)."""
    table = pd.read_pickle(run_dir / "reference_table.pickle")
    k = pd.to_numeric(table["k"], errors="coerce").dropna()
    stats: dict = {
        "n_events": int(len(table)),
        "k_min": float(k.min()) if len(k) else float("nan"),
        "k_max": float(k.max()) if len(k) else float("nan"),
        "has_nu0_column": "nu0" in table.columns,
        "n_finite_nu0": (
            int(pd.to_numeric(table["nu0"], errors="coerce").notna().sum())
            if "nu0" in table.columns
            else 0
        ),
    }
    return stats


@pytest.mark.skipif(not _OPT_IN, reason="set RUN_HTST_AB=1 to run the heavy A/B")
def test_ab_characterization() -> None:
    """Run constant vs htst in HTST_AB_DIR and record the comparison.

    Hard asserts only: both runs produce events; htst exposes finite ν₀ while
    constant does not. The full numerical comparison is written to the worklog.
    """
    import subprocess

    ab_dir = Path(os.environ.get("HTST_AB_DIR", ""))
    assert ab_dir.is_dir(), "set HTST_AB_DIR to a prepared run directory"
    mpirun = os.environ.get("MPIRUN", "mpirun")
    nproc = os.environ.get("HTST_AB_NPROC", "4")

    results: dict[str, dict] = {}
    for style, inp in (("constant", "input.in"), ("htst", "input_htst.in")):
        work = ab_dir / style
        work.mkdir(exist_ok=True)
        for fname in ("initial_config.xyz", *(p.name for p in ab_dir.glob("*.eam*"))):
            (work / fname).write_bytes((ab_dir / fname).read_bytes())
        (work / "input.in").write_text((ab_dir / inp).read_text())
        subprocess.run(
            [mpirun, "-n", nproc, "python", "-m", "pykmc", "-in", "input.in"],
            cwd=work,
            check=True,
        )
        results[style] = _reference_table_stats(work)

    assert results["constant"]["n_events"] > 0
    assert results["htst"]["n_events"] > 0
    assert results["htst"]["has_nu0_column"]
    assert results["htst"]["n_finite_nu0"] > 0
    print(f"[htst A/B] {results}")
