"""Task 23 (efficacy): ν₀ must actually vary across events.

HTST differs from a constant prefactor only through the spread of ν₀ across
events. If ν₀ collapsed to a near-constant value, ``style = htst`` would be
indistinguishable from ``style = constant``. This guard enriches a real EAM Ni
reference table and asserts a non-trivial coefficient of variation.

Serial (no MPI): reuses the offline enricher on the in-repo validation table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lammps")

import pandas as pd  # noqa: E402

from pykmc.htst.enrich import enrich_dataframe, lammps_forces_factory  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_POT = _ROOT / "basin_testing" / "NiAlH_jea.eam"
_REF = _ROOT / "basin_testing" / "validation" / "reference_table.pickle"


@pytest.mark.skipif(
    not _POT.exists() or not _REF.exists(),
    reason="EAM potential / validation reference table unavailable",
)
def test_nu0_has_spread() -> None:
    """Enriched validation table: ν₀ coefficient of variation exceeds 0.01."""
    df = pd.read_pickle(_REF)
    factory = lammps_forces_factory(str(_POT), "eam/alloy", ["Ni"])
    out = enrich_dataframe(
        df,
        engine_factory=factory,
        elements=["Ni"],
        free_radius=4.0,
        fd_step=0.01,
        nu0_min_hz=1.0e6,
        nu0_max_hz=1.0e15,
    )
    nu0 = pd.to_numeric(out["nu0"], errors="coerce").dropna().to_numpy()
    assert len(nu0) >= 3, "need several events with computed nu0"
    cv = float(nu0.std() / nu0.mean())
    # HTST only departs from a constant prefactor through nu0 spread.
    assert cv > 0.01, f"nu0 nearly constant (cv={cv:.4f}); HTST == constant prefactor"
