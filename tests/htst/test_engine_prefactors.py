"""Serial-LAMMPS smoke tests for the HTST engine ops (no MPI, no potential file).

Tests
-----
- test_get_forces_shape: gather_atoms("f") returns a finite (N,3) array.
- test_compute_event_prefactors_runs_on_engine: end-to-end call on a real
  LAMMPS engine returns an EventPrefactors dataclass without raising.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lammps")

from lammps import lammps  # noqa: E402

from pykmc.enginemanager.lmpi import lammps_operations as ops  # noqa: E402
from pykmc.htst.prefactor import EventPrefactors  # noqa: E402
from pykmc.htst.constants import hz_to_thz  # noqa: E402


class _SerialEngine:
    """Minimal engine shim exposing the .command/.lmp/.rank interface the ops use."""

    def __init__(self, lmp: object) -> None:
        self.lmp = lmp
        self.rank = 0

    def command(self, cmd: str) -> None:
        """Delegate to the underlying LAMMPS instance."""
        self.lmp.command(cmd)


def _build_lj_ni() -> _SerialEngine:
    """Create a 5-atom LJ system in a 12 Å box (no real potential file)."""
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    for cmd in [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "region box block 0 12 0 12 0 12",
        "create_box 1 box",
        # a small cluster around the center
        "create_atoms 1 single 6.0 6.0 6.0",
        "create_atoms 1 single 7.5 6.0 6.0",
        "create_atoms 1 single 6.0 7.5 6.0",
        "create_atoms 1 single 6.0 6.0 7.5",
        "create_atoms 1 single 4.5 6.0 6.0",
        "mass 1 58.69",
        "pair_style lj/cut 5.0",
        "pair_coeff 1 1 0.4 2.3",
        "run 0",
    ]:
        lmp.command(cmd)
    return _SerialEngine(lmp)


class _RC:
    style = "htst"
    free_radius = 5.0
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True


class _Cfg:
    rateconstant = _RC()


def test_get_forces_shape() -> None:
    """get_forces returns a finite (N, 3) array for all atoms."""
    eng = _build_lj_ni()
    f = ops.get_forces(eng)
    assert f.shape[1] == 3
    assert f.shape[0] >= 5
    assert np.isfinite(f).all()


def test_compute_event_prefactors_runs_on_engine() -> None:
    """compute_event_prefactors runs end-to-end and returns an EventPrefactors.

    min1==saddle==min2 (all identical) so there is no real saddle — the
    orchestrator falls back gracefully, but must not raise and must return
    the correct dataclass with n_free >= 1.
    """
    eng = _build_lj_ni()
    pos = ops.get_positions(eng)
    cell = np.diag([12.0, 12.0, 12.0])
    res = ops.compute_event_prefactors(
        eng,
        _Cfg(),
        central_atom_idx=0,
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        types=["Ni"] * pos.shape[0],
        cell=cell,
    )
    # min1==saddle==min2 (all minima) -> no real saddle -> graceful fallback,
    # but the op must run end-to-end on a real engine and return the dataclass.
    assert isinstance(res, EventPrefactors)
    assert res.n_free >= 1


_REPO_ROOT = Path(__file__).resolve().parents[2]
_POTENTIAL = _REPO_ROOT / "basin_testing" / "NiAlH_jea.eam"
_FIXTURE = Path(__file__).resolve().parents[1] / "data" / "htst_ni100_surface_hop.npz"


class _RCXval:
    """RateConstant shim for the cross-validation (smaller free radius)."""

    style = "htst"
    free_radius = 4.0  # keep free atoms inside the 130-atom subset (EAM cutoff ~5.65 A)
    fd_step = 0.01
    nu0_min_THz = 1e-6
    nu0_max_THz = 1e6
    require_one_negative_mode = True


class _CfgXval:
    """Config shim exposing only rateconstant."""

    rateconstant = _RCXval()


def _build_eam_engine(positions: np.ndarray) -> tuple[_SerialEngine, np.ndarray]:
    """Build a serial EAM-Ni engine holding positions (N,3) as type-1 Ni."""
    n = positions.shape[0]
    lo = positions.min(axis=0) - 15.0
    hi = positions.max(axis=0) + 15.0
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("atom_modify map array")
    lmp.command("boundary f f f")
    bounds = " ".join(f"{v:.3f}" for v in (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    lmp.command(f"region box block {bounds} units box")
    lmp.command("create_box 1 box")
    lmp.create_atoms(n, None, [1] * n, positions.astype(float).reshape(-1).tolist())
    lmp.command("mass 1 58.6934")
    lmp.command("pair_style eam/alloy")
    lmp.command(f"pair_coeff * * {_POTENTIAL} Ni")
    lmp.command("run 0")
    return _SerialEngine(lmp), np.diag(hi - lo)


@pytest.mark.skipif(not _POTENTIAL.exists(), reason="NiAlH_jea.eam unavailable")
@pytest.mark.skipif(not _FIXTURE.exists(), reason="surface-hop fixture unavailable")
def test_ni100_surface_1nn_prefactor_matches_analysis_canon() -> None:
    """Cross-validate pyKMC nu0 for the Ni(100) surface_1NN hop vs analysis canon.

    The analysis-side toolchain (apps/PyKMC_Analysis/Analysis/HTST.md) reports
    nu0 = 13.1 THz for surface_1NN_inplane on NiAlH_jea.eam. Computing nu0 on a
    real EAM surface-hop saddle (Ea ~= 0.60 eV) through the committed engine op
    must land in the same physical window, validating the whole Vineyard path.
    """
    data = np.load(_FIXTURE)
    init = data["initial_positions"]
    sad = data["saddle_positions"]
    fin = data["final_positions"]
    move = int(data["move_atom_idx"])
    n = int(data["n_atoms"])

    eng, cell = _build_eam_engine(sad)
    res = ops.compute_event_prefactors(
        eng,
        _CfgXval(),
        central_atom_idx=move,
        min1_positions=init,
        saddle_positions=sad,
        min2_positions=fin,
        types=["Ni"] * n,
        cell=cell,
    )

    assert res.ok_forward, f"forward nu0 failed: {res.reason}"
    assert res.nu0_forward is not None
    nu0_thz = hz_to_thz(res.nu0_forward)
    # canon 13.1 THz; ~12.6 THz computed at free_radius=4 A. Window rules out
    # non-physical values while tolerating the radius/geometry difference.
    assert 8.0 <= nu0_thz <= 20.0, f"nu0={nu0_thz:.2f} THz outside physical window"
    assert res.n_free >= 5
