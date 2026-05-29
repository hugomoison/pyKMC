"""Serial-LAMMPS smoke tests for the HTST engine ops (no MPI, no potential file).

Tests
-----
- test_get_forces_shape: gather_atoms("f") returns a finite (N,3) array.
- test_compute_event_prefactors_runs_on_engine: end-to-end call on a real
  LAMMPS engine returns an EventPrefactors dataclass without raising.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lammps")

from lammps import lammps  # noqa: E402

from pykmc.enginemanager.lmpi import lammps_operations as ops  # noqa: E402
from pykmc.htst.prefactor import EventPrefactors  # noqa: E402


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
