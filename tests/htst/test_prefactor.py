"""Unit tests for pykmc.htst.prefactor (per-event ν₀ orchestrator).

Strategy
--------
- test_success_path_populates_nu0: monkeypatched vineyard -> verifies dataclass
  fields and ok_forward/ok_backward are True.
- test_out_of_bounds_sets_none_with_reason: monkeypatched vineyard returning a
  value outside the acceptance window -> verifies None + reason populated.
- test_failure_path_never_raises: real vineyard on a purely harmonic "saddle"
  (zero negative modes) -> catches ValueError, returns None + reason.
- test_masses_from_types_and_free_region: two Fe atoms close enough that both
  are free -> verifies n_free == 2.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import pykmc.htst.prefactor as pf
from pykmc.htst.prefactor import EventPrefactors, compute_event_prefactors

_CELL = np.diag([20.0, 20.0, 20.0])
_PBC = np.array([True, True, True])
_EQ2 = np.array([[10.0, 10.0, 10.0], [11.5, 10.0, 10.0]])


def _harmonic(stiffness: float, eq: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a harmonic forces callable centred at ``eq``."""

    def fn(pos: np.ndarray) -> np.ndarray:
        return -stiffness * (pos - eq)

    return fn


def test_success_path_populates_nu0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatched vineyard returning 5 THz -> both prefactors populated and ok."""
    monkeypatch.setattr(
        pf, "vineyard_prefactor", lambda h_min, h_sad, n_zero_modes=0: 5.0e12
    )
    res = compute_event_prefactors(
        forces_fn=_harmonic(2.0, _EQ2),
        min1=_EQ2,
        saddle=_EQ2,
        min2=_EQ2,
        types=["Ni", "Ni"],
        central_index=0,
        free_radius=3.0,
        fd_step=1e-3,
        cell=_CELL,
        pbc=_PBC,
        nu0_min_hz=0.0,
        nu0_max_hz=1e30,
        require_one_negative_mode=False,
    )
    assert isinstance(res, EventPrefactors)
    assert res.nu0_forward == 5.0e12 and res.ok_forward
    assert res.nu0_backward == 5.0e12 and res.ok_backward
    assert res.n_free >= 1


def test_out_of_bounds_sets_none_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatched vineyard returning 5 THz rejected by impossible window -> None + reason."""
    monkeypatch.setattr(
        pf, "vineyard_prefactor", lambda h_min, h_sad, n_zero_modes=0: 5.0e12
    )
    res = compute_event_prefactors(
        forces_fn=_harmonic(2.0, _EQ2),
        min1=_EQ2,
        saddle=_EQ2,
        min2=_EQ2,
        types=["Ni", "Ni"],
        central_index=0,
        free_radius=3.0,
        fd_step=1e-3,
        cell=_CELL,
        pbc=_PBC,
        nu0_min_hz=1e30,
        nu0_max_hz=2e30,  # impossible window
        require_one_negative_mode=False,
    )
    assert res.nu0_forward is None and not res.ok_forward
    assert res.reason


def test_failure_path_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real vineyard on a purely harmonic saddle has 0 negative modes -> None + reason."""
    # Real vineyard: a purely harmonic "saddle" has 0 negative modes -> ValueError
    # inside vineyard -> orchestrator catches it and returns None + reason.
    res = compute_event_prefactors(
        forces_fn=_harmonic(2.0, _EQ2),
        min1=_EQ2,
        saddle=_EQ2,
        min2=_EQ2,
        types=["Ni", "Ni"],
        central_index=0,
        free_radius=3.0,
        fd_step=1e-3,
        cell=_CELL,
        pbc=_PBC,
        nu0_min_hz=0.0,
        nu0_max_hz=1e30,
        require_one_negative_mode=True,
    )
    assert res.nu0_forward is None and res.nu0_backward is None
    assert res.reason  # carries the exception text


def test_masses_from_types_and_free_region() -> None:
    """Two Fe atoms 1.5 Å apart with radius 3.0 -> both atoms are free (n_free==2)."""
    # Two Fe atoms 1.5 Angstrom apart, radius 3.0 -> both free.
    res = compute_event_prefactors(
        forces_fn=_harmonic(2.0, _EQ2),
        min1=_EQ2,
        saddle=_EQ2,
        min2=_EQ2,
        types=["Fe", "Fe"],
        central_index=0,
        free_radius=3.0,
        fd_step=1e-3,
        cell=_CELL,
        pbc=_PBC,
        nu0_min_hz=0.0,
        nu0_max_hz=1e30,
        require_one_negative_mode=True,
    )
    assert res.n_free == 2
