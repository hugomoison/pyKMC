"""KMC.attach_prefactors: no-op for constant, sets nu0 (and logs fallback) for htst."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pykmc.htst.prefactor import EventPrefactors
from pykmc.kmc import KMC
from pykmc.result import EventSearchOutput


def _event() -> EventSearchOutput:
    p = np.zeros((2, 3))
    return EventSearchOutput(
        central_atom_index=0,
        min1_positions=p,
        saddle_positions=p,
        min2_positions=p,
        dE_forward=0.5,
        dE_backward=0.6,
        move_atom_index=0,
        cell=np.diag([10.0, 10.0, 10.0]),
    )


def _kmc(style: str) -> KMC:
    """Build a KMC instance without running __init__, with the bits attach_prefactors needs."""
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(rateconstant=SimpleNamespace(style=style))
    # System is now owned by the State coordinator; KMC.system is a read-only
    # property delegating to self.state.system.
    kmc.state = SimpleNamespace(
        system=SimpleNamespace(types=["Ni", "Ni"], cell=np.diag([10.0, 10.0, 10.0]))
    )
    kmc.manager = Mock()
    kmc.loggers = Mock()
    return kmc


def test_attach_prefactors_noop_for_constant() -> None:
    """Constant style: no manager call, nu0 stays None."""
    kmc = _kmc("constant")
    ev = _event()
    kmc.attach_prefactors([ev])
    kmc.manager.compute_event_prefactors.assert_not_called()
    assert ev.nu0_forward is None
    assert ev.nu0_backward is None


def test_attach_prefactors_sets_nu0_for_htst() -> None:
    """HTST style: stores nu0_forward/backward from the EventPrefactors result."""
    kmc = _kmc("htst")
    ev = _event()
    fut = Mock()
    fut.result.return_value = EventPrefactors(
        nu0_forward=5.0e12,
        nu0_backward=3.0e12,
        n_free=10,
        n_neg_saddle=1,
        ok_forward=True,
        ok_backward=True,
        reason="",
    )
    kmc.manager.compute_event_prefactors.return_value = [fut]
    kmc.attach_prefactors([ev])
    assert ev.nu0_forward == 5.0e12
    assert ev.nu0_backward == 3.0e12


def test_attach_prefactors_logs_fallback() -> None:
    """HTST fallback (None nu0): event keeps None and a diagnostic is logged."""
    kmc = _kmc("htst")
    ev = _event()
    fut = Mock()
    fut.result.return_value = EventPrefactors(
        nu0_forward=None,
        nu0_backward=None,
        n_free=10,
        n_neg_saddle=0,
        ok_forward=False,
        ok_backward=False,
        reason="bad saddle spectrum",
    )
    kmc.manager.compute_event_prefactors.return_value = [fut]
    kmc.attach_prefactors([ev])
    assert ev.nu0_forward is None
    kmc.loggers.info.assert_called()
