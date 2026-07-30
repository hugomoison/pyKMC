"""Check that every pARTn setup path transmits the configured convergence property.

Covers the four ``artn.set`` setup blocks (search + refine, in the manager
operations module and in the ``LammpsEngine`` methods) with a fake ``pypARTn``
module, so no LAMMPS run or MPI launch is needed: a sentinel exception stops
each function at its ``minimize`` command, right after the setup block.
"""

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("lammps")
pytest.importorskip("pypARTn")

import pykmc.engine.lammps as engine_lammps  # noqa: E402
import pykmc.enginemanager.lmpi.lammps_operations as lammps_operations  # noqa: E402
from pykmc.config import PartnConfig  # noqa: E402
from pykmc.engine.lammps import LammpsEngine  # noqa: E402


class _SetupComplete(Exception):
    """Sentinel raised once the setup phase reaches the minimize run."""


class _FakeArtn:
    """Stand-in for a ``pypARTn.artn`` instance that records ``set`` calls."""

    def __init__(self) -> None:
        self.lib = SimpleNamespace(_name="libartn-fake.so")
        self.calls: list[tuple[str, Any]] = []

    def reset_input(self) -> None:
        """Match the pypARTn API; nothing to reset on the fake."""

    def set(self, key: str, value: Any) -> None:
        """Record one parameter transmission."""
        self.calls.append((key, value))


class _FakePypARTn:
    """Stand-in for the ``pypARTn`` module, exposing the created instance."""

    def __init__(self) -> None:
        self.instance = _FakeArtn()

    def artn(self, engine: str) -> _FakeArtn:
        """Return the recording fake instead of a real pARTn handle."""
        return self.instance


class _FakeCommandRunner:
    """Minimal engine/lmp stand-in: aborts at the first minimize command."""

    engine_id = 0

    def command(self, cmd: str) -> None:
        """Raise the sentinel once setup is complete (the minimize run)."""
        if cmd.startswith("minimize"):
            raise _SetupComplete


def _fake_os() -> SimpleNamespace:
    """Build an ``os`` stand-in so stdout redirection touches no real fds."""
    return SimpleNamespace(
        dup=lambda fd: 99,
        dup2=lambda fd, fd2: None,
        open=lambda path, flags: 98,
        close=lambda fd: None,
        devnull="/dev/null",
        O_WRONLY=0,
    )


@pytest.fixture
def config() -> SimpleNamespace:
    """Config stub with a non-default convergence_property."""
    return SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        eventsearch=SimpleNamespace(delr_thr=1.0),
        partn=PartnConfig(convergence_property="norm"),
        frozen_atoms=None,
    )


def _make_engine() -> LammpsEngine:
    """Build a LammpsEngine around the fake LAMMPS handle, without starting it."""
    engine = LammpsEngine(config=SimpleNamespace(verbosity=0), comm=None, engine_id=0)
    engine.lmp = _FakeCommandRunner()
    return engine


def test_manager_search_transmits_convergence_property(
    monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace
) -> None:
    """The manager-side partn_search sends converge_property to pARTn."""
    fake = _FakePypARTn()
    monkeypatch.setattr(lammps_operations, "pypARTn", fake)
    monkeypatch.setattr(lammps_operations, "os", _fake_os())
    with pytest.raises(_SetupComplete):
        lammps_operations.partn_search(_FakeCommandRunner(), config, 0)
    assert ("converge_property", "norm") in fake.instance.calls


def test_manager_refine_transmits_convergence_property(
    monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace
) -> None:
    """The manager-side partn_refine sends converge_property to pARTn."""
    fake = _FakePypARTn()
    monkeypatch.setattr(lammps_operations, "pypARTn", fake)
    with pytest.raises(_SetupComplete):
        lammps_operations.partn_refine(_FakeCommandRunner(), config, 0)
    assert ("converge_property", "norm") in fake.instance.calls


def test_engine_search_transmits_convergence_property(
    monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace
) -> None:
    """LammpsEngine.partn_search sends converge_property to pARTn."""
    fake = _FakePypARTn()
    monkeypatch.setattr(engine_lammps, "pypARTn", fake)
    monkeypatch.setattr(engine_lammps, "os", _fake_os())
    with pytest.raises(_SetupComplete):
        _make_engine().partn_search(config, 0)
    assert ("converge_property", "norm") in fake.instance.calls


def test_engine_refine_transmits_convergence_property(
    monkeypatch: pytest.MonkeyPatch, config: SimpleNamespace
) -> None:
    """LammpsEngine.partn_refine sends converge_property to pARTn."""
    fake = _FakePypARTn()
    monkeypatch.setattr(engine_lammps, "pypARTn", fake)
    with pytest.raises(_SetupComplete):
        _make_engine().partn_refine(config, 0)
    assert ("converge_property", "norm") in fake.instance.calls
