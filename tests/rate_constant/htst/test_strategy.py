"""Tests for the htst rate constant strategy."""

import math
from types import SimpleNamespace
from typing import Any

import ase.build
import numpy as np

from pykmc._core.result import Ok
from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import (
    PrefactorOutcome,
    PrefactorRequest,
    RateConstant,
    RateConstantStrategy,
)


def test_htst_strategy_uses_nu0_in_thz() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    rate_constant = RateConstant.create("htst", config=config)

    for dE, nu0 in ((0.0, 10.0), (0.5, 22.0), (1.2, 7.0)):
        expected = nu0 * math.exp(-dE / (PhysicalConstants.kb * config.T))
        assert rate_constant.compute_rate(dE, prefactor=nu0) == expected


def test_htst_prefactor_is_nu0_or_k0() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    strategy = RateConstantStrategy.create("htst", config=config)

    assert strategy.prefactor(22.0) == 22.0
    for nu0 in (None, math.nan, math.inf, -math.inf):
        assert strategy.prefactor(nu0) == config.k0


def test_htst_strategy_without_nu0_equals_constant() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    htst = RateConstant.create("htst", config=config)
    constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        for nu0 in (None, math.nan, math.inf):
            assert htst.compute_rate(dE, prefactor=nu0) == constant.compute_rate(dE)


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


class _Manager:
    """Answer the HTST calls of the strategy like a session pool would."""

    def __init__(self, nu0s: list[float]) -> None:
        self.broadcasts: list[str] = []
        self.jobs: list[dict] = []
        self._nu0s = list(nu0s)

    def broadcast(self, op: str, **kwargs: Any) -> None:
        self.broadcasts.append(op)

    def htst_capability_check(self, **kwargs: Any) -> _Future:
        info = {"force_cutoff": 2.5, "lammps_version": 1, "hessian_dir": "/scratch"}
        return _Future(info)

    def compute_htst_prefactors(self, **kwargs: Any) -> _Future:
        self.jobs.append(kwargs)
        return _Future([Ok(self._nu0s.pop(0)) for _ in kwargs["minima"]])


def _vacancy_hop() -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return a 3x3x3 fcc Ni system with a vacancy, min1, saddle, min2, mover."""
    atoms = ase.build.bulk("Ni", "fcc", a=3.52, cubic=True) * (3, 3, 3)
    positions = atoms.get_positions()
    vacancy = positions[0]
    del atoms[0]
    min1 = atoms.get_positions()
    mover = int(np.argmin(np.linalg.norm(min1 - vacancy, axis=1)))
    min2, saddle = min1.copy(), min1.copy()
    min2[mover] = vacancy
    saddle[mover] = 0.5 * (min1[mover] + vacancy)
    system = SimpleNamespace(
        cell=atoms.cell,
        pbc=atoms.pbc,
        types=atoms.get_chemical_symbols(),
        positions=min1,
    )
    return system, min1, saddle, min2, mover


def _set_up(nu0s: list[float]) -> tuple[RateConstant, _Manager, _Loggers, Any]:
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=False, n_sessions=2, basin=False),
        rateconstant=RateConstantConfig(
            style="htst", k0=10.0, T=300.0, free_radius=3.0
        ),
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full"),
        activevolume=None,
    )
    rate_constant = RateConstant.create("htst", config=config.rateconstant)
    manager, loggers = _Manager(nu0s), _Loggers()
    system = _vacancy_hop()[0]
    rate_constant.setup(config, manager, system, loggers)  # type: ignore[arg-type]
    return rate_constant, manager, loggers, system


def test_htst_setup_checks_every_session_and_records_the_region() -> None:
    rate_constant, manager, _, _ = _set_up([])

    assert rate_constant.per_event_prefactor is True
    assert manager.broadcasts == ["htst_capability_check"]
    # reduced mode: the frozen shell is two force cutoffs
    assert rate_constant.settings_record == "reduced free_radius=3 shell=5"


def test_htst_compute_prefactors_submits_one_job_per_event() -> None:
    rate_constant, manager, loggers, system = _set_up([22.0, 2000.0])
    _, min1, saddle, min2, mover = _vacancy_hop()
    request = PrefactorRequest(
        saddle=saddle,
        minima=[min1, min2],
        centre=mover,
        reference=True,
        names=["reference event 0 (forward)", "reference event 1 (backward)"],
    )
    frozen = PrefactorRequest(
        saddle=saddle, minima=[min1], centre=mover, reference=True, names=["frozen"]
    )

    outcomes = rate_constant.compute_prefactors(
        [request], system, frozen_ids=np.empty(0, dtype=np.int64)
    )
    blocked = rate_constant.compute_prefactors(
        [frozen], system, frozen_ids=np.array([mover])
    )

    # one job for both directions; the event with a frozen atom among the
    # displaced atoms is never sent
    assert len(manager.jobs) == 1
    job = manager.jobs[0]
    assert len(job["minima"]) == 2
    assert job["species"] == ["Ni"]
    assert outcomes == [[PrefactorOutcome(22.0, "ok"), PrefactorOutcome(2000.0, "ok")]]
    assert blocked == [[PrefactorOutcome(None, "PrefactorError.FROZEN_OVERLAP")]]
    warnings = [message for level, message in loggers.lines if level == "warning"]
    assert any("reference event 1 (backward): nu0 = 2000 THz" in m for m in warnings)
    assert any(
        m.startswith("HTST: frozen uses k0: the frozen region") for m in warnings
    )
    infos = [message for level, message in loggers.lines if level == "info"]
    assert "HTST: nu0 computed for 2 of 2 new reference directions" in infos
    assert "HTST: nu0 computed for 0 of 1 new reference directions" in infos
