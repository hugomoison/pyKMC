"""Tests for the htst rate constant strategy."""

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ase.build
import numpy as np
import pytest
from ase.geometry import find_mic

import pykmc
from pykmc import constants
from pykmc._core.result import Err, Ok
from pykmc.config import ActiveVolume, RateConstantConfig, RegionConfig
from pykmc.rate_constant import RateConstant, RateRequest
from pykmc.rate_constant.strategies.htst.errors import PrefactorError


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


def _set_up(
    nu0s: list[float], active_volume: bool = False, frozen: list[int] | None = None
) -> tuple[RateConstant, _Manager, _Loggers]:
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=active_volume, n_sessions=2),
        rateconstant=RateConstantConfig(
            style="htst", k0=10.0, T=300.0, free_radius=3.0
        ),
        atomicenvironment=SimpleNamespace(atom_coloring_mode="full"),
        activevolume=ActiveVolume(rmov=3.0, ract=5.0),
        frozen_atoms=None if frozen is None else RegionConfig(indices=frozen),
    )
    rate_constant = RateConstant.create("htst", config=config.rateconstant)
    manager, loggers = _Manager(nu0s), _Loggers()
    system = _vacancy_hop()[0]
    rate_constant.setup(config, manager, system, loggers)  # type: ignore[arg-type]
    return rate_constant, manager, loggers


def _event(dE_forward: float = 0.5, dE_backward: float = 0.7) -> list[RateRequest]:
    """Return the two requests of a reference event, as the reference table sends."""
    _, min1, saddle, min2, mover = _vacancy_hop()
    return [
        RateRequest(
            dE=dE_forward,
            initial_positions=min1,
            saddle_positions=saddle,
            central_atom=mover,
        ),
        RateRequest(
            dE=dE_backward,
            initial_positions=min2,
            saddle_positions=saddle,
            central_atom=mover,
        ),
    ]


def test_htst_without_geometry_equals_constant() -> None:
    """A known prefactor is used; a barrier alone gets k0. No setup is needed."""
    config = RateConstantConfig(style="htst", k0=10.0, T=500.0)
    htst = RateConstant.create("htst", config=config)
    constant = RateConstant.create("constant", config=config)
    requests = [RateRequest(dE=0.5), RateRequest(dE=1.2, prefactor=22.0)]

    results = htst.compute(requests)

    for result, expected in zip(results, constant.compute(requests), strict=True):
        assert isinstance(result, Ok)
        assert result.ok_value() == expected.ok_value()
    assert results[0].ok_value().prefactor == config.k0
    assert results[1].ok_value().prefactor == 22.0
    assert htst.compute_rate(0.5) == constant.compute_rate(0.5)
    assert htst.compute([]) == []


def test_htst_before_setup_fails_to_the_k0_backup() -> None:
    config = RateConstantConfig(style="htst", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("htst", config=config)

    results = rate_constant.compute(_event())

    for request, result in zip(_event(), results, strict=True):
        assert isinstance(result, Err)
        error = result.err_value()
        assert error.type == PrefactorError.ENGINE_FAILURE
        assert "setup was not called" in error.message
        assert error.variables["backup"].prefactor == config.k0
        assert error.variables["backup"].k == rate_constant.compute_rate(request.dE)


def test_constant_strategy_needs_no_setup() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    assert rate_constant.engine_extensions == ()
    rate_constant.setup(None, None, None, None)  # type: ignore[arg-type]


def test_htst_setup_checks_every_session() -> None:
    _, manager, loggers = _set_up([])

    assert manager.broadcasts == ["htst_capability_check"]
    # reduced mode: the frozen shell is two force cutoffs
    infos = [message for level, message in loggers.lines if level == "info"]
    assert any("a frozen shell of 5 A (2 x force cutoff 2.5 A" in m for m in infos)


def test_htst_setup_refuses_an_active_volume_without_buffer() -> None:
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=True, n_sessions=2),
        rateconstant=RateConstantConfig(style="htst", k0=10.0, T=300.0),
        activevolume=ActiveVolume(rmov=5.0, ract=5.0),
    )
    rate_constant = RateConstant.create("htst", config=config.rateconstant)
    manager, loggers = _Manager([]), _Loggers()

    with pytest.raises(RuntimeError, match="rmov = 5 A must be smaller than ract"):
        rate_constant.setup(config, manager, _vacancy_hop()[0], loggers)  # type: ignore[arg-type]

    assert manager.broadcasts == []
    assert [level for level, _ in loggers.lines] == ["error"]


def test_htst_compute_submits_one_job_per_saddle() -> None:
    rate_constant, manager, loggers = _set_up([22.0, 2000.0])
    forward, backward = _event()
    known = RateRequest(dE=0.6, prefactor=5.0)

    results = rate_constant.compute([forward, known, backward])

    # one job for both directions of the event, none for the known prefactor
    assert len(manager.jobs) == 1
    job = manager.jobs[0]
    assert len(job["minima"]) == 2
    assert job["species"] == ["Ni"]
    # the region is centred on the central atom of the request, at the saddle
    saddle, mover = forward.saddle_positions, forward.central_atom
    distances = find_mic(saddle - saddle[mover], np.eye(3) * 3 * 3.52, pbc=True)[1]
    assert len(job["saddle"]) == np.count_nonzero(distances <= 3.0 + 5.0)
    assert len(job["displaced"]) == np.count_nonzero(distances <= 3.0)

    assert all(isinstance(result, Ok) for result in results)
    outputs = [result.ok_value() for result in results]
    assert [output.prefactor for output in outputs] == [22.0, 5.0, 2000.0]
    for request, output in zip([forward, known, backward], outputs, strict=True):
        boltzmann = math.exp(-request.dE / (constants.KB * 300.0))
        assert output.k == output.prefactor * boltzmann
    warnings = [message for level, message in loggers.lines if level == "warning"]
    assert any("nu0 = 2000 THz is outside 0.1-1000 THz" in m for m in warnings)
    infos = [message for level, message in loggers.lines if level == "info"]
    assert "HTST: nu0 computed for 2 of 2 new event directions" in infos


def test_htst_frozen_atom_among_the_displaced_uses_the_backup() -> None:
    mover = _vacancy_hop()[4]
    rate_constant, manager, loggers = _set_up([], frozen=[mover])

    results = rate_constant.compute(_event())

    # the event is never sent
    assert manager.jobs == []
    for request, result in zip(_event(), results, strict=True):
        assert isinstance(result, Err)
        error = result.err_value()
        assert error.type == PrefactorError.FROZEN_OVERLAP
        assert error.message.startswith("the frozen region overlapped")
        assert error.variables["backup"].prefactor == 10.0
        assert error.variables["backup"].k == rate_constant.compute_rate(request.dE)
    infos = [message for level, message in loggers.lines if level == "info"]
    assert "HTST: nu0 computed for 0 of 2 new event directions" in infos


def test_htst_active_volume_region_has_the_radii_of_the_active_volume() -> None:
    rate_constant, manager, _ = _set_up([22.0, 23.0], active_volume=True)
    forward, backward = _event()

    results = rate_constant.compute([forward, backward])

    assert [result.ok_value().prefactor for result in results] == [22.0, 23.0]
    (job,) = manager.jobs
    # rmov = 3 and ract = 5 around the central atom of the request, at the minimum
    min1, mover = forward.initial_positions, forward.central_atom
    distances = find_mic(min1 - min1[mover], np.eye(3) * 3 * 3.52, pbc=True)[1]
    assert len(job["saddle"]) == np.count_nonzero(distances <= 5.0)
    assert len(job["displaced"]) == np.count_nonzero(distances <= 3.0)
    assert job["pbc"] == (True, True, True)


def test_import_registers_every_strategy_without_loading_lammps() -> None:
    """`import pykmc` registers htst and leaves LAMMPS unloaded.

    An eager LAMMPS import in the htst package would load LAMMPS in every run,
    and where LAMMPS is missing it would drop the style silently: a strategy
    module that fails to import is skipped. The check runs in a fresh
    interpreter, as other tests load LAMMPS.
    """
    code = (
        "import json, sys\n"
        "import pykmc\n"
        "from pykmc.rate_constant import RateConstantStrategy\n"
        "print(json.dumps({'file': pykmc.__file__,"
        " 'lammps': 'lammps' in sys.modules,"
        " 'registered': sorted(RateConstantStrategy._registry),"
        " 'import_errors': sorted(RateConstantStrategy._import_errors)}))\n"
    )
    root = Path(pykmc.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    found = json.loads(result.stdout.splitlines()[-1])

    assert Path(found["file"]).resolve() == Path(pykmc.__file__).resolve()
    assert not found["lammps"]
    assert {"constant", "htst"} <= set(found["registered"])
    assert found["import_errors"] == []
