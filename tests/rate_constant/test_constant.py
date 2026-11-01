"""Tests for the rate constant strategies and the constant strategy."""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

import pykmc
from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import RateConstant, RateConstantStrategy


class DoublingStrategy(RateConstantStrategy):
    """Test strategy: twice `k0` for every event."""

    name = "doubling_test_strategy"

    def prefactor(self, value: Optional[float] = None) -> float:
        """Return twice `k0`."""
        return 2.0 * self.config.k0


def test_constant_strategy_returns_k0_boltzmann_rate() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = config.k0 * math.exp(-dE / (PhysicalConstants.kb * config.T))
        assert rate_constant.compute_rate(dE) == expected
        assert rate_constant.compute_rate(dE, prefactor=22.0) == expected


def test_constant_prefactor_is_k0_whatever_nu0() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    strategy = RateConstantStrategy.create("constant", config=config)

    for nu0 in (None, math.nan, math.inf, 22.0):
        assert strategy.prefactor(nu0) == config.k0


def test_base_compute_rate_uses_the_strategy_prefactor() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    rate_constant = RateConstant.create("doubling_test_strategy", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = 2.0 * config.k0 * math.exp(-dE / (PhysicalConstants.kb * config.T))
        assert rate_constant.compute_rate(dE) == expected


def test_strategy_without_prefactor_is_abstract() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)

    with pytest.raises(TypeError):
        RateConstantStrategy(config)  # type: ignore[abstract]


def test_constant_strategy_has_no_per_event_prefactor() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    assert rate_constant.per_event_prefactor is False
    assert rate_constant.engine_extensions == ()
    assert rate_constant.settings_record == ""
    rate_constant.setup(None, None, None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        rate_constant.compute_prefactors([], None, None)  # type: ignore[arg-type]


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
