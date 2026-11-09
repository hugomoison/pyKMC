"""Tests for the rate constant strategies and the constant strategy."""

import math
from typing import Optional

import pytest

from pykmc import constants
from pykmc.config import RateConstantConfig
from pykmc.rate_constant import RateConstant, RateConstantStrategy


class DoublingStrategy(RateConstantStrategy):
    """Test strategy: twice `k0` for every event."""

    name = "doubling_test_strategy"

    def prefactor(self, nu0: Optional[float] = None) -> float:
        """Return twice `k0`."""
        return 2.0 * self.config.k0


def test_constant_strategy_returns_k0_boltzmann_rate() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = config.k0 * math.exp(-dE / (constants.KB * config.T))
        assert rate_constant.compute_rate(dE) == expected
        assert rate_constant.compute_rate(dE, nu0=22.0) == expected


def test_constant_prefactor_is_k0_whatever_nu0() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    strategy = RateConstantStrategy.create("constant", config=config)

    for nu0 in (None, math.nan, math.inf, 22.0):
        assert strategy.prefactor(nu0) == config.k0


def test_base_compute_rate_uses_the_strategy_prefactor() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    rate_constant = RateConstant.create("doubling_test_strategy", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = 2.0 * config.k0 * math.exp(-dE / (constants.KB * config.T))
        assert rate_constant.compute_rate(dE) == expected


def test_strategy_without_prefactor_is_abstract() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)

    with pytest.raises(TypeError):
        RateConstantStrategy(config)  # type: ignore[abstract]
