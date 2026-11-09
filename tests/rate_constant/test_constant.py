"""Tests for the rate constant strategies and the constant strategy."""

import math

import pytest

from pykmc import constants
from pykmc.config import RateConstantConfig
from pykmc.rate_constant import RateConstant, RateConstantStrategy
from pykmc.rate_constant.strategies.constant import ConstantConfig


class DoublingStrategy(RateConstantStrategy):
    """Test strategy: twice `k0` for every event."""

    name = "doubling_test_strategy"

    def __init__(self, config: ConstantConfig) -> None:
        self.config = config

    def prefactor(self) -> float:
        """Return twice `k0`."""
        return 2.0 * self.config.k0


def test_constant_strategy_returns_k0_boltzmann_rate() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = config.k0 * math.exp(-dE / (constants.KB * config.T))
        assert rate_constant.compute_rate(dE) == expected


def test_constant_prefactor_is_k0() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    strategy = RateConstantStrategy.create("constant", config=config)

    assert strategy.prefactor() == config.k0


def test_base_compute_rate_uses_the_strategy_prefactor() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=500.0)
    rate_constant = RateConstant.create("doubling_test_strategy", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = 2.0 * config.k0 * math.exp(-dE / (constants.KB * config.T))
        assert rate_constant.compute_rate(dE) == expected


def test_strategy_without_prefactor_is_abstract() -> None:
    with pytest.raises(TypeError):
        RateConstantStrategy()  # type: ignore[abstract]
