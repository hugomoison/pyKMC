"""Tests for the constant rate constant strategy."""

import math

from pykmc.config import PhysicalConstants, RateConstantConfig
from pykmc.rate_constant import RateConstant


def test_constant_strategy_returns_k0_boltzmann_rate() -> None:
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    for dE in (0.0, 0.5, 1.2):
        expected = config.k0 * math.exp(-dE / (PhysicalConstants.kb * config.T))
        assert rate_constant.compute_rate(dE) == expected
        assert rate_constant.compute_rate(dE, nu0=1.0e13) == expected
