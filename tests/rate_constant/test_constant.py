"""Tests for the rate constant strategies and the constant strategy."""

import math

import pytest

from pykmc import constants
from pykmc.config import RateConstantConfig
from pykmc._core.result import Ok
from pykmc.rate_constant import RateConstant, RateConstantStrategy, RateRequest
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


def test_compute_returns_one_ok_per_request_in_order() -> None:
    """`compute` returns one `Ok(k, prefactor)` per request, in the same order."""
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)
    barriers = (1.2, 0.0, 0.5)

    results = rate_constant.compute([RateRequest(dE=dE) for dE in barriers])

    assert len(results) == len(barriers)
    for dE, result in zip(barriers, results, strict=True):
        assert isinstance(result, Ok)
        output = result.ok_value()
        assert output.k == rate_constant.compute_rate(dE)
        assert output.prefactor == config.k0


def test_compute_given_prefactor_overwrites_k0() -> None:
    """A prefactor given in the request replaces `k0`, with the same barrier."""
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)
    dE = 0.5

    (result,) = rate_constant.compute([RateRequest(dE=dE, prefactor=3.0)])

    assert isinstance(result, Ok)
    output = result.ok_value()
    assert output.prefactor == 3.0
    assert output.k == pytest.approx(3.0 / config.k0 * rate_constant.compute_rate(dE))


def test_compute_of_no_request_is_empty() -> None:
    """`compute` of an empty list is an empty list."""
    config = RateConstantConfig(style="constant", k0=10.0, T=300.0)
    rate_constant = RateConstant.create("constant", config=config)

    assert rate_constant.compute([]) == []


def test_strategy_without_prefactor_is_abstract() -> None:
    with pytest.raises(TypeError):
        RateConstantStrategy()  # type: ignore[abstract]
