"""Constant prefactor rate constant strategy."""

from typing import Protocol

from .base import RateConstantStrategy


class ConstantConfig(Protocol):
    """Configuration required by `ConstantStrategy`."""

    k0: float  # ps^-1
    T: float  # K


class ConstantStrategy(RateConstantStrategy):
    r"""Rate constant with the same prefactor `k0` for every event.

    $$
    k = k_{0} \, e^{-\frac{dE}{k_{b}T}}
    $$
    """

    name = "constant"

    def __init__(self, config: ConstantConfig) -> None:
        self.config = config

    def prefactor(self) -> float:
        """Return `[RateConstant] k0`.

        Returns
        -------
        float
            `k0` (ps^-1).

        """
        return self.config.k0
