"""Constant prefactor rate constant strategy."""

import math
from typing import Optional, Protocol

from ...config import PhysicalConstants
from .base import RateConstantStrategy


class ConstantConfig(Protocol):
    """Configuration required by `ConstantStrategy`."""

    k0: float  # ps^-1
    T: float  # K


class ConstantStrategy(RateConstantStrategy):
    """Rate constant with the same prefactor `k0` for every event."""

    name = "constant"

    def __init__(self, config: ConstantConfig) -> None:
        self.config = config

    def compute_rate(self, dE: float, nu0: Optional[float] = None) -> float:
        r"""Compute the rate constant based on the energy barrier and the configured `k0` and `T`.

        It uses the following equation :
        $$
        k0*e^{-\frac{dE}{k_{b}T}}
        $$

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        nu0 : float, optional
            Ignored: every event uses `k0`.

        Returns
        -------
        float
            The rate constant (ps^-1).

        """
        return self.config.k0 * math.exp(-dE / (PhysicalConstants.kb * self.config.T))
