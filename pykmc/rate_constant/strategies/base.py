"""Base class for rate constant strategies."""

import math
from abc import abstractmethod

from pykmc import constants
from pykmc._core import Registrable


class RateConstantStrategy(Registrable, root=True):
    r"""Base class for rate constant computation methods.

    Every strategy computes the same functional form,
    $$
    k = k_{0} \, e^{-\frac{dE}{k_{b}T}}
    $$
    and differs only in the prefactor `k0` returned by `prefactor`.

    A concrete strategy declares its own config `Protocol` and stores it in
    `self.config`, which must expose the temperature `T` (K).
    """

    @abstractmethod
    def prefactor(self) -> float:
        """Return the pre-exponential factor `k0`.

        Returns
        -------
        float
            The prefactor (ps^-1).

        """

    def compute_rate(self, dE: float) -> float:
        r"""Compute the rate constant of an event.

        It uses the following equation :
        $$
        k = k_{0} \, e^{-\frac{dE}{k_{b}T}}
        $$
        where `k0` is `prefactor()` and `T` is `[RateConstant] T`.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).

        Returns
        -------
        float
            The rate constant (ps^-1).

        """
        return self.prefactor() * math.exp(-dE / (constants.KB * self.config.T))
