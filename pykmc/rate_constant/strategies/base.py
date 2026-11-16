"""Base class for rate constant strategies."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING

from pykmc import constants
from pykmc._core import Registrable
from pykmc._core.result import ErrorInfo, Ok, Result

from ..result import RateConstantOutput

if TYPE_CHECKING:
    from ..rate_constant import RateRequest


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
        return self.prefactor() * self._boltzmann_factor(dE)

    def compute(
        self, requests: list[RateRequest]
    ) -> list[Result[RateConstantOutput, ErrorInfo]]:
        """Compute the rate constants of several events in one call.

        Parameters
        ----------
        requests : list[RateRequest]
            The required data.

        Returns
        -------
        list[Result[RateConstantOutput, ErrorInfo]]
            One result per request, in the same order: `Ok` with the rate
            constant and its prefactor, or `Err` with the reason it could not be
            computed and a backup computed with `k0`.

        Notes
        -----
        Default: the prefactor of the request when given, `prefactor()`
        otherwise, times the Boltzmann factor of its barrier; it cannot fail. A
        strategy that needs the geometry, or sends work to the engines,
        overrides it.

        """
        results = []
        for request in requests:
            prefactor = (
                self.prefactor() if request.prefactor is None else request.prefactor
            )
            k = prefactor * self._boltzmann_factor(request.dE)
            results.append(Ok(RateConstantOutput(k=k, prefactor=prefactor)))
        return results

    def _boltzmann_factor(self, dE: float) -> float:
        r"""Return $e^{-\frac{dE}{k_{b}T}}$, with `T` the `[RateConstant] T`."""
        return math.exp(-dE / (constants.KB * self.config.T))
