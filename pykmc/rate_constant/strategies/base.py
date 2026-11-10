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
        return self.prefactor() * math.exp(-dE / (constants.KB * self.config.T))

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
        Default: `compute_rate` on the barrier of each request, which cannot
        fail. A strategy that needs the geometry, or sends work to the engines,
        overrides it.

        """
        return [
            Ok(
                RateConstantOutput(
                    k=self.compute_rate(request.dE), prefactor=self.prefactor()
                )
            )
            for request in requests
        ]
