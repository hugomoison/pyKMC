"""Base class for rate constant strategies."""

import math
from abc import abstractmethod
from typing import Optional, Protocol

from pykmc import constants
from pykmc._core import Registrable


class RateConstantConfigLike(Protocol):
    """Configuration read by every rate constant strategy."""

    k0: float  # ps^-1 (THz)
    T: float  # K


class RateConstantStrategy(Registrable, root=True):
    r"""Base class for rate constant computation methods.

    Every strategy computes the same functional form,
    $$
    k = A \, e^{-\frac{dE}{k_{b}T}}
    $$
    and differs only in the prefactor `A` returned by `prefactor`.

    Parameters
    ----------
    config : RateConstantConfigLike
        The `[RateConstant]` section of the configuration (`k0`, `T`).

    """

    def __init__(self, config: RateConstantConfigLike) -> None:
        self.config = config

    @abstractmethod
    def prefactor(self, nu0: Optional[float] = None) -> float:
        """Return the pre-exponential factor of one event.

        Parameters
        ----------
        nu0 : float, optional
            Attempt frequency of this event (THz, i.e. ps^-1). `None` or a
            non-finite value (a missing cell of a float table column reads back
            as NaN) means the event has none. A strategy that does not use
            per-event prefactors ignores it.

        Returns
        -------
        float
            The prefactor (ps^-1).

        """

    def compute_rate(self, dE: float, nu0: Optional[float] = None) -> float:
        r"""Compute the rate constant of an event.

        It uses the following equation :
        $$
        k = A \, e^{-\frac{dE}{k_{b}T}}
        $$
        where `A` is `prefactor(nu0)` and `T` is `[RateConstant] T`.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        nu0 : float, optional
            Attempt frequency of this event (THz, i.e. ps^-1), forwarded to
            `prefactor`.

        Returns
        -------
        float
            The rate constant (ps^-1).

        Notes
        -----
        A missing or non-finite `nu0` is not an error: the strategy falls back
        to `[RateConstant] k0`.

        """
        return self.prefactor(nu0) * math.exp(
            -dE / (constants.KB * self.config.T)
        )
