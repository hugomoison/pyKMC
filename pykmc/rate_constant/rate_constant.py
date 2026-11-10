"""Manages rate constant computation methods."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pykmc._core.result import ErrorInfo, Result

from .result import RateConstantOutput
from .strategies import RateConstantStrategy


@dataclass(frozen=True, eq=False)
class RateRequest:
    """Data required to compute the rate constant.

    Only `dE` is always needed, the configuration is read by the strategies that use it.
    Requests sharing the same `saddle_positions` array object have that saddle
    computed once by such strategies.

    Attributes
    ----------
    dE : float
        The energy barrier from the initial minimum to the saddle (eV).
    initial_positions : np.ndarray, optional
        Positions of the whole system at the initial minimum, shape (N, 3).
    saddle_positions : np.ndarray, optional
        Positions of the whole system at the saddle, shape (N, 3).
    central_atom : int, optional
        Index of the atom the that moves the most during the event. 

    """

    dE: float
    initial_positions: Optional[np.ndarray] = None
    saddle_positions: Optional[np.ndarray] = None
    central_atom: Optional[int] = None


class RateConstant:
    """Compute the rate constant of an event, using a pluggable strategy.

    Parameters
    ----------
    strategy : RateConstantStrategy
        The rate constant strategy to use.

    """

    def __init__(self, strategy: RateConstantStrategy) -> None:
        self._strategy = strategy

    def compute_rate(self, dE: float) -> float:
        """Compute the rate constant of an event using the configured strategy.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).

        Returns
        -------
        float
            The rate constant (ps^-1).

        """
        return self._strategy.compute_rate(dE)

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

        """
        return self._strategy.compute(requests)

    @classmethod
    def create(cls, style: str, **kwargs) -> "RateConstant":
        """Build a RateConstant wired with the requested strategy.

        Parameters
        ----------
        style : str
            Name of the registered strategy (e.g. "constant").
        **kwargs
            Forwarded to the strategy's constructor (e.g. ``config=config.rateconstant``).

        """
        return cls(strategy=RateConstantStrategy.create(style, **kwargs))
