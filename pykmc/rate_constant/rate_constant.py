"""Manages rate constant computation methods."""

from typing import Optional

from .strategies import RateConstantStrategy


class RateConstant:
    """Compute the rate constant of an event, using a pluggable strategy.

    Parameters
    ----------
    strategy : RateConstantStrategy
        The rate constant strategy to use.

    """

    def __init__(self, strategy: RateConstantStrategy) -> None:
        self._strategy = strategy

    def compute_rate(self, dE: float, nu0: Optional[float] = None) -> float:
        """Compute the rate constant of an event using the configured strategy.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        nu0 : float, optional
            Attempt frequency of this event (Hz). `None` or a non-finite value
            means the event has none.

        Returns
        -------
        float
            The rate constant (ps^-1).

        """
        return self._strategy.compute_rate(dE, nu0=nu0)

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
