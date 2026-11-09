"""Manages rate constant computation methods."""

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
