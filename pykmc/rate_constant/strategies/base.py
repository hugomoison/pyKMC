"""Base class for rate constant strategies."""

from abc import abstractmethod
from typing import Optional

from pykmc._core import Registrable


class RateConstantStrategy(Registrable, root=True):
    """Base class for rate constant computation methods.

    A strategy turns the energy barrier of an event into its rate constant.
    """

    @abstractmethod
    def compute_rate(self, dE: float, nu0: Optional[float] = None) -> float:
        """Compute the rate constant of an event.

        Parameters
        ----------
        dE : float
            The energy barrier (eV).
        nu0 : float, optional
            Attempt frequency of this event (Hz). `None` or a non-finite value
            (a missing cell of a float table column reads back as NaN) means the
            event has none, and the strategy uses `[RateConstant] k0` (ps^-1).
            A strategy that does not use per-event prefactors ignores it.

        Returns
        -------
        float
            The rate constant (ps^-1).

        Notes
        -----
        A missing or non-finite `nu0` is not an error: the strategy falls back
        to `k0`.

        """
