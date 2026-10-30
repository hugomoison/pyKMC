"""Constant prefactor rate constant strategy."""

from typing import Optional

from .base import RateConstantStrategy


class ConstantStrategy(RateConstantStrategy):
    r"""Rate constant with the same prefactor `k0` for every event.

    $$
    k = k_{0} \, e^{-\frac{dE}{k_{b}T}}
    $$
    """

    name = "constant"

    def prefactor(self, value: Optional[float] = None) -> float:
        """Return `[RateConstant] k0` (ps^-1).

        Parameters
        ----------
        value : float, optional
            Ignored: every event uses `k0`.

        Returns
        -------
        float
            `k0` (ps^-1).

        """
        return self.config.k0
