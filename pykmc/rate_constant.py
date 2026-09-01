"""Module defining function used to compute the rate constant."""

from .parameters import PhysicalConstants, Parameters
import math as m


def compute_rate_Eyring(dE: float, params: Parameters) -> float:
    r"""Compute the rate constant based on the energy barrier and parameters in the configuration.

    It uses the following equation :
    $$
    k0*e^{-\frac{dE}{k_{b}T}}
    $$

    Parameters
    ----------
    dE : float
        The energy barrier.
    params : Parameters
        The configuration of the simulation.

    Returns
    -------
    float
        the rate constant.

    """
    p = PhysicalConstants()
    T = params.rateconstant.T
    k0 = params.rateconstant.k0
    return k0 * m.exp(-dE / (p.kb * T))


def compute_htst() -> None:
    """Define a future operation to be implemented."""
    pass
