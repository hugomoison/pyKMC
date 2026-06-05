"""Module defining function used to compute the rate constant."""

from __future__ import annotations

from .config import PhysicalConstants, Config
import math as m


def compute_rate_Eyring(dE: float, config: Config) -> float:
    r"""Compute the rate constant based on the energy barrier and parameters in the configuration.

    It uses the following equation : 
    $$
    k0*e^{-\frac{dE}{k_{b}T}}
    $$

    Parameters
    ----------
    dE : float
        The energy barrier.
    config : Config
        The configuration of the simulation.

    Returns
    -------
    float
        the rate constant.

    """
    p = PhysicalConstants()
    T = config.rateconstant.T
    k0 = config.rateconstant.k0
    return k0 * m.exp(-dE / (p.kb * T))


def compute_rate(dE: float, config: Config, nu0: float | None = None) -> float:
    r"""Compute the rate constant, dispatching on ``rateconstant.style``.

    For ``style == "htst"`` with a finite ``nu0`` (Hz), the Vineyard prefactor
    replaces ``k0``; otherwise it falls back to the constant ``k0``. The Eyring
    exponential ``exp(-dE / (k_b T))`` is unchanged in both cases.

    Parameters
    ----------
    dE : float
        The energy barrier (eV).
    config : Config
        The configuration of the simulation.
    nu0 : float | None
        Vineyard prefactor (Hz) for the 'htst' style. When None or non-finite,
        the constant ``k0`` is used.

    Returns
    -------
    float
        The rate constant (s^-1 under the Hz convention).

    """
    p = PhysicalConstants()
    T = config.rateconstant.T
    if config.rateconstant.style == "htst" and nu0 is not None and m.isfinite(nu0):
        return nu0 * m.exp(-dE / (p.kb * T))
    return config.rateconstant.k0 * m.exp(-dE / (p.kb * T))
