"""Rate-constant computation: pluggable prefactor solvers + Eyring exponential.

The rate is ``k = prefactor * exp(-dE / (k_b T))``. The prefactor is produced by
a :class:`BasePrefactorSolver` so the ``constant`` / ``htst`` / ``rpa`` styles
plug in cleanly:

- ``constant`` : a fixed ``k0``.
- ``htst``     : the per-event Vineyard prefactor ``nu0`` (computed from a LAMMPS
  ``dynamical_matrix`` at search time), with a ``k0`` fallback when ``nu0`` is
  unavailable.
- ``rpa``      : HTST plus the Sharia & Henkelman recrossing correction
  ``kappa``; ``kappa`` is deferred, so this is currently bare-Vineyard
  (identical to ``htst``).

Hold a :class:`RateConstant` (see :func:`make_rate_constant`) and call
``.compute`` when producing many rates; :func:`compute_rate` is a thin shim that
builds one per call for legacy call sites.
"""

from __future__ import annotations

import math as m
from abc import ABC, abstractmethod

from .config import Config, PhysicalConstants


class BasePrefactorSolver(ABC):
    """Strategy returning the rate prefactor (Hz)."""

    @abstractmethod
    def prefactor(self, nu0: float | None = None) -> float:
        """Return the prefactor (Hz); ``nu0`` is the per-event Vineyard value."""


class ConstantPrefactorSolver(BasePrefactorSolver):
    """Fixed prefactor ``k0`` (the classic Eyring/Arrhenius constant)."""

    def __init__(self, k0: float) -> None:
        self.k0 = k0

    def prefactor(self, nu0: float | None = None) -> float:
        """Return the constant ``k0`` (``nu0`` ignored)."""
        return self.k0


class HtstPrefactorSolver(BasePrefactorSolver):
    """Harmonic TST: the per-event Vineyard ``nu0``, with a ``k0`` fallback.

    ``nu0`` is computed once per reference event (LAMMPS ``dynamical_matrix`` ->
    Vineyard) and stored in the catalog; this solver selects it, falling back to
    ``k0`` when ``nu0`` is missing or non-finite.
    """

    def __init__(self, k0: float) -> None:
        self.k0 = k0

    def prefactor(self, nu0: float | None = None) -> float:
        """Return ``nu0`` when finite, else the fallback ``k0``."""
        if nu0 is not None and m.isfinite(nu0):
            return nu0
        return self.k0


class RpaPrefactorSolver(HtstPrefactorSolver):
    """RPA recrossing-corrected prefactor (Sharia & Henkelman 2016).

    The recrossing factor ``kappa`` is deferred (the analysis-side ``kappa_rpa``
    breaks down numerically), so this is currently bare-Vineyard and identical to
    :class:`HtstPrefactorSolver`. Override :meth:`prefactor` to multiply by
    ``kappa`` once it is available.
    """


class RateConstant:
    """Rate constant ``k = solver.prefactor(nu0) * exp(-dE / (k_b T))``."""

    def __init__(self, solver: BasePrefactorSolver, temperature: float) -> None:
        self.solver = solver
        self.temperature = temperature
        self._kb = PhysicalConstants().kb

    def compute(self, dE: float, nu0: float | None = None) -> float:
        """Return the rate (s^-1) for barrier ``dE`` (eV) and Vineyard ``nu0`` (Hz)."""
        prefactor = self.solver.prefactor(nu0)
        return prefactor * m.exp(-dE / (self._kb * self.temperature))


def make_rate_constant(config: Config) -> RateConstant:
    """Build a :class:`RateConstant` from ``config.rateconstant.style``."""
    rc = config.rateconstant
    solvers: dict[str, type[BasePrefactorSolver]] = {
        "constant": ConstantPrefactorSolver,
        "htst": HtstPrefactorSolver,
        "rpa": RpaPrefactorSolver,
    }
    solver_cls = solvers.get(rc.style, ConstantPrefactorSolver)
    return RateConstant(solver_cls(rc.k0), rc.T)


def compute_rate_Eyring(dE: float, config: Config) -> float:
    r"""Constant-prefactor rate ``k = k0 * exp(-dE / (k_b T))`` (legacy helper).

    Parameters
    ----------
    dE : float
        The energy barrier (eV).
    config : Config
        The configuration of the simulation.

    Returns
    -------
    float
        The rate constant.

    """
    p = PhysicalConstants()
    return config.rateconstant.k0 * m.exp(-dE / (p.kb * config.rateconstant.T))


def compute_rate(dE: float, config: Config, nu0: float | None = None) -> float:
    r"""Rate constant dispatching on ``rateconstant.style`` (thin shim).

    Delegates to a :class:`RateConstant` built from ``config``. Prefer holding a
    ``RateConstant`` (see :func:`make_rate_constant`) when computing many rates.

    Parameters
    ----------
    dE : float
        The energy barrier (eV).
    config : Config
        The configuration of the simulation.
    nu0 : float | None
        Vineyard prefactor (Hz) for the 'htst'/'rpa' styles. When None or
        non-finite, the constant ``k0`` is used.

    Returns
    -------
    float
        The rate constant (s^-1 under the Hz convention).

    """
    return make_rate_constant(config).compute(dE, nu0)
