"""Physical constants and unit conversions.

Units
-----
- Mass-weighted Hessian and its eigenvalues `lambda`: eV / (amu A^2).
- `hbar * omega` of a mode: eV, `HBAR_OMEGA_EV * sqrt(lambda)`.
"""

import math
from dataclasses import dataclass


@dataclass
class PhysicalConstants:
    """Store physical constants.

    `kb` keeps its CODATA 2014 value, so that constant runs stay step-identical.
    """

    kb = 8.6173303e-05  # eV.K^-1
    h = 4.135667696e-3  # eV.ps (CODATA 2018)


# CODATA 2018 values, used only to derive HBAR_OMEGA_EV.
EV_J = 1.602176634e-19  # J
AMU_KG = 1.66053906660e-27  # kg
HBAR_EV_S = 6.582119569e-16  # eV.s

HBAR_OMEGA_EV = HBAR_EV_S * math.sqrt(EV_J / (AMU_KG * 1e-20))
"""`hbar * omega` in eV per `sqrt(eV / (amu A^2))`, about 0.0646541513."""

ESKM_METAL_CONVERSION = 9648.5
"""Factor LAMMPS applies in `dynamical_matrix eskm` with metal units.

Dividing an eskm matrix by it gives eV / (amu A^2).
"""
