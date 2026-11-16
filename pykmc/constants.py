"""Physical constants and unit conversion factors.

Units follow the LAMMPS `metal` convention: eV, ps, amu, Angstrom, K.
"""

import math
from typing import Final


# GENERAL
KB: Final = 8.6173303e-05  # eV.K^-1
"""Boltzmann constant, CODATA 2014 value kept so constant runs stay step-identical."""

H: Final = 4.135667696e-3  # eV.ps
"""Planck constant (CODATA 2018)."""


# FOR HTST(LAMMPS)
# CODATA 2018 values, used only to derive HBAR_OMEGA_EV.
EV_J: Final = 1.602176634e-19  # J
AMU_KG: Final = 1.66053906660e-27  # kg
HBAR_EV_S: Final = 6.582119569e-16  # eV.s

HBAR_OMEGA_EV: Final = HBAR_EV_S * math.sqrt(EV_J / (AMU_KG * 1e-20))
"""`hbar * omega` in eV per `sqrt(eV / (amu A^2))`, about 0.0646541513."""

ESKM_METAL_CONVERSION: Final = 9648.5
"""Factor LAMMPS applies in `dynamical_matrix eskm` with metal units."""
